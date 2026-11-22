"""Verify the caller's SpecSafe access token before the agent does any work.

The orchestrator's ``/invocations`` is reachable by anything holding AWS IAM
permission for ``InvokeAgentRuntime``. That proves the *caller* is allowed to talk
to the runtime; it says nothing about the *end user* whose question is in the
payload. Everything downstream is spent on that user's behalf -- their tenant's
credits, their ACL-filtered documents -- so the user has to be established before
any of it happens, not inferred from the fact that a payload arrived.

What is checked, and why each one matters:

* **Signature**, RS256 only, against the issuer's published JWKS. The algorithm is
  an allowlist rather than whatever the token's header asks for, which is what
  stops the two classic forgeries: ``alg: none``, and ``alg: HS256`` signed with
  the RSA *public* key as the HMAC secret.
* **Issuer**, resolved from the tenant we already decided from ``base_url`` -- never
  from the token's own ``iss``. This is the check that makes the token tenant-bound:
  without it a genuine, unexpired TENANT_B token would be accepted on a request whose
  ``base_url`` points at TENANT_A, and the holder would read another customer's corpus
  with a token that is valid in every other respect.
* **Expiry** (``exp``/``nbf``), with a little leeway for clock drift between the
  IdP, SpecSafe and this runtime.
* **Scope**, which must contain the API scope SpecSafe's own clients request. A
  token minted for some other scope on the same issuer is a valid token for a
  different job, and reusing it here would be a privilege upgrade.
* **Client id**, when configured, so only SpecSafe's registered clients are
  accepted rather than anything the issuer ever signed.

Failure is split in two on purpose, because the two mean opposite things to whoever
is debugging:

* ``TokenInvalid``   -- the token is bad. 401, and no retry will help.
* ``TokenUnverifiable`` -- we could not reach the issuer to find out. 503, and a
  retry probably will help.

Collapsing them into one status would make an IdP outage look like every user's
credentials expiring at once.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt

from rag_agentic.config import AgenticSettings
from rag_agentic.tenancy import TenantContext
from rag_core.logging import get_logger

_log = get_logger(__name__)

# Only RS256. The issuer's discovery document advertises exactly this
# (`id_token_signing_alg_values_supported: ["RS256"]`), and keeping it a constant
# rather than a setting means no deployment can widen it by accident.
_ALGORITHMS = ("RS256",)


class TokenInvalid(Exception):
    """The token is missing, malformed, expired, or not for us. 401."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason


class TokenUnverifiable(Exception):
    """The token could not be checked because the issuer was unreachable. 503."""


@dataclass(frozen=True, slots=True)
class VerifiedCaller:
    """Who the token says is asking, once the signature has been checked.

    Carried separately from ``AgentRequest`` so there is no way to confuse a claim
    we verified with a field the caller simply sent us.
    """

    subject: str
    client_id: str
    scopes: tuple[str, ...]
    issuer: str
    expires_at: int
    user_name: str = ""
    # Names of the token's claims whose value equals the request's UserCode. Names
    # only, never values: this exists so the right claim can be identified from the
    # logs of a real request without putting anyone's identifier in CloudWatch.
    user_code_claims_matched: tuple[str, ...] = ()

    def log_fields(self) -> dict[str, Any]:
        """Identity for the audit log. No token, no raw claims.

        A token in a log line is a usable credential for as long as it lives, so
        only what it *asserted* is recorded -- which is also the part an
        investigation actually needs.
        """
        return {
            "token_subject": self.subject,
            "token_client_id": self.client_id,
            "token_issuer": self.issuer,
        }


def _strip_bearer(token: str) -> str:
    """Accept both a bare token and a full `Authorization` header value.

    `AgentRequest.access_token` takes `Authorization` as one of its spellings, so
    the value may legitimately arrive with the scheme still attached.
    """
    value = (token or "").strip()
    if value[:7].lower() == "bearer ":
        return value[7:].strip()
    return value


def _claims_matching(claims: dict[str, Any], user_code: str) -> tuple[str, ...]:
    """Every claim whose value equals this UserCode.

    Scans all of them rather than a fixed shortlist, because the claim an IdP maps
    a user code onto is a deployment decision -- often a custom one -- and a
    shortlist that misses it reports "no match" and teaches nobody anything.

    A list-valued claim counts if the UserCode is one of its entries, since that is
    how multi-valued claims arrive.
    """
    wanted = user_code.strip().casefold()
    if not wanted:
        return ()
    matched: list[str] = []
    for name, value in claims.items():
        values = value if isinstance(value, (list, tuple)) else (value,)
        if any(str(item).strip().casefold() == wanted for item in values):
            matched.append(str(name))
    return tuple(sorted(matched))


def _scopes_of(claims: dict[str, Any]) -> tuple[str, ...]:
    """The token's scopes, whichever shape the issuer used.

    IdentityServer writes `scope` as a space-separated string in some
    configurations and as a JSON array in others, and a token is equally valid
    either way -- so both are read rather than one being declared correct.
    """
    raw = claims.get("scope") or claims.get("scp") or ()
    if isinstance(raw, str):
        return tuple(part for part in raw.split() if part)
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw if str(part))
    return ()


@dataclass
class _IssuerKeys:
    """One issuer's signing keys, and when they were last fetched."""

    keys: dict[str, jwt.PyJWK] = field(default_factory=dict)
    fetched_at: float = 0.0
    jwks_uri: str = ""


class JwksCache:
    """The issuers' public signing keys, fetched once and reused.

    Fetching per request would put the IdP in the hot path of every question and
    add its latency to every answer, so keys are cached for ``auth_jwks_ttl_s``.

    A key id this cache has never seen triggers one early refresh, because that is
    what a legitimate key rotation looks like from here. That refresh is rate
    limited: ``kid`` comes from an unverified header, so without a floor on how
    often an unknown one may cause a fetch, anyone who can reach this runtime could
    turn a stream of junk tokens into a denial-of-service attack on the IdP -- and
    the IdP is shared with everything else SpecSafe logs into.
    """

    def __init__(
        self,
        *,
        ttl_s: float,
        timeout_s: float,
        min_refresh_interval_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._ttl_s = ttl_s
        self._timeout_s = timeout_s
        self._min_refresh_interval_s = min_refresh_interval_s
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._cached: dict[str, _IssuerKeys] = {}
        # One lock per issuer: a burst of first requests should cause one fetch,
        # not one per request.
        self._locks: dict[str, asyncio.Lock] = {}

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _lock(self, issuer: str) -> asyncio.Lock:
        return self._locks.setdefault(issuer, asyncio.Lock())

    async def key(self, issuer: str, kid: str) -> jwt.PyJWK:
        """The signing key `kid` for `issuer`, fetching or refreshing as needed."""
        entry = self._cached.get(issuer)
        now = time.monotonic()
        if entry and kid in entry.keys and now - entry.fetched_at < self._ttl_s:
            return entry.keys[kid]

        async with self._lock(issuer):
            # Another request may have refreshed while we waited for the lock.
            entry = self._cached.get(issuer)
            now = time.monotonic()
            if entry and kid in entry.keys and now - entry.fetched_at < self._ttl_s:
                return entry.keys[kid]
            stale = bool(entry) and now - entry.fetched_at >= self._ttl_s
            unknown_kid = bool(entry) and kid not in entry.keys
            if entry is None or stale or (
                unknown_kid and now - entry.fetched_at >= self._min_refresh_interval_s
            ):
                await self._refresh(issuer)
            entry = self._cached.get(issuer)

        if entry and kid in entry.keys:
            return entry.keys[kid]
        # A signing key we do not have, and we are not allowed to ask again yet.
        # Invalid rather than unverifiable: the issuer answered, and this kid was
        # not among the keys it publishes.
        raise TokenInvalid("unknown_signing_key", f"issuer {issuer} does not publish kid {kid!r}")

    async def _refresh(self, issuer: str) -> None:
        jwks_uri = await self._discover(issuer)
        try:
            response = await self._client.get(jwks_uri)
            response.raise_for_status()
            document = response.json()
        except Exception as exc:
            raise TokenUnverifiable(f"could not fetch JWKS from {jwks_uri}: {exc}") from exc

        keys: dict[str, jwt.PyJWK] = {}
        for raw in document.get("keys") or ():
            if not isinstance(raw, dict):
                continue
            kid = str(raw.get("kid") or "")
            if not kid:
                continue
            try:
                keys[kid] = jwt.PyJWK.from_dict(raw)
            except Exception as exc:  # a key we cannot use is not a reason to fail
                _log.warning("jwks_key_unusable", issuer=issuer, kid=kid, error=str(exc)[:200])
        if not keys:
            raise TokenUnverifiable(f"JWKS at {jwks_uri} published no usable keys")

        previous = self._cached.get(issuer)
        self._cached[issuer] = _IssuerKeys(
            keys=keys, fetched_at=time.monotonic(), jwks_uri=jwks_uri
        )
        _log.info(
            "jwks_refreshed",
            issuer=issuer,
            keys=len(keys),
            first_fetch=previous is None,
        )

    async def _discover(self, issuer: str) -> str:
        """The issuer's `jwks_uri`, from its own discovery document.

        Not constructed from the issuer URL: this IdP serves its keys at
        `/.well-known/openid-configuration/jwks`, which is not where the
        conventional guess (`/.well-known/jwks.json`) would look. Asking is also
        the only way to notice that the host answering for this issuer does not
        agree about what it is called.
        """
        cached = self._cached.get(issuer)
        if cached and cached.jwks_uri and time.monotonic() - cached.fetched_at < self._ttl_s:
            return cached.jwks_uri

        url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            document = response.json()
        except Exception as exc:
            raise TokenUnverifiable(f"could not fetch discovery from {url}: {exc}") from exc

        advertised = str(document.get("issuer") or "")
        if advertised.rstrip("/") != issuer.rstrip("/"):
            # The host we were told to trust claims to be a different issuer. That
            # is a misconfiguration on our side or a redirect to somewhere else;
            # either way its keys cannot be used to decide who the user is.
            raise TokenUnverifiable(
                f"discovery at {url} reports issuer {advertised!r}, expected {issuer!r}"
            )
        jwks_uri = str(document.get("jwks_uri") or "")
        if not jwks_uri:
            raise TokenUnverifiable(f"discovery at {url} has no jwks_uri")
        return jwks_uri


class AccessTokenVerifier:
    """Turns the payload's access token into a verified caller, or refuses.

    Inert when ``auth_required`` is false, which is how the offline tests and local
    demos run. There is no middle setting on purpose: a verifier that checked the
    token when one happened to arrive and waved the request through when it did not
    would be satisfied by simply omitting the field.
    """

    def __init__(
        self,
        settings: AgenticSettings,
        *,
        jwks: JwksCache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._jwks = jwks or JwksCache(
            ttl_s=settings.auth_jwks_ttl_s,
            timeout_s=settings.auth_jwks_timeout_s,
            client=client,
        )

    @property
    def enabled(self) -> bool:
        return self._settings.auth_required

    async def aclose(self) -> None:
        await self._jwks.aclose()

    def issuer_for(self, tenant: TenantContext) -> str:
        """The one issuer whose tokens are accepted for this tenant.

        Resolved from configuration keyed by the tenant we already established from
        ``base_url``, so a token is only ever good for the tenant it was minted
        for. Checked in the same order as tenancy itself resolves: exact key
        ``environment/tenant``, then the bare tenant name, then a template.
        """
        mapping = self._settings.auth_issuer_map
        for key in (tenant.key, tenant.tenant, tenant.tenant.lower()):
            if issuer := mapping.get(key):
                return issuer.rstrip("/")
        template = self._settings.auth_issuer_template
        if template:
            return template.format(
                tenant=tenant.tenant, environment=tenant.environment
            ).rstrip("/")
        return ""

    async def verify(
        self,
        token: str | None,
        *,
        tenant: TenantContext,
        user_code: str | None = None,
    ) -> VerifiedCaller | None:
        """Verify the token for this tenant, or raise.

        ``user_code`` is the identity the payload claims. When
        ``auth_user_code_claims`` names the claim that should carry it, the two must
        agree; otherwise the match is only reported, so the right claim can be
        identified from a real request before the check is switched on.

        Returns None only when verification is switched off, so a caller can tell
        "not checked" from "checked and fine" instead of having to assume.
        """
        if not self._settings.auth_required:
            return None

        raw = _strip_bearer(token or "")
        if not raw:
            raise TokenInvalid("missing_token", "no AccessToken in the request")

        issuer = self.issuer_for(tenant)
        if not issuer:
            # Fail closed. With no issuer configured for this tenant there is
            # nothing to check the signature against, and the alternative --
            # trusting the token's own `iss` -- would let the token nominate its
            # own authority, which is not a check at all.
            raise TokenUnverifiable(
                f"no auth issuer configured for tenant {tenant.key!r}; "
                "set AGENTIC_AUTH_ISSUER_MAP or AGENTIC_AUTH_ISSUER_TEMPLATE"
            )

        try:
            header = jwt.get_unverified_header(raw)
        except Exception as exc:
            raise TokenInvalid("malformed_token", str(exc)[:200]) from exc
        kid = str(header.get("kid") or "")
        if not kid:
            raise TokenInvalid("no_kid", "token header carries no key id")

        signing_key = await self._jwks.key(issuer, kid)

        try:
            claims = jwt.decode(
                raw,
                key=signing_key.key,
                algorithms=list(_ALGORITHMS),
                issuer=issuer,
                audience=self._settings.auth_audience or None,
                leeway=self._settings.auth_leeway_s,
                options={
                    "require": ["exp", "iss"],
                    # Only when an audience is configured. This issuer does not put
                    # `aud` on access tokens for a bare API scope, so demanding one
                    # unconditionally would refuse every real token.
                    "verify_aud": bool(self._settings.auth_audience),
                },
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenInvalid("expired_token", str(exc)[:200]) from exc
        except jwt.InvalidIssuerError as exc:
            # The signature held, so this is a real token -- for somewhere else.
            raise TokenInvalid("wrong_issuer", str(exc)[:200]) from exc
        except jwt.InvalidTokenError as exc:
            raise TokenInvalid("invalid_token", str(exc)[:200]) from exc

        scopes = _scopes_of(claims)
        required = self._settings.auth_required_scope
        if required and required.lower() not in {scope.lower() for scope in scopes}:
            raise TokenInvalid(
                "missing_scope", f"token scopes {list(scopes)} do not include {required!r}"
            )

        client_id = str(claims.get("client_id") or claims.get("azp") or "")
        allowed = self._settings.auth_client_ids
        if allowed and client_id not in allowed:
            raise TokenInvalid(
                "unknown_client", f"client_id {client_id!r} is not an accepted client"
            )

        supplied = (user_code or "").strip()
        matched = _claims_matching(claims, supplied)
        required_claims = self._settings.auth_user_code_claims
        if required_claims:
            if not supplied:
                # The check cannot be satisfied by leaving the user out. Same
                # reasoning as auth_required having no "check it if present" mode:
                # a requirement that an omitted field passes is not a requirement.
                raise TokenInvalid(
                    "missing_user_code",
                    "UserCode is required when it must be matched against the token",
                )
            if not set(required_claims) & set(matched):
                # Deliberately does not name the two values. Whoever is debugging
                # has the token; whoever is reading the log should not be handed an
                # identifier, and the claim names are enough to act on.
                raise TokenInvalid(
                    "user_code_mismatch",
                    f"UserCode does not match any of {sorted(required_claims)} "
                    f"(it matched {list(matched)})",
                )

        return VerifiedCaller(
            subject=str(claims.get("sub") or ""),
            client_id=client_id,
            scopes=scopes,
            issuer=issuer,
            expires_at=int(claims.get("exp") or 0),
            user_name=str(claims.get("preferred_username") or claims.get("name") or ""),
            user_code_claims_matched=matched,
        )
