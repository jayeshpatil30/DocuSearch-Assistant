"""The caller's access token decides whether the agent runs at all.

IAM permission to invoke the runtime proves the *caller* may reach us. It says
nothing about the *end user* in the payload, and everything downstream is spent on
that user's behalf -- their tenant's credits, their ACL-filtered documents. So the
token is verified before the reservation and well before the model.

Two groups of tests, and the forgeries matter most:

* A token that is wrong in any way must not start a run. Not a shorter run, not a
  run that fails later -- no model call and no reservation held.
* A token that is *genuine but for somewhere else* must not work either. That is
  the tenant-binding check, and it is the one whose absence would be invisible:
  everything about such a token verifies, so only comparing the issuer against the
  tenant we resolved from base_url catches it.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.converse import ScriptedConverseModel, text_turn, tool_turn
from rag_agentic.credit_gate import CreditGate
from rag_agentic.factory import AgenticComponents
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.runtime.app import create_app
from rag_agentic.tenancy import TenantContext, TenantResolver
from rag_agentic.token_auth import (
    AccessTokenVerifier,
    JwksCache,
    TokenInvalid,
    TokenUnverifiable,
)

from .test_credit_gate import FakeLicensing

CLIENT_ID = "EXAMPLE_CLIENT_ID"
SCOPE = "SpecSafe.Core.Api"
TENANT_B_ISSUER = "https://TENANT_B.sso.example.com"
TENANT_A_ISSUER = "https://TENANT_A.sso.example.com"

# One keypair per issuer, generated once: 2048-bit keygen is slow enough that doing
# it per test would dominate the suite's runtime.
_KEYS: dict[str, Any] = {
    TENANT_B_ISSUER: rsa.generate_private_key(public_exponent=65537, key_size=2048),
    TENANT_A_ISSUER: rsa.generate_private_key(public_exponent=65537, key_size=2048),
}
_KIDS = {TENANT_B_ISSUER: "TENANT_B-key-1", TENANT_A_ISSUER: "TENANT_A-key-1"}


def _jwks_for(issuer: str) -> dict[str, Any]:
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(_KEYS[issuer].public_key()))
    jwk.update({"kid": _KIDS[issuer], "use": "sig", "alg": "RS256"})
    return {"keys": [jwk]}


def _token(
    *,
    issuer: str = TENANT_B_ISSUER,
    scope: str = SCOPE,
    client_id: str = CLIENT_ID,
    expires_in: int = 300,
    kid: str | None = None,
    signing_issuer: str | None = None,
    **extra: Any,
) -> str:
    """A token as this IdP would mint one, with each field overridable.

    `signing_issuer` separates *who signed it* from *what it claims* -- the knob a
    forgery needs and a real token never uses.
    """
    now = int(time.time())
    claims = {
        "iss": issuer,
        "sub": "user-123",
        "client_id": client_id,
        "scope": scope,
        "preferred_username": "auditor@TENANT_B.example",
        "iat": now,
        "nbf": now,
        "exp": now + expires_in,
        **extra,
    }
    signer = signing_issuer or issuer
    return jwt.encode(
        claims, _KEYS[signer], algorithm="RS256", headers={"kid": kid or _KIDS[signer]}
    )


class _IdP:
    """The issuers' discovery and JWKS endpoints, served in-process.

    Counts requests so the caching tests can prove the IdP is not in the hot path
    of every question.
    """

    def __init__(self, *, reachable: bool = True) -> None:
        self.reachable = reachable
        self.discovery_calls = 0
        self.jwks_calls = 0

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if not self.reachable:
            raise httpx.ConnectError("issuer unreachable", request=request)
        url = str(request.url)
        issuer = url.split("/.well-known", 1)[0]
        if url.endswith("/.well-known/openid-configuration"):
            self.discovery_calls += 1
            return httpx.Response(
                200,
                json={
                    "issuer": issuer,
                    "jwks_uri": f"{issuer}/.well-known/openid-configuration/jwks",
                    "id_token_signing_alg_values_supported": ["RS256"],
                },
            )
        if url.endswith("/jwks"):
            self.jwks_calls += 1
            if issuer not in _KEYS:
                return httpx.Response(404)
            return httpx.Response(200, json=_jwks_for(issuer))
        return httpx.Response(404)


def _verifier(idp: _IdP, settings: AgenticSettings, **over: Any) -> AccessTokenVerifier:
    tuned = settings.model_copy(
        update={
            "auth_required": True,
            "auth_issuer_map": {"dev/TENANT_B": TENANT_B_ISSUER, "dev/TENANT_A": TENANT_A_ISSUER},
            "auth_client_ids": [CLIENT_ID],
            **over,
        }
    )
    return AccessTokenVerifier(
        tuned,
        jwks=JwksCache(
            ttl_s=tuned.auth_jwks_ttl_s,
            timeout_s=tuned.auth_jwks_timeout_s,
            client=idp.client(),
        ),
    )


TENANT_B = TenantContext(environment="dev", tenant="TENANT_B", base_url="https://tenant-b.demo.com/")
TENANT_A = TenantContext(environment="dev", tenant="TENANT_A", base_url="https://tenant-a.demo.com/")


# ---- the happy path ---------------------------------------------------------------

async def test_a_genuine_token_identifies_the_caller(agentic_settings: AgenticSettings) -> None:
    caller = await _verifier(_IdP(), agentic_settings).verify(_token(), tenant=TENANT_B)
    assert caller is not None
    assert caller.subject == "user-123"
    assert caller.client_id == CLIENT_ID
    assert caller.issuer == TENANT_B_ISSUER
    assert SCOPE in caller.scopes
    assert caller.user_name == "auditor@TENANT_B.example"


async def test_a_bearer_prefix_is_accepted(agentic_settings: AgenticSettings) -> None:
    """`access_token` also accepts the `Authorization` spelling, scheme included."""
    caller = await _verifier(_IdP(), agentic_settings).verify(
        f"Bearer {_token()}", tenant=TENANT_B
    )
    assert caller is not None


async def test_the_verified_identity_never_carries_the_token(
    agentic_settings: AgenticSettings,
) -> None:
    """A token in a log line is a usable credential for as long as it lives."""
    raw = _token()
    caller = await _verifier(_IdP(), agentic_settings).verify(raw, tenant=TENANT_B)
    assert caller is not None
    assert raw not in json.dumps(caller.log_fields())


# ---- tenant binding: the check whose absence would be invisible -------------------

async def test_another_tenants_valid_token_is_refused_when_keys_differ(
    agentic_settings: AgenticSettings,
) -> None:
    """A real, unexpired, correctly scoped TENANT_B token, presented on a TENANT_A request.

    When the two issuers sign with different keys, the TENANT_B `kid` is simply not in
    TENANT_A's published JWKS, so this is caught at the key lookup -- before any claim is
    read. 401, `unknown_signing_key`.
    """
    tenant_b_token = _token(issuer=TENANT_B_ISSUER)
    with pytest.raises(TokenInvalid) as exc:
        await _verifier(_IdP(), agentic_settings).verify(tenant_b_token, tenant=TENANT_A)
    assert exc.value.reason == "unknown_signing_key"


async def test_another_tenants_valid_token_is_refused_when_they_share_a_key(
    agentic_settings: AgenticSettings,
) -> None:
    """The same cross-tenant attempt, with ONE signing key behind both issuers.

    This is not a hypothetical: as of 2026-11-11, `TENANT_B.sso.example.com` and
    `TENANT_A.sso.example.com` publish the *same* key -- kid
    BDA1EEDC63CBEC4687D937C58B3FAB61, identical modulus -- and resolve to the same
    addresses. One IdentityServer deployment, several issuer hostnames.

    So key separation catches nothing here. The signature on a TENANT_B token verifies
    perfectly against TENANT_A's JWKS, because it is the same key.

    What stops it is comparing `iss` against the tenant resolved from `base_url`.
    This test is the reason that check cannot be dropped: without it, a valid TENANT_B
    token reads TENANT_A's corpus and every other check still passes.
    """
    idp = _IdP()
    original = idp._handle

    def one_key_for_both(request: httpx.Request) -> httpx.Response:
        """Serve TENANT_B's key under TENANT_A's kid as well, as a shared-key IdP would."""
        url = str(request.url)
        if url.endswith("/jwks") and url.startswith(TENANT_A_ISSUER):
            jwk = json.loads(
                jwt.algorithms.RSAAlgorithm.to_jwk(_KEYS[TENANT_B_ISSUER].public_key())
            )
            jwk.update({"kid": _KIDS[TENANT_B_ISSUER], "use": "sig", "alg": "RS256"})
            return httpx.Response(200, json={"keys": [jwk]})
        return original(request)

    verifier = _verifier(idp, agentic_settings)
    verifier._jwks._client = httpx.AsyncClient(
        transport=httpx.MockTransport(one_key_for_both)
    )

    with pytest.raises(TokenInvalid) as exc:
        await verifier.verify(_token(issuer=TENANT_B_ISSUER), tenant=TENANT_A)
    assert exc.value.reason == "wrong_issuer"


async def test_the_matching_case_is_simply_allowed(
    agentic_settings: AgenticSettings,
) -> None:
    """A TENANT_B token on a TENANT_B request: nothing is flagged, the run proceeds.

    Spelled out beside the refusals because "what does the good case do" is the
    other half of the question, and a check that refuses the right things while
    quietly breaking the normal path is not usable.
    """
    verifier = _verifier(_IdP(), agentic_settings)
    caller = await verifier.verify(_token(issuer=TENANT_B_ISSUER), tenant=TENANT_B)
    assert caller is not None
    assert caller.issuer == TENANT_B_ISSUER
    # And the same token on the tenant it belongs to, for TENANT_A, with its own key.
    tenant_a_caller = await verifier.verify(_token(issuer=TENANT_A_ISSUER), tenant=TENANT_A)
    assert tenant_a_caller is not None
    assert tenant_a_caller.issuer == TENANT_A_ISSUER


async def test_the_issuer_comes_from_config_not_from_the_token(
    agentic_settings: AgenticSettings,
) -> None:
    """Claiming to be TENANT_B while signed by TENANT_A must not help.

    If the expected issuer were read from `iss`, the token would nominate its own
    authority -- which is not a check.
    """
    forged = _token(issuer=TENANT_B_ISSUER, signing_issuer=TENANT_A_ISSUER)
    with pytest.raises(TokenInvalid):
        await _verifier(_IdP(), agentic_settings).verify(forged, tenant=TENANT_B)


async def test_a_tenant_with_no_configured_issuer_fails_closed(
    agentic_settings: AgenticSettings,
) -> None:
    """No issuer configured means nothing to verify against, so nothing is run."""
    verifier = _verifier(_IdP(), agentic_settings, auth_issuer_map={})
    with pytest.raises(TokenUnverifiable):
        await verifier.verify(_token(), tenant=TENANT_B)


async def test_the_issuer_template_covers_tenants_the_map_does_not(
    agentic_settings: AgenticSettings,
) -> None:
    verifier = _verifier(
        _IdP(),
        agentic_settings,
        auth_issuer_map={},
        auth_issuer_template="https://{tenant}.sso.example.com",
    )
    assert verifier.issuer_for(TENANT_B) == TENANT_B_ISSUER
    assert verifier.issuer_for(TENANT_A) == TENANT_A_ISSUER
    assert (await verifier.verify(_token(), tenant=TENANT_B)) is not None


# ---- forgeries --------------------------------------------------------------------

async def test_an_unsigned_token_is_refused(agentic_settings: AgenticSettings) -> None:
    """`alg: none`: the oldest JWT forgery there is."""
    now = int(time.time())
    unsigned = jwt.encode(
        {"iss": TENANT_B_ISSUER, "sub": "attacker", "scope": SCOPE,
         "client_id": CLIENT_ID, "exp": now + 300},
        key="",
        algorithm="none",
        headers={"kid": _KIDS[TENANT_B_ISSUER]},
    )
    with pytest.raises(TokenInvalid):
        await _verifier(_IdP(), agentic_settings).verify(unsigned, tenant=TENANT_B)


async def test_a_token_hmac_signed_with_the_public_key_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    """Algorithm confusion: the RSA public key is published, so if HS256 were
    accepted anyone could read it from the JWKS and sign their own tokens.

    Assembled by hand because PyJWT refuses to *encode* this -- it rejects an
    asymmetric key as an HMAC secret. That refusal protects whoever signs a token,
    not whoever verifies one, so it says nothing about our side and the attack
    still has to be built and presented.
    """
    import hashlib
    import hmac

    from cryptography.hazmat.primitives import serialization

    public_pem = _KEYS[TENANT_B_ISSUER].public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    now = int(time.time())
    header = jwt.utils.base64url_encode(
        json.dumps({"alg": "HS256", "typ": "JWT", "kid": _KIDS[TENANT_B_ISSUER]}).encode()
    )
    payload = jwt.utils.base64url_encode(
        json.dumps({"iss": TENANT_B_ISSUER, "sub": "attacker", "scope": SCOPE,
                    "client_id": CLIENT_ID, "exp": now + 300}).encode()
    )
    signed = header + b"." + payload
    signature = jwt.utils.base64url_encode(
        hmac.new(public_pem, signed, hashlib.sha256).digest()
    )
    forged = (signed + b"." + signature).decode()

    with pytest.raises(TokenInvalid):
        await _verifier(_IdP(), agentic_settings).verify(forged, tenant=TENANT_B)


async def test_a_tampered_payload_is_refused(agentic_settings: AgenticSettings) -> None:
    """Editing a claim invalidates the signature over it."""
    header, _original_payload, signature = _token().split(".")
    other = jwt.utils.base64url_encode(
        json.dumps({"iss": TENANT_B_ISSUER, "sub": "someone-else", "scope": SCOPE,
                    "client_id": CLIENT_ID, "exp": int(time.time()) + 300}).encode()
    ).decode()
    with pytest.raises(TokenInvalid):
        await _verifier(_IdP(), agentic_settings).verify(
            f"{header}.{other}.{signature}", tenant=TENANT_B
        )


@pytest.mark.parametrize(
    "value", ["", "   ", "not-a-jwt", "a.b.c", "Bearer ", None]
)
async def test_a_missing_or_malformed_token_is_refused(
    agentic_settings: AgenticSettings, value: str | None
) -> None:
    with pytest.raises(TokenInvalid):
        await _verifier(_IdP(), agentic_settings).verify(value, tenant=TENANT_B)


async def test_an_expired_token_is_refused(agentic_settings: AgenticSettings) -> None:
    with pytest.raises(TokenInvalid) as exc:
        await _verifier(_IdP(), agentic_settings).verify(
            _token(expires_in=-120), tenant=TENANT_B
        )
    assert exc.value.reason == "expired_token"


async def test_leeway_does_not_stretch_far(agentic_settings: AgenticSettings) -> None:
    """Clock drift is tolerated; a token expired by minutes is not.

    The leeway extends the life of every expired token by exactly its value, so
    this pins that it stays small.
    """
    verifier = _verifier(_IdP(), agentic_settings, auth_leeway_s=60.0)
    assert (await verifier.verify(_token(expires_in=-30), tenant=TENANT_B)) is not None
    with pytest.raises(TokenInvalid):
        await verifier.verify(_token(expires_in=-600), tenant=TENANT_B)


async def test_a_key_the_issuer_does_not_publish_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    with pytest.raises(TokenInvalid) as exc:
        await _verifier(_IdP(), agentic_settings).verify(
            _token(kid="rotated-away"), tenant=TENANT_B
        )
    assert exc.value.reason == "unknown_signing_key"


async def test_a_token_without_a_key_id_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    now = int(time.time())
    no_kid = jwt.encode(
        {"iss": TENANT_B_ISSUER, "sub": "x", "scope": SCOPE, "exp": now + 300},
        _KEYS[TENANT_B_ISSUER],
        algorithm="RS256",
    )
    with pytest.raises(TokenInvalid) as exc:
        await _verifier(_IdP(), agentic_settings).verify(no_kid, tenant=TENANT_B)
    assert exc.value.reason == "no_kid"


# ---- scope and client -------------------------------------------------------------

async def test_a_token_for_a_different_scope_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    """Valid on the same issuer, minted for another job. Accepting it would be a
    privilege upgrade."""
    with pytest.raises(TokenInvalid) as exc:
        await _verifier(_IdP(), agentic_settings).verify(
            _token(scope="openid profile PIFExchange"), tenant=TENANT_B
        )
    assert exc.value.reason == "missing_scope"


async def test_scope_is_read_whether_it_is_a_string_or_a_list(
    agentic_settings: AgenticSettings,
) -> None:
    """IdentityServer writes `scope` both ways, and a token is valid either way."""
    verifier = _verifier(_IdP(), agentic_settings)
    assert (await verifier.verify(_token(scope=f"openid {SCOPE}"), tenant=TENANT_B)) is not None
    as_list = _token(scope=["openid", SCOPE])
    assert (await verifier.verify(as_list, tenant=TENANT_B)) is not None


async def test_an_unregistered_client_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    with pytest.raises(TokenInvalid) as exc:
        await _verifier(_IdP(), agentic_settings).verify(
            _token(client_id="SomeOtherApp"), tenant=TENANT_B
        )
    assert exc.value.reason == "unknown_client"


async def test_any_client_is_accepted_when_none_are_configured(
    agentic_settings: AgenticSettings,
) -> None:
    """Weaker, and the default -- so the config comment has to say so."""
    verifier = _verifier(_IdP(), agentic_settings, auth_client_ids=[])
    assert (await verifier.verify(_token(client_id="Anything"), tenant=TENANT_B)) is not None


# ---- the issuer is not in the hot path --------------------------------------------

async def test_signing_keys_are_fetched_once_and_reused(
    agentic_settings: AgenticSettings,
) -> None:
    """Fetching per request would add the IdP's latency to every answer."""
    idp = _IdP()
    verifier = _verifier(idp, agentic_settings)
    for _ in range(5):
        assert (await verifier.verify(_token(), tenant=TENANT_B)) is not None
    assert idp.jwks_calls == 1
    assert idp.discovery_calls == 1


async def test_an_unknown_key_id_does_not_let_anyone_hammer_the_issuer(
    agentic_settings: AgenticSettings,
) -> None:
    """`kid` comes from an unverified header.

    One unknown kid triggers one refresh, because that is what a key rotation looks
    like from here. A stream of them must not become a denial-of-service attack on
    an IdP that everything else at SpecSafe signs in through.
    """
    idp = _IdP()
    verifier = _verifier(idp, agentic_settings)
    await verifier.verify(_token(), tenant=TENANT_B)   # warms the cache
    assert idp.jwks_calls == 1
    for index in range(20):
        with pytest.raises(TokenInvalid):
            await verifier.verify(_token(kid=f"junk-{index}"), tenant=TENANT_B)
    assert idp.jwks_calls == 1


async def test_an_unreachable_issuer_is_unverifiable_not_invalid(
    agentic_settings: AgenticSettings,
) -> None:
    """Different meanings, different statuses: an IdP outage is not every user's
    credentials expiring at once."""
    verifier = _verifier(_IdP(reachable=False), agentic_settings)
    with pytest.raises(TokenUnverifiable):
        await verifier.verify(_token(), tenant=TENANT_B)


async def test_an_issuer_that_disagrees_about_its_own_name_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    """The host answering for this issuer claims to be someone else.

    A redirect, or our own misconfiguration. Either way its keys cannot be used to
    decide who the user is.
    """
    idp = _IdP()
    original = idp._handle

    def handle(request: httpx.Request) -> httpx.Response:
        response = original(request)
        if str(request.url).endswith("openid-configuration"):
            body = response.json()
            body["issuer"] = "https://somewhere-else.example"
            return httpx.Response(200, json=body)
        return response

    verifier = _verifier(idp, agentic_settings)
    verifier._jwks._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    with pytest.raises(TokenUnverifiable):
        await verifier.verify(_token(), tenant=TENANT_B)


async def test_verification_is_off_until_it_is_switched_on(
    agentic_settings: AgenticSettings,
) -> None:
    """Returns None rather than a caller, so "not checked" cannot be mistaken for
    "checked and fine"."""
    verifier = AccessTokenVerifier(agentic_settings.model_copy(
        update={"auth_required": False}
    ))
    assert verifier.enabled is False
    assert (await verifier.verify(None, tenant=TENANT_B)) is None


# ---- end to end through the runtime -----------------------------------------------

@pytest.fixture
def harness(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> Any:
    harness_tenant_key = resolver.resolve(base_url).key
    def build(**over: Any) -> tuple[TestClient, ScriptedConverseModel, FakeLicensing, _IdP]:
        model = ScriptedConverseModel(
            [
                tool_turn("search_documents", {"query": "ISO 9001"}),
                text_turn("Suppliers must hold a valid ISO 9001 certificate."),
            ]
        )
        settings = agentic_settings.model_copy(
            update={
                "licensing_enabled": True,
                "licensing_base_url": "https://tenantadmin.example.com",
                "licensing_tenant_codes": "TENANT_A=TENANT_A,TENANT_B=TENANT_B",
                "auth_required": True,
                # Keyed by whatever the fixture's base_url resolves to, so the
                # fixture stays the single source of truth for the tenant.
                "auth_issuer_map": {harness_tenant_key: TENANT_B_ISSUER},
                "auth_client_ids": [CLIENT_ID],
                **over,
            }
        )
        idp = _IdP(reachable=over.pop("_idp_reachable", True))
        licensing = FakeLicensing()
        agent = DocumentSearchAgent(
            settings=settings, model=model, embedder=embedder,
            index_store=index_store, resolver=resolver,
        )
        components = AgenticComponents(
            settings=settings, role=AgentRole.DOCUMENT_SEARCH, agent=agent,
            resolver=resolver, model=model,
            credit_gate=CreditGate(licensing, settings),
            token_verifier=AccessTokenVerifier(
                settings,
                jwks=JwksCache(ttl_s=3600.0, timeout_s=5.0, client=idp.client()),
            ),
        )
        app = create_app(AgentRole.DOCUMENT_SEARCH, components=components)
        return TestClient(app), model, licensing, idp

    return build


def _body(base_url: str, token: str | None, **over: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "baseUrl": base_url,
        "Query": "What does ISO 9001 require of suppliers?",
        "UserCode": "U-1",
        "ReservationId": "res-1",
        **over,
    }
    if token is not None:
        payload["AccessToken"] = token
    return payload


def test_a_verified_request_is_answered(harness: Any, base_url: str) -> None:
    client, model, _licensing, _idp = harness()
    with client:
        response = client.post("/invocations", json=_body(base_url, _token()))
    assert response.status_code == 200
    assert "ISO 9001" in response.json()["answer"]
    assert len(model.calls) > 0


def test_an_unauthenticated_request_never_reaches_the_model(
    harness: Any, base_url: str
) -> None:
    """The point of the whole exercise: refused before anything is spent."""
    client, model, _licensing, _idp = harness()
    with client:
        response = client.post("/invocations", json=_body(base_url, None))
    assert response.status_code == 401
    assert response.json()["reason"] == "missing_token"
    assert model.calls == []


def test_an_expired_token_never_reaches_the_model(harness: Any, base_url: str) -> None:
    client, model, _licensing, _idp = harness()
    with client:
        response = client.post(
            "/invocations", json=_body(base_url, _token(expires_in=-300))
        )
    assert response.status_code == 401
    assert response.json()["reason"] == "expired_token"
    assert model.calls == []


def test_a_refused_request_closes_the_callers_reservation(
    harness: Any, base_url: str
) -> None:
    """They paid for a hold on a request we are refusing. Leaving it to expire
    would hide the refusal from Licensing entirely."""
    client, _model, licensing, _idp = harness()
    with client:
        client.post("/invocations", json=_body(base_url, None))
    assert len(licensing.settles) == 1
    assert licensing.settles[0]["reservation_id"] == "res-1"


def test_an_unverifiable_token_is_a_503_and_runs_nothing(
    harness: Any, base_url: str
) -> None:
    client, model, _licensing, idp = harness()
    idp.reachable = False
    with client:
        response = client.post("/invocations", json=_body(base_url, _token()))
    assert response.status_code == 503
    assert model.calls == []


def test_the_response_never_echoes_the_token(harness: Any, base_url: str) -> None:
    client, _model, _licensing, _idp = harness()
    raw = _token()
    with client:
        response = client.post("/invocations", json=_body(base_url, raw))
    assert raw not in response.text


def test_with_auth_off_a_request_without_a_token_still_works(
    harness: Any, base_url: str
) -> None:
    """How the offline tests and local demos run -- and what every other test in
    this suite depends on."""
    client, model, _licensing, _idp = harness(auth_required=False)
    with client:
        response = client.post("/invocations", json=_body(base_url, None))
    assert response.status_code == 200
    assert len(model.calls) > 0


# ---- the misconfigurations that would silently reopen the hole --------------------
#
# Cross-tenant tokens are refused because each tenant resolves to its own issuer.
# That makes the issuer config load-bearing: get it wrong in one specific way and
# every check still passes while tenant isolation is gone. These are refused at
# startup rather than left to be discovered in production.

def test_an_issuer_template_without_a_tenant_placeholder_is_refused() -> None:
    """One issuer for every tenant means one tenant's token works for all."""
    with pytest.raises(ValueError, match=r"\{tenant\}"):
        AgenticSettings(
            auth_required=True,
            auth_issuer_template="https://sso.example.com",
        )


def test_an_issuer_template_with_the_placeholder_is_accepted() -> None:
    settings = AgenticSettings(
        auth_required=True,
        auth_issuer_template="https://{tenant}.sso.example.com",
    )
    assert settings.auth_issuer_template == "https://{tenant}.sso.example.com"


def test_two_tenants_sharing_one_issuer_is_refused() -> None:
    """The map version of the same mistake."""
    with pytest.raises(ValueError, match="same issuer"):
        AgenticSettings(
            auth_required=True,
            auth_issuer_map={"dev/TENANT_B": TENANT_B_ISSUER, "dev/TENANT_A": TENANT_B_ISSUER},
        )


def test_two_spellings_of_one_tenant_may_share_an_issuer() -> None:
    """`issuer_for` tries "environment/tenant" and then the bare tenant name, so
    both spellings of the same tenant pointing at one issuer is correct -- the
    guard compares tenants, not keys."""
    settings = AgenticSettings(
        auth_required=True,
        auth_issuer_map={"dev/TENANT_B": TENANT_B_ISSUER, "TENANT_B": TENANT_B_ISSUER},
    )
    assert len(settings.auth_issuer_map) == 2


def test_requiring_auth_with_no_issuer_at_all_is_refused() -> None:
    """Every request would fail closed, so there is nothing to serve. Say so at
    startup instead of 503-ing each caller in turn."""
    with pytest.raises(ValueError, match="no issuer is configured"):
        AgenticSettings(auth_required=True)


def test_auth_off_needs_no_issuer() -> None:
    """How the offline tests and local demos run."""
    assert AgenticSettings(auth_required=False).auth_issuer_template == ""


# ---- binding the payload's UserCode to the token ---------------------------------
#
# The signature proves that *a* user signed in. UserCode is what the audit trail and
# the credit record are attributed to, and it is whatever the caller typed -- so
# without this, a valid token for user A can be sent with `UserCode: B`.

async def test_a_user_code_matching_the_named_claim_is_accepted(
    agentic_settings: AgenticSettings,
) -> None:
    verifier = _verifier(
        _IdP(), agentic_settings, auth_user_code_claims=["preferred_username"]
    )
    caller = await verifier.verify(
        _token(), tenant=TENANT_B, user_code="auditor@TENANT_B.example"
    )
    assert caller is not None
    assert "preferred_username" in caller.user_code_claims_matched


async def test_a_user_code_the_token_does_not_assert_is_refused(
    agentic_settings: AgenticSettings,
) -> None:
    """The actual attack: a genuine token, someone else's UserCode."""
    verifier = _verifier(
        _IdP(), agentic_settings, auth_user_code_claims=["preferred_username"]
    )
    with pytest.raises(TokenInvalid) as exc:
        await verifier.verify(_token(), tenant=TENANT_B, user_code="someone-else@TENANT_B.example")
    assert exc.value.reason == "user_code_mismatch"


async def test_the_check_cannot_be_satisfied_by_omitting_the_user_code(
    agentic_settings: AgenticSettings,
) -> None:
    """A requirement that an omitted field passes is not a requirement."""
    verifier = _verifier(
        _IdP(), agentic_settings, auth_user_code_claims=["preferred_username"]
    )
    for empty in (None, "", "   "):
        with pytest.raises(TokenInvalid) as exc:
            await verifier.verify(_token(), tenant=TENANT_B, user_code=empty)
        assert exc.value.reason == "missing_user_code"


async def test_any_one_of_several_named_claims_may_carry_it(
    agentic_settings: AgenticSettings,
) -> None:
    """Every candidate is asserted by the IdP, so the caller cannot pick which one
    to satisfy -- listing several is for rollout, not a weakening."""
    verifier = _verifier(
        _IdP(), agentic_settings, auth_user_code_claims=["sub", "preferred_username"]
    )
    assert (await verifier.verify(_token(), tenant=TENANT_B, user_code="user-123")) is not None
    assert (
        await verifier.verify(_token(), tenant=TENANT_B, user_code="auditor@TENANT_B.example")
    ) is not None


async def test_a_custom_claim_is_found_too(agentic_settings: AgenticSettings) -> None:
    """The claim an IdP maps a user code onto is a deployment decision, often a
    custom one, so the scan covers every claim rather than a shortlist."""
    verifier = _verifier(
        _IdP(), agentic_settings, auth_user_code_claims=["specsafe_user_code"]
    )
    token = _token(specsafe_user_code="U-1")
    caller = await verifier.verify(token, tenant=TENANT_B, user_code="U-1")
    assert caller is not None
    assert caller.user_code_claims_matched == ("specsafe_user_code",)


async def test_a_multi_valued_claim_counts_if_the_code_is_one_of_its_entries(
    agentic_settings: AgenticSettings,
) -> None:
    verifier = _verifier(_IdP(), agentic_settings, auth_user_code_claims=["codes"])
    token = _token(codes=["U-9", "U-1"])
    assert (await verifier.verify(token, tenant=TENANT_B, user_code="U-1")) is not None


async def test_matching_is_case_insensitive(agentic_settings: AgenticSettings) -> None:
    """An IdP may normalise case, and two real user codes differing only by case
    would be a dangerous design in itself."""
    verifier = _verifier(
        _IdP(), agentic_settings, auth_user_code_claims=["specsafe_user_code"]
    )
    token = _token(specsafe_user_code="u-1")
    assert (await verifier.verify(token, tenant=TENANT_B, user_code="U-1")) is not None


async def test_the_matching_claims_are_reported_before_the_check_is_switched_on(
    agentic_settings: AgenticSettings,
) -> None:
    """How the right claim gets identified: run real traffic with the check off and
    read the claim NAMES out of `auth_verified`. No values are ever logged."""
    verifier = _verifier(_IdP(), agentic_settings, auth_user_code_claims=[])
    caller = await verifier.verify(
        _token(specsafe_user_code="U-1"), tenant=TENANT_B, user_code="U-1"
    )
    assert caller is not None
    assert caller.user_code_claims_matched == ("specsafe_user_code",)
    # Reported, not enforced: nothing was refused.
    mismatched = await verifier.verify(_token(), tenant=TENANT_B, user_code="whoever")
    assert mismatched is not None
    assert mismatched.user_code_claims_matched == ()


async def test_no_user_code_is_fine_while_the_check_is_off(
    agentic_settings: AgenticSettings,
) -> None:
    verifier = _verifier(_IdP(), agentic_settings, auth_user_code_claims=[])
    caller = await verifier.verify(_token(), tenant=TENANT_B, user_code=None)
    assert caller is not None
    assert caller.user_code_claims_matched == ()


def test_a_mismatched_user_code_never_reaches_the_model(
    harness: Any, base_url: str
) -> None:
    client, model, _licensing, _idp = harness(
        auth_user_code_claims=["preferred_username"]
    )
    with client:
        response = client.post(
            "/invocations", json=_body(base_url, _token(), UserCode="not-the-token-user")
        )
    assert response.status_code == 401
    assert response.json()["reason"] == "user_code_mismatch"
    assert model.calls == []


def test_a_matching_user_code_is_answered(harness: Any, base_url: str) -> None:
    client, model, _licensing, _idp = harness(
        auth_user_code_claims=["preferred_username"]
    )
    with client:
        response = client.post(
            "/invocations",
            json=_body(base_url, _token(), UserCode="auditor@TENANT_B.example"),
        )
    assert response.status_code == 200
    assert len(model.calls) > 0
