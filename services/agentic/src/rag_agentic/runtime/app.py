"""The HTTP surface Amazon Bedrock AgentCore Runtime invokes.

AgentCore's container contract is small and fixed:

* ``POST /invocations`` -- the JSON payload the caller sent; return JSON.
* ``GET /ping``         -- health check; must answer ``{"status": "Healthy"}``.
* listen on ``0.0.0.0:8080``, in a ``linux/arm64`` image.

Implementing it as plain FastAPI (rather than through ``BedrockAgentCoreApp``)
keeps the image dependency-light and means the exact same process runs locally
under uvicorn -- ``curl localhost:8080/invocations`` is byte-identical to what
AgentCore will send.

Both roles share this module; ``AGENTIC_ROLE`` picks which agent is built.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from rag_search.facet_filters import split_multi

from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.converse import reset_spent
from rag_agentic.credit_gate import CreditGate, caller_identity_fields
from rag_agentic.factory import AgenticComponents, build_components
from rag_agentic.guardrails import redact
from rag_agentic.index_store import IndexUnavailableError
from rag_agentic.licensing import LicensingDeclined, LicensingUnavailable
from rag_agentic.models import (
    OBJECT_CODES_OPERATION,
    AgentRequest,
    mentions_object_codes,
)
from rag_agentic.search_client import DelegateError, ensure_session_id
from rag_agentic.tenancy import TenancyError
from rag_agentic.token_auth import TokenInvalid, TokenUnverifiable
from rag_core.logging import bind_context, get_logger

_log = get_logger("rag_agentic.runtime")

# AgentCore forwards the caller's session id on this header. Sessions isolate one
# user's runtime state from another's, so it is echoed back on every response.
SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
# Correlates the orchestrator's log lines with the search agent's for one request.
TRACE_HEADER = "X-Amzn-Trace-Id"


def create_app(
    role: AgentRole | None = None, *, components: AgenticComponents | None = None
) -> FastAPI:
    """Build the runtime app.

    ``role`` overrides ``AGENTIC_ROLE`` (used by the two thin entrypoint modules so
    one image can serve either agent). ``components`` injects an already-built
    graph instead of constructing one from the environment -- the seam tests use
    to exercise the real HTTP contract with no AWS behind it.
    """

    settings = AgenticSettings()
    if role is not None:
        settings = settings.model_copy(update={"role": role})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        built = components or build_components(settings)
        app.state.components = built
        _log.info("agentic_runtime_started", role=built.role.value)
        # Live (Mongo) tenants are opened and queried once in the background, so
        # /ping answers at once and the first real request finds a warm pool.
        warming: asyncio.Task[None] | None = None
        warm = getattr(built.index_store, "warm", None)
        if warm is not None and built.settings.mongo_warm_on_start:
            interval = built.settings.mongo_keep_warm_s

            async def keep_warm() -> None:
                # Again every ``mongo_keep_warm_s`` so Atlas keeps the indexes in
                # memory between users; warm() itself logs and never raises.
                while True:
                    await warm()
                    if interval <= 0:
                        return
                    await asyncio.sleep(interval)

            warming = asyncio.create_task(keep_warm())
        try:
            yield
        finally:
            if warming is not None and not warming.done():
                warming.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await warming
            # An injected graph belongs to whoever passed it in; only tear down
            # what this app constructed itself.
            if components is None:
                await built.aclose()
            _log.info("agentic_runtime_stopped", role=built.role.value)

    app = FastAPI(title=f"SpecSafe {settings.role.value} agent", lifespan=lifespan)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        """AgentCore health probe. Must stay cheap -- no AWS calls, no index load."""
        return {"status": "Healthy"}

    @app.post("/invocations")
    async def invocations(
        request: Request,
        session_header: str | None = Header(default=None, alias=SESSION_HEADER),
    ) -> JSONResponse:
        components: AgenticComponents = request.app.state.components
        # Built once here: a payload can be refused before it is ever parsed, and
        # that refusal may still have a reservation to close.
        gate: CreditGate = components.credit_gate or CreditGate(None, components.settings)

        try:
            body: Any = await request.json()
        except Exception:
            return _error(400, "request body must be JSON")
        if not isinstance(body, dict):
            return _error(400, "request body must be a JSON object")

        # The header wins over a body field: it is what AgentCore actually used to
        # isolate this session, so honouring the body instead would split a
        # conversation across two sessions.
        session_id = ensure_session_id(session_header or body.get("session_id"))
        try:
            agent_request = AgentRequest.model_validate({**body, "session_id": session_id})
        except ValidationError as exc:
            # The payload is refused, but the caller may already have reserved
            # credits for it. Close that reservation rather than leave it hanging
            # until it expires, invisible to Licensing.
            await gate.settle_rejected(
                body,
                tenant=_tenant_of_url(components, str(body.get("BaseUrl") or body.get("base_url") or "")),
                trace_id=request.headers.get(TRACE_HEADER) or uuid.uuid4().hex,
                reason="invalid request payload",
            )
            return _error(400, "invalid request payload", detail=exc.errors())

        # SpecSafe asks twice for one question and cannot add a field saying which
        # call this is, so the shape of the payload has to answer it: call 1 names
        # a BusinessObject to enumerate, call 2 carries the codes its ACL allowed.
        # Deciding here means the rest of the request -- the agent, the delegation
        # hop and the settle -- sees an ordinary phase-1 lookup and needs no idea
        # that it was inferred.
        inferred_lookup = (
            components.settings.infer_object_codes_lookup
            and not agent_request.is_object_codes_lookup
            and bool(agent_request.business_object)
            and not mentions_object_codes(body)
        )
        if inferred_lookup:
            agent_request = agent_request.model_copy(
                update={"operation": OBJECT_CODES_OPERATION}
            )

        # One id on every line of this request, so an investigation can follow a
        # single call through routing, delegation, each tool and the answer --
        # across BOTH runtimes, because the orchestrator passes it to the search
        # agent and the search agent binds the one it was given.
        request_id = request.headers.get(TRACE_HEADER) or uuid.uuid4().hex
        bind_context(
            request_id=request_id,
            session_id=session_id,
            agent_role=components.role.value,
            tenant=agent_request.base_url,
            user_code=agent_request.user_code or "",
        )
        _log.info(
            "request_received",
            query=redact(agent_request.query)[:500],
            business_object=agent_request.business_object or "",
            object_codes=len(split_multi(agent_request.object_codes)),
            history_turns=len(agent_request.history),
            # "objectcodes" here may be the caller's own word or a conclusion
            # drawn from the payload shape; no separate key records which, because
            # the log schema is fixed and the distinction is not actionable -- the
            # caller that sends the field explicitly does not exist yet.
            operation=agent_request.operation or "answer",
            # Whether the credit fields arrived, never what they contain. A token in
            # a log line is a usable credential for its five-minute life, so only
            # its presence is recorded -- which is the thing an integration needs
            # to debug anyway.
            has_access_token=bool(agent_request.access_token),
            has_reservation_id=bool((agent_request.reservation_id or "").strip()),
            # The identity fields exactly as the caller spelled and valued them.
            # Parsed values cannot answer "did they send this, or did we lose it?",
            # and that question has already cost a round trip once. Secrets are
            # excluded by caller_identity_fields, never by this call site.
            caller_identity=caller_identity_fields(body),
            payload_keys=sorted(str(k) for k in body),
        )

        # Establish WHO is asking before anything is spent on their behalf. IAM
        # permission to invoke this runtime proves only that the caller may reach
        # us; the access token is what identifies the end user whose credits and
        # whose ACL the rest of this request consumes. So it is checked here --
        # ahead of the reservation and well ahead of the model -- because a request
        # we are going to refuse must not cost the tenant anything first.
        verifier = components.token_verifier
        if verifier is not None and verifier.enabled:
            try:
                # The issuer is chosen by the tenant that base_url resolves to, so
                # resolution has to happen before the token can be checked at all.
                tenant_context = components.resolver.resolve(agent_request.base_url)
            except TenancyError as exc:
                _log.warning("tenancy_rejected", base_url=agent_request.base_url, error=str(exc))
                await gate.settle_rejected(
                    body, tenant="", trace_id=request_id, reason="unknown base_url"
                )
                return _error(400, str(exc), session_id=session_id)

            try:
                caller = await verifier.verify(
                    agent_request.access_token.get_secret_value()
                    if agent_request.access_token
                    else None,
                    tenant=tenant_context,
                    # The identity the payload claims, so it can be held against
                    # the one the token asserts.
                    user_code=agent_request.user_code,
                )
            except TokenInvalid as exc:
                # The caller's reservation is closed on the way out: they paid for
                # a hold on a request we are refusing, and leaving it to expire
                # would hide the refusal from Licensing entirely.
                _log.warning(
                    "auth_rejected",
                    reason=exc.reason,
                    # Which scope or client id actually arrived, not just that one
                    # was wrong -- the difference between a one-line config fix and
                    # an afternoon. None of these messages quote the token itself:
                    # they name the claim that failed, which is already ours.
                    detail=str(exc)[:200],
                    tenant=tenant_context.key,
                    user_code=agent_request.user_code or "",
                )
                await gate.settle_rejected(
                    body, tenant=tenant_context.tenant, trace_id=request_id,
                    reason=f"unauthenticated: {exc.reason}",
                )
                # The reason is named so an integration can tell an expired token
                # from a wrong issuer; the detail is not, because it is only ever
                # read by whoever already has the token.
                return _error(401, "access token rejected", reason=exc.reason,
                              session_id=session_id)
            except TokenUnverifiable as exc:
                # Fail closed. We cannot say who this is, and a verifier that
                # waves the request through when the issuer is unreachable is not
                # a verifier -- anyone able to break that link would make the
                # whole check optional. 503, not 401: the token may be perfectly
                # good, and a retry may well succeed.
                _log.error("auth_unverifiable", error=str(exc), tenant=tenant_context.key)
                await gate.settle_rejected(
                    body, tenant=tenant_context.tenant, trace_id=request_id,
                    reason="token could not be verified",
                )
                return _error(503, "access token could not be verified",
                              session_id=session_id)

            if caller is not None:
                # Bound to every later line of this request, so the audit trail
                # records the identity we PROVED rather than the UserCode the
                # caller merely sent -- which is the one the logs showed before.
                bind_context(**caller.log_fields())
                _log.info(
                    "auth_verified",
                    tenant=tenant_context.key,
                    scopes=",".join(caller.scopes),
                    expires_in_s=max(0, caller.expires_at - int(time.time())),
                    # Which of the token's claims carry the UserCode that arrived.
                    # Names only, no values -- this is how the claim to put in
                    # AGENTIC_AUTH_USER_CODE_CLAIMS is identified from a real
                    # request without logging anyone's identifier. Empty means the
                    # UserCode matched nothing in the token, which is worth knowing
                    # before the check is switched on.
                    user_code_matches=",".join(caller.user_code_claims_matched),
                    user_code_bound=bool(components.settings.auth_user_code_claims),
                )

        # Count this request's spend from zero, so a failure reports its own cost
        # and not the previous request's.
        reset_spent()
        started = time.perf_counter()

        # Credits are reserved before the run and settled after it, so a run that
        # Licensing declines never starts and one that happens is always recorded.
        # The gate is inert unless licensing is configured.
        try:
            hold = await gate.reserve(
                agent_request, tenant=_tenant_of(components, agent_request), trace_id=request_id
            )
        except LicensingDeclined as exc:
            # Not an error on our side: the tenant is out of credits, or this
            # feature is not licensed. The reason is what the caller must act on.
            _log.warning("credit_declined", reason=exc.reason, user_code=agent_request.user_code or "")
            return _error(402, str(exc), reason=exc.reason, session_id=session_id)
        except LicensingUnavailable as exc:
            _log.error("credit_gate_failed", error=str(exc))
            return _error(503, f"credit check unavailable: {exc}", session_id=session_id)

        def _elapsed_ms() -> int:
            return round((time.perf_counter() - started) * 1000)

        try:
            response = await components.agent.handle(agent_request)
        except TenancyError as exc:
            # Caller-supplied base_url does not map to a tenant: their bug, not ours.
            _log.warning("tenancy_rejected", base_url=agent_request.base_url, error=str(exc))
            await gate.settle_failure(hold, execution_time_ms=_elapsed_ms())
            return _error(400, str(exc), session_id=session_id)
        except DelegateError as exc:
            # The search runtime failed; answer with its status, not a generic 500.
            _log.warning("delegate_failed", status=exc.status, error=str(exc))
            await gate.settle_failure(hold, execution_time_ms=_elapsed_ms())
            return _error(exc.status, str(exc), session_id=session_id)
        except IndexUnavailableError as exc:
            # Tenant is known but has no snapshot yet (never indexed, or mid-publish).
            _log.warning("index_unavailable", base_url=agent_request.base_url, error=str(exc))
            await gate.settle_failure(hold, execution_time_ms=_elapsed_ms())
            return _error(503, str(exc), session_id=session_id)
        except Exception as exc:
            _log.exception("agent_failed", role=components.role.value)
            await gate.settle_failure(hold, execution_time_ms=_elapsed_ms())
            return _error(500, f"{type(exc).__name__}: {exc}", session_id=session_id)

        # Telemetry the caller logs against its own request row. Filled here
        # because this is the only layer that sees the whole request -- including
        # the delegation hop, whose tokens the orchestrator has already folded in.
        response.input_tokens = response.usage.input_tokens
        response.output_tokens = response.usage.output_tokens
        response.total_tokens = response.usage.total_tokens
        response.cache_read_tokens = response.usage.cache_read_tokens
        response.cache_write_tokens = response.usage.cache_write_tokens
        response.embedding_tokens = response.usage.embedding_tokens
        response.model_id = response.usage.model
        response.agent_id = _agent_id(components)
        response.duration_ms = round((time.perf_counter() - started) * 1000, 1)

        _log.info(
            "request_completed",
            request_id=request_id,
            citations=len(response.citations),
            documents_returned=len(response.documents),
            tool_calls=len(response.trace),
            answer_chars=len(response.answer),
            agent_id=response.agent_id,
            duration_ms=response.duration_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
            embedding_tokens=response.embedding_tokens,
            model_id=response.model_id,
            user_code=agent_request.user_code or "",
            route=response.route,
        )

        # Settle after the telemetry above is filled, so Licensing is billed the
        # same numbers the caller is shown rather than a second, separate tally.
        if agent_request.is_object_codes_lookup:
            # Phase 1 read facet values: no model, no vector search, nothing to
            # bill. Settling it as a normal run would charge for a free call.
            await gate.settle_lookup(
                hold,
                # No model ran, so name the one that would have: Settle prices
                # against a configured cost and rejects a ModelId it cannot find.
                model_id=getattr(components.model, "model_id", ""),
                agent_id=response.agent_id,
                execution_time_ms=_elapsed_ms(),
            )
        else:
            await gate.settle_success(hold, response)

        payload = response.model_dump(mode="json")
        payload["session_id"] = session_id
        return JSONResponse(payload, headers={SESSION_HEADER: session_id})

    return app


def _agent_id(components: Any) -> str:
    """Which runtime answered.

    The AgentCore runtime ARN when the platform supplies it (that is what a
    CloudWatch log group is named after); otherwise the configured role, so the
    field is never empty in local runs.
    """
    for name in ("AGENT_RUNTIME_ARN", "BEDROCK_AGENTCORE_RUNTIME_ARN", "AGENTIC_AGENT_ID"):
        if value := os.environ.get(name, "").strip():
            return value
    return str(getattr(components.role, "value", components.role))


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    # The status is repeated in the body so it survives a hop through AgentCore,
    # which hands the orchestrator the body of a failed search invocation.
    return JSONResponse({"error": message, "status": status, **extra}, status_code=status)


def _tenant_of_url(components: Any, base_url: str) -> str:
    """Resolve a tenant from a raw base_url string, for the rejection path."""
    try:
        return components.resolver.resolve(base_url).tenant
    except Exception:
        return ""


def _tenant_of(components: Any, request: AgentRequest) -> str:
    """The tenant this request belongs to, for Licensing's TenantCode.

    Resolution can fail for a base_url we do not serve; that is the agent's error
    to report, not the gate's, so an unresolved tenant yields "" and the gate
    simply reserves against the configured default.
    """
    try:
        return components.resolver.resolve(request.base_url).tenant
    except Exception:
        return ""
