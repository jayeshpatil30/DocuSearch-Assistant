"""Reserve before the run, settle after -- including when things go wrong.

The interesting cases are all failures:

* Licensing says no          -> the run must not start, and the reason must survive.
* Licensing is unreachable   -> the run must not start either, unless an operator
                                has explicitly chosen availability over metering.
* Settle fails after the run -> the answer must still be returned. The work is
                                done and the user is owed it; an unsettled
                                reservation expires on its own.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import reset_spent
from rag_agentic.credit_gate import CreditGate
from rag_agentic.licensing import (
    LicensingDeclined,
    LicensingUnavailable,
    Reservation,
    ReservationCheck,
    ReserveOutcome,
    RunTelemetry,
)
from rag_agentic.models import AgentRequest, AgentResponse, Usage
from rag_search.embedder import reset_query_embedding_tokens


class FakeLicensing:
    """Records what Licensing was asked, and can be told to misbehave."""

    def __init__(
        self,
        *,
        allowed: bool = True,
        reason: str = "",
        reserve_raises: Exception | None = None,
        settle_raises: Exception | None = None,
        reservation_id: str = "res-1",
        lookup: ReservationCheck | None = None,
        lookup_raises: Exception | None = None,
    ) -> None:
        self._lookup = lookup or ReservationCheck(
            id=reservation_id, status="Reserved", usable=True
        )
        self._lookup_raises = lookup_raises
        self._allowed = allowed
        self._reason = reason
        self._reserve_raises = reserve_raises
        self._settle_raises = settle_raises
        self._reservation_id = reservation_id
        self.reserves: list[dict[str, Any]] = []
        self.lookups: list[dict[str, Any]] = []
        self.settles: list[dict[str, Any]] = []
        self.releases: list[dict[str, Any]] = []

    async def check_and_reserve(self, **kw: Any) -> ReserveOutcome:
        self.reserves.append(kw)
        if self._reserve_raises:
            raise self._reserve_raises
        if not self._allowed:
            return ReserveOutcome(allowed=False, reason=self._reason, message="no")
        return ReserveOutcome(
            allowed=True,
            reservation=Reservation(id=self._reservation_id, estimated_credits=5),
            estimated_credits=5,
        )

    async def settle(self, **kw: Any) -> dict[str, Any]:
        self.settles.append(kw)
        if self._settle_raises:
            raise self._settle_raises
        return {"Status": "Committed", "ActualCredits": 5}

    async def release(self, **kw: Any) -> dict[str, Any]:
        self.releases.append(kw)
        return {"Status": "Released"}

    async def get_reservation(self, **kw: Any) -> ReservationCheck:
        self.lookups.append(kw)
        if self._lookup_raises:
            raise self._lookup_raises
        return self._lookup

    async def aclose(self) -> None:
        return None


def _settings(**over: Any) -> AgenticSettings:
    return AgenticSettings(
        licensing_enabled=True, licensing_base_url="https://tenantadmin.example.com",
        licensing_tenant_codes="TENANT_A=TENANT_A,TENANT_B=TENANT_B", **over,
    )


def _request(**over: Any) -> AgentRequest:
    return AgentRequest(base_url="https://tenant-a.demo.com/", query="how many documents?", **over)


def _response() -> AgentResponse:
    response = AgentResponse(answer="1,359.", route="document_search",
                             environment="demo", tenant="TENANT_A")
    response.usage = Usage(input_tokens=3322, output_tokens=51, cache_read_tokens=1955,
                           embedding_tokens=7, model="claude", llm_calls=2)
    response.duration_ms = 29760.4
    response.model_id = "claude"
    response.agent_id = "orchestrator"
    return response


# ---- the happy path ---------------------------------------------------------------


async def test_a_run_reserves_then_settles_success() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="trace-1")
    await gate.settle_success(hold, _response())

    assert len(licensing.reserves) == 1
    [settle] = licensing.settles
    assert settle["result"] == "Success"
    assert settle["reservation_id"] == "res-1"
    assert settle["telemetry"].input_tokens == 3322
    assert settle["telemetry"].embedding_tokens == 7


async def test_the_tenant_code_is_the_one_licensing_knows() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert licensing.reserves[0]["tenant_code"] == "TENANT_A"


async def test_the_request_feature_code_wins_over_the_default() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings(licensing_feature_code="DOCUMENT_SEARCH"))

    await gate.reserve(_request(feature_code="OBJECT_COMPARE"), tenant="TENANT_A", trace_id="t")

    assert licensing.reserves[0]["feature_code"] == "OBJECT_COMPARE"


async def test_the_trace_id_ties_the_settle_to_the_request() -> None:
    """TraceId points back at the request; TransactionId identifies the settle.

    They are deliberately different values. Licensing stores TransactionId as the
    identity of the settle transaction, and one request can settle more than once
    -- so sharing the trace id between them would make two settles
    indistinguishable in the ledger.
    """
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="trace-42")
    await gate.settle_success(hold, _response())

    settle = licensing.settles[0]
    assert settle["trace_id"] == "trace-42", "the settle must point back at the request"
    assert settle["transaction_id"] != "trace-42", "the settle needs its own identity"
    assert settle["transaction_id"]


async def test_each_attempt_reserves_under_its_own_key() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    await gate.reserve(_request(), tenant="TENANT_A", trace_id="t1")
    await gate.reserve(_request(), tenant="TENANT_A", trace_id="t2")

    keys = [r["idempotency_key"] for r in licensing.reserves]
    assert keys[0] != keys[1], "a second attempt reusing the key reuses the reservation"


# ---- declined ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    ["insufficient_credits", "no_active_license", "no_credit_config",
     "invalid_license_unit", "validation_error", "declined"],
)
async def test_a_decline_stops_the_run_and_keeps_its_reason(reason: str) -> None:
    gate = CreditGate(FakeLicensing(allowed=False, reason=reason), _settings())

    with pytest.raises(LicensingDeclined) as caught:
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert caught.value.reason == reason


async def test_a_decline_settles_nothing() -> None:
    """There is no reservation to settle, and no run happened."""
    licensing = FakeLicensing(allowed=False, reason="insufficient_credits")
    gate = CreditGate(licensing, _settings())

    with pytest.raises(LicensingDeclined):
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert licensing.settles == []


async def test_allowed_without_a_reservation_is_treated_as_a_failure() -> None:
    """Running on a reservation that does not exist would be unbillable work."""

    class NoReservation(FakeLicensing):
        async def check_and_reserve(self, **kw: Any) -> ReserveOutcome:
            return ReserveOutcome(allowed=True, reservation=None)

    gate = CreditGate(NoReservation(), _settings())

    with pytest.raises(LicensingUnavailable):
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")


# ---- Licensing unreachable --------------------------------------------------------


async def test_an_unreachable_licensing_blocks_the_run_by_default() -> None:
    """A gate that fails open is not a gate."""
    gate = CreditGate(FakeLicensing(reserve_raises=LicensingUnavailable("boom")), _settings())

    with pytest.raises(LicensingUnavailable):
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")


async def test_fail_open_lets_the_run_proceed_unmetered() -> None:
    gate = CreditGate(
        FakeLicensing(reserve_raises=LicensingUnavailable("boom")),
        _settings(licensing_fail_open=True),
    )

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert hold is None, "an unmetered run has nothing to settle"


async def test_settling_an_unmetered_run_is_a_no_op() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings(licensing_fail_open=True))

    await gate.settle_success(None, _response())
    await gate.settle_failure(None)

    assert licensing.settles == []


# ---- settle failures --------------------------------------------------------------


async def test_a_failed_settle_does_not_raise() -> None:
    """The answer is already computed; losing it to a billing outage helps nobody
    and does not recover the credits either."""
    licensing = FakeLicensing(settle_raises=LicensingUnavailable("licensing down"))
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")
    await gate.settle_success(hold, _response())  # must not raise

    assert len(licensing.settles) == 1


async def test_a_broken_run_settles_as_fail() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")
    await gate.settle_failure(hold, execution_time_ms=900)

    [settle] = licensing.settles
    assert settle["result"] == "Fail"
    assert settle["telemetry"].execution_time_ms == 900


# ---- off by default ---------------------------------------------------------------


async def test_licensing_off_means_no_calls_at_all() -> None:
    gate = CreditGate(None, AgenticSettings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")
    await gate.settle_success(hold, _response())

    assert hold is None
    assert not gate.active


async def test_a_client_without_the_flag_stays_inert() -> None:
    """Configuring a client is not the same as switching metering on."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, AgenticSettings(licensing_enabled=False))

    await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert licensing.reserves == []


# ---- the forwarded token ----------------------------------------------------------


async def test_the_forwarded_token_reaches_licensing_and_nothing_else() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    request = _request(access_token="Bearer eyJsecret")
    hold = await gate.reserve(request, tenant="TENANT_A", trace_id="t")
    await gate.settle_success(hold, _response())

    assert licensing.reserves[0]["access_token"] == "Bearer eyJsecret"
    # ...and it is not something a response or a log could leak.
    assert "access_token" not in request.model_dump()
    assert "eyJsecret" not in repr(request)


async def test_a_release_gives_the_hold_back() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")
    await gate.release(hold, "Cancelled")

    assert licensing.releases[0]["reservation_id"] == "res-1"
    assert licensing.releases[0]["reason"] == "Cancelled"


def test_run_telemetry_tracks_the_model_call_count() -> None:
    """BatchCallCount comes from the loop, not from counting tool calls."""
    assert RunTelemetry.from_response(_response()).batch_call_count == 2


# ---- what was asked ---------------------------------------------------------------
#
# The credit record is the only durable trace of a run: CloudWatch expires, and the
# agent keeps no conversation store. Without the prompt a row shows that someone
# spent 12 credits but not what they asked, which is the first question anyone has.


async def test_the_prompt_travels_with_the_settle() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="trace-1")
    await gate.settle_success(hold, _response())

    [settle] = licensing.settles
    assert settle["user_prompt"] == "how many documents?"


async def test_a_failed_run_still_records_what_was_asked() -> None:
    """A failure is the row most worth reading back."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="trace-1")
    await gate.settle_failure(hold, model_id="claude")

    [settle] = licensing.settles
    assert settle["result"] == "Fail"
    assert settle["user_prompt"] == "how many documents?"


async def test_a_rejected_request_recovers_the_prompt_from_the_raw_body() -> None:
    """Validation failed, so no parsed request exists -- but the prompt is still there."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    await gate.settle_rejected(
        {"ReservationId": "res-9", "queryText": "  who approved PIF0001?  "},
        tenant="TENANT_A", trace_id="trace-1", reason="missing base url",
    )

    [settle] = licensing.settles
    assert settle["result"] == "Fail"
    assert settle["user_prompt"] == "who approved PIF0001?"


async def test_a_rejected_request_without_a_prompt_still_settles() -> None:
    """Recovering the prompt is a convenience; releasing the reservation is not."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    await gate.settle_rejected(
        {"ReservationId": "res-9"}, tenant="TENANT_A", trace_id="t", reason="empty body",
    )

    [settle] = licensing.settles
    assert settle["reservation_id"] == "res-9"
    assert settle["user_prompt"] == ""


async def test_the_answer_travels_with_the_settle() -> None:
    """Both halves of the record: what was asked, and what came back."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="trace-1")
    await gate.settle_success(hold, _response())

    [settle] = licensing.settles
    assert settle["user_prompt"] == "how many documents?"
    assert settle["response_text"] == "1,359."


async def test_a_broken_run_settles_without_an_answer() -> None:
    """There is no response to quote -- the exception took it with it."""
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="trace-1")
    await gate.settle_failure(hold, model_id="claude")

    [settle] = licensing.settles
    assert settle["response_text"] == ""


# ---- phase 1: the object-code lookup ----------------------------------------------
#
# SpecSafe asks twice for one user question: first which object codes exist, then
# the real question narrowed to the codes its ACL allowed. The first call reads
# facet values -- no model, no vector search -- so it must not be billed as a run.


async def test_an_object_codes_lookup_settles_at_zero_tokens() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(operation="objectCodes"), tenant="TENANT_A", trace_id="t")
    # The gate is exercised directly here, with no agent to reset the per-request
    # accumulators, so this test does it rather than inherit another test's spend.
    reset_spent()
    reset_query_embedding_tokens()
    await gate.settle_lookup(
        hold, model_id="claude", agent_id="document_search", execution_time_ms=84
    )

    [settle] = licensing.settles
    assert settle["result"] == "Success"
    telemetry = settle["telemetry"]
    assert (telemetry.input_tokens, telemetry.output_tokens, telemetry.total_tokens) == (0, 0, 0)
    assert telemetry.embedding_tokens == 0
    assert telemetry.agent_id == "document_search"
    assert telemetry.execution_time_ms == 84
    # Named so Settle can price it; zero tokens cost zero at any rate.
    assert telemetry.model_id == "claude"


async def test_a_lookup_can_be_left_unsettled_so_the_real_question_reuses_the_id() -> None:
    """With `licensing_settle_object_codes` off the reservation stays Reserved.

    Settling commits it, and a committed id is refused -- so an operator whose
    SpecSafe reuses one reservation across both calls needs the hold left alone.
    """
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings(licensing_settle_object_codes=False))

    hold = await gate.reserve(_request(operation="objectCodes"), tenant="TENANT_A", trace_id="t")
    await gate.settle_lookup(hold)

    assert licensing.settles == []
    assert licensing.releases == []


async def test_a_real_question_is_still_billed_in_full() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings(licensing_settle_object_codes=False))

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")
    await gate.settle_success(hold, _response())

    [settle] = licensing.settles
    assert settle["telemetry"].input_tokens == 3322


async def test_a_lookup_reports_tokens_if_it_ever_spends_any() -> None:
    """Zero because nothing ran, not because zero was assumed.

    The lookup path calls no model today. If that ever changes -- a reranker, a
    query rewrite, an embedding on the facet filter -- the spend must follow the
    tenant who incurred it rather than being absorbed by whoever runs the agent.
    """
    from rag_agentic.converse import _record_spent

    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())
    hold = await gate.reserve(_request(operation="objectCodes"), tenant="TENANT_A", trace_id="t")

    reset_spent()
    _record_spent(Usage(input_tokens=120, output_tokens=8, model="claude", llm_calls=1))
    try:
        await gate.settle_lookup(hold, model_id="claude")
    finally:
        reset_spent()

    telemetry = licensing.settles[0]["telemetry"]
    assert telemetry.input_tokens == 120
    assert telemetry.output_tokens == 8
    assert telemetry.total_tokens == 128
    assert telemetry.batch_call_count == 1
