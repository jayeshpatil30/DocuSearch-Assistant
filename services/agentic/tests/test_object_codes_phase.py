"""SpecSafe's two-phase call, end to end through the runtime.

One user question arrives as two requests sharing an X-Ray trace:

  1. `Operation: "objectCodes"` -- which object codes exist for this
     BusinessObject. A facet lookup: no model, no vector search.
  2. the real question, narrowed to the codes SpecSafe's ACL allowed.

Phase 1 used to arrive without the `Operation` field, so it was answered as a
normal question -- a full model run whose answer SpecSafe discarded -- and then
settled, which committed the reservation and made phase 2 unusable. These tests
pin both halves of that: phase 1 costs nothing, and phase 2 is billed normally.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings, AgentRole
from rag_agentic.converse import ScriptedConverseModel, text_turn, tool_turn
from rag_agentic.credit_gate import CreditGate
from rag_agentic.factory import AgenticComponents
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.runtime.app import create_app
from rag_agentic.tenancy import TenantResolver

from .test_credit_gate import FakeLicensing


@pytest.fixture
def harness(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
) -> Any:
    def build(**over: Any) -> tuple[TestClient, ScriptedConverseModel, FakeLicensing]:
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
                **over,
            }
        )
        licensing = FakeLicensing()
        agent = DocumentSearchAgent(
            settings=settings, model=model, embedder=embedder,
            index_store=index_store, resolver=resolver,
        )
        components = AgenticComponents(
            settings=settings, role=AgentRole.DOCUMENT_SEARCH, agent=agent,
            resolver=resolver, model=model,
            credit_gate=CreditGate(licensing, settings),
        )
        app = create_app(AgentRole.DOCUMENT_SEARCH, components=components)
        return TestClient(app), model, licensing

    return build


def _lookup_body(base_url: str) -> dict[str, Any]:
    """Phase 1 exactly as SpecSafe should send it -- no Query at all."""
    return {
        "baseUrl": base_url,
        "Operation": "objectCodes",
        "BusinessObject": "SUPPLIERDOCUMENTS",
        "ReservationId": "res-1",
        "UserCode": "DemoUser",
        "SessionId": "s" * 40,
    }


def test_a_lookup_calls_no_model(harness: Any, base_url: str) -> None:
    """The 60 seconds and ~630 tokens phase 1 used to burn, now zero of both."""
    client, model, _ = harness()
    with client:
        response = client.post("/invocations", json=_lookup_body(base_url))

    assert response.status_code == 200
    assert model.calls == []
    assert response.json()["answer"] == ""


def test_a_lookup_returns_the_object_codes(harness: Any, base_url: str) -> None:
    client, _, _ = harness()
    with client:
        payload = client.post("/invocations", json=_lookup_body(base_url)).json()

    # SpecSafe filters these against the user's ACL and sends the survivors back.
    assert sorted(payload["object_codes"].split(",")) == ["SQ-1", "SQ-2"]


def test_a_lookup_is_settled_at_zero_tokens(harness: Any, base_url: str) -> None:
    client, _, licensing = harness()
    with client:
        client.post("/invocations", json=_lookup_body(base_url))

    [settle] = licensing.settles
    assert settle["result"] == "Success"
    telemetry = settle["telemetry"]
    assert (telemetry.input_tokens, telemetry.output_tokens, telemetry.total_tokens) == (0, 0, 0)
    assert telemetry.embedding_tokens == 0
    # The model is named though none ran: Settle prices tokens against a
    # configured AIModelTokenCost and answers 400 missing_model_cost for a
    # ModelId it does not know, the empty string included.
    assert telemetry.model_id == "scripted"


def test_a_lookup_can_leave_the_reservation_for_phase_two(harness: Any, base_url: str) -> None:
    client, _, licensing = harness(licensing_settle_object_codes=False)
    with client:
        assert client.post("/invocations", json=_lookup_body(base_url)).status_code == 200

    # Nothing settled, so the id is still Reserved when the real question arrives.
    assert licensing.settles == []


def test_the_real_question_is_billed_in_full(harness: Any, base_url: str) -> None:
    """Phase 2 carries ObjectCodes and no Operation -- an ordinary billed run."""
    client, model, licensing = harness()
    with client:
        response = client.post(
            "/invocations",
            json={
                "baseUrl": base_url,
                "query": "Do suppliers need ISO 9001?",
                "BusinessObject": "PIF",
                "ObjectCodes": "PIF0002,PIF3961",
                "ReservationId": "res-2",
                "UserCode": "DemoUser",
                "SessionId": "s" * 40,
            },
        )

    assert response.status_code == 200
    assert model.calls, "the real question must reach the model"
    [settle] = licensing.settles
    assert settle["telemetry"].model_id == "scripted"
    assert settle["telemetry"].agent_id == "document_search"


# ---- inferring phase 1 when the caller cannot say so -------------------------------
#
# SpecSafe will not add an `Operation` field, so with
# `infer_object_codes_lookup` on, the payload shape decides: a BusinessObject to
# enumerate and no ObjectCodes key is call 1. These tests pin the boundary,
# because a false positive answers a real question with a list of codes.


def _phase_one(base_url: str) -> dict[str, Any]:
    """Call 1 exactly as SpecSafe sends it today -- Query present, no Operation."""
    return {
        "baseUrl": base_url,
        "query": "give me list of all active documents",
        "BusinessObject": "SUPPLIERDOCUMENTS",
        "ReservationId": "res-1",
        "AccessToken": "tok",
        "UserCode": "DemoUser",
        "SessionId": "s" * 40,
    }


def test_specsafes_first_call_is_inferred_and_costs_nothing(
    harness: Any, base_url: str
) -> None:
    """The 60s/628-token call SpecSafe throws away becomes a facet lookup."""
    client, model, licensing = harness(infer_object_codes_lookup=True)
    with client:
        payload = client.post("/invocations", json=_phase_one(base_url)).json()

    assert model.calls == []
    assert sorted(payload["object_codes"].split(",")) == ["SQ-1", "SQ-2"]
    [settle] = licensing.settles
    assert settle["telemetry"].total_tokens == 0


def test_the_inference_is_off_unless_asked_for(harness: Any, base_url: str) -> None:
    """Default behaviour is unchanged: the same payload is answered as a question."""
    client, model, _ = harness()
    with client:
        payload = client.post("/invocations", json=_phase_one(base_url)).json()

    assert model.calls, "without the flag this is an ordinary question"
    assert payload["answer"]


def test_a_second_call_with_object_codes_is_still_answered(
    harness: Any, base_url: str
) -> None:
    """Call 2 carries ObjectCodes, so it must reach the model even with the flag on."""
    client, model, licensing = harness(infer_object_codes_lookup=True)
    with client:
        payload = client.post(
            "/invocations",
            json={**_phase_one(base_url), "ObjectCodes": "SQ-1", "ReservationId": "res-2"},
        ).json()

    assert model.calls, "the real question must reach the model"
    assert payload["answer"]
    [settle] = licensing.settles
    assert settle["telemetry"].model_id == "scripted"


def test_an_empty_object_codes_is_not_a_lookup(harness: Any, base_url: str) -> None:
    """"The ACL allowed nothing" is an answer about this user, not a request to list.

    Inferring a lookup here would hand back every code in the BusinessObject --
    the exact set SpecSafe had just finished filtering down to none.
    """
    client, model, _ = harness(infer_object_codes_lookup=True)
    with client:
        response = client.post(
            "/invocations", json={**_phase_one(base_url), "ObjectCodes": ""}
        )

    assert response.status_code == 200
    assert model.calls, "an explicit empty ObjectCodes is still a question"


def test_a_question_without_a_business_object_is_not_a_lookup(
    harness: Any, base_url: str
) -> None:
    """Phase 1 exists to enumerate one BusinessObject; without one there is no
    lookup to do, and this is how a direct caller asks an ordinary question."""
    client, model, _ = harness(infer_object_codes_lookup=True)
    body = _phase_one(base_url)
    del body["BusinessObject"]
    with client:
        payload = client.post("/invocations", json=body).json()

    assert model.calls
    assert payload["answer"]
