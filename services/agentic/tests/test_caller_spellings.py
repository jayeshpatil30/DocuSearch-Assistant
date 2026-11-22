"""The agent accepts however SpecSafe spells a field.

Rejecting a caller's spelling bought nothing but failed integrations. The worst
case was `BussinessObject` (sic): an unknown field is ignored, so the request
succeeded with the WRONG SCOPE -- the whole tenant searched instead of four
business objects, with no error anywhere.

Names are compared with case, underscores, hyphens and spaces removed.
"""

from __future__ import annotations

import pytest
from rag_agentic.models import AgentRequest

# The real payload SpecSafe sends, verbatim.
SPECSAFE_PAYLOAD = {
    "BussinessObject": "INGREDIENTSPEC,PARAMETERGRID,AUGTEST,PIF",
    "BaseUrl": "https://tenant-b.demo.com/",
    "UserCode": "DemoUser",
    "QueryText": "Documents created by DemoUser",
    "SessionId": "1b94816b-3964-4eeb-99d8-c055df410236",
}


def test_the_payload_specsafe_actually_sends() -> None:
    request = AgentRequest.model_validate(SPECSAFE_PAYLOAD)

    assert request.base_url == "https://tenant-b.demo.com/"
    assert request.query == "Documents created by DemoUser"
    assert request.user_code == "DemoUser"
    assert request.session_id == "1b94816b-3964-4eeb-99d8-c055df410236"
    # The one that silently did nothing before.
    assert request.business_object == "INGREDIENTSPEC,PARAMETERGRID,AUGTEST,PIF"


@pytest.mark.parametrize(
    "spelling",
    ["query", "Query", "queryText", "QueryText", "query_text", "QUERY_TEXT", "question"],
)
def test_query_spellings(spelling: str) -> None:
    request = AgentRequest.model_validate(
        {"baseUrl": "https://tenant-a.demo.com/", spelling: "what is in the manual?"}
    )
    assert request.query == "what is in the manual?"


@pytest.mark.parametrize(
    "spelling",
    [
        "business_object",
        "businessObject",
        "BusinessObject",
        "bussinessObject",  # sic
        "BussinessObject",  # sic
        "BUSINESS-OBJECT",
    ],
)
def test_business_object_spellings(spelling: str) -> None:
    request = AgentRequest.model_validate(
        {"baseUrl": "https://tenant-a.demo.com/", "query": "q", spelling: "COMPLAINT"}
    )
    assert request.business_object == "COMPLAINT"


@pytest.mark.parametrize("spelling", ["base_url", "baseUrl", "BaseUrl", "BASE_URL"])
def test_base_url_spellings(spelling: str) -> None:
    request = AgentRequest.model_validate({spelling: "https://tenant-a.demo.com/", "query": "q"})
    assert request.base_url == "https://tenant-a.demo.com/"


def test_the_canonical_name_wins_over_a_synonym() -> None:
    """A payload carrying both must not depend on dictionary ordering."""
    request = AgentRequest.model_validate(
        {
            "baseUrl": "https://tenant-a.demo.com/",
            "query": "the real question",
            "QueryText": "the other one",
        }
    )
    assert request.query == "the real question"


def test_a_missing_query_is_still_an_error() -> None:
    """Being liberal about spelling must not mean accepting nothing at all."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"baseUrl": "https://tenant-a.demo.com/"})


def test_unknown_fields_are_still_ignored() -> None:
    """SpecSafe sends awsBedrockSettings; it must not break the request."""
    request = AgentRequest.model_validate(
        {
            "baseUrl": "https://tenant-a.demo.com/",
            "QueryText": "q",
            "awsBedrockSettings": {"region": "us-east-1"},
            "conversationTurnNumber": 2,
        }
    )
    assert request.query == "q"
