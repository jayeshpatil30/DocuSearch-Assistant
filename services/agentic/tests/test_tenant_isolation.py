"""Two tenants, one agent: TENANT_A must never see TENANT_B's documents, and vice versa.

Isolation normally rests on physical separation -- one S3 prefix, one index, one
set of files per tenant. That is sound, but it is a SINGLE barrier, and it has
already failed once in this project: TENANT_A's first index was built without a tenant
filter and contained TENANT_A documents.

So a snapshot now declares which tenant codes it holds, and the toolbox enforces
that on every query. These tests build a DELIBERATELY CONTAMINATED index -- three
TENANT_A documents and one TENANT_B document in the same snapshot -- and assert the foreign
document still cannot be reached by any route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rag_agentic.agent_loop import ToolError
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.tenancy import TenantResolver
from rag_search.embedder import FakeEmbedder

from .conftest import BASE_URL, BUCKET, FakeS3, _build_snapshot, _upload

MIXED_DOCS = [
    {
        "document_id": "dg1",
        "source_key": "TENANT_A/supplier_quality_manual.pdf",
        "title": "Supplier Quality Manual",
        "facets": {
            "tenant": "TENANT_A",
            "business_object": "SUPPLIERDOCUMENTS",
            "status": "Active",
            "object_code": "TENANT_A-1",
        },
        "text": "All suppliers must hold a valid ISO 9001 certificate before approval.",
    },
    {
        "document_id": "dg2",
        "source_key": "TENANT_A/complaint_policy.pdf",
        "title": "Complaint Handling Policy",
        "facets": {
            "tenant": "TENANT_A",
            "business_object": "COMPLAINT",
            "status": "Active",
            "object_code": "TENANT_A-2",
        },
        "text": "Every customer complaint is acknowledged within 48 hours of receipt.",
    },
    {
        "document_id": "dg3",
        "source_key": "TENANT_A/audit_schedule.pdf",
        "title": "Supplier Audit Schedule",
        "facets": {
            "tenant": "TENANT_A",
            "business_object": "SUPPLIERDOCUMENTS",
            "status": "Draft",
            "object_code": "TENANT_A-3",
        },
        "text": "Critical suppliers are audited annually using a remote questionnaire.",
    },
    {
        # The document that must never surface for TENANT_A.
        "document_id": "bee1",
        "source_key": "TENANT_B/confidential_pricing.pdf",
        "title": "TENANT_B Confidential Pricing Agreement",
        "facets": {
            "tenant": "TENANT_B",
            "business_object": "SUPPLIERDOCUMENTS",
            "status": "Active",
            "object_code": "TENANT_B-9",
        },
        "text": "Confidential TENANT_B pricing agreement: ISO 9001 certified supplier rates.",
    },
]


@pytest.fixture
async def mixed_tools(tmp_path: Path) -> Any:
    """A toolbox over one snapshot holding two tenants' documents."""
    root = tmp_path / "mixed"
    root.mkdir(parents=True, exist_ok=True)
    await _build_snapshot(root, MIXED_DOCS)
    objects: dict[str, bytes] = {}
    _upload(root, objects, "indexes/dev/acme")

    settings = AgenticSettings(
        index_bucket=BUCKET,
        index_prefix="indexes",
        cache_dir=str(tmp_path / "cache"),
        cache_ttl_s=0.0,
        tenant_pattern=r"^(?P<environment>[^.]+)\.specsafe\.io/(?P<tenant>[^/]+)",
        embed_provider="fake",
        embed_dim=256,
        collection="compliance",
        top_k=5,
    )
    store = TenantIndexStore(settings, FakeS3(objects))
    index = await store.acquire(TenantResolver(settings).resolve(BASE_URL))

    from rag_agentic.tools import DocumentSearchTools
    from rag_search.config import SearchSettings
    from rag_search.retrieval import SearchService

    def _make(tenant_codes: list[str] | None = None) -> Any:
        search = SearchService(
            embedder=FakeEmbedder(dim=256),
            vector=index.vector,
            keyword=index.keyword,
            settings=SearchSettings(collection="compliance", top_k=5, embed_dim=256),
        )
        return DocumentSearchTools(index, search, tenant_codes=tenant_codes, top_k=5)

    try:
        yield _make
    finally:
        await store.aclose()


async def test_a_contaminated_index_cannot_list_foreign_documents(mixed_tools: Any) -> None:
    payload, _ = await mixed_tools(["TENANT_A"]).invoke("list_documents", {})

    returned = {d["document_id"] for d in payload["documents"]}
    assert returned == {"dg1", "dg2", "dg3"}
    assert payload["total_matching"] == 3


async def test_search_cannot_reach_the_other_tenant(mixed_tools: Any) -> None:
    """The TENANT_B document is the best textual match for this query, by design."""
    payload, _ = await mixed_tools(["TENANT_A"]).invoke(
        "search_documents", {"query": "confidential TENANT_B pricing agreement ISO 9001"}
    )

    assert "bee1" not in {p["document_id"] for p in payload["passages"]}


async def test_the_other_tenant_sees_only_its_own(mixed_tools: Any) -> None:
    payload, _ = await mixed_tools(["TENANT_B"]).invoke("list_documents", {})

    assert {d["document_id"] for d in payload["documents"]} == {"bee1"}


async def test_a_guessed_document_id_from_another_tenant_is_refused(
    mixed_tools: Any,
) -> None:
    """Knowing an id is not authorisation: the last barrier before the answer."""
    with pytest.raises(ToolError, match="does not belong to this tenant"):
        await mixed_tools(["TENANT_A"]).invoke("get_document_details", {"document_id": "bee1"})


async def test_facet_values_do_not_leak_the_other_tenant(mixed_tools: Any) -> None:
    """Even the vocabulary is scoped -- TENANT_B's object code must not be suggested."""
    payload, _ = await mixed_tools(["TENANT_A"]).invoke("list_facet_values", {})

    assert "TENANT_B-9" not in payload["facets"].get("object_code", [])
    assert payload["facets"].get("tenant", []) in ([], ["TENANT_A"])


async def test_the_citations_returned_are_the_tenant_s_own(mixed_tools: Any) -> None:
    tools = mixed_tools(["TENANT_A"])
    await tools.invoke("search_documents", {"query": "ISO 9001 certificate supplier"})

    assert "bee1" not in {c.document_id for c in await tools.citations_with_urls()}


async def test_without_declared_tenant_codes_nothing_is_narrowed(
    mixed_tools: Any,
) -> None:
    """Snapshots published before the manifest carried tenant codes keep working.

    They fall back to physical separation alone -- which is why the agent logs a
    warning -- but they must not silently start returning nothing.
    """
    payload, _ = await mixed_tools().invoke("list_documents", {})

    assert payload["total_matching"] == 4
