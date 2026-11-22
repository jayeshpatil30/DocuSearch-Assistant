"""DocumentSync: one rule for every trigger (S3 event, SpecSafe call, sweep), offline.

Each test drives a document through a change -- its file, its catalogue record, or
time -- and checks what a user filtered the way the agent filters (by tenant) can see.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from rag_search.document_sync import DocumentSync, SyncAction
from rag_search.index_state import InMemoryIndexState, PendingFile
from rag_search.indexer import EXTRA_SOURCE_ETAG, METADATA_CHUNK_STRATEGY
from rag_search.models import IndexRequest
from rag_search.mongo_source import MongoMetadataSource

from rag_core.utils import chunk_id, document_id

Stack = Callable[..., Any]
BUCKET = "demo-bucket"
POLICY = "TENANT_A/security_policy.docx"
REGISTER = "TENANT_A/supplier_register.html"
AS_DG = {"tenant": ["TENANT_A"]}  # how every agent search is scoped


class Catalogue(MongoMetadataSource):
    """AttachmentMetaData in memory."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__("mongodb://x", database="d", collection="c", tenant_codes="TENANT_A")
        self.records = records

    def _find_sync(self, query: dict[str, Any]) -> Iterable[dict[str, Any]]:
        names: set[str] = set()
        for clause in query.get("$or", []):
            for cond in clause.values():
                names.update(cond.get("$in", []))
        return [
            r
            for r in self.records
            if r.get("GeneratedFileName") in names or r.get("OriginalFileName") in names
        ]


class Bucket:
    """S3 as DocumentSync sees it: which objects exist and their ETags. Shares the
    objects dict with the indexer's fake S3, so replacing a body is what it downloads."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.etags = dict.fromkeys(objects, "v1")

    def replace(self, key: str, body: bytes) -> None:
        self.objects[key] = body
        self.etags[key] = f"v{int(self.etags.get(key, 'v0')[1:]) + 1}"

    async def head_object_or_none(self, bucket: str, key: str) -> dict[str, Any] | None:
        if key not in self.objects:
            return None
        return {"ETag": f'"{self.etags.setdefault(key, "v1")}"'}


def _record(name: str, tenant: str = "TENANT_A", **extra: Any) -> dict[str, Any]:
    return {
        "GeneratedFileName": name,
        "TenantCode": tenant,
        "BusinessObject": "SUPPLIERDOCUMENTS",
        "ObjectCode": "SUP001",
        "Status": "Active",
        "DeleteMark": 0,
        **extra,
    }


def _values(**fields: str) -> list[dict[str, Any]]:
    return [{"FieldName": k, "DisplayName": k, "Value": v} for k, v in fields.items()]


class Harness:
    def __init__(self, build_stack: Stack, sample_objects: dict[str, bytes], records: list[Any]):
        self.objects = dict(sample_objects)
        self.mongo = Catalogue(records)
        self.indexer, self.search, self.meta, self.closeables = build_stack(
            self.objects, mongo=self.mongo
        )
        self.bucket = Bucket(self.objects)
        self.state = InMemoryIndexState()
        self.embedded: list[int] = []
        real_embed = self.indexer._embedder.embed

        async def counting_embed(texts: list[str]) -> list[list[float]]:
            self.embedded.append(len(texts))
            return await real_embed(texts)

        self.indexer._embedder.embed = counting_embed

    def sync(self, **kw: Any) -> DocumentSync:
        return DocumentSync(
            indexer=self.indexer,
            metadata_store=self.meta,
            mongo=self.mongo,
            state=self.state,
            s3=self.bucket,
            bucket=BUCKET,
            **kw,
        )

    async def run(self, key: str, **kw: Any) -> Any:
        [outcome] = await self.sync(**kw).sync_many([key])
        return outcome

    async def visible(self, query: str, **filters: list[str]) -> set[str]:
        hits = await self.search.search(query, filters={**AS_DG, **filters})
        return {p.source_key for p in hits.results}

    async def facets(self, key: str) -> dict[str, str]:
        doc_id = document_id(BUCKET, key)
        return (await self.meta.describe([doc_id])).get(doc_id, {}).get("facets", {})

    async def close(self) -> None:
        for c in self.closeables:
            await c.aclose()


@pytest.fixture
async def make(build_stack: Stack, sample_objects: dict[str, bytes]):  # type: ignore[no-untyped-def]
    made: list[Harness] = []

    def _make(*records: dict[str, Any]) -> Harness:
        made.append(Harness(build_stack, sample_objects, list(records)))
        return made[-1]

    yield _make
    for h in made:
        await h.close()


# --------------------------------------------------------------------------
# A new file, before and after its record
# --------------------------------------------------------------------------


async def test_a_file_without_a_record_is_embedded_hidden_then_revealed_in_place(make) -> None:
    h = make()

    first = await h.run(REGISTER)
    assert first.action is SyncAction.INDEXED_HIDDEN and first.hidden
    assert [p.key for p in await h.state.pending()] == [REGISTER]
    assert REGISTER not in await h.visible("ACME supplier Germany")
    unfiltered = await h.search.search("ACME supplier Germany")
    assert REGISTER in {p.source_key for p in unfiltered.results}  # embedded, just hidden

    # SpecSafe writes the record and calls us: applied in place, nothing re-embedded
    # except the one chunk holding the record's own field values.
    h.mongo.records.append(_record("supplier_register.html", MetaData=_values(Audit="passed 2026")))
    h.embedded.clear()
    second = await h.run(REGISTER)
    assert second.action is SyncAction.METADATA_UPDATED
    assert h.embedded == [1]
    assert await h.state.pending() == []
    assert REGISTER in await h.visible("ACME supplier Germany")
    assert REGISTER in await h.visible("audit passed 2026")
    assert (await h.facets(REGISTER))["object_code"] == "SUP001"


async def test_a_file_with_its_record_is_searchable_at_once(make) -> None:
    h = make(_record("security_policy.docx"))
    outcome = await h.run(POLICY)
    assert outcome.action is SyncAction.INDEXED and not outcome.hidden
    assert POLICY in await h.visible("encrypted at rest AES-256")
    assert await h.state.pending() == []


async def test_a_record_with_nothing_new_writes_nothing(make) -> None:
    h = make(_record("security_policy.docx"))
    await h.run(POLICY)
    h.embedded.clear()
    calls: list[Any] = []
    real_apply = h.meta.apply_facets

    async def spy(*args: Any, **kwargs: Any) -> bool:
        calls.append(args)
        return await real_apply(*args, **kwargs)

    h.meta.apply_facets = spy
    assert (await h.run(POLICY)).action is SyncAction.UNCHANGED
    assert calls == [] and h.embedded == []


# --------------------------------------------------------------------------
# Record changes
# --------------------------------------------------------------------------


async def test_record_changes_reach_the_record_and_every_chunk(make) -> None:
    h = make(_record("security_policy.docx", ObjectName="Fryer oil policy"))
    await h.run(POLICY)
    query = "encrypted at rest AES-256"
    assert POLICY in await h.visible(query, status=["Active"], object_name=["Fryer oil policy"])

    record = h.mongo.records[0]
    record["Status"] = "Expired"
    del record["ObjectName"]
    outcome = await h.run(POLICY)

    assert outcome.action is SyncAction.METADATA_UPDATED
    facets = await h.facets(POLICY)
    assert facets["status"] == "Expired" and "object_name" not in facets
    # Chunk payloads (what search filters read) moved with the record.
    assert POLICY in await h.visible(query, status=["Expired"])
    assert POLICY not in await h.visible(query, status=["Active"])
    assert POLICY not in await h.visible(query, object_name=["Fryer oil policy"])


async def test_metadata_values_are_re_embedded_when_changed_and_removed_when_gone(make) -> None:
    h = make(_record("security_policy.docx", MetaData=_values(Note="rancid oil smell")))
    await h.run(POLICY)
    cid = chunk_id(document_id(BUCKET, POLICY), METADATA_CHUNK_STRATEGY, 0)
    assert "rancid oil smell" in (await h.indexer._keyword.get([cid]))[cid][0]

    h.mongo.records[0]["MetaData"] = _values(Note="burnt fryer residue")
    h.embedded.clear()
    await h.run(POLICY)
    assert h.embedded == [1]
    assert "burnt fryer residue" in (await h.indexer._keyword.get([cid]))[cid][0]

    h.mongo.records[0]["MetaData"] = []
    await h.run(POLICY)
    assert await h.indexer._keyword.get([cid]) == {}


# --------------------------------------------------------------------------
# Removal
# --------------------------------------------------------------------------


async def test_delete_mark_purges_and_other_tenants_files_are_never_indexed(make) -> None:
    h = make(
        _record("security_policy.docx"), _record("supplier_register.html", tenant="TENANT_A")
    )
    await h.run(POLICY)

    h.mongo.records[0]["DeleteMark"] = 1
    deleted = await h.run(POLICY)
    assert (deleted.action, deleted.reason) == (SyncAction.PURGED, "deleted")
    assert await h.meta.indexed_ids() == set()

    foreign = await h.run(REGISTER)
    assert (foreign.action, foreign.reason) == (SyncAction.SKIPPED, "other_tenant")
    assert h.embedded == [1]  # only the policy, before it was deleted


async def test_a_file_removed_from_s3_is_purged(make) -> None:
    h = make(_record("security_policy.docx"))
    await h.run(POLICY)
    del h.objects[POLICY]
    outcome = await h.run(POLICY)
    assert (outcome.action, outcome.reason) == (SyncAction.PURGED, "file_removed")
    # A file that is not there and was never indexed: its upload event will come.
    assert (await h.run(POLICY)).action is SyncAction.AWAITING_FILE


async def test_a_removed_record_purges_a_catalogued_document(make) -> None:
    h = make(_record("security_policy.docx"))
    await h.run(POLICY)
    h.mongo.records.clear()
    outcome = await h.run(POLICY)
    assert (outcome.action, outcome.reason) == (SyncAction.PURGED, "record_removed")


async def test_a_record_that_never_arrives_is_given_up_after_max_age(make) -> None:
    h = make()
    await h.run(REGISTER)  # indexed, hidden, pending
    await h.state.put_pending([PendingFile(REGISTER, datetime.now(UTC) - timedelta(days=8))])
    outcome = await h.run(REGISTER, pending_max_age=timedelta(days=7))
    assert (outcome.action, outcome.reason) == (SyncAction.PURGED, "never_catalogued")
    assert await h.state.pending() == [] and await h.meta.indexed_ids() == set()


# --------------------------------------------------------------------------
# File replaced
# --------------------------------------------------------------------------


def _html(*paragraphs: str) -> bytes:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<html><body><h1>Register</h1>{body}</body></html>".encode()


async def test_a_replaced_file_is_re_indexed_and_its_old_chunks_removed(make) -> None:
    h = make(_record("supplier_register.html"))
    long_version = [f"Clause {i}: zanzibar pallet rotation rule number {i}. " * 6 for i in range(8)]
    h.bucket.replace(REGISTER, _html(*long_version))
    await h.run(REGISTER)
    doc_id = document_id(BUCKET, REGISTER)
    before = await h.indexer._vector.fetch_document(doc_id, limit=1000)
    assert len(before) > 2

    h.bucket.replace(REGISTER, _html("Supplier ACME moved to Austria."))
    outcome = await h.run(REGISTER)

    assert (outcome.action, outcome.reason) == (SyncAction.REINDEXED, "file_replaced")
    after = await h.indexer._vector.fetch_document(doc_id, limit=1000)
    assert len(after) < len(before)
    # The old version's text is gone from both stores, not just outranked.
    assert await h.indexer._keyword.search("zanzibar", limit=10) == []
    assert {p["_chunk_id"] for p in after} <= {p["_chunk_id"] for p in before}
    assert REGISTER in await h.visible("ACME Austria")
    meta = await h.meta.get(doc_id)
    assert meta is not None and meta.extra[EXTRA_SOURCE_ETAG] == h.bucket.etags[REGISTER]


async def test_a_document_indexed_before_etags_adopts_one_instead_of_re_embedding(make) -> None:
    h = make(_record("security_policy.docx"))
    await h.indexer.index(IndexRequest(reset=True))  # the full build: listing ETag "e"
    doc_id = document_id(BUCKET, POLICY)
    meta = await h.meta.get(doc_id)
    assert meta is not None
    await h.meta.apply_facets(doc_id, {}, extra={EXTRA_SOURCE_ETAG: None})  # as before ETags
    h.embedded.clear()

    outcome = await h.run(POLICY)
    assert outcome.action in (SyncAction.UNCHANGED, SyncAction.METADATA_UPDATED)
    assert h.embedded == []
    meta = await h.meta.get(doc_id)
    assert meta is not None and meta.extra[EXTRA_SOURCE_ETAG] == "v1"


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


async def test_a_document_another_worker_holds_is_left_for_retry(make) -> None:
    h = make(_record("security_policy.docx"))
    doc_id = document_id(BUCKET, POLICY)
    assert await h.state.acquire_doc(doc_id, "other-worker", timedelta(minutes=5))
    outcome = await h.run(POLICY)
    assert outcome.action is SyncAction.BUSY and outcome.retry
    assert h.embedded == []

    await h.state.release_doc(doc_id, "other-worker")
    assert (await h.run(POLICY)).action is SyncAction.INDEXED


# --------------------------------------------------------------------------
# Which key a file name means
# --------------------------------------------------------------------------


async def test_a_file_name_resolves_to_the_indexed_or_existing_spelling(make) -> None:
    """TENANT_B keeps files at ``TENANT_B//<name>`` as well as ``TENANT_B/<name>``: SpecSafe sends the
    name only, so the spelling that is indexed -- or else exists in S3 -- is the one."""
    h = make(_record("security_policy.docx"))
    double = "TENANT_A//security_policy.docx"
    sync = h.sync()
    assert await sync.resolve_key((double, POLICY)) == POLICY  # only this one exists
    assert await sync.resolve_key(("TENANT_A/nowhere.pdf", "TENANT_A//nowhere.pdf")) == "TENANT_A/nowhere.pdf"

    await h.run(POLICY)
    del h.objects[POLICY]  # indexed beats existing
    h.objects[double] = b"elsewhere"
    assert await sync.resolve_key((double, POLICY)) == POLICY


async def test_only_text_formats_are_embedded(make) -> None:
    """Images and unrecognised formats are skipped before any download."""
    h = make()
    sync = h.sync()
    keys = ["TENANT_A/photo~jpg", "TENANT_A/scan.PNG", "TENANT_A/archive.zip", "TENANT_A/clip~mp4", "TENANT_A/noextension"]
    legacy = ["TENANT_A/old_spec~doc", "TENANT_A/deck.PPT", "TENANT_A/costs~xls"]  # binary Office: no extractor
    outcomes = {o.key: (o.action, o.reason) for o in await sync.sync_many(keys + legacy)}
    assert {outcomes[k] for k in keys} == {(SyncAction.SKIPPED, "non_text")}
    assert {outcomes[k] for k in legacy} == {(SyncAction.SKIPPED, "unsupported_format")}
    assert h.embedded == [] and await h.meta.indexed_ids() == set()


async def test_a_file_name_finds_its_copy_in_a_subfolder(make) -> None:
    """Some tenant files live in subfolders (``TENANT_A/COMPDOCCAT/<name>``); SpecSafe sends
    only the name, so an update must still reach the indexed document."""
    nested = "TENANT_A/COMPDOCCAT/supplier_register.html"
    h = make(_record("supplier_register.html"))
    h.objects[nested] = h.objects.pop(REGISTER)
    await h.run(nested)
    sync = h.sync()
    assert await sync.resolve_key((REGISTER, "TENANT_A//supplier_register.html")) == nested
    # Another tenant folder's copy of the same name is not ours.
    assert await sync.resolve_key(("TENANT_B/supplier_register.html", "TENANT_B//x")) == (
        "TENANT_B/supplier_register.html"
    )
