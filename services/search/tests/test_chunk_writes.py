"""How the indexer stores a document while other users are searching.

Two rules keep a live index consistent during incremental indexing:

* A chunk is never searchable without its text. On Mongo, where the vector and the
  text live in the same record, each chunk is written once, whole.
* The metadata record is written last. It is what marks a document as indexed, so
  a document cut off part-way is indexed again instead of skipped for good.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from rag_search.indexer import remove_chunks, write_chunks
from rag_search.stores.mongo import CHUNK, MongoIndexDatabase, MongoKeywordIndex, MongoVectorIndex

from rag_core.utils import document_id

Stack = Callable[..., Any]
BUCKET = "demo-bucket"
POLICY = "TENANT_A/security_policy.docx"


class RecordingCollection:
    """The slice of a pymongo collection the chunk writers use: keeps every bulk write."""

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []
        self.deletes: list[dict[str, Any]] = []

    def bulk_write(self, ops: list[Any], ordered: bool = True) -> None:
        self.batches.append(list(ops))

    def delete_many(self, query: dict[str, Any]) -> Any:
        self.deletes.append(query)
        return SimpleNamespace(deleted_count=len(query["_id"]["$in"]))


def _database(coll: RecordingCollection) -> MongoIndexDatabase:
    # A client is anything indexable as client[database][collection].
    return MongoIndexDatabase(client={"d": {"c": coll}}, database="d", collection="c", dim=3)


def _set(op: Any) -> dict[str, Any]:
    """The ``$set`` of a pymongo ``UpdateOne``."""
    return op._doc["$set"]


# --------------------------------------------------------------------------
# One write per chunk
# --------------------------------------------------------------------------


async def test_mongo_writes_each_chunk_once_with_text_vector_and_payload() -> None:
    coll = RecordingCollection()
    db = _database(coll)
    payload = {"document_id": "d1", "source_key": POLICY, "status": "Active"}

    await write_chunks(
        MongoVectorIndex(db),
        MongoKeywordIndex(db),
        [
            ("c1", "data is encrypted at rest", [1.0, 0.0, 0.0], payload),
            ("c2", "keys are rotated yearly", [0.0, 1.0, 0.0], payload),
        ],
    )

    assert len(coll.batches) == 1
    ops = coll.batches[0]
    assert [op._filter for op in ops] == [{"_id": "c1"}, {"_id": "c2"}]
    assert all(op._upsert for op in ops)
    first = _set(ops[0])
    assert first["text"] == "data is encrypted at rest"
    assert list(first["embedding"].as_vector().data) == [1.0, 0.0, 0.0]
    assert first["kind"] == CHUNK and first["document_id"] == "d1"
    assert first["payload"] == payload  # the text is not copied into the payload
    assert "status=Active" in first["facet_kv"]


async def test_stores_without_a_shared_record_write_the_text_first() -> None:
    """Separate records (here: two database handles) cannot be written together, so
    the text goes first -- a chunk the vector search reaches always has text."""
    coll = RecordingCollection()
    vector, keyword = MongoVectorIndex(_database(coll)), MongoKeywordIndex(_database(coll))
    assert not vector.shares_records_with(keyword)

    await write_chunks(
        vector, keyword, [("c1", "some text", [1.0, 0.0, 0.0], {"document_id": "d1"})]
    )

    text_write, vector_write = (_set(batch[0]) for batch in coll.batches)
    assert text_write["text"] == "some text" and "embedding" not in text_write
    assert "embedding" in vector_write and "text" not in vector_write


async def test_nothing_is_written_for_no_chunks() -> None:
    coll = RecordingCollection()
    db = _database(coll)
    await write_chunks(MongoVectorIndex(db), MongoKeywordIndex(db), [])
    assert coll.batches == []


async def test_mongo_removes_chunks_by_id_in_one_delete() -> None:
    """A replaced file's leftovers: chunk records only, each id once."""
    coll = RecordingCollection()
    db = _database(coll)
    removed = await remove_chunks(MongoVectorIndex(db), MongoKeywordIndex(db), ["c1", "c2", "c1"])
    assert removed == 2
    assert coll.deletes == [{"_id": {"$in": ["c1", "c2"]}, "kind": CHUNK}]
    assert await remove_chunks(MongoVectorIndex(db), MongoKeywordIndex(db), []) == 0
    assert len(coll.deletes) == 1


# --------------------------------------------------------------------------
# Metadata record last
# --------------------------------------------------------------------------


async def test_a_document_whose_embedding_fails_is_not_marked_indexed(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    indexer, search, meta_store, closeables = build_stack(sample_objects)
    doc_id = document_id(BUCKET, POLICY)
    try:
        real_embed = indexer._embedder.embed

        async def throttled(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("ThrottlingException: Too many requests")

        indexer._embedder.embed = throttled  # type: ignore[method-assign]
        summary, _ = await indexer.index_keys([POLICY])
        assert summary.documents_failed == 1
        # No record, so nothing marks it indexed: the next run that sees it retries it.
        assert await meta_store.describe([doc_id]) == {}
        assert doc_id not in await meta_store.indexed_ids()

        indexer._embedder.embed = real_embed  # type: ignore[method-assign]
        summary, _ = await indexer.index_keys([POLICY])
        assert summary.documents_indexed == 1
        assert doc_id in await meta_store.indexed_ids()
        hits = (await search.search("encrypted at rest AES-256")).results
        assert hits and hits[0].source_key == POLICY
    finally:
        for c in closeables:
            await c.aclose()


async def test_a_document_with_no_text_still_gets_its_record(
    build_stack: Stack, sample_objects: dict[str, bytes]
) -> None:
    """Nothing to search, but it is still a document: listed, and not downloaded again."""
    blank = "TENANT_A/blank.txt"
    indexer, _search, meta_store, closeables = build_stack({**sample_objects, blank: b"   \n"})
    try:
        summary, _ = await indexer.index_keys([blank])
        assert summary.documents_indexed == 0 and summary.documents_failed == 0
        assert document_id(BUCKET, blank) in await meta_store.indexed_ids()
    finally:
        for c in closeables:
            await c.aclose()
