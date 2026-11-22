"""Incremental indexing and DeleteMark purging for a live tenant index.

Two scheduled jobs (``index_jobs.lambda_handler`` runs them from EventBridge; the CLI
is ``scripts/incremental_index.py``), one tenant collection each:

``IncrementalIndexer.run`` -- the safety net behind the S3 events and SpecSafe's calls
    1. List the tenant's S3 prefix. Candidates are objects modified since the last
       run's watermark, plus files still pending from earlier runs.
    2. Hand them to :class:`~rag_search.document_sync.DocumentSync` -- the same rules
       the event-driven path applies: index new files (hidden until their catalogue
       record exists), re-index replaced ones, apply changed records in place, purge
       deleted or other tenants' files, and give up on a file whose record never
       arrives after ``pending_max_age``.
    3. Advance the watermark once every candidate is done or parked, so a run cut off
       by the Lambda timeout repeats safely (finished documents come back unchanged).

``DeleteMarkPurger.run``
    Find this tenant's ``DeleteMark=1`` records and remove those documents' chunks,
    vectors and metadata from the index. No S3, no Bedrock.

State (watermark, pending files, run lock, per-document leases) lives in
``<collection>_IndexState`` -- see :mod:`rag_search.index_state`.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from rag_core.logging import get_logger
from rag_search.document_sync import DocumentSync, SyncAction, SyncOutcome
from rag_search.index_state import (
    LOCK_ID,
    PENDING,
    WATERMARK_ID,
    IndexState,
    InMemoryIndexState,
    MongoIndexState,
    PendingFile,
)
from rag_search.indexer import IndexingService
from rag_search.mongo_source import MongoMetadataSource, basename
from rag_search.stores import DocumentMetadataStore

_log = get_logger("rag_search.incremental")

# S3 LastModified is the upload START; a large upload still in flight when one run
# lists the prefix is seen by the next only if the windows overlap.
_WATERMARK_OVERLAP = timedelta(minutes=10)


class AlreadyRunning(RuntimeError):
    """Another run holds this index's lock."""


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


# --------------------------------------------------------------------------
# Incremental indexing
# --------------------------------------------------------------------------


@dataclass(slots=True)
class IncrementalSummary:
    collection: str
    bucket: str
    prefix: str
    listed: int = 0
    candidates: int = 0
    already_indexed: int = 0
    non_text: int = 0
    other_tenant: int = 0
    deleted: int = 0
    awaiting_metadata: int = 0  # indexed but hidden: no catalogue record yet
    pending_expired: int = 0
    indexed: int = 0  # new documents with their record: searchable
    hidden: int = 0  # new documents without one: embedded, not yet visible
    reindexed: int = 0
    metadata_updated: int = 0
    purged: int = 0
    chunks: int = 0
    failed: int = 0
    deferred: int = 0  # out of time this run; picked up by the next
    watermark: str | None = None
    duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}

    def add(self, outcome: SyncOutcome) -> None:
        action, reason = outcome.action, outcome.reason
        self.chunks += outcome.chunks
        if outcome.hidden:
            self.awaiting_metadata += 1
        if reason == "other_tenant":
            self.other_tenant += 1
        elif reason == "deleted":
            self.deleted += 1
        elif reason in ("non_text", "unsupported_format"):
            self.non_text += 1
        elif reason == "never_catalogued":
            self.pending_expired += 1
        if action is SyncAction.INDEXED:
            self.indexed += 1
        elif action is SyncAction.INDEXED_HIDDEN:
            self.hidden += 1
        elif action is SyncAction.REINDEXED:
            self.reindexed += 1
        elif action is SyncAction.METADATA_UPDATED:
            self.metadata_updated += 1
        elif action is SyncAction.UNCHANGED:
            self.already_indexed += 1
        elif action is SyncAction.PURGED:
            self.purged += 1
        elif action in (SyncAction.DEFERRED, SyncAction.BUSY):
            self.deferred += 1
        elif action is SyncAction.FAILED:
            self.failed += 1
            if len(self.errors) < 5:
                self.errors.append(f"{outcome.key}: {outcome.error}")


class IncrementalIndexer:
    def __init__(
        self,
        *,
        indexer: IndexingService,
        metadata_store: DocumentMetadataStore,
        mongo: MongoMetadataSource,
        state: IndexState,
        s3: Any,
        bucket: str,
        prefix: str,
        collection: str = "",
        namespace: str = "default",
        initial_lookback: timedelta = timedelta(days=1),
        pending_max_age: timedelta = timedelta(days=7),
        lock_ttl: timedelta = timedelta(minutes=20),
    ) -> None:
        self._state = state
        self._s3 = s3
        self._bucket = bucket
        self._prefix = prefix
        self._collection = collection
        self._initial_lookback = initial_lookback
        self._lock_ttl = lock_ttl
        self._sync = DocumentSync(
            indexer=indexer,
            metadata_store=metadata_store,
            mongo=mongo,
            state=state,
            s3=s3,
            bucket=bucket,
            namespace=namespace,
            pending_max_age=pending_max_age,
        )

    async def run(self, *, deadline: float | None = None) -> IncrementalSummary:
        """One sweep. ``deadline`` is a ``time.monotonic()`` instant to stop embedding by."""
        owner = uuid.uuid4().hex
        if not await self._state.acquire(owner, self._lock_ttl):
            raise AlreadyRunning(f"{self._collection or 'index'}: another run holds the lock")
        try:
            return await self._sweep(deadline)
        finally:
            await self._state.release(owner)

    async def _sweep(self, deadline: float | None) -> IncrementalSummary:
        started = time.perf_counter()
        now = datetime.now(UTC)
        out = IncrementalSummary(self._collection, self._bucket, self._prefix)

        # Candidates: new since the watermark, plus files still pending. The listing's
        # ETags vouch for each object, so the sync needs no HEAD for them.
        watermark = await self._state.watermark() or now - self._initial_lookback
        since = watermark - _WATERMARK_OVERLAP
        candidates: dict[str, None] = dict.fromkeys(p.key for p in await self._state.pending())
        etags: dict[str, str] = {}
        async for obj in self._s3.list_objects(self._bucket, self._prefix):
            out.listed += 1
            modified = obj.last_modified
            if obj.key in candidates or modified is None or (_aware(modified) or now) > since:
                candidates.setdefault(obj.key)
                if obj.etag:
                    etags[obj.key] = obj.etag
        out.candidates = len(candidates)

        for outcome in await self._sync.sync_many(candidates, etags=etags, deadline=deadline):
            out.add(outcome)

        # Failed files are retried only if they change again in S3 (a broken PDF would
        # otherwise be re-downloaded every run); deferred ones were parked by the sync,
        # so the next run resumes them without waiting for S3 to change.
        await self._state.set_watermark(now)
        out.watermark = now.isoformat()
        out.duration_s = round(time.perf_counter() - started, 3)
        _log.info("incremental_index_complete", **out.as_dict())
        return out


# --------------------------------------------------------------------------
# DeleteMark purge
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PurgeSummary:
    collection: str
    deleted_in_catalogue: int = 0
    indexed_documents: int = 0
    purged: int = 0
    dry_run: bool = False
    sample: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class DeleteMarkPurger:
    def __init__(
        self,
        *,
        indexer: IndexingService,
        metadata_store: DocumentMetadataStore,
        mongo: MongoMetadataSource,
        collection: str = "",
        namespace: str = "default",
    ) -> None:
        self._indexer = indexer
        self._metadata = metadata_store
        self._mongo = mongo
        self._collection = collection
        self._namespace = namespace

    async def run(self, *, dry_run: bool = False) -> PurgeSummary:
        started = time.perf_counter()
        out = PurgeSummary(self._collection, dry_run=dry_run)
        deleted = await self._mongo.deleted_names()
        out.deleted_in_catalogue = len(deleted)
        indexed = await self._metadata.all_documents()
        out.indexed_documents = len(indexed)
        doomed = [key for _, key in indexed if basename(key) in deleted]
        out.sample = [basename(k) for k in doomed[:10]]
        if not dry_run:
            for key in doomed:
                await self._indexer.purge_document(key, self._namespace)
        out.purged = len(doomed)
        out.duration_s = round(time.perf_counter() - started, 3)
        _log.info("delete_mark_purge_complete", **out.as_dict())
        return out


__all__ = [
    "LOCK_ID",
    "PENDING",
    "WATERMARK_ID",
    "AlreadyRunning",
    "DeleteMarkPurger",
    "InMemoryIndexState",
    "IncrementalIndexer",
    "IncrementalSummary",
    "IndexState",
    "MongoIndexState",
    "PendingFile",
    "PurgeSummary",
]
