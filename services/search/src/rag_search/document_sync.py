"""Bring documents in a live index into line with S3 and the catalogue.

Three things change a document: its file (uploaded, replaced, removed in S3), its
``AttachmentMetaData`` record (created, edited, ``DeleteMark`` set), and time (a file
whose record never arrives). Three triggers report them:

* S3 ``Object Created`` / ``Object Deleted`` events (EventBridge -> SQS);
* SpecSafe's "document changed" call (Function URL -> SQS), which names files only;
* the scheduled sweep, the safety net for anything the other two missed.

All of them call :meth:`DocumentSync.sync_many`. It never trusts a trigger's account
of what happened: it re-reads the file's existence and ETag from S3 and its record from
the catalogue, then applies one rule. So it does not matter which trigger arrives
first, twice, or not at all -- the next one converges on the same state.

A file with no record yet is embedded straight away but stays HIDDEN: every search
filters on ``tenant``, ``business_object`` and the user's ``object_code``, which only
the record supplies, and tenants share S3 folders, so until the record says whose file
it is, nobody may see it. When the record arrives its facets are applied in place
(:meth:`IndexingService.update_catalogue_metadata`) -- no re-download, no re-embedding
-- and the document is searchable at once.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from rag_core.logging import get_logger
from rag_core.utils import document_id
from rag_search.index_state import IndexState, PendingFile
from rag_search.indexer import (
    CATALOGUE_FACETS,
    EXTRA_CATALOGUE_HASH,
    EXTRA_SOURCE_ETAG,
    IndexingService,
    skip_reason,
)
from rag_search.mongo_source import MongoMetadataSource
from rag_search.stores import DocumentMetadataStore

_log = get_logger("rag_search.document_sync")

_DESCRIBE_BATCH = 1000


class SyncAction(StrEnum):
    INDEXED = "indexed"  # new document with its record: searchable
    INDEXED_HIDDEN = "indexed_hidden"  # new document, no record yet: embedded, invisible
    REINDEXED = "reindexed"  # the file was replaced in S3
    METADATA_UPDATED = "metadata_updated"  # record changed; applied in place
    UNCHANGED = "unchanged"  # already up to date
    AWAITING_METADATA = "awaiting_metadata"  # indexed, still hidden: no record yet
    AWAITING_FILE = "awaiting_file"  # nothing in S3 (yet): its upload event will follow
    PURGED = "purged"  # removed from the index -- see ``reason``
    SKIPPED = "skipped"  # not this index's to hold, and not in it -- see ``reason``
    FAILED = "failed"
    BUSY = "busy"  # another worker holds this document right now
    DEFERRED = "deferred"  # out of time; parked for the next run


# Outcomes a queue should redeliver; everything else is final for this trigger.
RETRY_ACTIONS = frozenset({SyncAction.FAILED, SyncAction.BUSY, SyncAction.DEFERRED})


@dataclass(slots=True)
class SyncOutcome:
    key: str
    action: SyncAction
    chunks: int = 0
    reason: str = ""
    error: str = ""
    # Indexed without a catalogue record: tracked as pending until one arrives.
    hidden: bool = False

    @property
    def retry(self) -> bool:
        return self.action in RETRY_ACTIONS


class DocumentSync:
    def __init__(
        self,
        *,
        indexer: IndexingService,
        metadata_store: DocumentMetadataStore,
        mongo: MongoMetadataSource,
        state: IndexState,
        s3: Any,  # needs head_object_or_none (rag_core.aws.S3Client)
        bucket: str,
        namespace: str = "default",
        pending_max_age: timedelta = timedelta(days=7),
        lease_ttl: timedelta = timedelta(minutes=5),
    ) -> None:
        self._indexer = indexer
        self._metadata = metadata_store
        self._mongo = mongo
        self._state = state
        self._s3 = s3
        self._bucket = bucket
        self._namespace = namespace
        self._pending_max_age = pending_max_age
        self._lease_ttl = lease_ttl

    async def resolve_key(self, candidates: Sequence[str]) -> str:
        """The S3 key meant among spellings of one file (``TENANT_B/x`` vs ``TENANT_B//x``): the one
        already indexed, else the one that exists in S3, else -- for a file kept in a
        subfolder (``TENANT_A/COMPDOCCAT/<name>``) -- an indexed document of that file name
        under the same tenant folder, else the first."""
        if len(candidates) <= 1:
            return candidates[0]
        ids = {document_id(self._bucket, k): k for k in candidates}
        indexed = await self._metadata.describe(list(ids), self._namespace)
        for doc_id, key in ids.items():
            if doc_id in indexed:
                return key
        for key in candidates:
            if await self._s3.head_object_or_none(self._bucket, key) is not None:
                return key
        folder, _, name = candidates[0].rpartition("/")
        elsewhere = [
            k
            for k in await self._metadata.source_keys_named(name, self._namespace)
            if k.startswith(f"{folder}/")
        ]
        return elsewhere[0] if elsewhere else candidates[0]

    async def sync_many(
        self,
        keys: Iterable[str],
        *,
        etags: Mapping[str, str] | None = None,
        deadline: float | None = None,
    ) -> list[SyncOutcome]:
        """Sync each key once (duplicates collapse). ``etags`` -- from a listing -- vouch
        that those objects exist; any other key is checked with a HEAD. ``deadline`` is
        a ``time.monotonic()`` instant: keys not started by then come back DEFERRED."""
        ordered = list(dict.fromkeys(k for k in keys if k))
        outcomes: list[SyncOutcome] = []
        text_keys: list[str] = []
        for key in ordered:
            if reason := skip_reason(key):  # not a text format we can extract
                outcomes.append(SyncOutcome(key, SyncAction.SKIPPED, reason=reason))
            else:
                text_keys.append(key)
        if not text_keys:
            return outcomes

        # Bulk reads for the batch: catalogue records, what is indexed, what is parked.
        await self._mongo.load_keys(text_keys)
        ids = {k: document_id(self._bucket, k) for k in text_keys}
        id_list = list(ids.values())
        indexed: set[str] = set()
        for start in range(0, len(id_list), _DESCRIBE_BATCH):
            chunk = id_list[start : start + _DESCRIBE_BATCH]
            indexed.update(await self._metadata.describe(chunk, self._namespace))
        parked = {p.key: p for p in await self._state.pending()}

        now = datetime.now(UTC)
        owner = uuid.uuid4().hex
        park: list[PendingFile] = []
        drop: list[str] = []
        for pos, key in enumerate(text_keys):
            if deadline is not None and time.monotonic() >= deadline:
                for rest in text_keys[pos:]:
                    outcomes.append(SyncOutcome(rest, SyncAction.DEFERRED))
                    park.append(parked.get(rest) or PendingFile(key=rest, first_seen=now))
                break
            doc_id = ids[key]
            if not await self._state.acquire_doc(doc_id, owner, self._lease_ttl):
                outcomes.append(SyncOutcome(key, SyncAction.BUSY))
                continue
            try:
                outcome = await self._sync_one(
                    key, doc_id, doc_id in indexed, (etags or {}).get(key), parked.get(key), now
                )
            except Exception as exc:  # one bad document never stops the batch
                outcome = SyncOutcome(key, SyncAction.FAILED, error=str(exc)[:300] or repr(exc))
            finally:
                await self._state.release_doc(doc_id, owner)
            outcomes.append(outcome)
            if outcome.hidden:
                park.append(parked.get(key) or PendingFile(key=key, first_seen=now))
            elif key in parked and outcome.action is not SyncAction.FAILED:
                drop.append(key)

        await self._state.drop_pending(drop)
        await self._state.put_pending(park)
        for outcome in outcomes:
            log = _log.warning if outcome.action is SyncAction.FAILED else _log.info
            log(
                "document_sync",
                key=outcome.key,
                action=outcome.action.value,
                reason=outcome.reason or None,
                chunks=outcome.chunks or None,
                error=outcome.error or None,
            )
        return outcomes

    async def _sync_one(
        self,
        key: str,
        doc_id: str,
        is_indexed: bool,
        etag: str | None,
        parked: PendingFile | None,
        now: datetime,
    ) -> SyncOutcome:
        ns = self._namespace

        # Not this index's document: the catalogue says another tenant's, or deleted.
        # Purge unconditionally -- cheap, and it also clears chunks a run cut off
        # before the metadata record left behind.
        if self._mongo.is_foreign(key) or self._mongo.is_deleted(key):
            reason = "other_tenant" if self._mongo.is_foreign(key) else "deleted"
            await self._indexer.purge_document(key, ns)
            return SyncOutcome(
                key, SyncAction.PURGED if is_indexed else SyncAction.SKIPPED, reason=reason
            )

        if etag is None:
            head = await self._s3.head_object_or_none(self._bucket, key)
            if head is None:
                if is_indexed:
                    await self._indexer.purge_document(key, ns)
                    return SyncOutcome(key, SyncAction.PURGED, reason="file_removed")
                return SyncOutcome(key, SyncAction.AWAITING_FILE)
            etag = str(head.get("ETag") or "").strip('"') or None

        record = self._mongo.lookup(key)
        if (
            record is None
            and parked is not None
            and now - parked.first_seen > self._pending_max_age
        ):
            # Waited long enough: a record is not coming. Don't keep (or start) paying
            # to hold an invisible document.
            await self._indexer.purge_document(key, ns)
            return SyncOutcome(
                key,
                SyncAction.PURGED if is_indexed else SyncAction.SKIPPED,
                reason="never_catalogued",
            )

        if not is_indexed:
            # ``replace`` also clears chunks an earlier, cut-off attempt left behind.
            chunks = await self._indexer.index_document(key, ns, etag=etag, replace=True)
            if record is None:
                return SyncOutcome(key, SyncAction.INDEXED_HIDDEN, chunks=chunks, hidden=True)
            return SyncOutcome(key, SyncAction.INDEXED, chunks=chunks)

        meta = await self._metadata.get(doc_id)
        stored = meta.extra.get(EXTRA_SOURCE_ETAG) if meta is not None else None
        if etag and stored and etag != stored:
            chunks = await self._indexer.index_document(key, ns, etag=etag, replace=True)
            return SyncOutcome(
                key,
                SyncAction.REINDEXED,
                chunks=chunks,
                reason="file_replaced",
                hidden=record is None,
            )
        if etag and not stored:
            # Indexed before ETags were recorded: adopt the current one rather than
            # re-embed every such document the first time anything touches it.
            await self._metadata.apply_facets(doc_id, {}, extra={EXTRA_SOURCE_ETAG: etag})

        if record is None:
            if meta is not None and _was_catalogued(meta.extra):
                # It had a record and the record is gone -- treat as deleted.
                await self._indexer.purge_document(key, ns)
                return SyncOutcome(key, SyncAction.PURGED, reason="record_removed")
            return SyncOutcome(key, SyncAction.AWAITING_METADATA, hidden=True)

        changed = await self._indexer.update_catalogue_metadata(key, record, ns)
        return SyncOutcome(key, SyncAction.METADATA_UPDATED if changed else SyncAction.UNCHANGED)


def _was_catalogued(extra: Mapping[str, str]) -> bool:
    """Whether a document was indexed or updated with a catalogue record: it carries
    the record's fingerprint, or (indexed before fingerprints) the record's values."""
    return bool(extra.get(EXTRA_CATALOGUE_HASH)) or any(k in extra for k in CATALOGUE_FACETS)


__all__ = ["RETRY_ACTIONS", "DocumentSync", "SyncAction", "SyncOutcome"]
