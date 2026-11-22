"""Index jobs: the Lambda entry points and the wiring they share with the CLI.

One image, one handler per function (``ImageConfig.Command``):

``sqs_handler`` -- the ``index-events`` queue: S3 ``Object Created`` / ``Object Deleted``
events (via EventBridge), SpecSafe "document changed" messages (queued by
``rag_search.sync_api.api_handler``, the Function URL) and this worker's own delayed
re-checks. Each names one file; :class:`~rag_search.document_sync.DocumentSync`
re-reads S3 and the catalogue and brings that document up to date.

``lambda_handler`` -- EventBridge Scheduler and manual runs::

    {"job": "purge", "target": "TENANT_B", "dry_run": false}  # DeleteMark=1 purge (daily)
    {"job": "index", "target": "TENANT_A"}                    # a full sweep, by hand only

``target`` names an entry of ``INDEX_TARGETS`` (JSON in the environment)::

    {
      "TENANT_A":  {"collection": "TENANT_A_DocumentSearch_DEV",  "metadata_database": "TENANT_A_DEV"},
      "TENANT_B": {"collection": "TENANT_B_DocumentSearch_DEV", "metadata_database": "TENANT_B_DEV"}
    }

Per target, ``bucket``/``prefix``/``tenant_codes`` default to what the collection's
manifest recorded at build time (``scripts/mongo_index.py manifest``), so the job
always feeds the index from the source it was built from. ``metadata_collection``
defaults to ``SEARCH_MONGO_COLLECTION``.

The Atlas connection string is the function's ``SEARCH_MONGO_URI`` environment
variable; AWS access is the function's IAM role -- no keys.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from urllib.parse import unquote_plus

from rag_core.config import S3Settings, Settings, get_settings
from rag_core.logging import get_logger
from rag_search.config import SearchSettings, StoreBackend
from rag_search.document_sync import DocumentSync
from rag_search.factory import Components, build_components, build_mongo_index
from rag_search.incremental import (
    AlreadyRunning,
    DeleteMarkPurger,
    IncrementalIndexer,
    MongoIndexState,
)
from rag_search.stores import MongoVectorIndex
from rag_search.sync_api import SPECSAFE_SOURCE, _codes, _configure_logging, load_targets

_log = get_logger("rag_search.index_jobs")


@dataclass(slots=True)
class Target:
    name: str
    collection: str
    bucket: str
    prefix: str
    tenant_codes: str
    metadata_database: str
    metadata_collection: str


async def resolve_target(name: str, spec: dict[str, Any], search: SearchSettings) -> Target:
    """Fill gaps in a target from its collection's manifest."""
    if not spec.get("collection"):
        raise ValueError(f"target {name}: 'collection' is required")
    manifest: dict[str, Any] = {}
    if not all(spec.get(k) for k in ("bucket", "prefix", "tenant_codes")):
        db = build_mongo_index(
            search.model_copy(update={"index_mongo_collection": spec["collection"]}),
            dim=search.embed_dim,
        )
        try:
            manifest = await db.read_manifest()
        finally:
            await db.aclose()
    codes = spec.get("tenant_codes") or manifest.get("tenant_codes") or ""
    target = Target(
        name=name,
        collection=spec["collection"],
        bucket=spec.get("bucket") or manifest.get("source_bucket") or "",
        prefix=str(spec["prefix"] if spec.get("prefix") is not None else manifest.get("source_prefix") or ""),
        tenant_codes=",".join(codes) if isinstance(codes, list) else str(codes),
        metadata_database=spec.get("metadata_database") or search.mongo_database,
        metadata_collection=spec.get("metadata_collection") or search.mongo_collection,
    )
    if not target.bucket:
        raise ValueError(f"target {name}: no bucket configured and none in the manifest")
    if not target.tenant_codes:
        # Tenants share S3 folders and catalogue databases: unfiltered, this index
        # would take in every tenant's files.
        raise ValueError(f"target {name}: no tenant_codes configured and none in the manifest")
    return target


def build_for_target(target: Target, core: Settings, search: SearchSettings) -> Components:
    core = core.model_copy(update={"s3": S3Settings(bucket=target.bucket, prefix=target.prefix)})
    search = search.model_copy(
        update={
            "store_backend": StoreBackend.MONGO,
            "index_mongo_collection": target.collection,
            "mongo_database": target.metadata_database,
            "mongo_collection": target.metadata_collection,
            "mongo_tenant_codes": target.tenant_codes,
        }
    )
    if not search.mongo_uri:
        raise ValueError("SEARCH_MONGO_URI is required")
    return build_components(core, search)


def _pending_max_age() -> timedelta:
    """How long a file indexed without a catalogue record is kept (hidden) waiting for
    one. ``INDEX_RECORD_WAIT_MIN`` (the event worker's setting) wins over the older
    ``INDEX_PENDING_MAX_AGE_D``."""
    if minutes := os.environ.get("INDEX_RECORD_WAIT_MIN", "").strip():
        return timedelta(minutes=float(minutes))
    return timedelta(days=float(os.environ.get("INDEX_PENDING_MAX_AGE_D", "7")))


class RefusedTarget(RuntimeError):
    """A target this process must not write to (see :func:`open_sync`)."""


@contextlib.asynccontextmanager
async def open_sync(
    target: Target, core: Settings, search: SearchSettings
) -> AsyncIterator[DocumentSync]:
    """A :class:`DocumentSync` over one target's collection, closed on exit.

    Refuses a collection whose manifest declares no tenant codes: the agent filters
    searches by the manifest's tenant codes, and without them it would not -- so a
    document indexed before its catalogue record (hidden only by that filter) would be
    visible to everyone.
    """
    components = build_for_target(target, core, search)
    try:
        assert isinstance(components.vector, MongoVectorIndex)
        db = components.vector._db
        await db.ensure_schema()
        manifest = await db.read_manifest()
        if not manifest.get("tenant_codes"):
            raise RefusedTarget(
                f"target {target.name}: collection {target.collection!r} has no tenant_codes "
                "in its manifest; run scripts/mongo_index.py manifest before syncing into it"
            )
        mongo = components.indexer._mongo
        assert mongo is not None  # build_for_target requires the catalogue
        yield DocumentSync(
            indexer=components.indexer,
            metadata_store=components.metadata,
            mongo=mongo,
            state=MongoIndexState(db),
            s3=components.indexer._s3,
            bucket=target.bucket,
            namespace=search.namespace,
            pending_max_age=_pending_max_age(),
        )
    finally:
        await components.aclose()


def object_key_candidates(target: Target, file: str) -> tuple[str, ...]:
    """The S3 keys SpecSafe may mean by ``file``, most likely first.

    A value with a ``/`` is already a key. A bare file name (``GeneratedFileName``) lives
    directly under the target's prefix -- written ``TENANT_A/<name>`` for TENANT_A, but ``TENANT_B//<name>``
    for much of TENANT_B -- so both spellings are offered and
    :meth:`DocumentSync.resolve_key` picks the one that is indexed or exists.
    """
    if "/" in file:
        return (file,)
    prefix = target.prefix.strip("/")
    return (f"{prefix}/{file}", f"{prefix}//{file}") if prefix else (file,)


def _under_prefix(target: Target, key: str) -> bool:
    prefix = target.prefix.strip("/")
    return not prefix or key.startswith(f"{prefix}/")


async def run_job(
    job: str,
    target_name: str,
    *,
    deadline: float | None = None,
    dry_run: bool = False,
    files: Iterable[str] = (),
    core: Settings | None = None,
    search: SearchSettings | None = None,
) -> dict[str, Any]:
    core = core or get_settings()
    search = search or SearchSettings()
    targets = load_targets()
    key = target_name.upper()
    if key not in targets:
        raise ValueError(f"unknown target {target_name!r}; configured: {sorted(targets)}")
    target = await resolve_target(key, targets[key], search)
    if job == "sync":
        # The event-driven path, run by hand: re-sync named files now.
        if not files:
            raise ValueError("job 'sync' needs at least one file")
        async with open_sync(target, core, search) as sync:
            keys = [await sync.resolve_key(object_key_candidates(target, f)) for f in files]
            outcomes = await sync.sync_many(keys, deadline=deadline)
        return {
            "status": "ok",
            "job": job,
            "target": key,
            "results": [
                {"key": o.key, "action": o.action.value, "reason": o.reason, "error": o.error}
                for o in outcomes
            ],
        }
    components = build_for_target(target, core, search)
    try:
        mongo = components.indexer._mongo
        assert mongo is not None  # build_for_target requires the catalogue
        if job == "index":
            assert isinstance(components.vector, MongoVectorIndex)
            db = components.vector._db
            await db.ensure_schema()
            indexer = IncrementalIndexer(
                indexer=components.indexer,
                metadata_store=components.metadata,
                mongo=mongo,
                state=MongoIndexState(db),
                s3=components.indexer._s3,
                bucket=target.bucket,
                prefix=target.prefix,
                collection=target.collection,
                namespace=search.namespace,
                initial_lookback=timedelta(hours=float(os.environ.get("INDEX_INITIAL_LOOKBACK_H", "24"))),
                pending_max_age=_pending_max_age(),
            )
            try:
                result = (await indexer.run(deadline=deadline)).as_dict()
            except AlreadyRunning as exc:
                _log.warning("incremental_index_skipped", reason=str(exc))
                return {"status": "skipped", "reason": str(exc), "target": key}
        elif job == "purge":
            purger = DeleteMarkPurger(
                indexer=components.indexer,
                metadata_store=components.metadata,
                mongo=mongo,
                collection=target.collection,
                namespace=search.namespace,
            )
            result = (await purger.run(dry_run=dry_run)).as_dict()
        else:
            raise ValueError(f"unknown job {job!r}; expected 'index', 'purge' or 'sync'")
    finally:
        await components.aclose()
    return {"status": "ok", "job": job, "target": key, **result}


# --------------------------------------------------------------------------
# Event-driven sync: S3 events + SpecSafe calls, through the index-events queue
# --------------------------------------------------------------------------

RECHECK_SOURCE = "recheck"
# A message to retry (document busy, out of time, failed) comes back after this long
# rather than after the queue's full visibility timeout.
RETRY_DELAY_S = 60
# A file indexed before its catalogue record is looked at again after 1, 2, 4, 8 and
# then every 15 minutes (SQS's longest delay), until the record arrives or the wait
# (INDEX_RECORD_WAIT_MIN) is over and it is purged. The cap is only a backstop.
_RECHECK_FIRST_DELAY_S = 60
_RECHECK_MAX_DELAY_S = 900
MAX_RECHECKS = 12
# Resolved targets are plain data: kept across invocations, manifests re-read now and then.
_TARGETS_TTL_S = 600.0
_resolved_targets: tuple[float, dict[str, Target]] | None = None


def recheck_delay_s(attempt: int) -> int:
    """Seconds before re-check number ``attempt`` (1-based)."""
    return min(_RECHECK_FIRST_DELAY_S * 2 ** max(attempt - 1, 0), _RECHECK_MAX_DELAY_S)


@dataclass(slots=True, frozen=True)
class SyncRequest:
    """One file a queue message asks to sync."""

    message_id: str
    bucket: str = ""  # S3 events name the object itself...
    key: str = ""
    target: str = ""  # ...SpecSafe calls (and re-checks) name a target and a file in it
    tenant_code: str = ""
    file: str = ""
    reason: str = ""
    attempt: int = 0  # re-checks: which one this is


@dataclass(slots=True)
class SyncBatchResult:
    """What a batch of queue messages left to do."""

    retry: set[str]  # message ids to redeliver (busy, out of time, failed)
    # Re-checks to queue for files still waiting for their catalogue record, each with
    # the message ids that led to it (redelivered if the re-check cannot be queued).
    recheck: list[tuple[dict[str, Any], set[str]]]


def parse_message(message_id: str, body: str) -> list[SyncRequest]:
    """The files one ``index-events`` message names.

    Four shapes arrive: EventBridge S3 events (the deployed path), classic S3 event
    notifications (if a bucket is ever wired to the queue directly; their keys are
    URL-encoded), SpecSafe messages queued by :func:`api_handler`, and this worker's
    own delayed re-checks. Raises ``ValueError`` for anything else.
    """
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    if data.get("source") == RECHECK_SOURCE:
        target, key = str(data.get("target") or ""), str(data.get("key") or "")
        if not target or not key:
            raise ValueError("re-check without a target or key")
        return [
            SyncRequest(
                message_id,
                target=target.upper(),
                key=key,
                reason="recheck",
                attempt=int(data.get("attempt") or 0),
            )
        ]
    if data.get("source") == SPECSAFE_SOURCE:
        target, file = str(data.get("target") or ""), str(data.get("file") or "")
        if not target or not file:
            raise ValueError("SpecSafe message without a target or file")
        return [
            SyncRequest(
                message_id,
                target=target.upper(),
                tenant_code=str(data.get("tenant_code") or ""),
                file=file,
                reason=str(data.get("reason") or ""),
            )
        ]
    if data.get("source") == "aws.s3":
        detail = data.get("detail") or {}
        bucket = str((detail.get("bucket") or {}).get("name") or "")
        key = str((detail.get("object") or {}).get("key") or "")
        if not bucket or not key:
            raise ValueError("S3 event without a bucket or key")
        return [SyncRequest(message_id, bucket=bucket, key=key, reason=str(data.get("detail-type") or ""))]
    if isinstance(data.get("Records"), list):
        out = []
        for rec in data["Records"]:
            s3 = (rec or {}).get("s3") or {}
            bucket = str((s3.get("bucket") or {}).get("name") or "")
            key = unquote_plus(str((s3.get("object") or {}).get("key") or ""))
            if bucket and key:
                out.append(SyncRequest(message_id, bucket=bucket, key=key, reason=str(rec.get("eventName") or "")))
        return out
    if data.get("Event") == "s3:TestEvent":  # sent once when a notification is configured
        return []
    raise ValueError("unrecognised message")


def _route(req: SyncRequest, targets: dict[str, Target]) -> list[tuple[Target, tuple[str, ...]]]:
    """The targets a request applies to, each with the candidate S3 key(s) it means."""
    if req.target:
        target = targets.get(req.target)
        if target is None:
            _log.error("sync_unknown_target", target=req.target, file=req.file or req.key)
            return []
        if req.key:  # a re-check: the key was resolved when the file was first synced
            return [(target, (req.key,))]
        if req.tenant_code.strip().upper() not in _codes(target.tenant_codes):
            _log.error(
                "sync_tenant_mismatch", target=target.name, tenant_code=req.tenant_code, file=req.file
            )
            return []
        candidates = object_key_candidates(target, req.file)
        if not all(_under_prefix(target, k) for k in candidates):
            _log.error("sync_key_outside_target", target=target.name, key=candidates[0])
            return []
        return [(target, candidates)]
    # An S3 event goes to every target reading this bucket and prefix. Tenants share
    # folders, so several can match; each one's catalogue decides whose file it is.
    matched = [
        (t, (req.key,))
        for t in targets.values()
        if t.bucket == req.bucket and _under_prefix(t, req.key)
    ]
    if not matched:
        _log.info("sync_event_ignored", bucket=req.bucket, key=req.key)
    return matched


OpenSync = Callable[[Target], contextlib.AbstractAsyncContextManager[DocumentSync]]


async def handle_sync_messages(
    messages: list[tuple[str, str]],
    *,
    targets: dict[str, Target],
    open_target: OpenSync,
    deadline: float | None = None,
) -> SyncBatchResult:
    """Sync every file the ``(message id, body)`` pairs name.

    A message naming one file in two targets is retried if either fails. Messages
    naming the same file collapse into one sync. An unreadable message is logged and
    dropped -- retrying cannot make it readable. A file still waiting for its catalogue
    record gets a delayed re-check (see ``recheck_delay_s``).
    """
    # target -> candidate keys -> (message ids, highest re-check attempt among them)
    work: dict[str, dict[tuple[str, ...], tuple[set[str], int]]] = {}
    for mid, body in messages:
        try:
            requests = parse_message(mid, body)
        except ValueError as exc:
            _log.error("sync_message_unreadable", message_id=mid, error=str(exc), body=body[:500])
            continue
        for req in requests:
            for target, candidates in _route(req, targets):
                ids, attempt = work.setdefault(target.name, {}).get(candidates, (set(), 0))
                work[target.name][candidates] = ({*ids, mid}, max(attempt, req.attempt))

    result = SyncBatchResult(retry=set(), recheck=[])
    for name, files in work.items():
        try:
            async with open_target(targets[name]) as sync:
                by_key: dict[str, tuple[set[str], int]] = {}
                for candidates, (ids, attempt) in files.items():
                    key = await sync.resolve_key(candidates)
                    seen_ids, seen_attempt = by_key.get(key, (set(), 0))
                    by_key[key] = (seen_ids | ids, max(seen_attempt, attempt))
                outcomes = await sync.sync_many(list(by_key), deadline=deadline)
        except Exception as exc:
            _log.error("sync_target_failed", target=name, error=str(exc)[:500])
            for ids, _ in files.values():
                result.retry.update(ids)
            continue
        for outcome in outcomes:
            ids, attempt = by_key.get(outcome.key, (set(), 0))
            if outcome.retry:
                result.retry.update(ids)
            elif outcome.hidden:
                if attempt < MAX_RECHECKS:
                    result.recheck.append(
                        ({"target": name, "key": outcome.key, "attempt": attempt + 1}, ids)
                    )
                else:
                    _log.warning("sync_recheck_gave_up", target=name, key=outcome.key)
    return result


async def _targets(search: SearchSettings) -> dict[str, Target]:
    global _resolved_targets
    now = time.monotonic()
    if _resolved_targets is not None and now - _resolved_targets[0] < _TARGETS_TTL_S:
        return _resolved_targets[1]
    resolved = {
        name: await resolve_target(name, spec, search) for name, spec in load_targets().items()
    }
    _resolved_targets = (now, resolved)
    return resolved


# --------------------------------------------------------------------------
# Lambda
# --------------------------------------------------------------------------

_bootstrapped = False


def _bootstrap() -> None:
    """Once per container: logging, and settings read fresh from the environment
    (``SEARCH_MONGO_URI`` is the function's own environment variable)."""
    global _bootstrapped
    if _bootstrapped:
        return
    _configure_logging()
    get_settings.cache_clear()
    _bootstrapped = True


def _deadline(context: Any) -> float | None:
    if context is None or not hasattr(context, "get_remaining_time_in_millis"):
        return None
    # Leave room for the document in flight (its own timeout) plus state writes.
    margin_s = SearchSettings().doc_timeout_s + 60
    return time.monotonic() + context.get_remaining_time_in_millis() / 1000 - margin_s


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    _bootstrap()
    job = str(event.get("job") or "index")
    target = str(event.get("target") or "")
    if not target:
        raise ValueError("event needs a 'target' (one of INDEX_TARGETS)")
    result = asyncio.run(
        run_job(job, target, deadline=_deadline(context), dry_run=bool(event.get("dry_run")))
    )
    _log.info("index_job_result", **{k: v for k, v in result.items() if k != "errors"})
    return result


def _retry_soon(records: list[dict[str, Any]]) -> None:
    """Bring messages to retry back after ``RETRY_DELAY_S`` instead of the queue's full
    visibility timeout. Best effort: if it fails they simply come back later."""
    queue_url = os.environ.get("INDEX_EVENTS_QUEUE_URL", "").strip()
    entries = [
        {"Id": str(i), "ReceiptHandle": r["receiptHandle"], "VisibilityTimeout": RETRY_DELAY_S}
        for i, r in enumerate(records)
        if r.get("receiptHandle")
    ]
    if not queue_url or not entries:
        return
    import boto3

    client = boto3.client("sqs")
    for start in range(0, len(entries), 10):
        try:
            client.change_message_visibility_batch(
                QueueUrl=queue_url, Entries=entries[start : start + 10]
            )
        except Exception as exc:
            _log.warning("sync_retry_visibility_failed", error=str(exc)[:300])


def _queue_rechecks(rechecks: list[dict[str, Any]]) -> set[int]:
    """Queue delayed re-checks; the indexes (into ``rechecks``) that could not be queued."""
    queue_url = os.environ.get("INDEX_EVENTS_QUEUE_URL", "").strip()
    if not rechecks:
        return set()
    if not queue_url:
        _log.error("sync_recheck_not_queued", error="INDEX_EVENTS_QUEUE_URL is not set")
        return set(range(len(rechecks)))
    import boto3

    client = boto3.client("sqs")
    failed: set[int] = set()
    for start in range(0, len(rechecks), 10):
        part = list(enumerate(rechecks))[start : start + 10]
        entries = [
            {
                "Id": str(i),
                "MessageBody": json.dumps({"source": RECHECK_SOURCE, **r}),
                "DelaySeconds": recheck_delay_s(int(r["attempt"])),
            }
            for i, r in part
        ]
        try:
            resp = client.send_message_batch(QueueUrl=queue_url, Entries=entries)
            failed.update(int(f["Id"]) for f in resp.get("Failed") or [])
        except Exception as exc:
            _log.error("sync_recheck_not_queued", error=str(exc)[:300])
            failed.update(i for i, _ in part)
    return failed


def sqs_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """The ``index-events`` queue: sync each named file, report the ones to retry."""
    _bootstrap()
    records = [r for r in event.get("Records") or [] if isinstance(r, dict)]
    messages = [(str(r.get("messageId") or ""), str(r.get("body") or "")) for r in records]
    deadline = _deadline(context)
    core, search = get_settings(), SearchSettings()

    async def _run() -> SyncBatchResult:
        return await handle_sync_messages(
            messages,
            targets=await _targets(search),
            open_target=lambda t: open_sync(t, core, search),
            deadline=deadline,
        )

    result = asyncio.run(_run())
    # A re-check that could not be queued: redeliver the message(s) that led to it.
    for i in _queue_rechecks([body for body, _ in result.recheck]):
        result.retry.update(result.recheck[i][1])
    if result.retry:
        _retry_soon([r for r in records if r.get("messageId") in result.retry])
    _log.info(
        "sync_batch_done",
        messages=len(messages),
        retried=len(result.retry),
        rechecks=len(result.recheck),
    )
    return {"batchItemFailures": [{"itemIdentifier": mid} for mid in sorted(result.retry)]}


