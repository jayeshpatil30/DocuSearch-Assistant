"""Bookkeeping for keeping a live index in step with S3: watermark, pending files, locks.

Lives in ``<collection>_IndexState`` beside the index collection -- never in the index
collection itself, whose Atlas search indexes would cover it. Holds:

* the sweep's **watermark** (the last run's start) and its **run lock**, so overlapping
  scheduled runs cannot double-embed;
* **pending** files: ones a run could not finish (out of time) or that are indexed but
  still waiting for their catalogue record, re-checked by every sweep;
* **per-document leases**, so the S3 event, SpecSafe's call and the sweep never work
  on the same document at the same time.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

WATERMARK_ID = "_watermark"
LOCK_ID = "_lock"
PENDING = "pending"
LEASE = "lease"


@dataclass(slots=True)
class PendingFile:
    key: str
    first_seen: datetime
    attempts: int = 0


class IndexState(Protocol):
    async def acquire(self, owner: str, ttl: timedelta) -> bool: ...
    async def release(self, owner: str) -> None: ...
    async def acquire_doc(self, doc_id: str, owner: str, ttl: timedelta) -> bool: ...
    async def release_doc(self, doc_id: str, owner: str) -> None: ...
    async def watermark(self) -> datetime | None: ...
    async def set_watermark(self, value: datetime) -> None: ...
    async def pending(self) -> list[PendingFile]: ...
    async def put_pending(self, files: list[PendingFile]) -> None: ...
    async def drop_pending(self, keys: list[str]) -> None: ...


class InMemoryIndexState:
    """For tests and dry runs."""

    def __init__(self) -> None:
        self._watermark: datetime | None = None
        self._pending: dict[str, PendingFile] = {}
        self._lock: tuple[str, datetime] | None = None
        self._leases: dict[str, tuple[str, datetime]] = {}

    async def acquire(self, owner: str, ttl: timedelta) -> bool:
        now = datetime.now(UTC)
        if self._lock and self._lock[0] != owner and self._lock[1] > now:
            return False
        self._lock = (owner, now + ttl)
        return True

    async def release(self, owner: str) -> None:
        if self._lock and self._lock[0] == owner:
            self._lock = None

    async def acquire_doc(self, doc_id: str, owner: str, ttl: timedelta) -> bool:
        now = datetime.now(UTC)
        held = self._leases.get(doc_id)
        if held and held[0] != owner and held[1] > now:
            return False
        self._leases[doc_id] = (owner, now + ttl)
        return True

    async def release_doc(self, doc_id: str, owner: str) -> None:
        if (held := self._leases.get(doc_id)) and held[0] == owner:
            del self._leases[doc_id]

    async def watermark(self) -> datetime | None:
        return self._watermark

    async def set_watermark(self, value: datetime) -> None:
        self._watermark = value

    async def pending(self) -> list[PendingFile]:
        return list(self._pending.values())

    async def put_pending(self, files: list[PendingFile]) -> None:
        self._pending.update({f.key: f for f in files})

    async def drop_pending(self, keys: list[str]) -> None:
        for key in keys:
            self._pending.pop(key, None)


def _aware(value: Any) -> datetime | None:
    """pymongo returns naive UTC datetimes unless the client is tz_aware."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class MongoIndexState:
    """State in ``<collection>_IndexState`` of the index database."""

    def __init__(self, db: Any) -> None:  # rag_search.stores.mongo.MongoIndexDatabase
        self._db = db
        self._name = f"{db.collection_name}_IndexState"

    async def _run(self, fn: Any) -> Any:
        return await self._db.run(lambda coll: fn(coll.database[self._name]))

    async def _claim(self, lock_id: str, owner: str, ttl: timedelta, **fields: Any) -> bool:
        from pymongo.errors import DuplicateKeyError

        now = datetime.now(UTC)

        def _op(coll: Any) -> bool:
            try:
                # Matches only a free or expired lock, or one this owner holds; otherwise
                # the upsert collides with the live lock's _id and raises.
                coll.update_one(
                    {"_id": lock_id, "$or": [{"expires": {"$lt": now}}, {"owner": owner}]},
                    {
                        "$set": {
                            "owner": owner,
                            "expires": now + ttl,
                            "host": socket.gethostname(),
                            **fields,
                        }
                    },
                    upsert=True,
                )
                return True
            except DuplicateKeyError:
                return False

        return bool(await self._run(_op))

    async def acquire(self, owner: str, ttl: timedelta) -> bool:
        return await self._claim(LOCK_ID, owner, ttl)

    async def release(self, owner: str) -> None:
        await self._run(lambda coll: coll.delete_one({"_id": LOCK_ID, "owner": owner}))

    async def acquire_doc(self, doc_id: str, owner: str, ttl: timedelta) -> bool:
        return await self._claim(f"{LEASE}::{doc_id}", owner, ttl, kind=LEASE)

    async def release_doc(self, doc_id: str, owner: str) -> None:
        await self._run(lambda coll: coll.delete_one({"_id": f"{LEASE}::{doc_id}", "owner": owner}))

    async def watermark(self) -> datetime | None:
        doc = await self._run(lambda coll: coll.find_one({"_id": WATERMARK_ID}))
        return _aware((doc or {}).get("value"))

    async def set_watermark(self, value: datetime) -> None:
        await self._run(
            lambda coll: coll.update_one(
                {"_id": WATERMARK_ID}, {"$set": {"value": value}}, upsert=True
            )
        )

    async def pending(self) -> list[PendingFile]:
        rows = await self._run(lambda coll: list(coll.find({"kind": PENDING})))
        return [
            PendingFile(
                key=r["key"],
                first_seen=_aware(r.get("first_seen")) or datetime.now(UTC),
                attempts=int(r.get("attempts") or 0),
            )
            for r in rows
        ]

    async def put_pending(self, files: list[PendingFile]) -> None:
        if not files:
            return
        from pymongo import UpdateOne

        ops = [
            UpdateOne(
                {"_id": f"pending::{f.key}"},
                {
                    "$set": {"kind": PENDING, "key": f.key, "attempts": f.attempts},
                    "$setOnInsert": {"first_seen": f.first_seen},
                },
                upsert=True,
            )
            for f in files
        ]
        await self._run(lambda coll: coll.bulk_write(ops, ordered=False))

    async def drop_pending(self, keys: list[str]) -> None:
        if keys:
            ids = [f"pending::{k}" for k in keys]
            await self._run(lambda coll: coll.delete_many({"_id": {"$in": ids}}))


__all__ = [
    "LEASE",
    "LOCK_ID",
    "PENDING",
    "WATERMARK_ID",
    "InMemoryIndexState",
    "IndexState",
    "MongoIndexState",
    "PendingFile",
]
