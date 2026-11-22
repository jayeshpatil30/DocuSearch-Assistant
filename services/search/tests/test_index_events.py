"""The event-driven entry points: the index-events queue and SpecSafe's Function URL.

Offline: message parsing, routing to targets, which messages are retried, the
manifest guard, and the request contract SpecSafe calls.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from rag_search import index_jobs
from rag_search.document_sync import SyncAction, SyncOutcome
from rag_search.index_jobs import (
    MAX_RECHECKS,
    RefusedTarget,
    Target,
    handle_sync_messages,
    open_sync,
    parse_message,
    recheck_delay_s,
)
from rag_search.stores import MongoVectorIndex
from rag_search.sync_api import handle_api_request

TARGETS = {
    "TENANT_A": Target("TENANT_A", "TENANT_A_DocumentSearch_DEV", "src", "TENANT_A/", "TENANT_A", "TENANT_A_DEV", "AttachmentMetaData"),
    "TENANT_B": Target(
        "TENANT_B", "TENANT_B_DocumentSearch_DEV", "src", "TENANT_B", "TENANT_B", "TENANT_B_DEV", "AttachmentMetaData"
    ),
}


def s3_event(key: str, bucket: str = "src", kind: str = "Object Created") -> str:
    return json.dumps(
        {
            "source": "aws.s3",
            "detail-type": kind,
            "detail": {"bucket": {"name": bucket}, "object": {"key": key, "etag": "abc"}},
        }
    )


def specsafe(file: str, target: str = "TENANT_A", tenant: str = "TENANT_A") -> str:
    return json.dumps(
        {
            "source": "specsafe",
            "target": target,
            "tenant_code": tenant,
            "file": file,
            "reason": "updated",
        }
    )


def recheck(key: str, attempt: int, target: str = "TENANT_A") -> str:
    return json.dumps({"source": "recheck", "target": target, "key": key, "attempt": attempt})


_HIDDEN = {SyncAction.INDEXED_HIDDEN, SyncAction.AWAITING_METADATA}


class FakeSync:
    """Stands in for DocumentSync: records the keys asked for, answers per key."""

    def __init__(
        self, answers: dict[str, SyncAction] | None = None, existing: set[str] | None = None
    ) -> None:
        self.calls: list[list[str]] = []
        self.answers = answers or {}
        self.existing = existing or set()

    async def resolve_key(self, candidates: tuple[str, ...]) -> str:
        return next((k for k in candidates if k in self.existing), candidates[0])

    async def sync_many(self, keys: list[str], **_: Any) -> list[SyncOutcome]:
        self.calls.append(sorted(keys))
        out = []
        for k in keys:
            action = self.answers.get(k, SyncAction.INDEXED)
            out.append(SyncOutcome(k, action, hidden=action in _HIDDEN))
        return out


def opener(syncs: dict[str, FakeSync], broken: set[str] = frozenset()):  # type: ignore[no-untyped-def]
    @contextlib.asynccontextmanager
    async def _open(target: Target) -> AsyncIterator[FakeSync]:
        if target.name in broken:
            raise RuntimeError("Atlas unreachable")
        yield syncs.setdefault(target.name, FakeSync())

    return _open


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------


def test_messages_of_every_shape_are_understood() -> None:
    [created] = parse_message("m1", s3_event("TENANT_A/a b.pdf"))
    assert (created.bucket, created.key, created.reason) == ("src", "TENANT_A/a b.pdf", "Object Created")

    # A bucket wired to the queue directly sends classic notifications, URL-encoded.
    classic = json.dumps(
        {
            "Records": [
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {"bucket": {"name": "src"}, "object": {"key": "TENANT_A/a+b%7E1.pdf"}},
                }
            ]
        }
    )
    assert [r.key for r in parse_message("m2", classic)] == ["TENANT_A/a b~1.pdf"]

    [call] = parse_message("m3", specsafe("x.pdf"))
    assert (call.target, call.tenant_code, call.file) == ("TENANT_A", "TENANT_A", "x.pdf")

    [again] = parse_message("m6", recheck("TENANT_A/x.pdf", 3))
    assert (again.target, again.key, again.attempt) == ("TENANT_A", "TENANT_A/x.pdf", 3)

    assert parse_message("m4", json.dumps({"Event": "s3:TestEvent"})) == []
    for junk in ("not json", "[]", json.dumps({"hello": 1}), json.dumps({"source": "specsafe"})):
        with pytest.raises(ValueError):
            parse_message("m5", junk)


async def test_events_reach_the_target_that_reads_their_prefix() -> None:
    syncs: dict[str, FakeSync] = {}
    result = await handle_sync_messages(
        [
            ("m1", s3_event("TENANT_A/a.pdf")),
            ("m2", s3_event("TENANT_B/b.pdf")),
            ("m3", s3_event("elsewhere/c.pdf")),  # no target reads it
            ("m4", s3_event("TENANT_A/a.pdf", bucket="other-bucket")),
            ("m5", specsafe("d.pdf")),  # a file name: under the TENANT_A prefix
            ("m6", specsafe("TENANT_B/e.pdf", target="TENANT_B", tenant="TENANT_B")),  # already a key
            ("m7", specsafe("f.pdf", tenant="TENANT_A")),  # not TENANT_A's tenant: dropped
            ("m8", specsafe("other/g.pdf")),  # outside TENANT_A's prefix: dropped
            ("m9", "garbage"),  # unreadable: dropped, never retried
        ],
        targets=TARGETS,
        open_target=opener(syncs),
    )
    assert result.retry == set() and result.recheck == []
    assert syncs["TENANT_A"].calls == [["TENANT_A/a.pdf", "TENANT_A/d.pdf"]]
    assert syncs["TENANT_B"].calls == [["TENANT_B/b.pdf", "TENANT_B/e.pdf"]]


async def test_a_file_name_resolves_to_the_key_actually_used() -> None:
    """Much of TENANT_B lives at ``TENANT_B//<name>``: the key that exists (or is indexed) wins."""
    syncs = {"TENANT_B": FakeSync(existing={"TENANT_B//x.pdf"})}
    await handle_sync_messages(
        [("m1", specsafe("x.pdf", target="TENANT_B", tenant="TENANT_B")),
         ("m2", specsafe("y.pdf", target="TENANT_B", tenant="TENANT_B"))],
        targets=TARGETS,
        open_target=opener(syncs),
    )
    assert syncs["TENANT_B"].calls == [["TENANT_B//x.pdf", "TENANT_B/y.pdf"]]


async def test_a_file_still_waiting_for_its_record_is_re_checked_with_backoff() -> None:
    syncs = {
        "TENANT_A": FakeSync(
            {
                "TENANT_A/new.pdf": SyncAction.INDEXED_HIDDEN,
                "TENANT_A/waiting.pdf": SyncAction.AWAITING_METADATA,
                "TENANT_A/last.pdf": SyncAction.AWAITING_METADATA,
            }
        )
    }
    result = await handle_sync_messages(
        [
            ("m1", s3_event("TENANT_A/new.pdf")),
            ("m2", recheck("TENANT_A/waiting.pdf", 3)),
            ("m3", recheck("TENANT_A/last.pdf", MAX_RECHECKS)),  # the backstop: no more
            ("m4", s3_event("TENANT_A/ok.pdf")),
        ],
        targets=TARGETS,
        open_target=opener(syncs),
    )
    assert result.retry == set()
    assert sorted(result.recheck, key=lambda r: r[0]["key"]) == [
        ({"target": "TENANT_A", "key": "TENANT_A/new.pdf", "attempt": 1}, {"m1"}),
        ({"target": "TENANT_A", "key": "TENANT_A/waiting.pdf", "attempt": 4}, {"m2"}),
    ]
    assert [recheck_delay_s(n) for n in range(1, 8)] == [60, 120, 240, 480, 900, 900, 900]


async def test_only_unfinished_documents_are_retried() -> None:
    syncs = {"TENANT_A": FakeSync({"TENANT_A/busy.pdf": SyncAction.BUSY, "TENANT_A/bad.pdf": SyncAction.FAILED})}
    result = await handle_sync_messages(
        [
            ("m1", s3_event("TENANT_A/ok.pdf")),
            ("m2", s3_event("TENANT_A/busy.pdf")),
            ("m3", specsafe("busy.pdf")),  # same document: one sync, both retried
            ("m4", s3_event("TENANT_A/bad.pdf")),
        ],
        targets=TARGETS,
        open_target=opener(syncs),
    )
    assert result.retry == {"m2", "m3", "m4"}
    assert syncs["TENANT_A"].calls == [["TENANT_A/bad.pdf", "TENANT_A/busy.pdf", "TENANT_A/ok.pdf"]]


async def test_a_target_that_cannot_be_opened_retries_all_of_its_messages() -> None:
    syncs: dict[str, FakeSync] = {}
    result = await handle_sync_messages(
        [("m1", s3_event("TENANT_A/a.pdf")), ("m2", s3_event("TENANT_B/b.pdf"))],
        targets=TARGETS,
        open_target=opener(syncs, broken={"TENANT_A"}),
    )
    assert result.retry == {"m1"}
    assert syncs["TENANT_B"].calls == [["TENANT_B/b.pdf"]]


async def test_a_collection_without_tenant_codes_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hidden documents are hidden only by the agent's tenant filter, which reads the
    manifest's tenant codes -- so no syncing into a collection that has none."""
    closed: list[bool] = []

    class Db:
        async def ensure_schema(self) -> None: ...

        async def read_manifest(self) -> dict[str, Any]:
            return {"collection": "TENANT_A_DocumentSearch_DEV"}  # no tenant_codes

    async def aclose() -> None:
        closed.append(True)

    components = SimpleNamespace(vector=MongoVectorIndex(Db()), aclose=aclose)  # type: ignore[arg-type]
    monkeypatch.setattr(index_jobs, "build_for_target", lambda *a: components)

    with pytest.raises(RefusedTarget):
        async with open_sync(TARGETS["TENANT_A"], None, None):  # type: ignore[arg-type]
            pass
    assert closed == [True]


# --------------------------------------------------------------------------
# SpecSafe's Function URL
# --------------------------------------------------------------------------


def call(body: Any, *, method: str = "POST", path: str = "/", b64: bool = False) -> dict[str, Any]:
    raw = body if isinstance(body, str) else json.dumps(body)
    if b64:
        raw = base64.b64encode(raw.encode()).decode()
    return {
        "rawPath": path,
        "body": raw,
        "isBase64Encoded": b64,
        "requestContext": {"requestId": "req-1", "http": {"method": method}},
    }


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.setenv(
        "INDEX_TARGETS",
        json.dumps(
            {
                "TENANT_A": {"collection": "TENANT_A_DocumentSearch_DEV"},  # TenantCode TENANT_A by its name
                "TENANT_B": {"collection": "TENANT_B_DocumentSearch_DEV", "tenant_codes": ["TENANT_B", "BEEUK"]},
            }
        ),
    )
    return []


def sender(queue: list[dict[str, Any]], fail: int = 0):  # type: ignore[no-untyped-def]
    def _send(bodies: list[str]) -> int:
        queue.extend(json.loads(b) for b in bodies)
        return fail

    return _send


def test_a_call_queues_one_message_per_file(queue: list[dict[str, Any]]) -> None:
    status, body = handle_api_request(
        call({"TenantCode": "TENANT_A", "Files": ["a.pdf", "b.pdf", "a.pdf"], "Reason": "updated"}),
        send=sender(queue),
    )
    assert (status, body) == (202, {"accepted": 2, "requestId": "req-1"})
    assert [(m["target"], m["tenant_code"], m["file"], m["source"]) for m in queue] == [
        ("TENANT_A", "TENANT_A", "a.pdf", "specsafe"),
        ("TENANT_A", "TENANT_A", "b.pdf", "specsafe"),
    ]


def test_field_names_are_forgiving_and_tenants_map_through_tenant_codes(
    queue: list[dict[str, Any]],
) -> None:
    status, _ = handle_api_request(
        call({"tenant_code": "BEEUK", "file": "x.pdf"}, path="/v1/documents/sync", b64=True),
        send=sender(queue),
    )
    assert status == 202 and queue[0]["target"] == "TENANT_B"


@pytest.mark.parametrize(
    ("event", "status"),
    [
        (call("not json"), 400),
        (call(["a.pdf"]), 400),
        (call({"Files": ["a.pdf"]}), 400),
        (call({"TenantCode": "TENANT_A", "Files": []}), 400),
        (call({"TenantCode": "TENANT_A", "Files": ["a.pdf"] * 101}), 400),
        (call({"TenantCode": "TENANT_A", "Files": ["../secrets"]}), 400),
        (call({"TenantCode": "TENANT_A", "Files": [""]}), 400),
        (call({"TenantCode": "NOPE", "Files": ["a.pdf"]}), 400),
        (call({"TenantCode": "TENANT_A", "Files": ["a.pdf"]}, method="GET"), 405),
        (call({"TenantCode": "TENANT_A", "Files": ["a.pdf"]}, path="/admin"), 404),
    ],
)
def test_bad_calls_are_rejected_and_queue_nothing(
    queue: list[dict[str, Any]], event: dict[str, Any], status: int
) -> None:
    got, body = handle_api_request(event, send=sender(queue))
    assert got == status and body["status"] == status and body["requestId"] == "req-1"
    assert queue == []


def test_a_queue_failure_asks_specsafe_to_retry(queue: list[dict[str, Any]]) -> None:
    status, body = handle_api_request(
        call({"TenantCode": "TENANT_A", "Files": ["a.pdf"]}), send=sender(queue, fail=1)
    )
    assert status == 503 and "retry" in body["error"]
