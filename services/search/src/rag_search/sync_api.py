"""SpecSafe's "document changed" endpoint: validate the call, queue one message per file.

The Lambda Function URL (IAM / SigV4) SpecSafe calls when it creates, edits or deletes
an ``AttachmentMetaData`` record::

    POST {"TenantCode": "TENANT_A", "Files": ["<GeneratedFileName>", ...], "Reason": "updated"}
    -> 202 {"accepted": 1, "requestId": "..."}

Kept apart from the indexing code on purpose: it imports nothing heavier than the
standard library and logging, so a cold start answers SpecSafe quickly. The queued
messages are processed by ``rag_search.index_jobs.sqs_handler``.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from collections.abc import Callable
from typing import Any

from rag_core.logging import configure_logging, get_logger

_log = get_logger("rag_search.sync_api")

SPECSAFE_SOURCE = "specsafe"

MAX_FILES_PER_CALL = 100

_MAX_FILE_CHARS = 1024

_logging_ready = False

def load_targets() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("INDEX_TARGETS", "").strip()
    if not raw:
        raise ValueError("INDEX_TARGETS is not set (JSON: {name: {collection, ...}})")
    targets = json.loads(raw)
    if not isinstance(targets, dict) or not targets:
        raise ValueError("INDEX_TARGETS must be a non-empty JSON object")
    return {str(k).upper(): v for k, v in targets.items()}


def _codes(value: Any) -> set[str]:
    items = value.split(",") if isinstance(value, str) else list(value or [])
    return {str(c).strip().upper() for c in items if str(c).strip()}


def _configure_logging() -> None:
    global _logging_ready
    if not _logging_ready:
        configure_logging(
            level=os.environ.get("LOG_LEVEL", "INFO"), json_logs=True, service_name="index-jobs"
        )
        _logging_ready = True


def _error(status: int, message: str, request_id: str) -> tuple[int, dict[str, Any]]:
    return status, {"error": message, "status": status, "requestId": request_id}


def _target_for(tenant_code: str, targets: dict[str, dict[str, Any]]) -> str | None:
    """The target serving a SpecSafe TenantCode: one listing it in ``tenant_codes``, or
    -- when a target lists none -- the target of that name."""
    code = tenant_code.strip().upper()
    for name, spec in targets.items():
        if code in (_codes(spec.get("tenant_codes")) or {name}):
            return name
    return None


def handle_api_request(
    event: dict[str, Any], *, send: Callable[[list[str]], int]
) -> tuple[int, dict[str, Any]]:
    """Validate one SpecSafe call and queue a message per file.

    ``send`` queues message bodies and returns how many could NOT be queued. Only file
    names travel: the worker re-reads the catalogue, so a call can ask for a document
    to be looked at again, never put values into the index.
    """
    context = event.get("requestContext") or {}
    request_id = str(context.get("requestId") or uuid.uuid4().hex)
    method = str((context.get("http") or {}).get("method") or "POST").upper()
    path = "/" + str(event.get("rawPath") or "/").strip("/")
    if path not in {"/", "/v1/documents/sync"}:
        return _error(404, "not found", request_id)
    if method != "POST":
        return _error(405, "use POST", request_id)

    raw = event.get("body") or ""
    try:
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return _error(400, "body must be a JSON object", request_id)
    if not isinstance(payload, dict):
        return _error(400, "body must be a JSON object", request_id)
    fields = {str(k).lower().replace("_", "").replace("-", ""): v for k, v in payload.items()}

    tenant_code = fields.get("tenantcode")
    if not isinstance(tenant_code, str) or not tenant_code.strip() or len(tenant_code) > 64:
        return _error(400, "TenantCode is required", request_id)
    files = fields.get("files", fields.get("filenames", fields.get("file")))
    if isinstance(files, str):
        files = [files]
    if not isinstance(files, list) or not files:
        return _error(400, "Files must be a non-empty list of file names", request_id)
    if len(files) > MAX_FILES_PER_CALL:
        return _error(400, f"at most {MAX_FILES_PER_CALL} Files per call", request_id)
    if not all(
        isinstance(f, str) and f.strip() and len(f) <= _MAX_FILE_CHARS and ".." not in f
        for f in files
    ):
        return _error(400, "every entry in Files must be a file name", request_id)
    reason = fields.get("reason")
    reason = reason[:64] if isinstance(reason, str) else ""

    try:
        targets = load_targets()
    except ValueError as exc:
        _log.error("sync_api_misconfigured", error=str(exc))
        return _error(500, "service is not configured", request_id)
    name = _target_for(tenant_code, targets)
    if name is None:
        return _error(400, f"unknown TenantCode {tenant_code.strip()!r}", request_id)

    unique = list(dict.fromkeys(f.strip() for f in files))
    bodies = [
        json.dumps(
            {
                "source": SPECSAFE_SOURCE,
                "target": name,
                "tenant_code": tenant_code.strip().upper(),
                "file": f,
                "reason": reason,
                "request_id": request_id,
            }
        )
        for f in unique
    ]
    failed = send(bodies)
    _log.info(
        "sync_requested", request_id=request_id, target=name, files=len(unique),
        reason=reason, failed=failed,
    )
    if failed:
        return _error(503, f"{failed} of {len(unique)} files could not be queued; retry", request_id)
    return 202, {"accepted": len(unique), "requestId": request_id}


def _send_to_queue(bodies: list[str]) -> int:
    """Queue message bodies on ``index-events``; how many could not be queued."""
    queue_url = os.environ.get("INDEX_EVENTS_QUEUE_URL", "").strip()
    if not queue_url:
        _log.error("sync_api_misconfigured", error="INDEX_EVENTS_QUEUE_URL is not set")
        return len(bodies)
    import boto3

    client = boto3.client("sqs")
    failed = 0
    for start in range(0, len(bodies), 10):
        part = bodies[start : start + 10]
        try:
            resp = client.send_message_batch(
                QueueUrl=queue_url,
                Entries=[{"Id": str(i), "MessageBody": b} for i, b in enumerate(part)],
            )
            failed += len(resp.get("Failed") or [])
        except Exception as exc:
            _log.error("sync_api_queue_failed", error=str(exc)[:300])
            failed += len(part)
    return failed


def api_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """The Function URL SpecSafe calls (IAM-authenticated before it gets here)."""
    _configure_logging()
    status, body = handle_api_request(event, send=_send_to_queue)
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
