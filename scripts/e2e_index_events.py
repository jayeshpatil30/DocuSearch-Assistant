"""End-to-end test of event-driven indexing against the DEPLOYED stack.

    uv run python scripts/e2e_index_events.py [--stack docsearch-dev] [--region ap-south-1] [--keep]

Runs on the isolated E2E target (deploy with ``deploy.py --with-e2e``): its own Atlas
collection, its own catalogue collection and its own S3 prefix -- never TENANT_A or TENANT_B data.
It drives the real path end to end:

    S3 upload -> EventBridge -> SQS -> index-events Lambda -> Atlas
    SpecSafe call -> Function URL (SigV4) -> SQS -> index-events Lambda -> Atlas

and checks every step with the same hybrid search the agent runs (Titan + Atlas Vector
Search + Atlas Search), filtered by tenant the way the agent filters.

  1. record, then upload                 -> indexed and searchable
  2. record edited + SpecSafe call        -> updated in place; searches never stop meanwhile
  3. upload, then record + SpecSafe call  -> hidden, then searchable
  4. upload, then record, NO call         -> found by the worker's own re-check
  5. DeleteMark=1 + SpecSafe call         -> removed (vectors, keywords, metadata)
  6. scanned (image-only) PDF             -> OCR'd by Textract in Lambda, searchable
  7. file in a subfolder, call by name    -> SpecSafe's bare file name still finds it
  8. CloudWatch                           -> the worker's log lines for this run

Cleans up its files and records afterwards unless ``--keep``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from dotenv import dotenv_values
from pymongo import MongoClient
from rag_search.config import SearchSettings
from rag_search.index_jobs import Target, build_for_target

from rag_core.config import get_settings
from rag_core.utils import document_id

ROOT = Path(__file__).resolve().parents[1]
# Kept in step with deploy/index-jobs/deploy.py.
E2E = {
    "collection": "E2E_DocumentSearch_DEV",
    "bucket": "demo-docsearch-bucket",
    "prefix": "docsearch-e2e/",
    "tenant_codes": ["E2E"],
    "metadata_database": "DOCSEARCH_DEMO",
    "metadata_collection": "AttachmentMetaData_E2E",
}
AS_E2E = {"tenant": ["E2E"]}  # how the agent scopes every search


class Failed(AssertionError):
    pass


def docx(title: str, *paragraphs: str) -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading(title, level=1)
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def scanned_pdf(*lines: str) -> bytes:
    """An image-only PDF: the text exists only as pixels, so it must be OCR'd."""
    import fitz  # PyMuPDF

    src = fitz.open()
    page = src.new_page(width=595, height=842)
    for i, line in enumerate(lines):
        page.insert_text((72, 96 + 28 * i), line, fontsize=16)
    pixels = page.get_pixmap(dpi=200).tobytes("png")
    out = fitz.open()
    out.new_page(width=595, height=842).insert_image(fitz.Rect(0, 0, 595, 842), stream=pixels)
    return out.tobytes()


def html(title: str, *paragraphs: str) -> bytes:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<html><body><h1>{title}</h1>{body}</body></html>".encode()


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = time.strftime("%Y%m%d%H%M%S")
        self.session = boto3.Session(region_name=args.region)
        self.s3 = self.session.client("s3")
        outputs = self.session.client("cloudformation").describe_stacks(StackName=args.stack)
        self.outputs = {o["OutputKey"]: o["OutputValue"] for o in outputs["Stacks"][0]["Outputs"]}
        self.url = self.outputs["SyncApiUrl"]
        env = dotenv_values(ROOT / ".env")
        self.ca = env.get("AWS__CA_BUNDLE") or ""
        mongo = MongoClient(env["SEARCH_MONGO_URI"])
        self.catalogue = mongo[E2E["metadata_database"]][E2E["metadata_collection"]]
        self.target = Target(
            "E2E", E2E["collection"], E2E["bucket"], E2E["prefix"], "E2E",
            E2E["metadata_database"], E2E["metadata_collection"],
        )
        self.components = build_for_target(self.target, get_settings(), SearchSettings())
        self.db = self.components.vector._db  # type: ignore[attr-defined]
        self.keys: list[str] = []
        self.results: list[tuple[str, bool, str]] = []
        self.started_ms = int(time.time() * 1000)

    # ---- plumbing --------------------------------------------------------

    def key(self, name: str) -> str:
        return f"{E2E['prefix']}{name}"

    def record(self, name: str, **fields: Any) -> dict[str, Any]:
        return {
            "GeneratedFileName": name,
            "OriginalFileName": name.split("_", 1)[-1].replace("~", "."),
            "TenantCode": "E2E",
            "BusinessObject": "SUPPLIERDOCUMENTS",
            "ObjectCode": "E2E-001",
            "ObjectName": "E2E supplier",
            "Status": "Active",
            "DeleteMark": 0,
            "MetaData": [{"FieldName": "Note", "DisplayName": "Note", "Value": f"run {self.run_id}"}],
            "_e2e_run": self.run_id,
            **fields,
        }

    def upload(self, name: str, body: bytes) -> str:
        key = self.key(name)
        self.s3.put_object(Bucket=E2E["bucket"], Key=key, Body=body)
        self.keys.append(key)
        return key

    def call(self, files: list[str], reason: str) -> tuple[int, Any]:
        """SpecSafe's call: a SigV4-signed POST to the Function URL."""
        body = json.dumps({"TenantCode": "E2E", "Files": files, "Reason": reason}).encode()
        creds = self.session.get_credentials().get_frozen_credentials()
        req = AWSRequest(method="POST", url=self.url, data=body,
                         headers={"Content-Type": "application/json"})
        SigV4Auth(creds, "lambda", self.args.region).add_auth(req)
        ctx = ssl.create_default_context()
        if self.ca and Path(self.ca).exists():
            ctx.load_verify_locations(cafile=self.ca)
        request = urllib.request.Request(self.url, data=body, headers=dict(req.headers), method="POST")
        try:
            with urllib.request.urlopen(request, context=ctx, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")

    async def indexed(self, key: str) -> dict[str, Any] | None:
        doc_id = document_id(E2E["bucket"], key)
        return (await self.components.metadata.describe([doc_id])).get(doc_id)

    async def chunk_count(self, key: str) -> int:
        doc_id = document_id(E2E["bucket"], key)
        return await self.db.run(
            lambda c: c.count_documents({"kind": "chunk", "document_id": doc_id})
        )

    async def found(self, query: str, key: str, **filters: list[str]) -> bool:
        hits = await self.components.search.search(query, filters={**AS_E2E, **filters})
        return key in {p.source_key for p in hits.results}

    async def wait(self, what: str, check: Callable[[], Awaitable[bool]], limit_s: float) -> float:
        start = time.monotonic()
        while True:
            if await check():
                return time.monotonic() - start
            if time.monotonic() - start > limit_s:
                raise Failed(f"timed out after {limit_s:.0f}s waiting for: {what}")
            await asyncio.sleep(3)

    def ok(self, step: str, detail: str) -> None:
        self.results.append((step, True, detail))
        print(f"  PASS  {step}: {detail}", flush=True)

    # ---- steps -----------------------------------------------------------

    async def setup(self) -> None:
        await self.db.ensure_schema()
        await self.db.write_manifest({
            "collection": E2E["collection"],
            "tenant_codes": E2E["tenant_codes"],
            "source_bucket": E2E["bucket"],
            "source_prefix": E2E["prefix"],
            "namespace": "default",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "purpose": "end-to-end test target (scripts/e2e_index_events.py)",
        })
        await self.db.wait_until_searchable(timeout_s=900)
        print(f"  E2E collection ready; run {self.run_id}", flush=True)

    async def step1_record_then_upload(self) -> str:
        name = f"e2e{self.run_id}a_quality_manual~docx"
        word = f"zephyrine{self.run_id}"
        self.catalogue.insert_one(self.record(name))
        t0 = time.monotonic()
        key = self.upload(name, docx(
            f"E2E Quality Manual {self.run_id}",
            f"The {word} calibration protocol requires weekly torque checks on line 4.",
            "Deviations are logged in the quality register and reviewed by the plant QA lead.",
        ))
        await self.wait("metadata record", lambda: self._truthy(self.indexed(key)), 300)
        indexed_s = time.monotonic() - t0
        await self.wait("searchable", lambda: self.found(word, key), 180)
        facets = (await self.indexed(key) or {}).get("facets", {})
        if facets.get("object_code") != "E2E-001" or facets.get("tenant") != "E2E":
            raise Failed(f"catalogue facets missing: {facets}")
        self.ok("1 upload with record", f"indexed in {indexed_s:.0f}s, searchable in "
                f"{time.monotonic() - t0:.0f}s, {await self.chunk_count(key)} chunks, facets {facets}")
        return key

    async def step2_metadata_update(self, key: str) -> None:
        name, word = key.rsplit("/", 1)[-1], f"zephyrine{self.run_id}"
        stop, stats = asyncio.Event(), {"searches": 0, "missed": 0, "errors": 0}

        async def keep_searching() -> None:  # another user, searching throughout
            while not stop.is_set():
                try:
                    stats["searches"] += 1
                    stats["missed"] += not await self.found(word, key)
                except Exception:
                    stats["errors"] += 1
                await asyncio.sleep(0.25)

        searcher = asyncio.create_task(keep_searching())
        self.catalogue.update_one(
            {"GeneratedFileName": name},
            {"$set": {"Status": "Approved",
                      "MetaData": [{"FieldName": "Note", "DisplayName": "Note",
                                    "Value": f"approved in routing {self.run_id}"}]}},
        )
        status, body = self.call([name], "updated")
        if status != 202:
            raise Failed(f"SpecSafe call returned {status}: {body}")
        took = await self.wait(
            "status Approved",
            lambda: self._facet(key, "status", "Approved"), 180,
        )
        await self.wait("chunks filterable by the new status",
                        lambda: self.found(word, key, status=["Approved"]), 120)
        await asyncio.sleep(3)
        stop.set()
        await searcher
        if await self.found(word, key, status=["Active"]):
            raise Failed("old status still matches the chunks")
        if stats["missed"] or stats["errors"]:
            raise Failed(f"searches disturbed during the update: {stats}")
        self.ok("2 metadata update", f"202 {body}; applied in {took:.0f}s; "
                f"{stats['searches']} concurrent searches, 0 missed, 0 errors")

    async def step3_upload_then_record_and_call(self) -> None:
        name, word = f"e2e{self.run_id}b_supplier_note~html", f"quillback{self.run_id}"
        key = self.upload(name, html("Supplier note", f"The {word} pallets moved to bay 7."))
        await self.wait("indexed (hidden)", lambda: self._truthy(self.indexed(key)), 300)
        await asyncio.sleep(5)
        if await self.found(word, key):
            raise Failed("visible before its catalogue record exists")
        self.catalogue.insert_one(self.record(name))
        status, body = self.call([name], "created")
        if status != 202:
            raise Failed(f"SpecSafe call returned {status}: {body}")
        took = await self.wait("visible after the call", lambda: self.found(word, key), 180)
        self.ok("3 upload before record", f"hidden until the record; visible {took:.0f}s after the call")

    async def step4_recheck_without_call(self) -> None:
        name, word = f"e2e{self.run_id}c_audit_note~txt", f"marrowind{self.run_id}"
        key = self.upload(name, f"Audit note: {word} cold-chain readings were in range.".encode())
        await self.wait("indexed (hidden)", lambda: self._truthy(self.indexed(key)), 300)
        self.catalogue.insert_one(self.record(name))  # SpecSafe forgets to call
        took = await self.wait("found by the worker's re-check", lambda: self.found(word, key), 300)
        self.ok("4 re-check (no call)", f"visible {took:.0f}s after the record, with no call")

    async def step5_delete_mark(self, key: str) -> None:
        name = key.rsplit("/", 1)[-1]
        self.catalogue.update_one({"GeneratedFileName": name}, {"$set": {"DeleteMark": 1}})
        status, body = self.call([name], "deleted")
        if status != 202:
            raise Failed(f"SpecSafe call returned {status}: {body}")
        took = await self.wait("removed", lambda: self._gone(key), 180)
        await self.wait("out of search", lambda: self._not(self.found(f"zephyrine{self.run_id}", key)), 120)
        self.ok("5 DeleteMark", f"vectors, keywords and metadata removed {took:.0f}s after the call")

    async def step6_scanned_pdf(self) -> None:
        name, word = f"e2e{self.run_id}d_scanned_certificate~pdf", f"vellumcrest{self.run_id}"
        self.catalogue.insert_one(self.record(name))
        t0 = time.monotonic()
        key = self.upload(name, scanned_pdf(
            "Certificate of Analysis", f"Batch reference {word}", "Moisture within limits."
        ))
        await self.wait("OCR'd and searchable", lambda: self.found(word, key), 420)
        self.ok("6 scanned PDF (OCR)", f"Textract text searchable {time.monotonic() - t0:.0f}s "
                f"after upload, {await self.chunk_count(key)} chunks")

    async def step7_subfolder_by_name(self) -> None:
        name, word = f"e2e{self.run_id}e_policy~html", f"tamarisk{self.run_id}"
        self.catalogue.insert_one(self.record(name))
        key = self.upload(f"sub/{name}", html("Policy", f"The {word} waste policy applies."))
        await self.wait("indexed in a subfolder", lambda: self.found(word, key), 300)
        self.catalogue.update_one({"GeneratedFileName": name}, {"$set": {"Status": "Expired"}})
        status, body = self.call([name], "updated")  # the bare name, as SpecSafe sends it
        if status != 202:
            raise Failed(f"SpecSafe call returned {status}: {body}")
        took = await self.wait("update applied", lambda: self._facet(key, "status", "Expired"), 180)
        self.ok("7 subfolder by name", f"'{name}' resolved to {key}; update applied in {took:.0f}s")

    def step8_logs(self) -> None:
        logs = self.session.client("logs")
        group = f"/aws/lambda/{self.args.stack}-index-events"
        events = logs.filter_log_events(
            logGroupName=group, startTime=self.started_ms, filterPattern=f'"{self.run_id}"'
        ).get("events", [])
        lines = []
        for e in events:
            try:
                msg = json.loads(e["message"])
            except (ValueError, TypeError):
                continue
            if msg.get("event") == "document_sync":
                lines.append(f"{msg.get('action'):<18} {msg.get('reason') or '':<14} {msg.get('key')}")
        for line in lines:
            print("        " + line)
        if not lines:
            raise Failed(f"no document_sync lines for this run in {group}")
        self.ok("8 CloudWatch", f"{len(lines)} document_sync lines in {group}")

    async def cleanup(self) -> None:
        for key in self.keys:
            self.s3.delete_object(Bucket=E2E["bucket"], Key=key)
        self.catalogue.delete_many({"_e2e_run": self.run_id})
        # Object Deleted events purge what DeleteMark did not.
        for key in self.keys:
            await self.wait(f"{key} purged", lambda k=key: self._gone(k), 240)
        print("  cleaned up: test files, records and index entries removed", flush=True)

    # ---- small async predicates ---------------------------------------------

    async def _truthy(self, value: Awaitable[Any]) -> bool:
        return bool(await value)

    async def _not(self, value: Awaitable[bool]) -> bool:
        return not await value

    async def _facet(self, key: str, name: str, expected: str) -> bool:
        return ((await self.indexed(key)) or {}).get("facets", {}).get(name) == expected

    async def _gone(self, key: str) -> bool:
        return await self.indexed(key) is None and await self.chunk_count(key) == 0


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", default="docsearch-dev")
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--keep", action="store_true", help="leave the test files and records")
    args = p.parse_args()

    run = Run(args)
    failed = None
    try:
        print("setup", flush=True)
        await run.setup()
        key_a = await run.step1_record_then_upload()
        await run.step2_metadata_update(key_a)
        await run.step3_upload_then_record_and_call()
        await run.step4_recheck_without_call()
        await run.step5_delete_mark(key_a)
        await run.step6_scanned_pdf()
        await run.step7_subfolder_by_name()
        await asyncio.sleep(10)  # CloudWatch ingestion lag
        run.step8_logs()
    except Exception as exc:
        failed = exc
        print(f"  FAIL  {type(exc).__name__}: {exc}", flush=True)
    finally:
        if not args.keep:
            try:
                await run.cleanup()
            except Exception as exc:
                print(f"  cleanup incomplete: {exc}", flush=True)
        await run.components.aclose()
    print(f"\n{len(run.results)} of 8 steps passed" + ("" if failed is None else " -- FAILED"))
    return 0 if failed is None and len(run.results) == 8 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
