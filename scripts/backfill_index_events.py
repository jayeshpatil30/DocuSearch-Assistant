"""Queue files that never produced an S3 event for indexing by the deployed worker.

    uv run python scripts/backfill_index_events.py --target TENANT_B              # dry run: counts only
    uv run python scripts/backfill_index_events.py --target TENANT_B --send --wait

For files uploaded while no S3 event reached the index (before the event-driven stack was
deployed, or while ``EventsState=DISABLED``). Finds the target's text files modified since
``--since`` (default: the collection manifest's ``built_at``) that are not in the index,
keeps only those with a catalogue record of this tenant that is not ``DeleteMark``-ed, and
queues them on ``<stack>-index-events`` as SpecSafe-style messages -- so they go through
exactly the code a live upload does (Textract, the VPC's Atlas access, leases). Files
without a record are reported and left alone. ``--wait`` follows the queue until every
queued file is in the index.

Reads ``.env`` (Atlas URI); AWS access is the default profile.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime

import boto3
from rag_search.config import SearchSettings
from rag_search.factory import build_mongo_index
from rag_search.index_jobs import SPECSAFE_SOURCE, resolve_target
from rag_search.indexer import skip_reason
from rag_search.mongo_source import MongoMetadataSource
from rag_search.stores import MongoMetadataStore

from rag_core.utils import document_id

DEFAULT_TARGETS = {
    "TENANT_A": {"collection": "TENANT_A_DocumentSearch_DEV", "metadata_database": "TENANT_A_DEV"},
    "TENANT_B": {"collection": "TENANT_B_DocumentSearch_DEV", "metadata_database": "TENANT_B_DEV"},
}


async def _indexed(md: MongoMetadataStore, keys: list[str], bucket: str) -> set[str]:
    ids = {document_id(bucket, k): k for k in keys}
    found: set[str] = set()
    id_list = list(ids)
    for start in range(0, len(id_list), 1000):
        found.update(await md.describe(id_list[start : start + 1000]))
    return {ids[d] for d in found}


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, choices=sorted(DEFAULT_TARGETS))
    p.add_argument("--since", help="ISO date/time; default = the collection manifest's built_at")
    p.add_argument("--send", action="store_true", help="queue them (default: dry run)")
    p.add_argument("--wait", action="store_true", help="after --send, follow until all are indexed")
    p.add_argument("--stack", default="docsearch-dev")
    p.add_argument("--region", default="ap-south-1")
    args = p.parse_args()

    targets = json.loads(os.environ["INDEX_TARGETS"]) if os.environ.get("INDEX_TARGETS") else DEFAULT_TARGETS
    search = SearchSettings()
    target = await resolve_target(args.target, targets[args.target], search)
    codes = [c.strip() for c in target.tenant_codes.split(",") if c.strip()]

    db = build_mongo_index(search.model_copy(update={"index_mongo_collection": target.collection}), dim=search.embed_dim)
    md = MongoMetadataStore(db)
    catalogue = MongoMetadataSource(
        search.mongo_uri, database=target.metadata_database,
        collection=target.metadata_collection, tenant_codes=target.tenant_codes,
    )
    try:
        since = datetime.fromisoformat(args.since) if args.since else datetime.fromisoformat(
            (await db.read_manifest())["built_at"]
        )
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)

        session = boto3.Session(region_name=args.region)
        s3 = session.client("s3")
        recent: list[str] = []
        skipped = 0
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=target.bucket, Prefix=target.prefix):
            for obj in page.get("Contents", []):
                if obj["LastModified"] < since:
                    continue
                if skip_reason(obj["Key"]):
                    skipped += 1
                else:
                    recent.append(obj["Key"])
        missing = sorted(set(recent) - await _indexed(md, recent, target.bucket))

        await catalogue.load_keys(missing)
        deleted = [k for k in missing if catalogue.is_deleted(k)]
        foreign = [k for k in missing if catalogue.is_foreign(k)]
        queue = [k for k in missing if k not in deleted and k not in foreign and catalogue.lookup(k)]
        no_record = [k for k in missing if k not in deleted and k not in foreign and not catalogue.lookup(k)]

        print(f"{args.target} ({target.collection}) since {since:%Y-%m-%d %H:%M} UTC, s3://{target.bucket}/{target.prefix}")
        print(f"  text files uploaded since: {len(recent)} (+{skipped} images/unsupported, ignored)")
        print(f"  not in the index:          {len(missing)}")
        print(f"    with a catalogue record: {len(queue)}  <- to queue")
        print(f"    no record (left alone):  {len(no_record)}")
        print(f"    DeleteMark / other tenant: {len(deleted)} / {len(foreign)}")
        for k in queue[:5]:
            print(f"      e.g. {k}")
        if not args.send or not queue:
            print("  dry run: nothing queued (add --send)" if not args.send else "  nothing to queue")
            return 0

        outputs = session.client("cloudformation").describe_stacks(StackName=args.stack)["Stacks"][0]["Outputs"]
        queue_url = next(o["OutputValue"] for o in outputs if o["OutputKey"] == "IndexEventsQueueUrl")
        sqs = session.client("sqs")
        failed = 0
        for start in range(0, len(queue), 10):
            part = queue[start : start + 10]
            resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=[
                {"Id": str(i), "MessageBody": json.dumps({
                    "source": SPECSAFE_SOURCE, "target": args.target, "tenant_code": codes[0],
                    "file": key, "reason": "backfill",
                })}
                for i, key in enumerate(part)
            ])
            failed += len(resp.get("Failed") or [])
        print(f"  queued {len(queue) - failed} on {queue_url.rsplit('/', 1)[-1]} ({failed} failed)")

        if args.wait:
            started, remaining = time.monotonic(), set(queue)
            while remaining and time.monotonic() - started < 3600:
                await asyncio.sleep(30)
                remaining -= await _indexed(md, sorted(remaining), target.bucket)
                attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=[
                    "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
                print(f"  {time.monotonic() - started:5.0f}s  indexed {len(queue) - len(remaining)}/{len(queue)}"
                      f"  queue: {attrs['ApproximateNumberOfMessages']} waiting,"
                      f" {attrs['ApproximateNumberOfMessagesNotVisible']} in progress", flush=True)
                if not remaining or (attrs["ApproximateNumberOfMessages"] == "0"
                                     and attrs["ApproximateNumberOfMessagesNotVisible"] == "0"):
                    break
            print(f"  done: {len(queue) - len(remaining)} indexed, {len(remaining)} not indexed")
            for k in sorted(remaining)[:10]:
                print(f"      not indexed: {k}  (see /aws/lambda/{args.stack}-index-events)")
        return 0
    finally:
        await db.aclose()
        await catalogue.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
