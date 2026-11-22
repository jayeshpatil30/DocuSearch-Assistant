"""Run the scheduled index jobs locally -- the same code the Lambda runs.

    uv run python scripts/incremental_index.py index --target TENANT_A
    uv run python scripts/incremental_index.py purge --target TENANT_B --dry-run
    uv run python scripts/incremental_index.py index --target TENANT_A --minutes 10
    uv run python scripts/incremental_index.py sync --target TENANT_A --file <GeneratedFileName>

``sync`` does for the named files exactly what an S3 event or SpecSafe's call does:
re-reads S3 and the catalogue, then indexes, updates, re-indexes or purges.

Targets come from INDEX_TARGETS (JSON), e.g. in PowerShell::

    $env:INDEX_TARGETS = '{"TENANT_A": {"collection": "TENANT_A_DocumentSearch_DEV", "metadata_database": "TENANT_A_DEV"},
                           "TENANT_B": {"collection": "TENANT_B_DocumentSearch_DEV", "metadata_database": "TENANT_B_DEV"}}'

Bucket, prefix and tenant codes default to each collection's manifest. Mongo and AWS
credentials come from .env as for every other script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from rag_search.index_jobs import run_job

from rag_core.logging import configure_logging


def main() -> None:
    configure_logging(level="INFO", json_logs=False, service_name="index-jobs-cli")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("job", choices=["index", "purge", "sync"])
    parser.add_argument("--target", required=True, help="a key of INDEX_TARGETS, e.g. TENANT_A or TENANT_B")
    parser.add_argument("--dry-run", action="store_true", help="purge: report only, delete nothing")
    parser.add_argument(
        "--minutes", type=float, default=0, help="index: stop starting new documents after this long"
    )
    parser.add_argument(
        "--file", action="append", default=[],
        help="sync: a file name under the target's prefix, or a full S3 key (repeatable)",
    )
    args = parser.parse_args()
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    result = asyncio.run(
        run_job(args.job, args.target, deadline=deadline, dry_run=args.dry_run, files=args.file)
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
