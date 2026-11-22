# Incremental indexing (S3 events + SpecSafe calls → Lambda → Atlas)

Keeps the per-tenant Atlas collections (`TENANT_A_DocumentSearch_DEV`, `TENANT_B_DocumentSearch_DEV`
in `DOCSEARCH_DEMO`) in step with S3 and the `AttachmentMetaData` catalogue, one
document at a time, as things happen. Every write -- vectors, keyword text, metadata --
goes to those Atlas collections; nothing else stores the index.

```
SpecSafe uploads a file ──► S3 (demo-docsearch-bucket, ap-south-1)
     └─(EventBridge: Object Created / Deleted)──────────────────► SQS index-events (+ DLQ)
SpecSafe changes a record ──► Function URL (IAM) ──► sync-api ──► SQS index-events
                                    index-events ──► index-events Lambda ──► Atlas collection
EventBridge Scheduler ──► index-jobs Lambda: daily DeleteMark purge (safety net only)
```

| Trigger | Function | What it does |
|---|---|---|
| S3 `Object Created` / `Object Deleted` | `index-events` | Sync that file: index, re-index, or remove |
| SpecSafe `POST` to the Function URL | `sync-api` → `index-events` | Sync the named files: apply the record, or remove on `DeleteMark=1` |
| File indexed before its record | `index-events` (delayed re-checks) | Look again after 1, 2, 4, 8, then every 15 min, up to `RecordWaitMinutes` (60) |
| `purge-TENANT_A`, `purge-TENANT_B` — daily 02:00 UTC | `index-jobs` | Safety net: remove `DeleteMark=1` documents a call may have missed |

There is no periodic sweep: uploads are picked up by their S3 event, record changes by
SpecSafe's call.

Code: [`rag_search/document_sync.py`](../../services/search/src/rag_search/document_sync.py)
(the one rule every trigger applies), [`rag_search/index_jobs.py`](../../services/search/src/rag_search/index_jobs.py)
(the Lambda handlers), [`deploy.py`](deploy.py) (deploy/update),
[`scripts/e2e_index_events.py`](../../scripts/e2e_index_events.py) (end-to-end test on
the deployed stack), [`scripts/incremental_index.py`](../../scripts/incremental_index.py)
(`sync` / `purge` by hand).

## What happens to a document

Every trigger only says *"look at this file"*. The worker re-reads the truth — the
object in S3 (exists? ETag?) and its `AttachmentMetaData` record — and applies one rule,
so the order events arrive in, duplicates and missed events do not matter.

| Record | File in S3 | In the index | Result |
|---|---|---|---|
| any | not a text format | — | skipped before download (see below) |
| another tenant's, or `DeleteMark=1` | — | — | removed / never indexed |
| — | gone | yes | removed |
| present | new | no | **indexed → searchable** |
| none yet | new | no | **embedded, hidden** until the record exists; re-checked |
| any | replaced (new ETag) | yes | re-indexed in place; the old version's leftover chunks removed |
| changed | same | yes | **updated in place** — record, every chunk's filters, the metadata chunk (one embedding at most) |
| unchanged | same | yes | nothing written |
| none, after `RecordWaitMinutes` | same | yes, hidden | removed (SpecSafe's call re-adds it if a record comes later) |
| was present, now gone | same | yes | removed |

**Text formats only.** PDF (scanned ones are OCR'd with Textract's asynchronous API,
up to `SEARCH_DOC_TIMEOUT_S` = 360 s per document), Word, PowerPoint, Excel, XML, HTML
and plain text (`.pdf .docx .pptx .xlsx .xml .htm(l) .txt`, with `.` or `~` before the
extension). Images and every other or missing extension are skipped before anything is
downloaded (`non_text`); so are the old binary Office formats `.doc .ppt .xls`
(`unsupported_format` — the extractors read only the OOXML ones).

**Why "hidden".** Every agent search is filtered by `tenant`, `business_object` and the
user's `object_code`, and only the catalogue record supplies those. Tenants share S3
folders, so until the record says whose file it is, nobody may see it. It is embedded on
upload anyway, so when the record appears the facets are applied in place and the
document is searchable at once — no download, no re-embedding. For a file without a
record the facet-extraction LLM call is skipped (the record supplies those facets); the
short supplier-name call still runs. The worker refuses a collection whose manifest has
no `tenant_codes`: without them the agent would not filter by tenant.

**Searches never stop.** Atlas reads are never blocked by these writes. Each chunk is
written whole (text, vector and filters in one write), the metadata record last, and an
update rewrites values in place rather than deleting and re-adding — a document is never
missing while it is being updated. A per-document lease keeps an S3 event and a SpecSafe
call from working on one document at once; the second waits a minute and retries.

## SpecSafe "document changed" call

SpecSafe creates the `AttachmentMetaData` record automatically on upload, and the S3
event (or, if the record lands a little later, the worker's re-check) indexes the file with
it — **no call on create**. SpecSafe calls only when a user later **changes** the record
(routing steps, status, fields) or sets / clears `DeleteMark`. The body names files only;
values are re-read from the catalogue, so the call cannot put anything into the index,
only ask for a re-check. The document for the SpecSafe team is
`DocuSearch_SpecSafe_Metadata_Service.docx` (repo root).

```
POST <SyncApiUrl output>          (SigV4-signed: service "lambda", region ap-south-1)
Content-Type: application/json

{"TenantCode": "TENANT_A", "Files": ["<GeneratedFileName>", "..."], "Reason": "updated"}
```

| Field | |
|---|---|
| `TenantCode` | The record's `TenantCode`. Maps to the target of the same name, or to one whose `tenant_codes` in `INDEX_TARGETS` lists it. |
| `Files` | 1–100 `GeneratedFileName`s (the S3 object names under the tenant's folder), or full S3 keys. |
| `Reason` | Optional, logged only: `updated`, `deleted`. |

| Response | Meaning |
|---|---|
| `202 {"accepted": 2, "requestId": "..."}` | Queued; applied within seconds (a minute for a new file with OCR). |
| `400` | Bad body, unknown `TenantCode`, more than 100 files. Fix the request. |
| `403` | The caller's IAM principal is not allowed (or the request is not signed). |
| `503` | Could not queue every file — retry the same call. |

## Deploy

From the repo root, with Docker running and AWS credentials for the account:

```powershell
# 1. Build the image (arm64; the CA secret is for the managed TLS proxy)
docker build -f deploy/index-jobs/Dockerfile --platform linux/arm64 --provenance=false `
  --secret id=extra_ca,src=certs/ca-bundle.pem -t specsafe-index-jobs:<tag> .

# 2. Push, create/update the stack (Atlas URI from .env -> the functions' SEARCH_MONGO_URI),
#    turn on S3 -> EventBridge
uv run python deploy/index-jobs/deploy.py --image specsafe-index-jobs:<tag> `
  --specsafe-principal arn:aws:iam::123456789012:user/docsearch-service --with-e2e `
  --subnet-ids subnet-00000000 --vpc-id vpc-00000000

# 3. Prove it end to end (isolated E2E target; cleans up after itself)
uv run python scripts/e2e_index_events.py
```

`deploy.py` deploys to **ap-south-1**, the region of `demo-docsearch-bucket`: S3 sends its
events only to EventBridge in its own region. Embeddings still use Titan v2 in
us-east-1 (`--bedrock-region`), the model the collections were built with. Left out on an
update, `--specsafe-principal` keeps the caller already granted (on create it grants the
whole account). Leave out `--image` to change only the principal (the deployed image is kept).

Bucket, prefix and tenant codes per target come from each collection's **manifest**
(`scripts/mongo_index.py manifest ...`). TENANT_A reads `demo-docsearch-bucket/TENANT_A/`, TENANT_B
`demo-docsearch-bucket/TENANT_B/`. A bare file name from SpecSafe is resolved to the key actually
used: `<prefix>/<name>` or `<prefix>//<name>` (much of TENANT_B), whichever is indexed or
exists, else an indexed copy in a subfolder of the tenant folder (`TENANT_A/COMPDOCCAT/<name>`).
Keys with spaces, `+` or brackets arrive from EventBridge unencoded (verified).

## Logs and alarms (CloudWatch, ap-south-1, kept `LogRetentionDays` = 90)

| Log group | What is in it |
|---|---|
| `/aws/lambda/<stack>-index-events` | one JSON `document_sync` line per document: `action` (indexed, indexed_hidden, metadata_updated, reindexed, purged, skipped, …), `reason`, `key`, `chunks` |
| `/aws/lambda/<stack>-sync-api` | one `sync_requested` line per SpecSafe call: `request_id`, `target`, `files`, `reason` |
| `/aws/lambda/<stack>-index-jobs` | the daily purge |

Alarms: `<stack>-index-events-dead-letters` (a document failed 5 times — fix, then
redrive the DLQ) and `<stack>-index-jobs-errors`.

## Before going live

- **Who may call** — SpecSafe signs with the feature's service account, IAM user
  `docsearch-service` (`arn:aws:iam::123456789012:user/docsearch-service`), deployed as `--specsafe-principal`
  (2026-11-17); the function grants invoke to it alone. Identities whose own IAM policies
  allow Lambda invoke on everything (account admins, the deployer) can still call it, as
  with any function in the account; `scripts/e2e_index_events.py` relies on that.
- **Tenant codes** — `sync-api` maps `TenantCode` to a target by name (`TENANT_A`, `TENANT_B`). If
  SpecSafe's codes differ, add `"tenant_codes": [...]` to that target in `INDEX_TARGETS`.
- **Atlas network access** — Atlas accepts only listed IPs, and a Lambda outside a VPC
  has none fixed. The two functions that talk to Atlas (`index-events`, `index-jobs`)
  therefore run in `example-private-subnet` (VPC `example-vpc`) with
  their own outbound-only security group, and reach Atlas from that VPC's NAT gateway
  IP **43.205.74.169** — which must be on the Atlas project's IP Access List. The
  SpecSafe endpoint (`sync-api`, [`rag_search/sync_api.py`](../../services/search/src/rag_search/sync_api.py))
  needs only SQS and stays outside the VPC.
- **Catalogue indexes** — lookups are `$in` on `GeneratedFileName` / `OriginalFileName`,
  the purge filters on `DeleteMark`. Ask the catalogue owner for indexes on those fields.
- **Supplier names** — voted per document on this path; a full re-index makes them
  consistent per object code.
