# Document Search — "document changed" service (for the SpecSafe team)

> The version to send to the SpecSafe team is the Word document
> `DocuSearch_SpecSafe_Metadata_Service.docx` (repo root); this file mirrors it.

Document Search keeps its search index in step with SpecSafe attachments. A new upload
and its automatically created `AttachmentMetaData` record are picked up from S3 with no
call. **Later changes to the record** (routing steps, status, object code, MetaData
values) and **deletions (`DeleteMark = 1`)** are picked up when SpecSafe calls this
service.

## When to call

Only for changes after the upload:

| SpecSafe event | Call? | `Reason` |
|---|---|---|
| File uploaded; record created automatically | No — automatic (a record that lands shortly after the file is picked up by Document Search's re-check, within about a minute) | — |
| User changes the record (routing step, status, fields, MetaData values) | Yes | `updated` |
| `DeleteMark` set to 1 | Yes | `deleted` |
| `DeleteMark` set back to 0 | Yes | `updated` |

Calling when nothing changed is harmless.

## Endpoint

| | |
|---|---|
| URL | `https://example.lambda-url.ap-south-1.on.aws/` |
| Method | `POST` |
| Auth | AWS IAM — sign the request with **SigV4**, service **`lambda`**, region **`ap-south-1`** |
| Calling identity | The Document Search service account, IAM user **`docsearch-service`** (`arn:aws:iam::123456789012:user/docsearch-service`) |
| Content-Type | `application/json` |
| Function ARN | `arn:aws:lambda:ap-south-1:123456789012:function:docsearch-dev-sync-api` |

SpecSafe signs every call with the access key of the service account `docsearch-service` —
the same IAM user it uses to invoke the Document Search agent. The function grants
`lambda:InvokeFunctionUrl` and `lambda:InvokeFunction` to that user only, so it needs no
further IAM policy. Keep the key in SpecSafe's protected configuration, never in code.

## Request

```json
{
  "TenantCode": "TENANT_A",
  "Files": ["0a1b2c3d4e5f_Supplier Quality Manual~pdf"],
  "Reason": "updated"
}
```

| Field | Required | Description |
|---|---|---|
| `TenantCode` | yes | The record's `TenantCode` (e.g. `TENANT_A`, `TENANT_B`). |
| `Files` | yes | 1–100 `GeneratedFileName` values — the S3 object names. A full S3 key (`TENANT_A/<name>`) also works. |
| `Reason` | no | `updated` or `deleted`. Logged for tracing only. |

Only file names are sent. Document Search reads the current values from
`AttachmentMetaData` itself, so a call never needs the metadata in its body, and calls
arriving out of order or twice still end in the right state. Field names are not
case-sensitive (`tenantCode`, `tenant_code` also work).

More examples:

```json
{"TenantCode": "TENANT_B", "Files": ["0510e0e763dc_PIF-0beddd0c-31b6-44c6-a557-6e7f68ab1bf3~pdf"], "Reason": "updated"}
```
```json
{"TenantCode": "TENANT_A", "Files": ["a1b2c3_COA~pdf", "d4e5f6_Spec~docx"], "Reason": "deleted"}
```

## Responses

| Status | Body | Meaning / what to do |
|---|---|---|
| `202` | `{"accepted": 1, "requestId": "4f0c..."}` | Accepted. Applied within seconds (up to a minute for a new scanned PDF). |
| `400` | `{"error": "TenantCode is required", "status": 400, "requestId": "..."}` | Bad request: missing field, more than 100 files, unknown `TenantCode`. Fix and resend. |
| `403` | `{"Message":"Forbidden"}` | Request not signed, or not signed with the `docsearch-service` service account. |
| `503` | `{"error": "1 of 1 files could not be queued; retry", ...}` | Temporary. Retry the same call (safe to repeat). |

Keep the `requestId` in your logs; it identifies the call in Document Search's logs.

## What happens after the call

| Record state | Result in search |
|---|---|
| File uploaded with its auto-created record (no call) | Searchable for users whose object codes allow it, ~9 s after upload. |
| Record lands shortly after the file (no call) | Picked up by Document Search's re-check, within about a minute (checked for up to 60 min). |
| Metadata changed | Filters (status, object code, business object, …) and the metadata text are updated in place. Other users' searches keep working throughout. |
| `DeleteMark = 1` | Document removed from search (vectors, keywords, metadata). |
| `DeleteMark` back to 0 | Document indexed again. |

Only text documents are indexed: PDF, Word, PowerPoint, Excel, XML, HTML, TXT. Images
and other formats are ignored.

## Signing examples

**Python (boto3/botocore)**

```python
import json, boto3, requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

url = "https://example.lambda-url.ap-south-1.on.aws/"
body = json.dumps({"TenantCode": "TENANT_A", "Files": ["<GeneratedFileName>"], "Reason": "updated"})
req = AWSRequest(method="POST", url=url, data=body, headers={"Content-Type": "application/json"})
SigV4Auth(boto3.Session().get_credentials(), "lambda", "ap-south-1").add_auth(req)
print(requests.post(url, data=body, headers=dict(req.headers)).json())
```

**Command line**

```
awscurl --service lambda --region ap-south-1 -X POST -H "Content-Type: application/json" \
  -d '{"TenantCode":"TENANT_A","Files":["<GeneratedFileName>"],"Reason":"updated"}' https://example.lambda-url.ap-south-1.on.aws/
```

**.NET** — sign the `HttpRequestMessage` with any SigV4 signer for service `lambda`,
region `ap-south-1`, using the `docsearch-service` service account's credentials (the
same ones SpecSafe uses to call the Document Search agent).
