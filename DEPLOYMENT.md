# Deploying DocuSearch Agentic

This is the end-to-end deployment process for the SpecSafe Document Search feature:
the two AgentCore agents, the Bedrock guardrail they run behind, the per-tenant search
index, and the Lambda stack that keeps the index in step with S3 and SpecSafe.

Deeper detail lives next to each component; this file gives the order and the commands.

| Component | Detail |
|---|---|
| Agents (Orchestrator, Document Search) | [services/agentic/README.md](services/agentic/README.md) |
| Incremental indexing (Lambda stack) | [deploy/index-jobs/README.md](deploy/index-jobs/README.md) |
| SpecSafe "document changed" call | [deploy/index-jobs/SPECSAFE_INTEGRATION.md](deploy/index-jobs/SPECSAFE_INTEGRATION.md) |

---

## 1. What gets deployed

```
SpecSafe ──InvokeAgentRuntime──▶ Orchestrator runtime        (AgentCore, us-east-1)
                                      │ InvokeAgentRuntime
                                      ▼
                                 Document Search runtime      (AgentCore, us-east-1)
                                      │ $vectorSearch + Atlas Search
                                      ▼
                                 MongoDB Atlas: one collection per tenant
                                      ▲
SpecSafe uploads ─▶ S3 (demo-docsearch-bucket) ─EventBridge─▶ SQS ─▶ index-events Lambda   (ap-south-1)
SpecSafe edits   ─▶ sync-api Function URL (SigV4) ────▶ SQS ─┘
EventBridge Scheduler ─▶ index-jobs Lambda (daily DeleteMark purge)
```

| # | Piece | Where | Built from |
|---|---|---|---|
| 1 | Search index: one Atlas collection per tenant (`TENANT_A_DocumentSearch_DEV`, `TENANT_B_DocumentSearch_DEV`) in `DOCSEARCH_DEMO` | MongoDB Atlas | `scripts/mongo_index.py`, `scripts/index_s3.py` |
| 2 | Bedrock guardrail `specsafe-document-search` | Bedrock, us-east-1 | `scripts/create_agent_guardrail.py` |
| 3 | Document Search Agent runtime | AgentCore | `services/agentic` |
| 4 | Orchestrator Agent runtime | AgentCore | `services/agentic` (same code, different `AGENTIC_ROLE`) |
| 5 | Index-jobs stack (`docsearch-dev`) | CloudFormation, ap-south-1 | `deploy/index-jobs/` |

### Regions — read this before choosing one

- **AgentCore runtimes, Bedrock model, guardrail: one region (us-east-1).** `AGENTIC_REGION`
  sets the region for both the agents' Bedrock Converse calls (with the guardrail) and the
  orchestrator's `InvokeAgentRuntime` client
  ([converse.py](services/agentic/src/rag_agentic/converse.py),
  [search_client.py](services/agentic/src/rag_agentic/search_client.py)). The search
  runtime, the model, and the guardrail therefore have to share that region, and
  `create_agent_guardrail.py` creates the guardrail in us-east-1.
- **Index-jobs stack: ap-south-1**, the region of the source bucket `demo-docsearch-bucket`.
  S3 sends its events only to EventBridge in its own region.
- **Embeddings: Titan v2 in us-east-1** for both indexing and queries. The collections
  were built with it; a different embedding model makes every stored vector useless.

---

## 2. Prerequisites

**Tools on the deploying machine**

- `uv` (the repo is a uv workspace): `uv sync --all-packages` once (`--native-tls`
  behind the managed proxy).
- Docker with BuildKit/buildx, able to build `linux/arm64` (AgentCore and the Lambdas run
  on Graviton).
- AWS CLI v2, recent enough to have `aws bedrock-agentcore-control`.
- AWS credentials for account `123456789012` with rights to create IAM roles,
  CloudFormation stacks, ECR repositories, Lambda, SQS, EventBridge, AgentCore runtimes
  and Bedrock guardrails.

**Behind the TLS-intercepting proxy**

```powershell
uv run python scripts/export_ca_bundle.py     # writes certs/ca-bundle.pem
```

Set `AWS__CA_BUNDLE=./certs/ca-bundle.pem` in `.env`. Docker builds take the same file
as a build secret (`--secret id=extra_ca,src=certs/ca-bundle.pem`), which is never
written to an image layer.

**`.env` at the repo root** (gitignored; copy `.env.example`). The deploy scripts read:

| Key | Used by |
|---|---|
| `AWS__ACCESS_KEY_ID`, `AWS__SECRET_ACCESS_KEY`, `AWS__CA_BUNDLE` | every script (leave the keys blank to use your normal AWS profile) |
| `AWS__ENDPOINT_URL` | **must be blank** for real AWS (`.env.example` points it at LocalStack) |
| `SEARCH_MONGO_URI` | Atlas connection string; `deploy/index-jobs/deploy.py` refuses to run without it |
| `S3__BUCKET`, `S3__PREFIX` | index build: where the tenant's source documents are |
| `BEDROCK__REGION`, `BEDROCK__TITAN_EMBED_MODEL`, `BEDROCK__LLM_MODEL` | index build and agents |

**MongoDB Atlas network access.** Atlas accepts connections only from IPs on the
project's IP Access List. Anything that talks to Atlas needs a fixed, listed egress IP:
your machine for the index build, the Lambdas (NAT IP `43.205.74.169`, already set up),
and the Document Search runtime (see §6.1).

---

## 3. Deployment order

First-time deployment, in this order. Each step depends on the one before it.

1. Build the per-tenant index in Atlas (§4)
2. Create the guardrail (§5)
3. Create the two IAM execution roles (§6)
4. Package the agent code (§7)
5. Create the **Document Search** runtime, then the **Orchestrator** runtime, which needs the search runtime's ARN (§8)
6. Smoke-test the agents (§9)
7. Deploy the index-jobs stack and run its end-to-end test (§10)
8. Hand SpecSafe the orchestrator ARN and the sync-api URL (§11)
9. Go through the go-live checklist (§14)

---

## 4. Build the search index (MongoDB Atlas)

One collection per tenant, named `<TenantCode>_DocumentSearch_<ENV>` in database
`DOCSEARCH_DEMO`. Each collection holds the chunks, their vectors, the keyword
text, the metadata, and a **manifest** the agent reads (tenant codes, source bucket and
region, counts). The agents read Atlas live: no snapshot to publish, no cold start.

Once the index-jobs stack (§10) is running, uploads and record changes keep the
collection current. Do this full build **once per tenant**, or to rebuild from scratch.

```powershell
# Which tenant, and where its documents are
$env:S3__BUCKET = "demo-docsearch-bucket"; $env:S3__PREFIX = "TENANT_A/"
$env:SEARCH_MONGO_DATABASE = "TENANT_A_DEV"          # the tenant's AttachmentMetaData catalogue
$env:SEARCH_MONGO_TENANT_CODES = "TENANT_A"          # only this TenantCode's records
$env:SEARCH_STORE_BACKEND = "mongo"
$env:SEARCH_INDEX_MONGO_COLLECTION = "TENANT_A_DocumentSearch_DEV"

# 1. Collection, regular indexes, and both Atlas search indexes (vector + text)
uv run python scripts/mongo_index.py setup --collection TENANT_A_DocumentSearch_DEV

# 2. Extract, chunk, embed (Titan) and write every catalogued document.
#    This RESETS the target. Use it for a fresh build only.
uv run python scripts/index_s3.py index --mongo-only --only pdf,docx,xlsx

# 3. Write the manifest the agent reads
uv run python scripts/mongo_index.py manifest --collection TENANT_A_DocumentSearch_DEV `
    --environment demo --tenant TENANT_A --tenant-codes TENANT_A

# 4. Check: counts, and both search indexes READY / queryable
uv run python scripts/mongo_index.py status --collection TENANT_A_DocumentSearch_DEV
```

**`--environment` / `--tenant` must equal what the tenant's `base_url` resolves to**
(§8.3). With the production pattern, `https://tenant-a.demo.com/` resolves to `demo/tenant-a`, so
this collection is mapped as `"demo/tenant-a"` in `AGENTIC_MONGO_COLLECTIONS`.

**Already have a local index?** Copy it instead of re-embedding (no Bedrock cost):
`uv run python scripts/mongo_index.py migrate --collection TENANT_A_DocumentSearch_DEV --from-dir ./data/search --environment demo --tenant TENANT_A --tenant-codes TENANT_A`.

**One index per `TenantCode`.** Tenants share S3 folders and catalogue databases (dev
`TENANT_A_DEV` is mostly `TENANT_A` records), so always set `SEARCH_MONGO_TENANT_CODES`.
Two tenants mapped to one collection is refused at agent startup.

> **Alternative: S3 snapshots.** Tenants *not* listed in `AGENTIC_MONGO_COLLECTIONS` are
> served from a snapshot under `s3://$AGENTIC_INDEX_BUCKET/$AGENTIC_INDEX_PREFIX/<environment>/<tenant>/`:
> `scripts/publish_file_snapshot.py` (Qdrant + SQLite files; works with code deployment)
> or `scripts/publish_index_snapshot.py` (`pg_dump`; **container deployment only**,
> because the restore needs PostgreSQL binaries). See
> [services/agentic/README.md](services/agentic/README.md#per-tenant-index-snapshots-in-s3).

---

## 5. Create the guardrail

Applied to every model call in both agents: denied topics (legal, medical, financial
advice), prompt attacks at HIGH, profanity, prompt-leak phrases, and blocked credentials
and financial identifiers.

```powershell
New-Item -ItemType Directory -Force data/logs | Out-Null   # the script writes data/logs/guardrail.json
uv run python scripts/create_agent_guardrail.py --publish
```

It creates or updates the guardrail and, with `--publish`, an immutable version. Then it
prints the three values you need next:

```
AGENTIC_GUARDRAIL_ID=<id>
AGENTIC_GUARDRAIL_VERSION=<n>
The execution roles need bedrock:ApplyGuardrail on: <guardrail ARN>
```

Use a published version number in deployed runtimes, never `DRAFT`. Otherwise an
edit to the draft changes production behaviour with no deployment.

---

## 6. IAM execution roles

Create one role per runtime, trusted by `bedrock-agentcore.amazonaws.com`.

**Trust policy (both roles)**

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "bedrock-agentcore.amazonaws.com" },
    "Action": "sts:AssumeRole",
    "Condition": { "StringEquals": { "aws:SourceAccount": "123456789012" } }
  }]
}
```

**Document Search role (`docsearch-search-role`)**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "Models",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/*"
      ] },
    { "Sid": "Guardrail",
      "Effect": "Allow",
      "Action": "bedrock:ApplyGuardrail",
      "Resource": "<guardrail ARN from §5>" },
    { "Sid": "SourceDocumentsForPresignedLinks",
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": ["arn:aws:s3:::demo-docsearch-bucket/TENANT_A/*", "arn:aws:s3:::demo-docsearch-bucket/TENANT_B/*"] },
    { "Sid": "Logs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                 "logs:DescribeLogStreams", "logs:DescribeLogGroups"],
      "Resource": "arn:aws:logs:us-east-1:123456789012:log-group:/aws/bedrock-agentcore/*" }
  ]
}
```

- **Source documents:** citations carry presigned S3 links, and a presigned URL works
  only if the role that signed it can read the object itself. Without this
  statement, every link returns 403.
- **S3 snapshots:** if you use them (§4 alternative), add `s3:GetObject` + `s3:ListBucket`
  on the index prefix.
- **Container deployment:** add ECR pull: `ecr:GetAuthorizationToken` on `*`, and
  `ecr:BatchGetImage` + `ecr:GetDownloadUrlForLayer` on the repository.

**Orchestrator role (`docsearch-orchestrator-role`)**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "Models",
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/*"
      ] },
    { "Sid": "Guardrail",
      "Effect": "Allow",
      "Action": "bedrock:ApplyGuardrail",
      "Resource": "<guardrail ARN from §5>" },
    { "Sid": "DelegateToSearchAgent",
      "Effect": "Allow",
      "Action": "bedrock-agentcore:InvokeAgentRuntime",
      "Resource": "<Document Search runtime ARN>*" },
    { "Sid": "Logs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                 "logs:DescribeLogStreams", "logs:DescribeLogGroups"],
      "Resource": "arn:aws:logs:us-east-1:123456789012:log-group:/aws/bedrock-agentcore/*" }
  ]
}
```

The orchestrator never retrieves, so it needs **no S3 and no Atlas access**. Add the
`DelegateToSearchAgent` statement after §8.1, once the search runtime's ARN exists.

**SpecSafe's caller (IAM user `docsearch-service`)** needs `bedrock-agentcore:InvokeAgentRuntime`
on the **Orchestrator** runtime ARN only. The same user signs the sync-api calls (§10);
that permission is granted by the Lambda stack itself.

### 6.1 Network: the Document Search runtime must reach Atlas

The search runtime is the only runtime that talks to Atlas, and Atlas allows listed IPs
only. Create it in **VPC network mode**, in private subnets in the runtime's region
(us-east-1) behind a NAT gateway, and add that NAT gateway's IP to the Atlas IP Access
List. The Lambdas' NAT (`43.205.74.169`) is in the ap-south-1 VPC, so it cannot be reused
for us-east-1 runtimes. The orchestrator needs only AWS APIs and can stay in `PUBLIC` mode.

---

## 7. Package the agent code

Both runtimes run the **same package**; `AGENTIC_ROLE` decides which agent each one is.

There are two ways to ship it. Both serve the same app,
[rag_agentic.runtime.app](services/agentic/src/rag_agentic/runtime/app.py): plain
FastAPI on `0.0.0.0:8080` with `POST /invocations` and `GET /ping` → `{"status": "Healthy"}`.

| | **Code deployment (zip)**, recommended with Atlas | **Container image** |
|---|---|---|
| Entry point | [services/agentic/deploy/runtime.py](services/agentic/deploy/runtime.py) | `uvicorn ${AGENT_MODULE}:app` ([Dockerfile](services/agentic/Dockerfile)) |
| Built by | `scripts/package_agent_code.py` | `docker build` |
| Index sources it can serve | Atlas, file snapshots | Atlas, file snapshots, **Postgres `pg_dump` snapshots** (bundles PostgreSQL 16 + pgvector) |

> The AWS `bedrock-agentcore` SDK (`bedrock_agentcore.runtime.BedrockAgentCoreApp`) is
> **deliberately not used**. The only SDK release compatible with `aioboto3`'s botocore
> pin (1.3.2) turns every handler response into HTTP 200, so a 400/503 would reach
> SpecSafe looking like a successful answer. Do not add it to the package.

### 7a. Code deployment (zip)

```powershell
# Build: vendors linux/aarch64 wheels for Python 3.12 + rag_agentic, rag_search, rag_core
uv run python scripts/package_agent_code.py

# Build + upload. The bucket MUST be in the runtime's region (us-east-1).
# Use a versioned key so you can roll back (§13).
uv run python scripts/package_agent_code.py --upload `
    --bucket demo-deployment-TENANT_A-bedrock `
    --key agent_core/specsafe_docsearch/<yyyymmdd>-<git-sha>/deployment.zip
```

- **Check the `--bucket` flag.** The script defaults to `demo-docsearch-bucket`, which is in
  ap-south-1, so pass a us-east-1 bucket explicitly.
- **What the build prints.** It shows the zipped and unpacked sizes and the biggest
  packages. `rag_processing` (the OCR and extraction stack) is left out on purpose.
- **Adding a runtime dependency.** Add it to `REQUIREMENTS` in
  [package_agent_code.py](scripts/package_agent_code.py). A missing import crashes the
  runtime at startup, and AgentCore reports that as an **initialisation timeout**, not an
  ImportError.

### 7b. Container image

Build **from the repo root**, for `linux/arm64`:

```powershell
$ECR = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR

docker build -f services/agentic/Dockerfile --platform linux/arm64 `
  --build-arg AGENT_MODULE=rag_agentic.runtime.search_app `
  --secret id=extra_ca,src=certs/ca-bundle.pem `
  -t $ECR/specsafe-doc-search:<tag> .
docker push $ECR/specsafe-doc-search:<tag>

docker build -f services/agentic/Dockerfile --platform linux/arm64 `
  --build-arg AGENT_MODULE=rag_agentic.runtime.orchestrator_app `
  --secret id=extra_ca,src=certs/ca-bundle.pem `
  -t $ECR/specsafe-orchestrator:<tag> .
docker push $ECR/specsafe-orchestrator:<tag>
```

Never push an image built with `--build-arg PLATFORM=linux/amd64` (for local testing
only). ECR accepts it, and the runtime then fails to start.

---

## 8. Create the AgentCore runtimes

Create the **Document Search runtime first**: the orchestrator needs its ARN.

You can use the console or the CLI. The CLI shapes below match the
`bedrock-agentcore-control` API. Check them against
`aws bedrock-agentcore-control create-agent-runtime help` for your CLI version.

Runtime names allow letters, digits and underscores only, no hyphens.

### 8.1 Document Search runtime

`search-env.json`:

```json
{
  "AGENTIC_ROLE": "document_search",
  "AGENTIC_REGION": "us-east-1",
  "AGENTIC_TENANT_PATTERN": "^(?P<tenant>[a-z0-9][a-z0-9-]*)\\.(?P<environment>demo|qa|test|staging|portal)\\.com$",

  "AGENTIC_MONGO_URI": "<Atlas connection string>",
  "AGENTIC_MONGO_DATABASE": "DOCSEARCH_DEMO",
  "AGENTIC_MONGO_COLLECTIONS": "{\"demo/tenant-a\": \"TENANT_A_DocumentSearch_DEV\", \"demo/tenant-b\": \"TENANT_B_DocumentSearch_DEV\"}",

  "AGENTIC_GUARDRAIL_ID": "<from §5>",
  "AGENTIC_GUARDRAIL_VERSION": "<from §5>",

  "AGENTIC_DOCUMENT_URL_MODE": "presigned",
  "AGENTIC_SOURCE_REGION": "ap-south-1",

  "BEDROCK__REGION": "us-east-1",
  "BEDROCK__LLM_MODEL": "<inference profile id enabled in the account>",
  "BEDROCK__TITAN_EMBED_MODEL": "amazon.titan-embed-text-v2:0",

  "APP_ENV": "dev",
  "LOG_JSON": "true"
}
```

Code deployment:

```powershell
aws bedrock-agentcore-control create-agent-runtime --region us-east-1 `
  --agent-runtime-name specsafe_docsearch_search `
  --role-arn arn:aws:iam::123456789012:role/docsearch-search-role `
  --agent-runtime-artifact '{\"codeConfiguration\": {\"code\": {\"s3\": {\"bucket\": \"demo-deployment-TENANT_A-bedrock\", \"prefix\": \"agent_core/specsafe_docsearch/<version>/deployment.zip\"}}, \"runtime\": \"PYTHON_3_12\", \"entryPoint\": [\"runtime.py\"]}}' `
  --network-configuration '{\"networkMode\": \"VPC\", \"networkModeConfig\": {\"subnets\": [\"<subnet-id>\"], \"securityGroups\": [\"<sg-id>\"]}}' `
  --protocol-configuration serverProtocol=HTTP `
  --environment-variables file://search-env.json
```

Container deployment: replace `--agent-runtime-artifact` with
`'{\"containerConfiguration\": {\"containerUri\": \"<ECR>/specsafe-doc-search:<tag>\"}}'`.

Wait until `aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id <id>` shows
`READY`, then note the **`agentRuntimeArn`**. Add it to the orchestrator role's
`DelegateToSearchAgent` statement (§6).

### 8.2 Orchestrator runtime

`orchestrator-env.json`:

```json
{
  "AGENTIC_ROLE": "orchestrator",
  "AGENTIC_REGION": "us-east-1",
  "AGENTIC_TENANT_PATTERN": "<same value as the search runtime>",

  "AGENTIC_SEARCH_TRANSPORT": "agentcore",
  "AGENTIC_SEARCH_AGENT_ARN": "<Document Search agentRuntimeArn from 8.1>",
  "AGENTIC_SEARCH_AGENT_QUALIFIER": "DEFAULT",

  "AGENTIC_GUARDRAIL_ID": "<from §5>",
  "AGENTIC_GUARDRAIL_VERSION": "<from §5>",

  "AGENTIC_AUTH_REQUIRED": "true",
  "AGENTIC_AUTH_ISSUER_TEMPLATE": "https://{tenant}.sso.example.com",
  "AGENTIC_AUTH_CLIENT_IDS": "[\"<SpecSafe client id>\"]",

  "AGENTIC_LICENSING_ENABLED": "true",
  "AGENTIC_LICENSING_BASE_URL": "https://tenantadmin.example.com",
  "AGENTIC_LICENSING_SETTLE_ONLY": "true",

  "BEDROCK__REGION": "us-east-1",
  "BEDROCK__LLM_MODEL": "<inference profile id enabled in the account>",

  "APP_ENV": "dev",
  "LOG_JSON": "true"
}
```

```powershell
aws bedrock-agentcore-control create-agent-runtime --region us-east-1 `
  --agent-runtime-name specsafe_docsearch_orchestrator `
  --role-arn arn:aws:iam::123456789012:role/docsearch-orchestrator-role `
  --agent-runtime-artifact '<same artifact as 8.1>' `
  --network-configuration networkMode=PUBLIC `
  --protocol-configuration serverProtocol=HTTP `
  --environment-variables file://orchestrator-env.json
```

### 8.3 Settings that matter

**Both runtimes**

| Setting | Why it matters |
|---|---|
| `AGENTIC_ROLE` | `document_search` or `orchestrator`. The same code otherwise. |
| `AGENTIC_TENANT_PATTERN` (or `AGENTIC_TENANT_MAP`) | The **only** thing that decides which tenant's documents a request reads. Keep it anchored (`^…$`) with the environment domains listed. A loose pattern accepts `tenant-a.test.com.attacker.net`. An unresolvable `base_url` is a 400, never a fallback. |
| `AGENTIC_REGION` | Region for Bedrock Converse and `InvokeAgentRuntime`; see §1. |
| `AGENTIC_GUARDRAIL_ID` / `_VERSION` | Blank disables the guardrail. Always set in a deployed runtime. |
| `AGENTIC_MODEL` | Optional per-agent model override; blank uses `BEDROCK__LLM_MODEL`. |

**Document Search runtime**

| Setting | Why it matters |
|---|---|
| `AGENTIC_MONGO_COLLECTIONS` | `"environment/tenant"` → collection. Keys must equal what `base_url` resolves to. Unlisted tenants fall back to S3 snapshots. One collection per tenant is enforced at startup. |
| `AGENTIC_SOURCE_REGION` | Region of the source bucket, for manifests that don't record one. A link signed for the wrong region gets HTTP 400 from S3. |

**Orchestrator runtime**

| Setting | Why it matters |
|---|---|
| `AGENTIC_SEARCH_AGENT_ARN` | Required with `AGENTIC_SEARCH_TRANSPORT=agentcore`. |
| `AGENTIC_AUTH_*` | Verifies SpecSafe's `AccessToken`. Set on the **orchestrator only**: the token is not forwarded to the search runtime. The issuer template must contain `{tenant}`, and two tenants sharing an issuer is refused at startup. Leave `AGENTIC_AUTH_USER_CODE_CLAIMS` empty on the first deploy, then read `user_code_matches` in the `auth_verified` log line to learn which claim carries the UserCode. |
| `AGENTIC_LICENSING_*` | Credit reserve/settle against TenantAdmin. `SETTLE_ONLY=true` when SpecSafe reserves and sends the `ReservationId`. `FAIL_OPEN` stays `false` unless availability is explicitly preferred over metering. |

All settings with their defaults:
[services/agentic/src/rag_agentic/config.py](services/agentic/src/rag_agentic/config.py).

**Secrets.** AgentCore environment variables are visible to anyone who can read the
runtime configuration, and `AGENTIC_MONGO_URI` and `AGENTIC_LICENSING_CLIENT_SECRET`
contain credentials. Restrict `bedrock-agentcore-control:GetAgentRuntime` accordingly,
and use an Atlas database user that can only read `DOCSEARCH_DEMO`.

---

## 9. Smoke-test the agents

```python
# uv run python smoke.py  — run with credentials allowed to invoke the orchestrator
import json, uuid, boto3

client = boto3.client("bedrock-agentcore", region_name="us-east-1")
resp = client.invoke_agent_runtime(
    agentRuntimeArn="<Orchestrator agentRuntimeArn>",
    runtimeSessionId=f"smoke-{uuid.uuid4()}",          # >= 33 characters
    qualifier="DEFAULT",
    payload=json.dumps({
        "base_url": "https://tenant-a.demo.com/",
        "query": "What documents do we have for supplier approval?",
        "UserCode": "DEPLOY-SMOKE",
        "AccessToken": "<a real SpecSafe token, if AGENTIC_AUTH_REQUIRED=true>",
    }).encode(),
)
body = json.loads(resp["response"].read())
print(body.get("route"), body.get("tenant"), len(body.get("citations", [])))
print(body.get("answer") or body)
```

**Expected:** `route` is `document_search`, `tenant` is `TENANT_A`, and there are citations
with working `download_url`s. Open one link to confirm the S3 permission.

**Then:**

- **Second tenant.** Repeat with `https://tenant-b.demo.com/`, and confirm none of TENANT_A's
  documents appear.
- **Bad tenant.** Send `https://unknown.example.com/` and expect **400**.
- **Logs.** Check CloudWatch (us-east-1, `/aws/bedrock-agentcore/runtimes/<runtime-id>-DEFAULT`).
  The orchestrator's and the search agent's lines share one `trace_id` per request.

**Local UI.** The search service's UI can send questions to the deployed orchestrator
(`SEARCH_AGENT_RUNTIME_ARN`, `SEARCH_AGENT_REGION`). Those calls carry no `AccessToken`,
so they are refused once `AGENTIC_AUTH_REQUIRED=true`. Also set `SEARCH_AGENT_REGION=us-east-1`:
it defaults to `ap-south-1`.

---

## 10. Deploy the index-jobs stack (incremental indexing)

Deploys to **ap-south-1** as CloudFormation stack `docsearch-dev`
([template.yaml](deploy/index-jobs/template.yaml)):

| Resource | Purpose |
|---|---|
| `index-events` Lambda + SQS queue + DLQ | Indexes, updates or removes one document per S3 event or SpecSafe call |
| EventBridge rule on `demo-docsearch-bucket` | S3 `Object Created` / `Object Deleted` → queue |
| `sync-api` Lambda + Function URL (IAM auth) | SpecSafe's "document changed" call → queue |
| `index-jobs` Lambda + 2 schedules | Daily 02:00 UTC `DeleteMark` purge for TENANT_A and TENANT_B (a safety net) |
| Alarms | DLQ messages, Lambda errors |

From the repo root, with Docker running and `SEARCH_MONGO_URI` in `.env`:

```powershell
# 1. Build the Lambda image (arm64). --provenance=false is required: Lambda rejects
#    the multi-manifest index BuildKit produces by default.
docker build -f deploy/index-jobs/Dockerfile --platform linux/arm64 --provenance=false `
  --secret id=extra_ca,src=certs/ca-bundle.pem -t specsafe-index-jobs:<tag> .

# 2. Push to ECR (pinned by digest), create/update the stack, turn on S3 -> EventBridge
uv run python deploy/index-jobs/deploy.py --image specsafe-index-jobs:<tag> `
  --specsafe-principal arn:aws:iam::123456789012:user/docsearch-service `
  --subnet-ids subnet-00000000 --vpc-id vpc-00000000 `
  --with-e2e

# 3. Prove it end to end on the isolated E2E target (cleans up after itself)
uv run python scripts/e2e_index_events.py
```

`deploy.py` prints the stack outputs: `SyncApiUrl`, `IndexEventsQueueUrl`,
`IndexEventsDlqUrl`, `FunctionArn`.

**Pass these flags on every run, including updates:**

- **`--subnet-ids` and `--vpc-id`.** If you leave them out, the parameter is cleared and
  the functions move **out of the VPC**. They then lose the NAT IP `43.205.74.169`,
  which Atlas allows, and every Atlas call fails.
- **`--with-e2e`.** Leaving it out removes the E2E target, and `e2e_index_events.py`
  stops working.

**These can be left out:**

- **`--specsafe-principal`.** On an update the deployed principal is kept. On first
  create, leaving it out grants the whole account, so always pass it then.
- **`--image`.** Without it, the deployed image is kept (for example, to change only the
  principal).

The E2E target has its own collection, catalogue and S3 prefix (`docsearch-e2e/`), so the
end-to-end test never touches TENANT_A or TENANT_B data. The 8 scenarios it checks are listed in
[e2e_index_events.py](scripts/e2e_index_events.py).

Logs (ap-south-1, 90 days): `/aws/lambda/docsearch-dev-index-events` (one
`document_sync` line per document), `-sync-api` (one `sync_requested` line per call),
`-index-jobs`.

---

## 11. Hand over to SpecSafe

| SpecSafe needs | Value |
|---|---|
| Orchestrator runtime ARN + region | from §8.2, `us-east-1`, qualifier `DEFAULT` |
| Request/response contract | [services/agentic/README.md](services/agentic/README.md#how-specsafe-invokes-it-and-reads-the-response) |
| Sync-api URL + signing details | `SyncApiUrl` output from §10; contract in [SPECSAFE_INTEGRATION.md](deploy/index-jobs/SPECSAFE_INTEGRATION.md) (send `DocuSearch_SpecSafe_Metadata_Service.docx`) |
| Calling identity | IAM user `docsearch-service` signs both calls |

If the Function URL changes (the stack was recreated), update `SPECSAFE_INTEGRATION.md`
and the `.docx` before sending them.

---

## 12. Updating a deployment

| Change | What to run |
|---|---|
| Agent code | §7 with a **new** `--key` (or image tag), then `aws bedrock-agentcore-control update-agent-runtime --agent-runtime-id <id>` with the new artifact, the same role, network and protocol settings, and the **full** environment variable set. Uploading a zip alone changes nothing. AgentCore creates a new runtime version and moves the `DEFAULT` endpoint to it. Update the search runtime first if both change. |
| Agent setting only | `update-agent-runtime` with the same artifact and the changed environment variables. |
| Guardrail policy | `create_agent_guardrail.py --publish`, then set the new `AGENTIC_GUARDRAIL_VERSION` on **both** runtimes. |
| New tenant | §4 for its collection, add it to `AGENTIC_MONGO_COLLECTIONS` on the search runtime, add it to `IndexTargetsJson` / the template's targets for the Lambda stack, and add its source prefix to the search role's S3 statement. |
| New environment domain | Add it to the alternation in `AGENTIC_TENANT_PATTERN` on both runtimes. This is deliberately a manual change. |
| Index-jobs code | Rebuild the image with a new tag, then `deploy.py --image ...` with the same flags as §10. |
| Index rebuild | §4. The live agents read the collection directly, so rebuild into a **new** collection, check it with `mongo_index.py status` / `compare`, then switch `AGENTIC_MONGO_COLLECTIONS`. Don't reset the collection that is serving. |

Run the tests before shipping agent changes. They are fully offline:

```powershell
uv run python -m pytest services/agentic/tests services/search/tests -q
```

---

## 13. Rollback

- **Agents.** Run `update-agent-runtime` pointing at the **previous** artifact: the
  previous versioned zip key or image tag, which is why §7 uses versioned keys. Then
  re-run the §9 smoke test.
- **Index-jobs stack.** Run `deploy.py --image specsafe-index-jobs:<previous tag>` with
  the same flags. To pause indexing on upload without changing code, update the stack in
  the CloudFormation console (use the current template) with `EventsState=DISABLED`.
  This turns off the S3 events rule. SpecSafe's sync-api calls are still processed,
  and the agents keep serving the current index.
- **Guardrail.** Set `AGENTIC_GUARDRAIL_VERSION` back to the previous published version.

---

## 14. Go-live checklist

- [ ] Both runtimes are `READY`. `/ping` is healthy, and the §9 smoke test passes for every tenant.
- [ ] `AGENTIC_TENANT_PATTERN` is anchored and lists only real environment domains.
- [ ] Every tenant in `AGENTIC_MONGO_COLLECTIONS` has its own collection, and its manifest has `tenant_codes`. The worker refuses a collection without them.
- [ ] A published guardrail version (not `DRAFT`) is set on both runtimes.
- [ ] `AGENTIC_AUTH_REQUIRED=true` on the orchestrator, `AGENTIC_AUTH_CLIENT_IDS` set, and `AGENTIC_AUTH_USER_CODE_CLAIMS` set once the claim is known.
- [ ] Licensing is configured, and a request has been seen to settle.
- [ ] The search role can `s3:GetObject` every tenant's source prefix, and a citation link opens.
- [ ] The search runtime's egress IP and the Lambdas' NAT IP `43.205.74.169` are on the Atlas IP Access List.
- [ ] `e2e_index_events.py` passes. The DLQ alarm and errors alarm are routed to someone.
- [ ] The catalogue owner has indexed `GeneratedFileName`, `OriginalFileName` and `DeleteMark` on `AttachmentMetaData`.
- [ ] SpecSafe has the orchestrator ARN, the sync-api URL and the integration document.

---

## 15. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Runtime stuck creating, or "initialisation timeout" | An import failed at startup: a dependency is missing from `REQUIREMENTS` in `package_agent_code.py`, or the image was built for amd64. Check the runtime's CloudWatch log for the traceback. |
| Every request returns **400** | `base_url` doesn't match `AGENTIC_TENANT_PATTERN` / `AGENTIC_TENANT_MAP`, or the payload is malformed. |
| **503** for one tenant | Its index is unavailable: not in `AGENTIC_MONGO_COLLECTIONS` and no S3 snapshot, Atlas unreachable (IP list), or the search indexes aren't `READY` (`mongo_index.py status`). |
| **502** from the orchestrator | The search runtime failed. Look at its log for the same `trace_id`. |
| Answers are keyword-only, or slow after a quiet period | A cold Atlas vector index. The search runtime warms mapped tenants at start and every `AGENTIC_MONGO_KEEP_WARM_S` (240 s); confirm the runtime isn't recycling constantly. |
| Citation link → **403** | The search role lacks `s3:GetObject` on that source prefix. |
| Citation link → **400** `AuthorizationQueryParametersError` | Signed for the wrong region: set `AGENTIC_SOURCE_REGION=ap-south-1`, or re-run `mongo_index.py manifest` so the manifest records `source_region`. |
| Citation link stops working before its TTL | Presigned URLs die with the role session that signed them. Use `AGENTIC_DOCUMENT_URL_MODE=template` for long-lived links. |
| `boto3/botocore does not know the 'bedrock-agentcore' service` | The AWS SDK in the package or image is too old. Rebuild. |
| Every request → **401** `access token rejected` after enabling auth | The issuer template doesn't produce the token's real issuer, a client id isn't in `AGENTIC_AUTH_CLIENT_IDS`, or the token's scope isn't `SpecSafe.Core.Api`. The response's `reason` field names which one. |
| **503** with `auth_unverifiable` in the log | The issuer's signing keys couldn't be fetched (IdP unreachable from the runtime). The token was not judged invalid. |
| Lambda image rejected on update | Built without `--provenance=false`, or for the wrong architecture. |
| Lambda timeouts talking to Atlas | The functions are out of the VPC (`deploy.py` was run without `--subnet-ids`/`--vpc-id`), or the NAT IP isn't on the Atlas list. |
| A document stays unsearchable after upload | It is "hidden" until its `AttachmentMetaData` record exists (re-checked for 60 min). Search the `-index-events` log for its key. A message in the DLQ means it failed 5 times: fix the cause, then redrive the DLQ. |

---

## 16. Script reference

| Script | Purpose |
|---|---|
| `scripts/package_agent_code.py` | Build (and `--upload`) the AgentCore code-deployment zip |
| `services/agentic/deploy/runtime.py` | Entry point inside that zip |
| `scripts/create_agent_guardrail.py` | Create or update the guardrail; `--publish` for a version |
| `scripts/mongo_index.py` | Atlas collections: `setup`, `migrate`, `manifest`, `move`, `status`, `compare` |
| `scripts/index_s3.py` | Full index build from S3 + catalogue |
| `scripts/publish_file_snapshot.py` | S3 file snapshot (Qdrant + SQLite) for one tenant |
| `scripts/publish_index_snapshot.py` | S3 `pg_dump` snapshot for one tenant (container deployment only) |
| `deploy/index-jobs/deploy.py` | Create or update the index-jobs stack (idempotent) |
| `scripts/e2e_index_events.py` | End-to-end test of the deployed index-jobs stack |
| `scripts/incremental_index.py` | Run `index` / `sync` / `purge` locally, with the same code as the Lambdas |
| `scripts/export_ca_bundle.py` | CA bundle for the managed TLS proxy |
