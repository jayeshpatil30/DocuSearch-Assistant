"""Agentic-service settings (env prefix ``AGENTIC_``).

Everything deployment-specific lives here so the same image runs as either agent
in any environment: which S3 bucket holds the per-tenant indexes, how a base URL
maps to (environment, tenant), and — for the orchestrator — the ARN of the
document-search runtime it delegates to.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict
from rag_core.config import DotenvFirstSettings


class AgentRole(StrEnum):
    ORCHESTRATOR = "orchestrator"
    DOCUMENT_SEARCH = "document_search"


class SearchTransport(StrEnum):
    """How the orchestrator reaches the document-search agent."""

    AGENTCORE = "agentcore"   # separate runtime, InvokeAgentRuntime (production)
    IN_PROCESS = "in_process"  # same process (local dev, tests, single-container demo)


class PostgresSource(StrEnum):
    """Where a Postgres-backed snapshot is read from."""

    # Download the snapshot's pg_dump from S3 and restore it into a PostgreSQL server
    # inside this container. S3 is the only storage; no database runs anywhere else.
    S3 = "s3"
    # Connect to an already-running database at ``postgres_dsn`` (RDS, or the
    # docker-compose database for local dev on a machine without PG binaries).
    DSN = "dsn"


class DocumentUrlMode(StrEnum):
    """How each citation's ``download_url`` is produced."""

    # Time-limited S3 GET URL for the source object, opened directly by the browser.
    # Works with no extra infrastructure, but the URL carries the bucket, key and a
    # credential id, and cannot outlive the credentials that signed it.
    PRESIGNED = "presigned"
    # A SpecSafe-served URL built from ``document_url_template``. SpecSafe then
    # authenticates the user and streams the file, so nothing about storage leaks.
    TEMPLATE = "template"
    NONE = "none"


class AgenticSettings(DotenvFirstSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_", env_file=".env", extra="ignore", case_sensitive=False
    )

    role: AgentRole = AgentRole.ORCHESTRATOR
    region: str = "us-east-1"

    # ---- Per-tenant index storage -------------------------------------------
    # Layout in the bucket (see index_store.py):
    #   s3://{index_bucket}/{index_prefix}/{environment}/{tenant}/manifest.json
    #                                                    /qdrant/**      vectors
    #                                                    /keyword.db     FTS5
    #                                                    /metadata.db    facets
    # A manifest declaring "backend": "postgres" instead points at ``postgres_dsn``;
    # only the manifest is read from S3 in that case.
    index_bucket: str = ""
    index_prefix: str = "indexes"
    # Read ONE tenant from this exact prefix instead of deriving
    # {index_prefix}/{environment}/{tenant}. It bypasses per-tenant path isolation,
    # so it must be bound to the single tenant it belongs to: any other tenant is
    # refused rather than served that tenant's documents. Blank = per-tenant paths.
    index_prefix_override: str = ""
    # "environment/tenant" the override belongs to. Required with the override.
    index_prefix_override_tenant: str = ""
    # Snapshots whose manifest declares "backend": "postgres" are a pg_dump plus the
    # manifest. ``postgres_source`` decides how they are read (see PostgresSource).
    postgres_source: PostgresSource = PostgresSource.S3
    # Only for postgres_source=dsn. A SINGLE database, therefore a single tenant:
    # the `chunks` table carries no tenant column, so two tenants sharing one
    # database would see each other's documents. The index store enforces that --
    # a second distinct tenant against this DSN is refused, not silently served.
    postgres_dsn: str = ""
    # Multi-tenant form: one database per tenant on one server. Must contain
    # `{tenant}`; `{environment}` is also substituted. Isolation is physical, so no
    # query can cross a tenant boundary even if a filter is forgotten later.
    #   postgresql://user:pw@host:5432/{environment}_{tenant}
    postgres_dsn_template: str = ""
    # Only for postgres_source=s3: where initdb/pg_ctl/pg_restore live. Blank searches
    # PATH and /usr/lib/postgresql/<major>/bin.
    postgres_bin_dir: str = ""
    # Parallel pg_restore workers. The HNSW index build dominates restore time.
    postgres_restore_jobs: int = Field(default=2, ge=1, le=16)
    # ---- Live MongoDB Atlas index (no snapshot, no cold start) ----------------
    # Tenants listed in ``mongo_collections`` are served straight from their Atlas
    # collection: nothing is downloaded or restored, and the manifest is a record in
    # that collection. Unlisted tenants keep the S3 snapshot path above.
    #   AGENTIC_MONGO_COLLECTIONS='{"demo/tenant-a": "TENANT_A_DocumentSearch_DEV"}'
    # Naming: database OB_<FEATURE>_<ENV>, collection <TenantCode>_DocumentSearch_<ENV>.
    mongo_uri: str = ""
    mongo_database: str = "DOCSEARCH_DEMO"
    # "environment/tenant" -> collection. One collection per tenant: isolation is
    # physical, exactly like one database per tenant on Postgres.
    mongo_collections: dict[str, str] = Field(default_factory=dict)
    mongo_tls_ca_file: str | None = None
    mongo_pool_max: int = Field(default=20, ge=1)
    # Exact vector search reads every vector per query (30-60 s in Atlas); see
    # SEARCH_INDEX_MONGO_EXACT_VECTORS. Approximate + rescoring is the default.
    mongo_exact_vectors: bool = False
    # Server-side cap per search query; see SEARCH_INDEX_MONGO_QUERY_TIMEOUT_MS.
    mongo_query_timeout_ms: int = Field(default=8_000, ge=100)
    # Re-probe every mapped tenant this often so Atlas keeps their indexes in
    # memory; see SEARCH_INDEX_MONGO_KEEP_WARM_S. 0 disables.
    mongo_keep_warm_s: float = Field(default=240.0, ge=0.0)
    # How long a question may wait for a cold index to become resident before
    # going ahead without it. A cold $vectorSearch on a large collection can
    # exceed its own time limit and silently degrade to keyword-only, so waiting
    # buys a correct answer with time the request would have spent anyway.
    # 0 disables the wait and restores the previous behaviour.
    mongo_warm_wait_s: float = Field(default=60.0, ge=0.0)
    # Open the connection pool and run one query per mapped tenant at startup, in
    # the background, so the first real request finds everything warm.
    mongo_warm_on_start: bool = True

    # Where a downloaded snapshot is materialised inside the container. AgentCore
    # gives each runtime a writable /tmp; anything else is read-only.
    cache_dir: str = "/tmp/rag-agentic/tenants"
    # Snapshots held open at once. Each costs a Qdrant dir + two SQLite handles,
    # so this caps container memory/FDs regardless of how many tenants call in.
    cache_max_tenants: int = Field(default=4, ge=1)
    # Re-check the manifest ETag after this long; a changed snapshot is re-pulled.
    cache_ttl_s: float = Field(default=300.0, ge=0.0)

    # ---- Tenancy resolution (base_url -> environment + tenant) ---------------
    # Checked in order: exact map, then regex, then defaults. See tenancy.py.
    #   AGENTIC_TENANT_MAP='{"https://demo.example.com/acme": "dev/acme"}'
    tenant_map: dict[str, str] = Field(default_factory=dict)
    #   host-style:  AGENTIC_TENANT_PATTERN='^(?P<tenant>[^.]+)\.(?P<environment>[^.]+)\.specsafe\.io'
    #   path-style:  AGENTIC_TENANT_PATTERN='^(?P<environment>[^.]+)\.specsafe\.io/(?P<tenant>[^/]+)'
    tenant_pattern: str = ""
    # Used only when neither the map nor the pattern matches. Leave blank to make
    # an unrecognised base_url a hard error rather than a silent cross-tenant read.
    default_environment: str = ""
    default_tenant: str = ""

    # ---- Caller authentication (SpecSafe access token) ------------------------
    # Verify the payload's AccessToken before anything is reserved or run. IAM
    # permission to invoke the runtime proves only that the *caller* is allowed to
    # reach us; the token is what establishes the *end user* whose credits and ACL
    # the rest of the request spends. Off by default so the offline tests and local
    # demos run, and there is deliberately no "check it if present" middle setting:
    # that would be satisfied by omitting the field.
    #
    # Set on the ORCHESTRATOR only. The search runtime is reached through
    # InvokeAgentRuntime from the orchestrator, and `AgentRequest.access_token` is
    # excluded from the delegated envelope by design (a token that need not travel
    # should not), so requiring one there would refuse every delegated call.
    auth_required: bool = False
    # Which issuer's tokens are good for which tenant, keyed by "environment/tenant"
    # or by the bare tenant name. Resolved from the tenant we already decided from
    # base_url -- NEVER from the token's own `iss`, which the holder controls.
    # Without this, a genuine token for one tenant would be accepted against
    # another tenant's corpus.
    #   AGENTIC_AUTH_ISSUER_MAP='{"dev/TENANT_B": "https://TENANT_B.sso.example.com"}'
    auth_issuer_map: dict[str, str] = Field(default_factory=dict)
    # Used when the map has no entry. Placeholders: {tenant} {environment}.
    #   AGENTIC_AUTH_ISSUER_TEMPLATE='https://{tenant}.sso.example.com'
    auth_issuer_template: str = ""
    # The API scope SpecSafe's own clients request. A token minted on the same
    # issuer for a different scope is valid for a different job, and accepting it
    # here would be a privilege upgrade.
    auth_required_scope: str = "SpecSafe.Core.Api"
    # Registered client ids allowed to call us. Empty accepts any client the issuer
    # signed for, which is weaker -- set it to SpecSafe's own client id.
    #   AGENTIC_AUTH_CLIENT_IDS='["EXAMPLE_CLIENT_ID"]'
    auth_client_ids: list[str] = Field(default_factory=list)
    # Tie the payload's UserCode to the verified token, by naming the claim(s) that
    # should carry it. Without this the token proves that *a* user signed in, while
    # UserCode -- which is what the audit trail and the credit record are attributed
    # to -- remains whatever the caller typed. A token for user A can then be sent
    # with `UserCode: B`.
    #
    # Several claims may be listed and a match against any one is accepted: every
    # candidate is asserted by the IdP, so the caller cannot choose which one to
    # satisfy. Empty disables the check.
    #
    # Which claim is right depends on the IdP's mapping, so leave this empty for the
    # first deploy: `auth_verified` logs `user_code_matches` with the NAMES of every
    # claim whose value equals the UserCode that arrived, and that names the claim to
    # put here. Claim names only -- no values, so nothing identifying is logged.
    #   AGENTIC_AUTH_USER_CODE_CLAIMS='["preferred_username"]'
    auth_user_code_claims: list[str] = Field(default_factory=list)
    # Blank skips the `aud` check, which is the correct default here: this issuer
    # does not put an audience on access tokens issued for a bare API scope, so
    # requiring one would refuse every real token. Set it only if that changes.
    auth_audience: str = ""
    # Clock drift allowed between the IdP, SpecSafe and this runtime. Small: it
    # extends the life of every expired token by exactly this much.
    auth_leeway_s: float = Field(default=60.0, ge=0.0, le=300.0)
    # How long the issuer's signing keys are reused. Fetching per request would put
    # the IdP in the hot path of every question; an unknown key id still triggers
    # one rate-limited early refresh, which is what a key rotation looks like.
    auth_jwks_ttl_s: float = Field(default=3600.0, ge=60.0)
    auth_jwks_timeout_s: float = Field(default=5.0, gt=0)

    # ---- Document links on citations ------------------------------------------
    document_url_mode: DocumentUrlMode = DocumentUrlMode.PRESIGNED
    # Presigned-URL lifetime. Signed with temporary credentials (an AgentCore
    # execution role), a URL stops working when those credentials expire even if
    # this is longer -- so for links that must live for hours, use the template.
    document_url_ttl_s: int = Field(default=3600, ge=60, le=604_800)
    # Placeholders: {base_url} {environment} {tenant} {document_id} {source_key}
    # {filename}. e.g. "{base_url}/documents/{document_id}"
    document_url_template: str = ""
    # Bucket holding the SOURCE documents, used only when a snapshot's manifest does
    # not name one. The manifest is authoritative: it is per tenant, this is not.
    source_bucket: str = ""
    # Region of that bucket, for manifests that do not record ``source_region``. A
    # presigned URL signed for the wrong region is rejected by S3 (HTTP 400,
    # AuthorizationQueryParametersError) -- dev's demo-docsearch-bucket is ap-south-1.
    source_region: str = ""
    # Citations returned per answer. Search-grounded sources come first (best score
    # first); documents a listing question returned follow, so "list all Draft
    # documents" links every document it names rather than the top 8.
    max_citations: int = Field(default=25, ge=1, le=100)

    # How many rows of a document listing travel back to the caller. The model
    # still sees only the window it asked for; this is what SpecSafe renders.
    max_list_documents: int = 1000

    # ---- Guardrails -----------------------------------------------------------
    # An Amazon Bedrock guardrail applied to EVERY model call in both agents:
    # denied topics (legal/medical/financial advice), harmful content, prompt
    # attacks, and high-risk identifiers. Blank disables it, which is how the
    # offline tests and local demos run.
    guardrail_id: str = ""
    guardrail_version: str = "DRAFT"
    # 'enabled' returns the guardrail's assessment alongside the answer, which is
    # what makes a block auditable in CloudWatch.
    guardrail_trace: str = "enabled"

    # ---- Model / agent loop --------------------------------------------------
    # Blank = inherit rag_core's BEDROCK__LLM_MODEL (an inference-profile id already
    # verified in the target account). Override per agent to run a cheaper model on
    # the orchestrator than on the search agent.
    model: str = ""
    max_tokens: int = Field(default=2048, ge=64)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # Hard ceiling on model<->tool round trips. Reaching it ends the turn with
    # whatever the model has said so far rather than looping on the caller's dime.
    max_tool_iterations: int = Field(default=6, ge=1, le=20)
    # Cache the (large, stable) system prompt so follow-up turns re-read it at the
    # discounted cache rate instead of full price.
    prompt_cache: bool = True

    # ---- Retrieval defaults --------------------------------------------------
    # Passages per search. Deliberately generous: each extra SEARCH costs a model
    # round trip over a transcript that has already grown, while extra PASSAGES in
    # one search cost almost nothing (retrieval measured at ~0.2s). Five searches
    # of 8 passages took 98s; the same evidence in one search does not.
    top_k: int = Field(default=15, ge=1, le=50)
    candidate_k: int = Field(default=30, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    embed_dim: int = Field(default=1024, ge=1)
    collection: str = "compliance"
    # "fake" swaps in the deterministic hashing embedder — no AWS, for tests/demo.
    embed_provider: str = "titan"

    # ---- Orchestrator -> document-search wiring ------------------------------
    search_transport: SearchTransport = SearchTransport.AGENTCORE
    # arn:aws:bedrock-agentcore:<region>:<acct>:runtime/<name>-<id>
    search_agent_arn: str = ""
    search_agent_qualifier: str = "DEFAULT"
    search_invoke_timeout_s: float = Field(default=120.0, gt=0)


    # ---- SpecSafe Licensing (credits) ---------------------------------------
    # Reserve before a run, settle after. Off until a base URL is configured, so
    # an environment that has not been given Licensing credentials keeps answering
    # rather than failing every request at the gate.
    licensing_enabled: bool = False
    # TenantAdmin API host, e.g. https://tenantadmin.example.com
    licensing_base_url: str = ""
    # Must be one Licensing knows (COMPLIANCE_AI_ASSISTANT, OBJECT_COMPARE,
    # DOCUMENT_SEARCH); a request may name its own, this is the default.
    licensing_feature_code: str = "DOCUMENT_SEARCH"
    licensing_timeout_s: float = Field(default=10.0, gt=0)
    # What to do when Licensing cannot be reached at all. False (the default)
    # refuses the run: a credit gate that fails open is not a gate, since anyone
    # able to make Licensing unreachable would make credits optional. True trades
    # that for availability, and every bypass is logged at error level.
    licensing_fail_open: bool = False
    # Report usage only: SpecSafe performs CheckAndReserve itself and sends the
    # ReservationId with the request, so the agent just settles it. In this mode
    # the agent never gates a request -- a missing reservation is logged and the
    # run proceeds, because metering must not be able to refuse an answer.
    licensing_settle_only: bool = False
    # Cross-check a supplied ReservationId against GET /Credits/GetReservation
    # before running. Without it the agent trusts whatever id arrives and only
    # finds out at Settle -- after the tokens are spent -- that it was already
    # committed, released or expired. This DOES gate the run, which is a
    # deliberate exception to the settle-only rule above: an id that cannot be
    # settled means the work could never be billed to anyone.
    licensing_verify_reservation: bool = True
    # Covers both UserPrompt and Response: the guide flags the prompt as possibly
    # carrying PII, and an answer can quote the documents behind it, so the two
    # travel together under one deliberate opt-in rather than separately.
    licensing_send_user_prompt: bool = False
    # A phase-1 `Operation: "objectcodes"` call is a facet lookup: no model, no
    # vector search, nothing to bill. It still arrives with a ReservationId, so
    # the question is what to do with that reservation afterwards.
    #   True  -- settle it at zero tokens. Leaves an auditable row and closes the
    #            reservation, so phase 2 MUST arrive with a fresh ReservationId.
    #   False -- leave it untouched and still `Reserved`, so phase 2 may reuse the
    #            same id. Costs nothing, but the hold lingers until it expires.
    licensing_settle_object_codes: bool = True
    # SpecSafe splits one user question into two calls and cannot add a field to
    # say which is which: call 1 names a BusinessObject and omits ObjectCodes,
    # call 2 carries the codes its ACL allowed. With this on, a request naming a
    # BusinessObject and carrying no ObjectCodes key at all is treated as call 1
    # -- a facet lookup rather than a question, which is what the caller does with
    # the result anyway. Off by default: it is an inference about one caller's
    # protocol, and a client that genuinely asks a question without ObjectCodes
    # would get a code list back instead of an answer. Enable it only where every
    # caller is known to be two-phase.
    infer_object_codes_lookup: bool = False
    # Licensing identifies tenants by its own code. Blank means "use the resolved
    # tenant", which is right when they agree; otherwise map them explicitly as
    # "TENANT_A=TENANT_A,TENANT_B=TENANT_B".
    licensing_tenant_codes: str = ""
    # Only needed when SpecSafe cannot forward the user's IAM token on the request.
    # Forwarding is preferred: it attributes the spend to the real user and keeps
    # per-tenant client secrets out of this runtime's environment.
    licensing_token_endpoint: str = ""
    licensing_client_id: str = ""
    licensing_client_secret: SecretStr = SecretStr("")
    licensing_scope: str = ""

    def licensing_tenant_code(self, tenant: str) -> str:
        """The code Licensing knows this tenant by."""
        for pair in self.licensing_tenant_codes.split(","):
            name, _, code = pair.partition("=")
            if name.strip().lower() == tenant.strip().lower() and code.strip():
                return code.strip()
        return tenant

    @model_validator(mode="after")
    def _auth_issuer_isolates_tenants(self) -> AgenticSettings:
        """Every tenant must get its own issuer, or the token check stops working.

        The issuer is what binds a token to a tenant. Two tenants resolving to one
        issuer means a token minted for either is accepted for both -- and every
        other check still passes, so the hole is invisible: the signature is
        genuine, the scope is right, the expiry is fine, and the holder reads
        another customer's corpus.

        The same shape of mistake as ``postgres_dsn_template`` below, and refused
        the same way: at startup, not in production. A deployment that cannot
        isolate tenants should fail to come up rather than serve one of them the
        other's documents.
        """
        template = self.auth_issuer_template.strip()
        if template and "{tenant}" not in template:
            raise ValueError(
                "AGENTIC_AUTH_ISSUER_TEMPLATE must contain '{tenant}' so each tenant "
                "gets its own issuer; without it every tenant resolves to the same "
                "issuer and one tenant's token would be accepted for another "
                f"(got {self.auth_issuer_template!r})"
            )

        # A key may be "environment/tenant" or the bare tenant name, and both
        # spellings of the SAME tenant pointing at one issuer is correct -- so
        # compare tenants, not keys.
        tenants_by_issuer: dict[str, set[str]] = {}
        for key, issuer in self.auth_issuer_map.items():
            tenant = str(key).rsplit("/", 1)[-1].strip().lower()
            tenants_by_issuer.setdefault(str(issuer).rstrip("/"), set()).add(tenant)
        for issuer, tenants in tenants_by_issuer.items():
            if len(tenants) > 1:
                raise ValueError(
                    f"AGENTIC_AUTH_ISSUER_MAP gives tenants {sorted(tenants)} the same "
                    f"issuer {issuer!r}; a token for any one of them would then be "
                    "accepted for the others. Give each tenant its own issuer, or "
                    "leave the shared ones out of the map."
                )

        if self.auth_required and not self.auth_issuer_map and not template:
            # Requiring a token with no issuer to check it against cannot serve a
            # single request -- every one would fail closed. Say so now.
            raise ValueError(
                "AGENTIC_AUTH_REQUIRED is set but no issuer is configured; set "
                "AGENTIC_AUTH_ISSUER_TEMPLATE (e.g. 'https://{tenant}.sso.example.com') "
                "or AGENTIC_AUTH_ISSUER_MAP"
            )
        return self

    @model_validator(mode="after")
    def _dsn_template_isolates_tenants(self) -> AgenticSettings:
        """A template without ``{tenant}`` is one database shared by every tenant.

        The ``chunks`` table has no tenant column, so sharing a database means a
        search returns other customers' documents. Refuse at startup rather than
        discover it in production. Use ``postgres_dsn`` for a genuine single-tenant
        deployment -- that path is guarded separately, at load time.
        """
        template = self.postgres_dsn_template.strip()
        if template and "{tenant}" not in template:
            raise ValueError(
                "AGENTIC_POSTGRES_DSN_TEMPLATE must contain '{tenant}' so each tenant "
                "gets its own database; without it every tenant would share one "
                "database and see each other's documents "
                f"(got {self.postgres_dsn_template!r})"
            )
        if template and self.postgres_dsn.strip():
            raise ValueError(
                "Set AGENTIC_POSTGRES_DSN or AGENTIC_POSTGRES_DSN_TEMPLATE, not both -- "
                "which one applies to a given tenant would be ambiguous"
            )
        return self

    @model_validator(mode="after")
    def _mongo_collections_isolate_tenants(self) -> AgenticSettings:
        """Each mapped tenant needs its own collection, and a connection to reach it.

        Collections carry no tenant column, so two tenants mapped to one collection
        would read each other's documents. Refuse at startup.
        """
        if not self.mongo_collections:
            return self
        if not self.mongo_uri.strip():
            raise ValueError("AGENTIC_MONGO_COLLECTIONS is set but AGENTIC_MONGO_URI is not")
        seen: dict[str, str] = {}
        for key, collection in self.mongo_collections.items():
            environment, _, tenant = key.partition("/")
            if not environment or not tenant or "/" in tenant:
                raise ValueError(
                    f"AGENTIC_MONGO_COLLECTIONS keys must be 'environment/tenant', got {key!r}"
                )
            if not collection.strip():
                raise ValueError(f"AGENTIC_MONGO_COLLECTIONS[{key!r}] names no collection")
            if collection in seen:
                raise ValueError(
                    f"tenants {seen[collection]!r} and {key!r} both map to collection "
                    f"{collection!r}; each tenant needs its own so neither sees the other's "
                    "documents"
                )
            seen[collection] = key
        return self

    @model_validator(mode="after")
    def _override_is_bound_to_one_tenant(self) -> AgenticSettings:
        """An unbound override would serve one index to every tenant that resolves.

        Checked at startup so a misconfigured deployment never takes a request.
        """
        if not self.index_prefix_override.strip("/"):
            return self
        environment, _, tenant = self.index_prefix_override_tenant.strip().partition("/")
        if not environment or not tenant or "/" in tenant:
            raise ValueError(
                "AGENTIC_INDEX_PREFIX_OVERRIDE serves a single tenant, so it needs "
                "AGENTIC_INDEX_PREFIX_OVERRIDE_TENANT='environment/tenant' naming which one "
                f"(got {self.index_prefix_override_tenant!r})"
            )
        return self
