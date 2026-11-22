"""SpecSafe's real base URLs -> (environment, tenant), and nothing else.

The URL shape is ``https://<tenant>.<environment-domain>.com``. The recommended
pattern anchors the whole host and allow-lists the environment domains, so a
lookalike host can never borrow a real tenant's identity. An unanchored or
wildcard pattern would accept ``tenant-a.test.com.attacker.net`` as tenant tenant-a.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_agentic.config import AgenticSettings
from rag_agentic.index_store import TenantIndexStore
from rag_agentic.tenancy import TenancyError, TenantContext, TenantResolver

# The value to deploy as AGENTIC_TENANT_PATTERN.
SPECSAFE_TENANT_PATTERN = (
    r"^(?P<tenant>[a-z0-9][a-z0-9-]*)"
    r"\.(?P<environment>demo|qa|test|staging|portal)\.com$"
)


@pytest.fixture
def resolver() -> TenantResolver:
    return TenantResolver(AgenticSettings(tenant_pattern=SPECSAFE_TENANT_PATTERN))


@pytest.mark.parametrize(
    ("base_url", "environment", "tenant", "prefix"),
    [
        ("https://tenant-a.demo.com/", "demo", "tenant-a", "DocumentSearch/demo/tenant-a"),
        ("https://tenant-b.qa.com/", "qa", "tenant-b", "DocumentSearch/qa/tenant-b"),
        ("https://tenant-a.test.com/", "test", "tenant-a", "DocumentSearch/test/tenant-a"),
        ("https://tenant-a.staging.com/", "staging", "tenant-a", "DocumentSearch/staging/tenant-a"),
        ("https://tenant-a.portal.com/", "portal", "tenant-a", "DocumentSearch/portal/tenant-a"),
    ],
)
def test_real_specsafe_urls_resolve(
    resolver: TenantResolver, base_url: str, environment: str, tenant: str, prefix: str
) -> None:
    context = resolver.resolve(base_url)
    assert (context.environment, context.tenant) == (environment, tenant)
    # Each (environment, tenant) reads its own S3 folder -- tenant-a's three environments
    # never share an index.
    assert context.s3_prefix("DocumentSearch") == prefix


@pytest.mark.parametrize(
    "base_url",
    [
        "HTTPS://TENANT-A.test.com",  # case and a missing trailing slash
        "https://tenant-a.test.com:443/",  # explicit port
        "tenant-a.test.com",  # no scheme
    ],
)
def test_equivalent_spellings_resolve_the_same(resolver: TenantResolver, base_url: str) -> None:
    assert resolver.resolve(base_url).key == "test/tenant-a"


@pytest.mark.parametrize(
    "base_url",
    [
        "https://tenant-a.test.com.attacker.net/",  # real host as a prefix of another
        "https://tenant-a.test.co/",  # different TLD
        "https://tenant-a.demox.com/",  # lookalike of the dev domain
        "https://tenant-a.testx.com/",  # lookalike environment domain
        "https://tenant-a.evil.com/",  # unknown environment
        "https://test.com/",  # no tenant
        "https://a.tenant-a.test.com/",  # extra subdomain level
        "https://tenant-a.test.com/other/app",  # a path: not a base URL
    ],
)
def test_lookalikes_are_rejected_not_guessed(resolver: TenantResolver, base_url: str) -> None:
    with pytest.raises(TenancyError):
        resolver.resolve(base_url)


# --------------------------------------------------------------------------
# The flat-prefix override belongs to exactly one tenant
# --------------------------------------------------------------------------


def test_override_without_a_bound_tenant_fails_at_startup() -> None:
    with pytest.raises(ValidationError, match="AGENTIC_INDEX_PREFIX_OVERRIDE_TENANT"):
        AgenticSettings(index_prefix_override="DocumentSearch/TENANT_A-Dev")


async def test_override_refuses_every_other_tenant() -> None:
    """With one flat folder configured, Example Supplier must not be served TENANT_A's documents."""

    class NoS3:
        async def get_bytes(self, *a: object, **k: object) -> bytes:
            raise AssertionError("must be refused before touching S3")

        async def head_object(self, *a: object, **k: object) -> dict[str, object]:
            raise AssertionError("must be refused before touching S3")

        def list_objects(self, *a: object, **k: object) -> object:
            raise AssertionError("must be refused before touching S3")

    settings = AgenticSettings(
        index_bucket="demo-deployment-TENANT_A-bedrock",
        index_prefix_override="DocumentSearch/TENANT_A-Dev",
        index_prefix_override_tenant="demo/tenant-a",
        tenant_pattern=SPECSAFE_TENANT_PATTERN,
    )
    store = TenantIndexStore(settings, NoS3())  # type: ignore[arg-type]
    tenant-b = TenantResolver(settings).resolve("https://tenant-b.qa.com/")

    with pytest.raises(TenancyError, match="serves only 'demo/tenant-a'"):
        await store.acquire(tenant-b)


def test_override_still_serves_its_own_tenant() -> None:
    settings = AgenticSettings(
        index_prefix_override="DocumentSearch/TENANT_A-Dev", index_prefix_override_tenant="demo/tenant-a"
    )
    store = TenantIndexStore(settings, object())  # type: ignore[arg-type]
    own = TenantContext(environment="TENANT_A-Dev", tenant="TENANT_A", base_url="x")
    assert store._prefix_for(own) == "DocumentSearch/TENANT_A-Dev"
