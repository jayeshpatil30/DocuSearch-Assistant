"""The publisher must not take a snapshot's source location from ambient config.

``source_bucket``/``source_prefix`` in a manifest are what document download links
are minted from. They used to come from the shell's .env, so publishing TENANT_B from a
shell configured for TENANT_A wrote TENANT_A's bucket into TENANT_B's manifest -- every TENANT_B download
link would have pointed into another tenant's bucket. Caught in a dry run, but only
because someone read the output.

The prefix is now derived from the index's own ``source_key`` values and compared
with whatever configuration claims; a disagreement refuses the publish.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "publish_file_snapshot.py"
_spec = importlib.util.spec_from_file_location("publish_file_snapshot", _SCRIPT)
assert _spec and _spec.loader
_publisher = importlib.util.module_from_spec(_spec)
sys.modules["publish_file_snapshot"] = _publisher
_spec.loader.exec_module(_publisher)

prefix_disagrees = _publisher.prefix_disagrees


def test_the_real_incident_is_refused() -> None:
    """TENANT_B's index published from a TENANT_A-configured shell."""
    assert prefix_disagrees("TENANT_B/", "TENANT_A/") is True


def test_the_reverse_is_refused_too() -> None:
    assert prefix_disagrees("TENANT_A/", "TENANT_B/") is True


@pytest.mark.parametrize(
    ("indexed", "configured"),
    [
        ("TENANT_B/", "TENANT_B/"),
        ("TENANT_B/", "TENANT_B"),
        ("TENANT_B", "TENANT_B/"),
        ("TENANT_B/", "/TENANT_B/"),
        ("TENANT_B/", "  TENANT_B/  "),
    ],
)
def test_matching_prefixes_agree_however_they_are_written(
    indexed: str, configured: str
) -> None:
    """Trailing and leading slashes are formatting, not a different location."""
    assert prefix_disagrees(indexed, configured) is False


@pytest.mark.parametrize("configured", ["", "   "])
def test_saying_nothing_is_not_a_disagreement(configured: str) -> None:
    """Nothing configured means the index's own value stands unopposed."""
    assert prefix_disagrees("TENANT_B/", configured) is False


def test_a_similar_prefix_is_still_a_disagreement() -> None:
    """'TENANT_B' and 'TENANT_B-archive' are different buckets' worth of documents."""
    assert prefix_disagrees("TENANT_B/", "TENANT_B-archive/") is True
