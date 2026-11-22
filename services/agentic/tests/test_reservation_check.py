"""A supplied ReservationId is checked before anything is spent.

SpecSafe reserves the credits and sends the id with the request. Until now the
agent took that id on trust and only discovered at Settle -- after the Bedrock
tokens were spent -- that it had already been committed, released or expired.
There is nobody to bill that work to. One read beforehand costs far less.

The states come from the Licensing guide: proceed only on Reserved; Committed,
Released and Expired are terminal; and a row past ExpiresAt can still read
Reserved until cleanup catches up, so it is treated as stale rather than
trusted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from rag_agentic.config import AgenticSettings
from rag_agentic.credit_gate import CreditGate
from rag_agentic.licensing import (
    LicensingDeclined,
    LicensingUnavailable,
    ReservationCheck,
    read_reservation,
)
from rag_agentic.models import AgentRequest

from .test_credit_gate import FakeLicensing

SOON = (datetime.now(UTC) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
PAST = (datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")


def _settings(**over: Any) -> AgenticSettings:
    return AgenticSettings(
        licensing_enabled=True, licensing_base_url="https://tenantadmin.example.com",
        licensing_tenant_codes="TENANT_A=TENANT_A,TENANT_B=TENANT_B", licensing_settle_only=True, **over,
    )


def _request(reservation_id: str = "res-1") -> AgentRequest:
    return AgentRequest(base_url="https://tenant-a.demo.com/", query="how many documents?",
                        reservation_id=reservation_id)


# ---- reading the response ----------------------------------------------------------


def test_a_live_reservation_is_usable() -> None:
    check = read_reservation({"_id": "res-1", "Status": "Reserved", "ExpiresAt": SOON,
                              "EstimatedCredits": 5, "FeatureCode": "DOCUMENT_SEARCH"})

    assert check.usable
    assert check.reason == ""
    assert check.estimated_credits == 5


@pytest.mark.parametrize("status", ["Committed", "Released", "Expired"])
def test_terminal_states_are_not_usable(status: str) -> None:
    check = read_reservation({"Status": status})

    assert not check.usable
    assert check.reason == status.lower()


def test_reserved_but_past_expiry_is_stale() -> None:
    """Cleanup lags, so Reserved alone is not proof it is still live."""
    check = read_reservation({"Status": "Reserved", "ExpiresAt": PAST})

    assert not check.usable
    assert check.reason == "stale"


def test_an_unknown_status_is_refused_rather_than_assumed_good() -> None:
    assert read_reservation({"Status": "Pending"}).reason == "unknown_status"
    assert read_reservation({}).reason == "unknown_status"


def test_an_unreadable_expiry_does_not_fail_a_reserved_row() -> None:
    """Status is authoritative; a format change must not become an outage."""
    check = read_reservation({"Status": "Reserved", "ExpiresAt": "not a date"})

    assert check.usable


def test_every_refusal_carries_a_message() -> None:
    for status in ("Committed", "Released", "Expired"):
        assert read_reservation({"Status": status}).message


# ---- the gate ----------------------------------------------------------------------


async def test_a_valid_reservation_is_looked_up_then_accepted() -> None:
    licensing = FakeLicensing()
    gate = CreditGate(licensing, _settings())

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert hold is not None
    assert hold.reservation_id == "res-1"
    [lookup] = licensing.lookups
    assert lookup["reservation_id"] == "res-1"
    assert lookup["tenant_code"] == "TENANT_A"


@pytest.mark.parametrize("reason", ["committed", "released", "expired", "stale", "not_found"])
async def test_an_unusable_reservation_stops_the_run(reason: str) -> None:
    """Before any tokens are spent -- that is the whole point of the check."""
    licensing = FakeLicensing(lookup=ReservationCheck(id="res-1", usable=False, reason=reason))
    gate = CreditGate(licensing, _settings())

    with pytest.raises(LicensingDeclined) as caught:
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert caught.value.reason == reason
    assert licensing.reserves == [], "a settle-only run must not reserve for itself"


async def test_a_mismatched_feature_is_logged_but_still_runs(capsys: Any) -> None:
    """Billable, just booked against another capability -- not worth losing an answer."""
    licensing = FakeLicensing(lookup=ReservationCheck(
        id="res-1", status="Reserved", usable=True, feature_code="WIKI"))
    gate = CreditGate(licensing, _settings(licensing_feature_code="DOCUMENT_SEARCH"))

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert hold is not None
    assert "credit_reservation_feature_mismatch" in capsys.readouterr().out


# ---- when Licensing is down --------------------------------------------------------


async def test_an_outage_refuses_by_default() -> None:
    """A gate that fails open is not a gate; the existing setting decides."""
    licensing = FakeLicensing(lookup_raises=LicensingUnavailable("connect timeout"))
    gate = CreditGate(licensing, _settings())

    with pytest.raises(LicensingUnavailable):
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")


async def test_an_outage_is_waved_through_when_the_operator_chose_availability() -> None:
    licensing = FakeLicensing(lookup_raises=LicensingUnavailable("connect timeout"))
    gate = CreditGate(licensing, _settings(licensing_fail_open=True))

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert hold is not None, "fail-open should still run, and still settle later"
    assert hold.reservation_id == "res-1"


async def test_a_bad_id_is_refused_even_when_failing_open() -> None:
    """not_found is an answer, not an outage: fail-open must not rescue it."""
    licensing = FakeLicensing(lookup=ReservationCheck(usable=False, reason="not_found"))
    gate = CreditGate(licensing, _settings(licensing_fail_open=True))

    with pytest.raises(LicensingDeclined):
        await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")


async def test_the_check_can_be_turned_off() -> None:
    licensing = FakeLicensing(lookup=ReservationCheck(usable=False, reason="expired"))
    gate = CreditGate(licensing, _settings(licensing_verify_reservation=False))

    hold = await gate.reserve(_request(), tenant="TENANT_A", trace_id="t")

    assert hold is not None
    assert licensing.lookups == []
