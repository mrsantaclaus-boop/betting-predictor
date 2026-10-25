"""Tests for The Odds API credit tracking, circuit breaker and disabled competitions."""
import os
from datetime import datetime, timezone

import pytest

from data import odds_quota
from data.odds_api import OddsAPIClient


NOW = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    s = odds_quota.SqliteStore(str(tmp_path / "quota.db"))
    odds_quota.set_store(s)
    yield s
    odds_quota.set_store(None)


def test_next_reset_rolls_to_next_month():
    assert odds_quota.next_reset(NOW, reset_day=1).isoformat() == "2026-11-01"
    assert odds_quota.next_reset(NOW, reset_day=28).isoformat() == "2026-10-28"
    assert odds_quota.next_reset(datetime(2026, 12, 5, tzinfo=timezone.utc), reset_day=1).isoformat() == "2027-01-01"


def test_record_and_summary(store):
    odds_quota.record(remaining=120, used=380, now=NOW)
    s = odds_quota.summary(now=NOW)
    assert s["remaining"] == 120 and s["used"] == 380
    assert s["status"] == "ok"
    assert s["days_to_reset"] == 7
    assert s["daily_allowance"] == 17          # 120 // 7
    assert "BSA" in s["disabled_competitions"]


def test_low_quota_status(store):
    odds_quota.record(remaining=40, used=460, now=NOW)
    assert odds_quota.is_low()
    assert odds_quota.summary(now=NOW)["status"] == "low"


def test_exhausted_blocks_until_reset(store):
    odds_quota.record_exhausted(now=NOW)
    assert odds_quota.is_blocked(now=NOW)
    assert odds_quota.summary(now=NOW)["status"] == "exhausted"
    after_reset = datetime(2026, 11, 1, 0, 5, tzinfo=timezone.utc)
    assert not odds_quota.is_blocked(now=after_reset)


def test_unknown_state_is_not_blocked(store):
    assert not odds_quota.is_blocked(now=NOW)
    assert odds_quota.summary(now=NOW)["status"] == "unknown"


def test_disabled_competition_makes_no_request(store, monkeypatch):
    client = OddsAPIClient(api_key="x")
    monkeypatch.setattr(client.session, "get", lambda *a, **k: pytest.fail("no HTTP call expected"))
    out = client.get_fixture_odds("BSA", "Santos", "Bahia")
    assert out.get("disabled") is True
    assert client.get_all_odds("WCQE").get("disabled") is True


class _Resp:
    def __init__(self, status, payload, headers=None, text=""):
        self.status_code, self._payload, self.headers, self.text = status, payload, headers or {}, text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def test_request_records_quota_headers(store, monkeypatch):
    client = OddsAPIClient(api_key="x")
    monkeypatch.setattr(client.session, "get", lambda *a, **k: _Resp(
        200, [], {"x-requests-remaining": "321", "x-requests-used": "179"}))
    client._request("https://example/odds", {"markets": "h2h"})
    assert odds_quota.summary()["remaining"] == 321


def test_out_of_credits_is_detected_and_blocks_next_calls(store, monkeypatch):
    client = OddsAPIClient(api_key="x")
    body = '{"message":"Usage quota has been reached","error_code":"OUT_OF_USAGE_CREDITS"}'
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return _Resp(401, {}, {}, body)

    monkeypatch.setattr(client.session, "get", fake_get)
    out = client._request("https://example/odds", {"markets": "h2h"})
    assert "quota exhausted" in out["error"].lower()
    assert odds_quota.is_blocked()
    out2 = client._request("https://example/odds", {"markets": "h2h"})
    assert "quota exhausted" in out2["error"].lower()
    assert len(calls) == 1          # second call short-circuited, no HTTP request


def test_btts_skipped_when_quota_low(store, monkeypatch):
    odds_quota.record(remaining=10, used=490)
    client = OddsAPIClient(api_key="x")
    event = {"id": "ev1", "home_team": "Inter", "away_team": "Fiorentina", "bookmakers": []}
    monkeypatch.setattr(client, "_get_odds", lambda sport, markets: [event])
    monkeypatch.setattr(client, "_get_event_odds", lambda *a, **k: pytest.fail("BTTS call should be skipped"))
    out = client.get_fixture_odds("SA", "Inter", "Fiorentina")
    assert out["consensus"]["btts_yes"] is None


def test_quota_endpoint(store):
    import api_server
    odds_quota.record(remaining=200, used=300)
    res = api_server.app.test_client().get("/api/odds/quota")
    assert res.status_code == 200
    data = res.get_json()
    assert data["remaining"] == 200 and data["status"] in ("ok", "low")
