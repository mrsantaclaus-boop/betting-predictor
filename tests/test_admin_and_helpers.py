from datetime import datetime, timezone, timedelta

import pytest

from api_server import app, _kickoff_of, _has_kicked_off


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_write_endpoint_disabled_without_admin_token(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    r = client.post("/api/cache/clear")
    assert r.status_code == 503
    assert "ADMIN_TOKEN" in r.get_json()["error"]


def test_write_endpoint_rejects_wrong_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/api/cache/clear").status_code == 401
    assert client.post("/api/cache/clear", headers={"X-Admin-Token": "nope"}).status_code == 401


def test_write_endpoint_accepts_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    r = client.post("/api/cache/clear", headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200
    assert "deleted" in r.get_json()


def test_protected_routes_all_require_token(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/api/results/backfill-stats").status_code == 401
    assert client.delete("/api/predictions/unplayed").status_code == 401
    assert client.get("/api/odds/sports").status_code == 401
    assert client.get("/api/odds/probe/SA/123?markets=btts").status_code == 401


def test_read_endpoints_stay_public(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.get("/api/predictions").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").get_json()["admin_configured"] is False


def test_kickoff_from_match_date():
    p = {"match_date": "2026-10-10T13:00:00+00:00"}
    assert _kickoff_of(p) == datetime(2026, 10, 10, 13, 0, tzinfo=timezone.utc)


def test_kickoff_falls_back_to_odds_commence_time():
    p = {"live_odds": {"commence_time": "2026-07-19T19:00:00Z"}}
    assert _kickoff_of(p) == datetime(2026, 7, 19, 19, 0, tzinfo=timezone.utc)


def test_kickoff_falls_back_to_label_date():
    p = {"match": "AC Milan vs US Lecce — Serie A 20/09/2026"}
    ko = _kickoff_of(p)
    assert (ko.year, ko.month, ko.day) == (2026, 9, 20)


def test_kickoff_unknown():
    assert _kickoff_of({}) is None
    assert _has_kicked_off({}) is False


def test_has_kicked_off():
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    assert _has_kicked_off({"match_date": past}) is True
    assert _has_kicked_off({"match_date": future}) is False


def test_model_calibration_rejects_bad_since(client):
    assert client.get("/api/model-calibration?since=not-a-date").status_code == 400


def test_model_calibration_since_ok(client):
    r = client.get("/api/model-calibration?since=2026-09-28")
    assert r.status_code == 200
    assert r.get_json()["_since"] == "2026-09-28"
