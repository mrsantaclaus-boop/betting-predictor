# Fedor's proposals (18/11/2026): market error only on real odds, flat-stake ledger,
# Wilson intervals, CLV. DB paths for the import-time _init_pred_db() are set in tests/conftest.py
import pytest

from api_server import _market_implied_pct, _simulate_ledger, _wilson, _model_version, app


def _pred(fid, home=60.0, draw=25.0, away=15.0, odds=None, outcomes=None, code="SA", closing=None):
    p = {
        "fixture_id": fid, "match_label": f"Home{fid} vs Away{fid}", "competition_code": code,
        "created_at": f"2026-11-{fid:02d}T10:00:00+00:00",
        "home_win_pct": home, "draw_pct": draw, "away_win_pct": away,
        "over_2_5_pct": 50.0, "under_2_5_pct": 50.0, "btts_yes_pct": 50.0, "btts_no_pct": 50.0,
        "outcomes": outcomes or {"home_win": True, "draw": False, "away_win": False},
    }
    if odds:
        p["live_odds"] = {"consensus": odds}
    if closing:
        p["closing_odds"] = {"consensus": closing}
    return p


ODDS = {"home": 2.0, "draw": 3.5, "away": 4.0}  # de-vigged home ≈ 48.3%


def test_market_implied_none_without_odds():
    assert _market_implied_pct("home_win", {}) is None
    assert _market_implied_pct("over_9_5_corners", ODDS) is None  # no odds market for corners
    assert _market_implied_pct("home_win", ODDS) == pytest.approx(48.3, abs=0.1)


def test_wilson_interval():
    lo, hi = _wilson(50, 100)
    assert lo == pytest.approx(40.4, abs=0.2) and hi == pytest.approx(59.6, abs=0.2)
    assert _wilson(0, 0) == (None, None)
    lo, hi = _wilson(0, 10)
    assert lo == 0.0 and hi > 0


def test_ledger_flat_stake_profit_and_roi():
    preds = [
        _pred(1, odds=ODDS, outcomes={"home_win": True, "draw": False, "away_win": False}),   # win +1.0
        _pred(2, odds=ODDS, outcomes={"home_win": False, "draw": True, "away_win": False}),   # loss -1.0
        _pred(3, odds=ODDS, outcomes={"home_win": True, "draw": False, "away_win": False}),   # win +1.0
    ]
    out = _simulate_ledger(preds, min_edge=3.0)
    hw = out["by_market"]["home_win"]
    assert hw["bets"] == 3 and hw["wins"] == 2
    assert hw["profit"] == pytest.approx(1.0)
    assert hw["roi_pct"] == pytest.approx(33.3, abs=0.1)
    assert hw["breakeven_pct"] == pytest.approx(50.0)  # 1/2.0
    # draw/away are below the market (25 < 26.6, 15 < 23.3): no bets there
    assert "draw" not in out["by_market"] and "away_win" not in out["by_market"]
    assert out["total"]["bets"] == 3


def test_ledger_ignores_predictions_without_real_odds():
    out = _simulate_ledger([_pred(1), _pred(2, odds={})])
    assert out["total"]["bets"] == 0 and out["total"]["roi_pct"] is None


def test_ledger_respects_min_edge():
    # model 60 vs implied 48.3 -> edge 11.7: in at 3, out at 15
    assert _simulate_ledger([_pred(1, odds=ODDS)], min_edge=15.0)["total"]["bets"] == 0


def test_ledger_closing_line_value():
    out = _simulate_ledger([_pred(1, odds=ODDS, closing={"home": 1.8, "draw": 3.6, "away": 4.4})])
    # took 2.0, closed 1.8 -> +11.1% CLV
    assert out["total"]["clv_n"] == 1
    assert out["total"]["avg_clv_pct"] == pytest.approx(11.11, abs=0.01)
    assert out["recent"][0]["clv_pct"] == pytest.approx(11.11, abs=0.01)


def test_model_version_has_commit_and_knobs():
    v = _model_version()
    assert "commit" in v and "llm" in v and "shrinkage" in v


def test_edge_endpoint_separates_market_and_model(monkeypatch):
    import api_server
    preds = [_pred(i, odds=ODDS if i <= 2 else None) for i in range(1, 5)]
    monkeypatch.setattr(api_server, "_load_resolved_preds", lambda: preds)
    data = app.test_client().get("/api/edge").get_json()
    # market: only the 2 matches with odds
    assert data["home_win"]["sample_size"] == 2
    assert data["home_win"]["implied_pct"] == pytest.approx(48.3, abs=0.1)
    # model: all 4 matches, implied = the model's own 60%
    assert data["_model"]["home_win"]["sample_size"] == 4
    assert data["_model"]["home_win"]["implied_pct"] == pytest.approx(60.0)
    # corners have no odds: no market sample at all
    assert data["over_9_5_corners"]["sample_size"] == 0
    assert data["_meta"]["with_odds"] == 2 and data["_meta"]["settled"] == 4
    assert "significant" in data["home_win"] and "actual_low" in data["home_win"]


def test_ledger_endpoint(monkeypatch):
    import api_server
    monkeypatch.setattr(api_server, "_load_resolved_preds_full", lambda: [_pred(1, odds=ODDS)])
    data = app.test_client().get("/api/ledger?min_edge=3").get_json()
    assert data["total"]["bets"] == 1 and data["total"]["profit"] == pytest.approx(1.0)


def test_closing_snapshot_requires_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret-test")
    r = app.test_client().post("/api/odds/closing-snapshot")
    assert r.status_code == 401
