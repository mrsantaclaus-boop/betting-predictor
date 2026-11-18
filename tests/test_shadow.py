# Shadow trial (18/11 -> 23/12/2026). DB paths are set in tests/conftest.py
import pytest

from football.models import TeamForm
from predictor import shadow
from predictor.poisson import compute_poisson
from api_server import _shadow_report, app


def test_poisson_defaults_unchanged(avg_team):
    a = compute_poisson(avg_team, avg_team, "SA")
    b = compute_poisson(avg_team, avg_team, "SA", home_adv_factor=1.0, rho_override=None)
    assert (a.home_win_pct, a.draw_pct, a.away_win_pct) == (b.home_win_pct, b.draw_pct, b.away_win_pct)


def test_shadow_fix_raises_home_and_lowers_draw(avg_team):
    base = compute_poisson(avg_team, avg_team, "SA")
    fix = compute_poisson(avg_team, avg_team, "SA", home_adv_factor=shadow.HOME_ADV_FACTOR, rho_override=shadow.RHO)
    assert fix.home_win_pct > base.home_win_pct
    assert fix.draw_pct < base.draw_pct


def test_rho_override_ignored_for_neutral_tournaments(avg_team):
    a = compute_poisson(avg_team, avg_team, "WC", is_neutral=True)
    b = compute_poisson(avg_team, avg_team, "WC", is_neutral=True, home_adv_factor=1.24, rho_override=0.05)
    assert a.draw_pct == b.draw_pct and a.home_win_pct == b.home_win_pct


def test_blend_and_calibrate_sum_to_100():
    b = shadow.blend_1x2((50.0, 25.0, 25.0), (40.0, 30.0, 30.0), 0.4)
    assert b == (44.0, 28.0, 28.0)
    assert shadow.blend_1x2(None, (40.0, 30.0, 30.0), 0.4) == (40.0, 30.0, 30.0)
    c = shadow.calibrate((40.0, 30.0, 30.0))
    assert sum(c) == pytest.approx(100, abs=0.2) and c[1] < 30.0


def test_form_snapshot():
    f = TeamForm(team_name="X", results=["W", "D", "L", "W", "W", "W"], goals_scored=[2, 1, 0, 3, 1, 2], goals_conceded=[0, 1, 2, 1, 1, 0])
    s = shadow.form_snapshot(f)
    assert s["n"] == 5 and s["results"] == "DLWWW" and s["points"] == 10 and s["goals_for"] == 7


def _p(i, prod, var, out):
    return {"created_at": f"2026-11-{20 + i % 9:02d}T10:00:00+00:00",
            "home_win_pct": prod[0], "draw_pct": prod[1], "away_win_pct": prod[2],
            "outcomes": {"home_win": out == 0, "draw": out == 1, "away_win": out == 2},
            "shadow": {"version": "x", "variants": {"fix": {"blend": var}}}}


def test_shadow_report_detects_a_better_variant():
    # home always wins; the variant gives home 60% instead of 40%
    preds = [_p(i, (40, 30, 30), [60, 20, 20], 0) for i in range(200)]
    r = _shadow_report(preds, n_boot=200)
    v = r["variants"]["fix"]
    assert r["settled"] == 200 and v["gain_vs_production"] > 0.01 and v["gain_low"] > 0
    assert v["passes"] is True
    assert r["happened"] == [100.0, 0.0, 0.0]


def test_shadow_report_needs_enough_matches_and_skips_pre_trial():
    preds = [_p(i, (40, 30, 30), [60, 20, 20], 0) for i in range(50)]
    preds.append({**_p(0, (40, 30, 30), [60, 20, 20], 0), "created_at": "2026-11-01T10:00:00+00:00"})
    r = _shadow_report(preds, n_boot=100)
    assert r["settled"] == 50 and r["variants"]["fix"]["passes"] is False


def test_shadow_report_endpoint(monkeypatch):
    import api_server
    monkeypatch.setattr(api_server, "_load_resolved_preds_full", lambda: [])
    data = app.test_client().get("/api/shadow-report").get_json()
    assert data["settled"] == 0 and data["trial_end"] == "2026-12-23"


def test_orchestrator_shadow_variants_end_to_end(avg_team):
    """_shadow_variants on a stub report: three variants, form snapshot, prediction untouched."""
    from types import SimpleNamespace
    from predictor.orchestrator import BettingOrchestrator
    from predictor.result_parser import BettingPrediction

    form = TeamForm(team_name="Test FC", results=["W"] * 5, goals_scored=[3] * 5, goals_conceded=[0] * 5)
    report = SimpleNamespace(home_stats=avg_team.model_copy(deep=True), away_stats=avg_team.model_copy(deep=True),
                             home_form=form, away_form=form, head_to_head=None)
    fixture = SimpleNamespace(competition_code="SA", is_neutral=False)
    pred = BettingPrediction()
    pred.llm_home_win_pct, pred.llm_draw_pct, pred.llm_away_win_pct = 45.0, 25.0, 30.0
    orch = BettingOrchestrator.__new__(BettingOrchestrator)  # no network clients needed
    s = orch._shadow_variants(fixture, report, pred, avg_team.model_copy(deep=True), avg_team.model_copy(deep=True))
    assert set(s["variants"]) == {"fix", "fix_cal", "fix_form45"}
    assert s["form"]["home"]["points"] == 15
    for v in s["variants"].values():
        assert sum(v["blend"]) == pytest.approx(100, abs=0.3)
    assert pred.home_win_pct == 0.0  # production fields not touched
    assert "shadow" in {**pred.to_dict(), "shadow": s}
