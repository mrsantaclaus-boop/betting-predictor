import pytest
from football.models import TeamStats
from predictor.poisson import (
    compute_poisson,
    compute_corner_poisson,
    compute_cards_poisson,
    _poisson_pmf,
    _dixon_coles_tau,
    MAX_GOALS,
)


def _make_stats(xg: float, xga: float, **kwargs) -> TeamStats:
    return TeamStats(team_name="Test FC", competition="SA",
                     xg_pg=xg, xga_pg=xga, **kwargs)


# ── Poisson PMF ────────────────────────────────────────────────────────────────

def test_poisson_pmf_zero_lambda():
    assert _poisson_pmf(0, 0.0) == 1.0
    assert _poisson_pmf(1, 0.0) == 0.0


def test_poisson_pmf_known_value():
    import math
    lam = 1.5
    assert abs(_poisson_pmf(2, lam) - (math.exp(-lam) * lam**2 / 2)) < 1e-10


# ── Dixon-Coles tau ────────────────────────────────────────────────────────────

def test_dc_tau_high_scores_unchanged():
    assert _dixon_coles_tau(3, 2, 1.5, 1.2, -0.13) == 1.0


def test_dc_tau_0_0_increases_probability():
    # rho < 0 means 0-0 is MORE common than pure Poisson predicts.
    # tau = 1 - lh * la * rho = 1 - 1.5*1.2*(-0.13) = 1 + 0.234 = 1.234 > 1
    tau = _dixon_coles_tau(0, 0, 1.5, 1.2, -0.13)
    assert tau > 1.0
    assert abs(tau - (1.0 - 1.5 * 1.2 * (-0.13))) < 1e-10


# ── compute_poisson ────────────────────────────────────────────────────────────

def test_grid_sums_to_one():
    home = _make_stats(xg=1.4, xga=1.1)
    away = _make_stats(xg=1.2, xga=1.3)
    result = compute_poisson(home, away, "SA")
    total = sum(p for row in result.grid for p in row)
    assert abs(total - 1.0) < 1e-6


def test_home_advantage_sa_higher_than_cl():
    home = _make_stats(xg=1.4, xga=1.1)
    away = _make_stats(xg=1.2, xga=1.3)
    sa_res = compute_poisson(home, away, "SA")
    cl_res = compute_poisson(home, away, "CL")
    assert sa_res.lambda_home > cl_res.lambda_home


def test_pcts_sum_to_100():
    home = _make_stats(xg=1.6, xga=1.0)
    away = _make_stats(xg=1.0, xga=1.5)
    r = compute_poisson(home, away, "SA")
    assert abs(r.home_win_pct + r.draw_pct + r.away_win_pct - 100.0) < 0.5
    assert abs(r.over_2_5_pct + r.under_2_5_pct - 100.0) < 0.1
    assert abs(r.btts_yes_pct + r.btts_no_pct - 100.0) < 0.1


def test_top_scorelines_length_and_sorted():
    home = _make_stats(xg=1.4, xga=1.1)
    away = _make_stats(xg=1.2, xga=1.3)
    r = compute_poisson(home, away)
    top = r.top_scorelines(5)
    assert len(top) == 5
    assert top[0][1] >= top[1][1]


def test_fallback_to_goals_when_no_xg():
    home = TeamStats(team_name="H", competition="SA",
                     xg_pg=0.0, xga_pg=0.0,
                     goals_scored_pg=1.5, goals_conceded_pg=1.2)
    away = TeamStats(team_name="A", competition="SA",
                     xg_pg=0.0, xga_pg=0.0,
                     goals_scored_pg=1.0, goals_conceded_pg=1.4)
    r = compute_poisson(home, away, "SA")
    assert r.lambda_home > 0
    assert r.lambda_away > 0


# ── compute_corner_poisson ─────────────────────────────────────────────────────

def test_corner_returns_none_when_no_data():
    assert compute_corner_poisson(0.0, 0.0) is None


def test_corner_over_under_sum_to_100():
    r = compute_corner_poisson(5.5, 5.0, "SA")
    assert abs(r.over_9_5_corners_pct + r.under_9_5_corners_pct - 100.0) < 0.1


def test_corner_fallback_when_one_side_missing():
    r = compute_corner_poisson(0.0, 5.5, "SA")
    assert r is not None
    assert r.lambda_corners > 0


# ── compute_cards_poisson ──────────────────────────────────────────────────────

def test_cards_returns_none_when_no_data():
    assert compute_cards_poisson(0.0, 0.0, 0.0, 0.0) is None


def test_cards_pcts_sum_to_100():
    r = compute_cards_poisson(2.2, 2.2, 0.08, 0.08, "SA")
    assert abs(r.over_3_5_yellow_pct + r.under_3_5_yellow_pct - 100.0) < 0.1
    assert abs(r.over_4_5_yellow_pct + r.under_4_5_yellow_pct - 100.0) < 0.1


def test_cards_over_4_5_less_than_over_3_5():
    r = compute_cards_poisson(2.2, 2.2, 0.08, 0.08, "SA")
    assert r.over_4_5_yellow_pct < r.over_3_5_yellow_pct
