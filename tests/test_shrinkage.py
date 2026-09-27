from predictor.shrinkage import shrink, apply_shrinkage, SHRINKAGE_WEIGHT, LEAGUE_AVERAGES
from football.models import TeamStats
import pytest


def test_shrink_no_games_returns_prior():
    assert shrink(actual=3.0, league_avg=1.35, games_played=0) == 1.35


def test_shrink_many_games_stays_close_to_actual():
    result = shrink(actual=2.0, league_avg=1.35, games_played=30)
    # (8*1.35 + 30*2.0) / 38 = 70.8/38 ≈ 1.863
    assert abs(result - 1.863) < 0.01


def test_shrink_few_games_pulled_toward_prior():
    result = shrink(actual=3.0, league_avg=1.35, games_played=4)
    # (8*1.35 + 4*3.0) / 12 = 22.8/12 = 1.9
    assert abs(result - 1.9) < 0.01
    assert 1.35 < result < 3.0


def test_apply_shrinkage_reduces_inflated_stat():
    team = TeamStats(
        team_name="Early FC", competition="SA", games_played=4,
        goals_scored_pg=3.0, goals_conceded_pg=0.5,
        xg_pg=2.8, xga_pg=0.6, corners_pg=8.0,
        yellow_cards_pg=4.0, red_cards_pg=0.5,
    )
    original_goals = team.goals_scored_pg
    apply_shrinkage(team, "SA")
    assert team.goals_scored_pg < original_goals
    assert team.goals_scored_pg > 1.35  # not fully at prior


def test_apply_shrinkage_noop_when_at_league_avg():
    # Build the team from the SA prior itself, so the test follows the
    # constants in predictor/shrinkage.py instead of hard-coding them.
    sa = LEAGUE_AVERAGES["SA"]
    team = TeamStats(team_name="Avg FC", competition="SA", games_played=20, **sa)
    apply_shrinkage(team, "SA")
    # When actual == league_avg, shrinkage changes nothing
    for field, prior in sa.items():
        assert abs(getattr(team, field) - prior) < 0.01, field
