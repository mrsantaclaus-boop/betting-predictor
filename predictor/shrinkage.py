"""
predictor/shrinkage.py — Bayesian shrinkage for small-sample team stats.

Pulls per-game statistics toward the league average when a team has played
few games. Formula:
    adjusted = (W * league_avg + gp * actual) / (W + gp)
where W = SHRINKAGE_WEIGHT (virtual games from prior).

After W games the stat is 50% real data, 50% prior.
After 20 games it is ~71% real data.
"""

from __future__ import annotations
from football.models import TeamStats

SHRINKAGE_WEIGHT = 8  # equivalent to 8 games worth of prior evidence

# League-average per-game stats used as the Bayesian prior
LEAGUE_AVERAGES: dict[str, dict[str, float]] = {
    "SA":    {"goals_scored_pg": 1.35, "goals_conceded_pg": 1.35, "xg_pg": 1.35,
              "xga_pg": 1.35, "corners_pg": 5.1, "yellow_cards_pg": 2.2, "red_cards_pg": 0.08},
    "SB":    {"goals_scored_pg": 1.25, "goals_conceded_pg": 1.25, "xg_pg": 1.25,
              "xga_pg": 1.25, "corners_pg": 5.1, "yellow_cards_pg": 2.4, "red_cards_pg": 0.09},
    "CL":    {"goals_scored_pg": 1.40, "goals_conceded_pg": 1.40, "xg_pg": 1.40,
              "xga_pg": 1.40, "corners_pg": 4.8, "yellow_cards_pg": 1.8, "red_cards_pg": 0.06},
    "EL":    {"goals_scored_pg": 1.38, "goals_conceded_pg": 1.38, "xg_pg": 1.38,
              "xga_pg": 1.38, "corners_pg": 4.9, "yellow_cards_pg": 1.9, "red_cards_pg": 0.065},
    "USC":   {"goals_scored_pg": 1.42, "goals_conceded_pg": 1.42, "xg_pg": 1.42,
              "xga_pg": 1.42, "corners_pg": 5.2, "yellow_cards_pg": 1.7, "red_cards_pg": 0.05},
    "ECL":   {"goals_scored_pg": 1.30, "goals_conceded_pg": 1.30, "xg_pg": 1.30,
              "xga_pg": 1.30, "corners_pg": 4.6, "yellow_cards_pg": 2.0, "red_cards_pg": 0.07},
    "WC":    {"goals_scored_pg": 1.20, "goals_conceded_pg": 1.20, "xg_pg": 1.20,
              "xga_pg": 1.20, "corners_pg": 4.4, "yellow_cards_pg": 2.0, "red_cards_pg": 0.08},
    "WCQE":  {"goals_scored_pg": 1.25, "goals_conceded_pg": 1.25, "xg_pg": 1.25,
              "xga_pg": 1.25, "corners_pg": 4.5, "yellow_cards_pg": 2.3, "red_cards_pg": 0.09},
    "WCQA":  {"goals_scored_pg": 1.30, "goals_conceded_pg": 1.30, "xg_pg": 1.30,
              "xga_pg": 1.30, "corners_pg": 4.3, "yellow_cards_pg": 2.4, "red_cards_pg": 0.10},
    "WCQC":  {"goals_scored_pg": 1.20, "goals_conceded_pg": 1.20, "xg_pg": 1.20,
              "xga_pg": 1.20, "corners_pg": 4.2, "yellow_cards_pg": 2.3, "red_cards_pg": 0.09},
    "WCQAS": {"goals_scored_pg": 1.15, "goals_conceded_pg": 1.15, "xg_pg": 1.15,
              "xga_pg": 1.15, "corners_pg": 4.2, "yellow_cards_pg": 2.2, "red_cards_pg": 0.08},
    "WCQAF": {"goals_scored_pg": 1.10, "goals_conceded_pg": 1.10, "xg_pg": 1.10,
              "xga_pg": 1.10, "corners_pg": 4.0, "yellow_cards_pg": 2.5, "red_cards_pg": 0.10},
    "BSA":   {"goals_scored_pg": 1.45, "goals_conceded_pg": 1.45, "xg_pg": 1.45,
              "xga_pg": 1.45, "corners_pg": 5.5, "yellow_cards_pg": 2.3, "red_cards_pg": 0.10},
}
_DEFAULT_AVERAGES = LEAGUE_AVERAGES["SA"]

_SHRINK_FIELDS = [
    "goals_scored_pg", "goals_conceded_pg",
    "xg_pg", "xga_pg",
    "corners_pg", "yellow_cards_pg", "red_cards_pg",
]


def shrink(actual: float, league_avg: float, games_played: int) -> float:
    """Return the shrinkage-adjusted value of a per-game stat."""
    if games_played <= 0:
        return league_avg
    return round(
        (SHRINKAGE_WEIGHT * league_avg + games_played * actual)
        / (SHRINKAGE_WEIGHT + games_played),
        3,
    )


def apply_shrinkage(stats: TeamStats, competition_code: str) -> None:
    """Shrink all key per-game stats in-place toward the league average."""
    avgs = LEAGUE_AVERAGES.get(competition_code, _DEFAULT_AVERAGES)
    gp = stats.games_played
    for field in _SHRINK_FIELDS:
        actual = getattr(stats, field, 0.0)
        avg = avgs.get(field, 0.0)
        if avg > 0:
            setattr(stats, field, shrink(actual, avg, gp))
