"""
predictor/poisson.py — Poisson models for scoreline and corner probabilities.

Uses xG data from FBref to estimate expected goals per team,
then applies independent Poisson distributions to compute the full
scoreline probability matrix (Dixon-Coles style attack/defense adjustment).

Also provides a corner model: total corners ~ Poisson(home_cpg + away_cpg).

No external dependencies beyond the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from football.models import TeamStats, HeadToHead

# Empirical average xG per team per game, per competition
LEAGUE_AVG_XG: dict[str, float] = {
    "SA":    1.35,
    "CL":    1.40,
    "ECL":   1.30,
    "WC":    1.20,
    "WCQE":  1.25,
    "WCQA":  1.30,
    "WCQC":  1.20,
    "WCQAS": 1.15,
    "WCQAF": 1.10,
}
_DEFAULT_AVG_XG = 1.30

# Per-competition home advantage (home attack multiplier).
# Derived from historical home/away goals ratios per competition.
HOME_ADVANTAGE: dict[str, float] = {
    "SA":    1.12,   # Serie A: strong crowd factor
    "CL":    1.06,   # CL: competitive away sides, historically flatter
    "ECL":   1.10,
    "WC":    1.05,   # Often semi-neutral venues
    "WCQE":  1.12,
    "WCQA":  1.15,   # South America: very strong home factor
    "WCQC":  1.10,
    "WCQAS": 1.08,
    "WCQAF": 1.12,
}
_DEFAULT_HOME_ADVANTAGE = 1.10

# Compute grid up to MAX_GOALS × MAX_GOALS
MAX_GOALS = 6

# Dixon-Coles ρ — correction for joint low-score outcomes (D&C 1997)
_DC_RHO = -0.13


# ── Core math ──────────────────────────────────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam). Returns 1.0 at k=0 if lam ≤ 0."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dixon_coles_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    """
    Correction factor τ for the (h, a) joint outcome.
    Only adjusts the four low-score cells: (0,0), (1,0), (0,1), (1,1).
    All other scorelines return 1.0 (unchanged).
    """
    if h == 0 and a == 0:
        return 1.0 - lh * la * rho
    if h == 1 and a == 0:
        return 1.0 + la * rho
    if h == 0 and a == 1:
        return 1.0 + lh * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def _h2h_factors(h2h: HeadToHead | None) -> tuple[float, float]:
    """
    Compute (home_factor, away_factor) from head-to-head history.
    Each factor multiplies the respective lambda.
    Uses a maximum ±10% adjustment, shrunk by sample size.
    Returns (1.0, 1.0) when H2H is unavailable or has fewer than 3 meetings.
    """
    if h2h is None or h2h.total_meetings < 3:
        return 1.0, 1.0
    n = h2h.total_meetings
    home_rate = h2h.home_wins / n
    away_rate = h2h.away_wins / n
    # Weight: reaches full strength at 10 meetings, caps at 10%
    w = min(n / 10, 1.0) * 0.10
    home_f = round(1.0 + (home_rate - 0.45) * w / 0.45, 4)
    away_f = round(1.0 + (away_rate - 0.45) * w / 0.45, 4)
    # Clamp to ±10%
    home_f = max(0.90, min(home_f, 1.10))
    away_f = max(0.90, min(away_f, 1.10))
    return home_f, away_f


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class PoissonResult:
    """Full output of the Poisson scoreline model."""

    lambda_home: float
    lambda_away: float
    # grid[h][a] = P(home scores h AND away scores a)
    grid: list[list[float]] = field(default_factory=list)

    # ── Derived market probabilities ──────────────────────────────────────────

    @property
    def most_likely_scoreline(self) -> str:
        best_h = best_a = 0
        best_p = 0.0
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                if self.grid[h][a] > best_p:
                    best_p = self.grid[h][a]
                    best_h, best_a = h, a
        return f"{best_h}-{best_a}"

    @property
    def home_win_pct(self) -> float:
        return round(sum(
            self.grid[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h > a
        ) * 100, 1)

    @property
    def draw_pct(self) -> float:
        return round(sum(
            self.grid[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h == a
        ) * 100, 1)

    @property
    def away_win_pct(self) -> float:
        return round(100.0 - self.home_win_pct - self.draw_pct, 1)

    @property
    def over_2_5_pct(self) -> float:
        return round(sum(
            self.grid[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h + a > 2
        ) * 100, 1)

    @property
    def under_2_5_pct(self) -> float:
        return round(100.0 - self.over_2_5_pct, 1)

    @property
    def over_3_5_pct(self) -> float:
        return round(sum(
            self.grid[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h + a > 3
        ) * 100, 1)

    @property
    def under_3_5_pct(self) -> float:
        return round(100.0 - self.over_3_5_pct, 1)

    @property
    def btts_yes_pct(self) -> float:
        return round(sum(
            self.grid[h][a]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
            if h > 0 and a > 0
        ) * 100, 1)

    @property
    def btts_no_pct(self) -> float:
        return round(100.0 - self.btts_yes_pct, 1)

    def top_scorelines(self, n: int = 8) -> list[list]:
        """Return the N most probable scorelines as [[score, pct], ...]."""
        items = [
            [f"{h}-{a}", round(self.grid[h][a] * 100, 1)]
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    def asian_handicap(self, line: float) -> dict:
        """
        Compute Asian Handicap probabilities for a given home team handicap line.

        Positive line  (+1.0): home RECEIVES goals — underdog side.
        Negative line  (-1.5): home GIVES goals   — favourite side.
        Zero line      (0.0):  Draw No Bet.

        For integer lines a push is possible (e.g. home wins by exactly |line|).
        For half-ball lines (±0.5, ±1.5 …) there is no push.

        Returns {"home_win_pct", "push_pct", "away_win_pct"}.
        """
        home_win = push = away_win = 0.0
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                p = self.grid[h][a]
                adj = (h + line) - a      # adjusted margin (home perspective)
                if adj > 1e-9:
                    home_win += p
                elif abs(adj) < 1e-9:     # integer line: exact push
                    push += p
                else:
                    away_win += p
        return {
            "home_win_pct": round(home_win * 100, 1),
            "push_pct":     round(push     * 100, 1),
            "away_win_pct": round(away_win * 100, 1),
        }

    @property
    def top_ah_lines(self) -> list[dict]:
        """
        Return AH probabilities for the 5 most relevant lines given the
        expected goal spread (lambda_home − lambda_away).

        Lines are chosen so that the mid-line is near the "fair" handicap,
        giving the user context around both sides of the market.
        """
        spread = self.lambda_home - self.lambda_away
        # Pick centre line: round to nearest 0.5
        centre = -round(spread * 2) / 2          # e.g. spread=1.3 → centre=-1.5
        candidates = [centre + 0.5 * i for i in range(-2, 3)]  # 5 lines

        _LABEL = {
            -3.0: "AH -3",  -2.5: "AH -2.5", -2.0: "AH -2",  -1.5: "AH -1.5",
            -1.0: "AH -1",  -0.5: "AH -0.5",  0.0: "DNB",
             0.5: "AH +0.5", 1.0: "AH +1",    1.5: "AH +1.5",  2.0: "AH +2",
             2.5: "AH +2.5", 3.0: "AH +3",
        }
        result = []
        for line in candidates:
            ah = self.asian_handicap(line)
            result.append({
                "line":  line,
                "label": _LABEL.get(line, f"AH {line:+.1f}"),
                **ah,
            })
        return result


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_poisson(
    home_stats: TeamStats,
    away_stats: TeamStats,
    competition_code: str = "",
    head_to_head: HeadToHead | None = None,
    is_neutral: bool = False,
) -> PoissonResult:
    """
    Compute the Poisson scoreline probability grid.

    Attack/defense adjustment (Dixon-Coles style):
      λ_home = (home_attack / avg) × (away_defense / avg) × avg × home_advantage
      λ_away = (away_attack / avg) × (home_defense / avg) × avg

    Falls back from xG → goals_scored if xG data is unavailable.
    H2H history applies a ±10% max adjustment to both lambdas.
    is_neutral=True forces home_advantage=1.0 (WC, final-stage matches).
    """
    avg = LEAGUE_AVG_XG.get(competition_code, _DEFAULT_AVG_XG)

    def best(xg: float, goals: float) -> float:
        return xg if xg > 0 else (goals if goals > 0 else avg)

    home_att = best(home_stats.xg_pg,  home_stats.goals_scored_pg)
    away_att = best(away_stats.xg_pg,  away_stats.goals_scored_pg)
    home_def = best(home_stats.xga_pg, home_stats.goals_conceded_pg)
    away_def = best(away_stats.xga_pg, away_stats.goals_conceded_pg)

    ha = 1.0 if is_neutral else HOME_ADVANTAGE.get(competition_code, _DEFAULT_HOME_ADVANTAGE)
    lambda_home = (home_att / avg) * (away_def / avg) * avg * ha
    lambda_away = (away_att / avg) * (home_def / avg) * avg

    # H2H adjustment (±10% max, shrunk toward 1.0 for small samples)
    h2h_home_f, h2h_away_f = _h2h_factors(head_to_head)
    lambda_home *= h2h_home_f
    lambda_away *= h2h_away_f

    # Clamp to realistic range
    lambda_home = max(0.20, min(lambda_home, 5.0))
    lambda_away = max(0.20, min(lambda_away, 5.0))

    # Raw independent Poisson grid
    grid = [
        [_poisson_pmf(h, lambda_home) * _poisson_pmf(a, lambda_away)
         * _dixon_coles_tau(h, a, lambda_home, lambda_away, _DC_RHO)
         for a in range(MAX_GOALS + 1)]
        for h in range(MAX_GOALS + 1)
    ]

    # Renormalize so probabilities sum to 1 (DC correction perturbs the total)
    total = sum(p for row in grid for p in row)
    if total > 0:
        grid = [[p / total for p in row] for row in grid]

    return PoissonResult(
        lambda_home=round(lambda_home, 2),
        lambda_away=round(lambda_away, 2),
        grid=grid,
    )


# ── Corner model ───────────────────────────────────────────────────────────────

# Empirical average total corners per game, per competition
LEAGUE_AVG_CORNERS: dict[str, float] = {
    "SA":    10.5,
    "CL":    10.0,
    "ECL":    9.5,
    "WC":     9.0,
    "WCQE":   9.5,
    "WCQA":   9.0,
    "WCQC":   9.0,
    "WCQAS":  9.0,
    "WCQAF":  8.5,
}
_DEFAULT_AVG_CORNERS = 10.0

@dataclass
class CornerResult:
    lambda_corners: float
    over_9_5_corners_pct: float
    under_9_5_corners_pct: float


def compute_corner_poisson(
    home_corners_pg: float,
    away_corners_pg: float,
    competition_code: str = "",
) -> CornerResult | None:
    """
    Estimate over/under 9.5 corners using a Poisson model on total corners.

    λ = home_corners_pg + away_corners_pg, adjusted toward the league average
    when per-team data is sparse (either value is zero → fall back to average).

    Returns None if no usable data is available.
    """
    if home_corners_pg <= 0 and away_corners_pg <= 0:
        return None

    avg = LEAGUE_AVG_CORNERS.get(competition_code, _DEFAULT_AVG_CORNERS)

    # If only one side has data, substitute the other with half the league avg
    h = home_corners_pg if home_corners_pg > 0 else avg / 2
    a = away_corners_pg if away_corners_pg > 0 else avg / 2

    lambda_total = max(0.5, h + a)

    # P(total corners ≤ 9) = sum_{k=0}^{9} Poisson(k | lambda_total)
    under_9_5 = sum(_poisson_pmf(k, lambda_total) for k in range(10))
    over_9_5  = 1.0 - under_9_5

    return CornerResult(
        lambda_corners=round(lambda_total, 2),
        over_9_5_corners_pct=round(over_9_5 * 100, 1),
        under_9_5_corners_pct=round(under_9_5 * 100, 1),
    )


# ── Cards model ────────────────────────────────────────────────────────────────

# Empirical average total yellow cards per game, per competition
LEAGUE_AVG_CARDS: dict[str, dict[str, float]] = {
    "SA":    {"yellow": 4.4, "red": 0.16},
    "CL":    {"yellow": 3.6, "red": 0.12},
    "ECL":   {"yellow": 4.0, "red": 0.14},
    "WC":    {"yellow": 4.0, "red": 0.16},
    "WCQE":  {"yellow": 4.6, "red": 0.18},
    "WCQA":  {"yellow": 4.8, "red": 0.20},
    "WCQC":  {"yellow": 4.6, "red": 0.18},
    "WCQAS": {"yellow": 4.4, "red": 0.16},
    "WCQAF": {"yellow": 5.0, "red": 0.20},
}
_DEFAULT_AVG_CARDS = {"yellow": 4.4, "red": 0.16}


@dataclass
class CardsResult:
    lambda_yellow: float
    lambda_red: float
    over_3_5_yellow_pct: float
    under_3_5_yellow_pct: float
    over_4_5_yellow_pct: float
    under_4_5_yellow_pct: float


def compute_cards_poisson(
    home_yellow_pg: float,
    away_yellow_pg: float,
    home_red_pg: float,
    away_red_pg: float,
    competition_code: str = "",
    referee_factor: float = 1.0,
) -> CardsResult | None:
    """
    Estimate card markets using independent Poisson models.

    λ_yellow = (home_yellow_pg + away_yellow_pg) × referee_factor
    referee_factor = ref_yellow_pg / league_avg_yellow_pg (1.0 if unknown).

    Returns None if no usable data is available.
    """
    if home_yellow_pg <= 0 and away_yellow_pg <= 0:
        return None

    avgs = LEAGUE_AVG_CARDS.get(competition_code, _DEFAULT_AVG_CARDS)

    hy = home_yellow_pg if home_yellow_pg > 0 else avgs["yellow"] / 2
    ay = away_yellow_pg if away_yellow_pg > 0 else avgs["yellow"] / 2
    hr = home_red_pg if home_red_pg > 0 else avgs["red"] / 2
    ar = away_red_pg if away_red_pg > 0 else avgs["red"] / 2

    referee_factor = max(0.5, min(referee_factor, 2.0))  # clamp to realistic range
    lambda_yellow = max(0.5, (hy + ay) * referee_factor)
    lambda_red    = max(0.01, hr + ar)

    under_3_5_y = sum(_poisson_pmf(k, lambda_yellow) for k in range(4))
    over_3_5_y  = 1.0 - under_3_5_y

    under_4_5_y = sum(_poisson_pmf(k, lambda_yellow) for k in range(5))
    over_4_5_y  = 1.0 - under_4_5_y

    return CardsResult(
        lambda_yellow=round(lambda_yellow, 2),
        lambda_red=round(lambda_red, 2),
        over_3_5_yellow_pct=round(over_3_5_y * 100, 1),
        under_3_5_yellow_pct=round(under_3_5_y * 100, 1),
        over_4_5_yellow_pct=round(over_4_5_y * 100, 1),
        under_4_5_yellow_pct=round(under_4_5_y * 100, 1),
    )
