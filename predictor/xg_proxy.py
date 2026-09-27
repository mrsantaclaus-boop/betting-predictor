"""
predictor/xg_proxy.py — expected goals estimated from shots on target.

FBref (our xG source) is blocked from Render, and every real-xG API costs
money. ESPN match summaries are free and give shots / shots on target per
team, so we estimate xG from them.

Calibration (2026-09-27, ESPN summaries, Serie A 2025/26 + 2026/27 and
Champions League 2025/26 + 2026/27, n = 1,268 team-match observations):

    goals ~ a*shots + b*shots_on_target -> a = -0.018, b = 0.356 (shots add nothing)
    goals ~ b*shots_on_target           -> b = 0.309, corr 0.61, MAE 0.80
    goals ~ a*shots                     -> a = 0.104, corr 0.35
    constant (league mean)              -> MAE 1.04

So the proxy is simply  xG ≈ 0.31 × shots on target. It is a proxy, not
Opta-style xG: it ignores shot location and quality. It tracks Serie A
scoring almost exactly (1.24 vs 1.24 goals/game) and slightly under-reads
the Champions League (1.60 vs 1.75), which the Poisson league averages
already compensate for.
"""

from __future__ import annotations

SOT_TO_XG = 0.31   # goals per shot on target, see calibration above


def xg_from_shots(shots_on_target_pg: float) -> float:
    """xG per game estimated from shots on target per game (0 when unknown)."""
    if not shots_on_target_pg or shots_on_target_pg <= 0:
        return 0.0
    return round(SOT_TO_XG * shots_on_target_pg, 2)
