"""
predictor/shadow.py — shadow trial of a corrected model (18/11/2026 → 23/12/2026).

Production predictions are unchanged. Every new prediction also stores, under
`shadow`, the 1X2 probabilities of three candidate variants plus a snapshot of
both teams' recent form, so that after 5 weeks /api/shadow-report can compare
them with production on matches none of them was tuned on.

Why: on 442 settled matches the model gave home wins 39% (48.4% happened) and
draws 27% (19.2% happened). Poisson and the MiroFish LLM share the bias
independently (the LLM never sees the Poisson numbers), so it comes from the
shared inputs. Parameters below were fitted on the first half of the settled
matches (chronological) and checked on the second half:
    blend Brier 1X2 0.5771 -> 0.5629 (fix), 0.5617 (fix + calibration).

Variants (the MiroFish LLM probabilities are reused, never re-run):
    fix          Poisson with home advantage x HOME_ADV_FACTOR and Dixon-Coles
                 rho = RHO, blended with the same LLM output at the same weight
    fix_cal      fix, then fixed per-outcome multipliers CALIBRATION
    fix_form45   fix, with recent-form weight FORM_WEIGHT_ALT instead of 0.30
Decision rule (agreed with Mimmo and Fedor): adopt a variant only if, on at
least 150 settled matches predicted during the trial, its Brier is lower than
production by at least 0.01 with a paired bootstrap interval excluding zero,
and the per-outcome calibration does not get worse.
"""
from __future__ import annotations

SHADOW_VERSION = "2026-11-18"
TRIAL_START = "2026-11-18"
TRIAL_END = "2026-12-23"

HOME_ADV_FACTOR = 1.24          # e.g. Serie A 1.08 -> 1.34
RHO = 0.05                      # was -0.13 (fewer extra low-score draws)
CALIBRATION = (1.01, 0.89, 1.08)  # home, draw, away multipliers (fitted 18/11, frozen)
FORM_WEIGHT_ALT = 0.45          # production uses 0.30
MIN_SETTLED = 150
MIN_GAIN = 0.01


def blend_1x2(llm: tuple[float, float, float] | None, poisson: tuple[float, float, float],
              llm_weight: float) -> tuple[float, float, float]:
    """Same rule as production: weighted average when the LLM has values, else Poisson."""
    if not llm or (llm[0] <= 0 and llm[1] <= 0):
        return tuple(round(x, 1) for x in poisson)
    h = llm[0] * llm_weight + poisson[0] * (1 - llm_weight)
    d = llm[1] * llm_weight + poisson[1] * (1 - llm_weight)
    return (round(h, 1), round(d, 1), round(100 - h - d, 1))


def calibrate(p: tuple[float, float, float]) -> tuple[float, float, float]:
    w = [p[i] * CALIBRATION[i] for i in range(3)]
    s = sum(w) or 1.0
    return tuple(round(x / s * 100, 1) for x in w)


def form_snapshot(form) -> dict:
    """Compact record of the recent form the model saw (for the post-trial analysis)."""
    res = list(getattr(form, "results", []) or [])[-5:]
    gs = list(getattr(form, "goals_scored", []) or [])[-5:]
    gc = list(getattr(form, "goals_conceded", []) or [])[-5:]
    pts = {"W": 3, "D": 1, "L": 0}
    return {"n": len(gs), "results": "".join(res), "points": sum(pts.get(r, 0) for r in res),
            "goals_for": sum(gs), "goals_against": sum(gc),
            "home_record": getattr(form, "home_record", "") or "",
            "away_record": getattr(form, "away_record", "") or ""}
