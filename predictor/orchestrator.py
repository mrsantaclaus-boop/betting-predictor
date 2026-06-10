"""
predictor/orchestrator.py

Main entry point: given a fixture, fetches all data, builds the seed,
runs MiroFish, and returns structured betting predictions.

Can be called:
  - Via HTTP API (see api_server.py)
  - Directly from CLI: python -m predictor.orchestrator --fixture-id 12345
"""

from __future__ import annotations

import math
import os
import sys
import json
import logging
import argparse
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from football import FootballDataClient, FBrefScraper, ApiFootballClient, EspnClient, MatchReport, Fixture
from football.models import TeamStats, InjuryReport
from seed import build_seed_document
from predictor.mirofish_client import MiroFishClient
from predictor.result_parser import ResultParser, BettingPrediction
from predictor.poisson import compute_poisson, compute_corner_poisson, compute_cards_poisson
from predictor.shrinkage import apply_shrinkage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# MiroFish (LLM) runs by default alongside Poisson and results are blended.
# Set USE_LLM=false to disable and run Poisson-only (faster, no API key needed).
_USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

# Blend weights: how much the LLM contributes vs Poisson (0.0 = Poisson only, 1.0 = LLM only).
# Only applied when the LLM produces a non-zero value for that market.
_LLM_WEIGHT_1X2   = float(os.getenv("LLM_WEIGHT_1X2",   "0.40"))
_LLM_WEIGHT_GOALS = float(os.getenv("LLM_WEIGHT_GOALS",  "0.35"))
_LLM_WEIGHT_BTTS  = float(os.getenv("LLM_WEIGHT_BTTS",   "0.40"))


def _blend(llm_val: float, poisson_val: float, llm_weight: float) -> float:
    """Weighted average; falls back to Poisson when the LLM has no value."""
    if llm_val <= 0:
        return poisson_val
    return round(llm_val * llm_weight + poisson_val * (1.0 - llm_weight), 1)


class BettingOrchestrator:
    """Coordinates the full data → prediction pipeline."""

    # No API Football key available — route all previously AF-backed competitions via ESPN
    _API_FOOTBALL_CODES: set[str] = set()
    _ESPN_CODES = {"ECL", "BSA", "WC", "WCQA", "WCQC", "WCQAS", "WCQAF"}

    def __init__(self):
        self.fd_client = FootballDataClient()
        self.af_client = ApiFootballClient()
        self.espn_client = EspnClient()
        self.fbref = FBrefScraper()
        self.mf_client = MiroFishClient()
        self.parser = ResultParser()
        # Cache of fd.org team IDs keyed by competition code (ESPN IDs ≠ fd.org IDs)
        self._fd_competition_teams: dict[str, dict[str, int]] = {}
        # Cache of ESPN WC+WCQ results for team stats fallback
        self._espn_wc_results_cache: Optional[list] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_fixture(self, fixture: Fixture) -> BettingPrediction:
        """
        Run a full prediction for a single fixture.

        Pipeline:
          1. Data gathering
          1b. Bayesian shrinkage
          1c. Form weighting (exponential decay on last 5 games)
          1d. FBref data quality validation
          [2-4. LLM path — only when USE_LLM=true]
          5. Poisson goals model (H2H + neutral venue aware)
          6. Poisson corners model
          7. Poisson cards model (referee factor)
        """
        logger.info("=== Predicting: %s vs %s (%s) ===",
                    fixture.home_team, fixture.away_team, fixture.competition)

        match_label = (
            f"{fixture.home_team} vs {fixture.away_team} — "
            f"{fixture.competition} {fixture.match_date.strftime('%d/%m/%Y')}"
        )

        # 1. Gather all data
        report = self._build_match_report(fixture)

        # 1b. Bayesian shrinkage
        apply_shrinkage(report.home_stats, fixture.competition_code)
        apply_shrinkage(report.away_stats, fixture.competition_code)

        # 1c. Blend in recent form (exponential decay)
        self._apply_form_weighting(report.home_stats, report.home_form)
        self._apply_form_weighting(report.away_stats, report.away_form)

        # 1d. Warn on data gaps
        self._validate_fbref_stats(report.home_stats)
        self._validate_fbref_stats(report.away_stats)

        # ── LLM analysis (MiroFish) ───────────────────────────────────────────
        prediction = BettingPrediction()
        prediction.match = match_label
        prediction.competition = fixture.competition

        if _USE_LLM:
            seed_text, pred_prompt = build_seed_document(report)
            result = self.mf_client.run_match_prediction(
                seed_text=seed_text,
                prediction_prompt=pred_prompt,
                match_label=match_label,
                simulation_rounds=10,
            )
            if result["status"] == "success":
                llm_pred = self.parser.parse(result["report_markdown"])
                logger.info("LLM prediction parsed (source: %s)", llm_pred.parse_source)
                if llm_pred.parse_source != "fallback":
                    prediction.llm_home_win_pct       = llm_pred.home_win_pct
                    prediction.llm_draw_pct            = llm_pred.draw_pct
                    prediction.llm_away_win_pct        = llm_pred.away_win_pct
                    prediction.llm_over_2_5_pct        = llm_pred.over_2_5_pct
                    prediction.llm_btts_yes_pct        = llm_pred.btts_yes_pct
                    prediction.llm_most_likely_scoreline = llm_pred.most_likely_scoreline
                    prediction.llm_analysis            = result["report_markdown"]
                    prediction.raw_report              = result["report_markdown"]
            else:
                logger.warning("MiroFish failed: %s — continuing with Poisson-only", result.get("error"))
        else:
            logger.info("LLM skipped (USE_LLM=false) — using Poisson-only pipeline")

        # ── 5. Poisson goals model ────────────────────────────────────────────
        try:
            poisson = compute_poisson(
                report.home_stats,
                report.away_stats,
                fixture.competition_code,
                head_to_head=report.head_to_head,
                is_neutral=fixture.is_neutral,
            )
            has_llm = prediction.llm_home_win_pct > 0 or prediction.llm_draw_pct > 0

            # Scoreline and lambdas always come from Poisson (most precise)
            prediction.most_likely_scoreline  = poisson.most_likely_scoreline
            prediction.poisson_lambda_home    = poisson.lambda_home
            prediction.poisson_lambda_away    = poisson.lambda_away
            prediction.poisson_top_scorelines = poisson.top_scorelines(8)

            # Store raw Poisson-only values for model comparison tracking
            prediction.poisson_home_win_pct  = poisson.home_win_pct
            prediction.poisson_draw_pct      = poisson.draw_pct
            prediction.poisson_away_win_pct  = poisson.away_win_pct
            prediction.poisson_over_2_5_pct  = poisson.over_2_5_pct
            prediction.poisson_btts_yes_pct  = poisson.btts_yes_pct

            # 1X2 — blend LLM + Poisson when LLM is available
            if has_llm:
                prediction.home_win_pct = _blend(prediction.llm_home_win_pct, poisson.home_win_pct, _LLM_WEIGHT_1X2)
                prediction.draw_pct     = _blend(prediction.llm_draw_pct,     poisson.draw_pct,     _LLM_WEIGHT_1X2)
                prediction.away_win_pct = round(100.0 - prediction.home_win_pct - prediction.draw_pct, 1)
            else:
                prediction.home_win_pct = poisson.home_win_pct
                prediction.draw_pct     = poisson.draw_pct
                prediction.away_win_pct = poisson.away_win_pct

            # Goals markets
            if has_llm:
                prediction.over_2_5_pct  = _blend(prediction.llm_over_2_5_pct, poisson.over_2_5_pct, _LLM_WEIGHT_GOALS)
                prediction.under_2_5_pct = round(100.0 - prediction.over_2_5_pct, 1)
            else:
                prediction.over_2_5_pct  = poisson.over_2_5_pct
                prediction.under_2_5_pct = poisson.under_2_5_pct

            # Over/Under 3.5 — Poisson only (LLM rarely predicts this sub-market)
            prediction.over_3_5_pct  = poisson.over_3_5_pct
            prediction.under_3_5_pct = poisson.under_3_5_pct

            # BTTS
            if has_llm:
                prediction.btts_yes_pct = _blend(prediction.llm_btts_yes_pct, poisson.btts_yes_pct, _LLM_WEIGHT_BTTS)
                prediction.btts_no_pct  = round(100.0 - prediction.btts_yes_pct, 1)
            else:
                prediction.btts_yes_pct = poisson.btts_yes_pct
                prediction.btts_no_pct  = poisson.btts_no_pct

            prediction.confidence = self._poisson_confidence(
                poisson, report.home_stats, report.away_stats, report.head_to_head
            )
            logger.info(
                "Poisson goals: λ_home=%.2f λ_away=%.2f → %s "
                "(1X2: %.1f/%.1f/%.1f, O2.5: %.1f%%, O3.5: %.1f%%, BTTS: %.1f%%) [llm_blend=%s]",
                poisson.lambda_home, poisson.lambda_away, poisson.most_likely_scoreline,
                prediction.home_win_pct, prediction.draw_pct, prediction.away_win_pct,
                prediction.over_2_5_pct, poisson.over_3_5_pct, prediction.btts_yes_pct,
                "yes" if has_llm else "no",
            )
        except Exception as e:
            logger.warning("Poisson goals model failed (non-fatal): %s", e)

        # ── Motivational context ──────────────────────────────────────────────
        prediction.motivation_context = self._motivation_context(
            fixture, report.home_standing, report.away_standing
        )
        if prediction.motivation_context:
            logger.info("Motivation context: %s", prediction.motivation_context)

        # ── 6. Poisson corners ────────────────────────────────────────────────
        try:
            home_cpg = (report.home_stats.corners_home_pg
                        if report.home_stats.corners_home_pg > 0
                        else report.home_stats.corners_pg)
            away_cpg = (report.away_stats.corners_away_pg
                        if report.away_stats.corners_away_pg > 0
                        else report.away_stats.corners_pg)
            corner_poisson = compute_corner_poisson(home_cpg, away_cpg, fixture.competition_code)
            if corner_poisson:
                prediction.over_9_5_corners_pct  = corner_poisson.over_9_5_corners_pct
                prediction.under_9_5_corners_pct = corner_poisson.under_9_5_corners_pct
                prediction.poisson_corners_lambda = corner_poisson.lambda_corners
                logger.info(
                    "Poisson corners: λ=%.2f → O9.5: %.1f%%",
                    corner_poisson.lambda_corners, corner_poisson.over_9_5_corners_pct,
                )
        except Exception as e:
            logger.warning("Poisson corners model failed (non-fatal): %s", e)

        # ── 7. Poisson cards (with referee factor) ────────────────────────────
        try:
            referee_factor = self._referee_factor(fixture)
            cards_poisson = compute_cards_poisson(
                report.home_stats.yellow_cards_pg,
                report.away_stats.yellow_cards_pg,
                report.home_stats.red_cards_pg,
                report.away_stats.red_cards_pg,
                fixture.competition_code,
                referee_factor=referee_factor,
            )
            if cards_poisson:
                prediction.over_3_5_cards_pct   = cards_poisson.over_3_5_yellow_pct
                prediction.under_3_5_cards_pct  = cards_poisson.under_3_5_yellow_pct
                prediction.over_4_5_yellow_pct  = cards_poisson.over_4_5_yellow_pct
                prediction.under_4_5_yellow_pct = cards_poisson.under_4_5_yellow_pct
                prediction.poisson_cards_lambda  = cards_poisson.lambda_yellow
                prediction.red_card_pct = round(
                    (1.0 - math.exp(-cards_poisson.lambda_red)) * 100, 1
                )
                logger.info(
                    "Poisson cards: λ_y=%.2f (ref_factor=%.2f) → O3.5Y: %.1f%% red: %.1f%%",
                    cards_poisson.lambda_yellow, referee_factor,
                    cards_poisson.over_3_5_yellow_pct, prediction.red_card_pct,
                )
        except Exception as e:
            logger.warning("Poisson cards model failed (non-fatal): %s", e)

        # ── Summary ──────────────────────────────────────────────────────────
        prediction.prediction_summary = self._build_summary(fixture, report, prediction)

        logger.info("Prediction complete: %s", json.dumps(prediction.to_dict(), indent=2))
        return prediction

    def get_upcoming_fixtures(self) -> list[Fixture]:
        """Return all upcoming fixtures across all supported competitions."""
        fixtures = []
        for code in ("SA", "SB", "CL", "EL", "ECL", "USC", "WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF", "BSA"):
            try:
                if code in self._ESPN_CODES:
                    upcoming = self.espn_client.get_upcoming_fixtures(code, days_ahead=14)
                elif code in self._API_FOOTBALL_CODES:
                    upcoming = self.af_client.get_upcoming_fixtures(code, days_ahead=14)
                else:
                    upcoming = self.fd_client.get_upcoming_fixtures(code, days_ahead=14)
                fixtures.extend(upcoming)
            except Exception as e:
                logger.warning("Could not fetch %s fixtures: %s", code, e)
        fixtures.sort(key=lambda f: f.match_date)
        return fixtures

    # ── Data gathering ────────────────────────────────────────────────────────

    def _build_match_report(self, fixture: Fixture) -> MatchReport:
        code = fixture.competition_code

        # For ESPN-sourced competitions (WC, ECL, BSA), fixture team IDs are ESPN
        # internal IDs which don't match fd.org IDs. Resolve the correct fd.org IDs
        # via the competition standings (or team list as fallback) so that form
        # lookups return real data instead of a 404 / wrong team.
        home_team_id = fixture.home_team_id
        away_team_id = fixture.away_team_id
        home_standing = away_standing = None

        if code in self._ESPN_CODES:
            try:
                standings = self.fd_client.get_standings(code)
                home_standing = self._fuzzy_find_standing(fixture.home_team, standings)
                away_standing = self._fuzzy_find_standing(fixture.away_team, standings)
                if home_standing:
                    home_team_id = home_standing.team_id
                if away_standing:
                    away_team_id = away_standing.team_id
            except Exception:
                pass

            # If standings didn't cover both teams (pre-tournament or new comp),
            # fall back to the /competitions/{code}/teams endpoint.
            if home_team_id == fixture.home_team_id or away_team_id == fixture.away_team_id:
                fd_teams = self._get_fd_teams_for_competition(code)
                if home_team_id == fixture.home_team_id:
                    fd_id = self._match_team_name(fixture.home_team, fd_teams)
                    if fd_id:
                        home_team_id = fd_id
                if away_team_id == fixture.away_team_id:
                    fd_id = self._match_team_name(fixture.away_team, fd_teams)
                    if fd_id:
                        away_team_id = fd_id

        # Team forms (from football-data.org, using resolved IDs)
        home_form = self._safe(
            lambda: self.fd_client.get_team_form(home_team_id, fixture.home_team),
            default_factory=lambda: self._empty_form(fixture.home_team),
        )
        away_form = self._safe(
            lambda: self.fd_client.get_team_form(away_team_id, fixture.away_team),
            default_factory=lambda: self._empty_form(fixture.away_team),
        )

        # Standings for non-ESPN competitions (already fetched above for ESPN)
        if code not in self._ESPN_CODES:
            try:
                standings = self.fd_client.get_standings(code)
                standings_map = {s.team_name: s for s in standings}
                home_standing = standings_map.get(fixture.home_team)
                away_standing = standings_map.get(fixture.away_team)
            except Exception:
                home_standing = away_standing = None

        # Head to head
        h2h = self._safe(
            lambda: self.fd_client.get_head_to_head(
                fixture.fixture_id, fixture.home_team, fixture.away_team
            )
        )

        # FBref detailed stats (corners, cards, xG)
        all_fbref = self._safe(lambda: self.fbref.get_team_stats(code), default_factory=list)
        fbref_map = {s.team_name: s for s in all_fbref}

        home_stats = self._merge_stats(
            fixture.home_team, code, home_form, fbref_map
        )
        away_stats = self._merge_stats(
            fixture.away_team, code, away_form, fbref_map
        )

        # If stats are still empty but standings data is available (in-tournament),
        # use standings goals for/against to differentiate teams.
        for stats, standing in (
            (home_stats, home_standing), (away_stats, away_standing)
        ):
            if stats.games_played == 0 and standing and standing.played > 0:
                gp = standing.played
                stats.games_played = gp
                if stats.goals_scored_pg == 0.0:
                    stats.goals_scored_pg = round(standing.goals_for / gp, 2)
                if stats.goals_conceded_pg == 0.0:
                    stats.goals_conceded_pg = round(standing.goals_against / gp, 2)

        # WC fallback chain when tournament hasn't started yet or no stats available.
        # 1. FBref WCQ — uses confederation map so only 1 request per team (~4s each).
        # 2. ESPN historical results — covers any finished WC/WCQ games on ESPN.
        if code == "WC" and (home_stats.games_played == 0 or away_stats.games_played == 0):
            for stats, name in ((home_stats, fixture.home_team), (away_stats, fixture.away_team)):
                if stats.games_played == 0:
                    wcq = self._safe(lambda n=name: self.fbref.get_wcq_stats(n))
                    if wcq and wcq.games_played > 0:
                        stats.goals_scored_pg   = wcq.goals_scored_pg
                        stats.goals_conceded_pg  = wcq.goals_conceded_pg
                        stats.xg_pg             = wcq.xg_pg
                        stats.xga_pg            = wcq.xga_pg
                        stats.games_played      = wcq.games_played
                        stats.corners_pg        = wcq.corners_pg
                        stats.yellow_cards_pg   = wcq.yellow_cards_pg
                        stats.red_cards_pg      = wcq.red_cards_pg
                        logger.info("WCQ fallback: %s — %d games (via %s)",
                                    name, wcq.games_played, wcq.competition)

        # ESPN results fallback (covers once group stage matches start).
        if code == "WC" and (home_stats.games_played == 0 or away_stats.games_played == 0):
            espn_results = self._get_espn_wc_results()
            for stats, name in ((home_stats, fixture.home_team), (away_stats, fixture.away_team)):
                if stats.games_played == 0:
                    self._fill_stats_from_espn_results(stats, name, espn_results)

        # BTTS / clean sheet rates from FBref schedule
        btts_data = self._safe(
            lambda: self.fbref.get_btts_and_clean_sheets(code),
            default_factory=dict,
        )
        self._apply_btts(home_stats, fixture.home_team, btts_data)
        self._apply_btts(away_stats, fixture.away_team, btts_data)

        # Injuries — only available via API-Football (WCQ competitions)
        if code in self._API_FOOTBALL_CODES:
            home_injuries, away_injuries = self._safe(
                lambda: self.af_client.get_injuries(
                    fixture.fixture_id, fixture.home_team, fixture.away_team
                ),
                default_factory=lambda: (
                    InjuryReport(team_name=fixture.home_team),
                    InjuryReport(team_name=fixture.away_team),
                ),
            )
        else:
            home_injuries = InjuryReport(team_name=fixture.home_team)
            away_injuries = InjuryReport(team_name=fixture.away_team)

        return MatchReport(
            fixture=fixture,
            home_form=home_form,
            away_form=away_form,
            home_stats=home_stats,
            away_stats=away_stats,
            home_standing=home_standing,
            away_standing=away_standing,
            head_to_head=h2h,
            home_injuries=home_injuries,
            away_injuries=away_injuries,
        )

    # ── WC / ESPN team-ID resolution ──────────────────────────────────────────

    def _get_fd_teams_for_competition(self, code: str) -> dict[str, int]:
        """Lazy-load and cache fd.org team IDs for an ESPN-sourced competition."""
        if code not in self._fd_competition_teams:
            teams = self._safe(
                lambda: self.fd_client.get_competition_teams(code),
                default_factory=dict,
            )
            self._fd_competition_teams[code] = teams or {}
        return self._fd_competition_teams[code]

    @staticmethod
    def _fuzzy_find_standing(team_name: str, standings: list):
        """Return the first Standing whose name shares a keyword with team_name."""
        stop = {"fc", "ac", "as", "ss", "sc", "cf", "the", "de", "1."}
        kw = {w.lower() for w in team_name.split() if len(w) > 2 and w.lower() not in stop}
        if not kw:
            return None
        for s in standings:
            s_kw = {w.lower() for w in s.team_name.split() if len(w) > 2 and w.lower() not in stop}
            if kw & s_kw:
                return s
        return None

    def _get_espn_wc_results(self) -> list:
        """Aggregate ESPN results across WC + all WCQ competitions (cached per instance)."""
        if self._espn_wc_results_cache is None:
            results = []
            for code in ("WC", "WCQA", "WCQC", "WCQAS", "WCQAF", "WCQE"):
                partial = self._safe(
                    lambda c=code: self.espn_client.get_competition_results(c, days_back=730),
                    default_factory=list,
                )
                results.extend(partial or [])
            self._espn_wc_results_cache = results
        return self._espn_wc_results_cache

    @staticmethod
    def _fill_stats_from_espn_results(stats, team_name: str, results: list) -> None:
        """Fill goals_scored_pg / goals_conceded_pg / games_played from ESPN results."""
        scored, conceded = [], []
        for f in results:
            if f.home_team == team_name:
                scored.append(f.home_score)
                conceded.append(f.away_score)
            elif f.away_team == team_name:
                scored.append(f.away_score)
                conceded.append(f.home_score)
        if not scored:
            return
        n = len(scored)
        stats.goals_scored_pg  = round(sum(scored)   / n, 2)
        stats.goals_conceded_pg = round(sum(conceded) / n, 2)
        stats.games_played     = n
        logger.info("ESPN WCQ fallback for %s: %d games, %.2f GF, %.2f GA",
                    team_name, n, stats.goals_scored_pg, stats.goals_conceded_pg)

    @staticmethod
    def _match_team_name(team_name: str, teams: dict[str, int]) -> Optional[int]:
        """Find a team ID by exact then keyword-fuzzy name match."""
        if team_name in teams:
            return teams[team_name]
        stop = {"fc", "ac", "as", "ss", "sc", "cf", "the", "de"}
        kw = {w.lower() for w in team_name.split() if len(w) > 2 and w.lower() not in stop}
        for name, tid in teams.items():
            name_kw = {w.lower() for w in name.split() if len(w) > 2 and w.lower() not in stop}
            if kw & name_kw:
                return tid
        return None

    def _merge_stats(self, team_name: str, code: str, form, fbref_map: dict) -> TeamStats:
        """Combine football-data.org form data with FBref detailed stats."""
        # Start from FBref if available (more detailed)
        base = fbref_map.get(team_name)

        if base is None:
            # Fuzzy match
            keywords = set(w.lower() for w in team_name.split()
                           if len(w) > 2 and w.lower() not in {"fc", "ac", "as", "ss"})
            for name, stats in fbref_map.items():
                name_words = set(w.lower() for w in name.split())
                if keywords & name_words:
                    base = stats
                    break

        if base is None:
            base = TeamStats(team_name=team_name, competition=code)

        # Fill goals from form if FBref has zeros
        if base.goals_scored_pg == 0.0 and form.goals_scored:
            n = len(form.goals_scored)
            base.goals_scored_pg = round(sum(form.goals_scored) / n, 2)
        if base.goals_conceded_pg == 0.0 and form.goals_conceded:
            n = len(form.goals_conceded)
            base.goals_conceded_pg = round(sum(form.goals_conceded) / n, 2)

        # Use form game count as games_played when FBref has no data.
        # This prevents apply_shrinkage from collapsing all stats to the league
        # average (which happens whenever games_played == 0).
        if base.games_played == 0 and form.goals_scored:
            base.games_played = len(form.goals_scored)

        return base

    @staticmethod
    def _apply_btts(stats: TeamStats, team_name: str, btts_data: dict) -> None:
        entry = btts_data.get(team_name)
        if not entry:
            # Fuzzy
            for k, v in btts_data.items():
                if any(w in k for w in team_name.split() if len(w) > 3):
                    entry = v
                    break
        if entry and entry.get("games", 0) > 0:
            stats.btts_count = entry["btts"]
            stats.clean_sheets = entry["clean_sheets"]
            if stats.games_played == 0:
                stats.games_played = entry["games"]

    @staticmethod
    def _empty_form(team_name: str):
        from football.models import TeamForm
        return TeamForm(team_name=team_name)

    @staticmethod
    def _safe(fn, default_factory=None):
        try:
            return fn()
        except Exception as e:
            logger.warning("Data fetch failed (non-fatal): %s", e)
            if default_factory:
                return default_factory()
            return None

    # ── New helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _apply_form_weighting(
        stats: TeamStats,
        form,
        decay: float = 0.75,
        max_recent_weight: float = 0.30,
    ) -> None:
        """
        Blend season-average stats with exponentially-weighted recent form (in-place).

        Each game back from most-recent gets multiplied by `decay`.
        The blend weight reaches max_recent_weight (30%) when ≥5 recent games available.
        Using actual recent goals as a proxy for attacking/defensive form trend.
        """
        def _weighted_avg(values: list) -> float | None:
            if not values:
                return None
            n = len(values)
            weights = [decay ** (n - 1 - i) for i in range(n)]
            w_sum = sum(weights)
            return sum(v * w for v, w in zip(values, weights)) / w_sum

        recent_scored   = _weighted_avg(getattr(form, "goals_scored",   []))
        recent_conceded = _weighted_avg(getattr(form, "goals_conceded", []))

        n_games = len(getattr(form, "goals_scored", []) or [])
        w = min(n_games / 5, 1.0) * max_recent_weight  # 0 → max_recent_weight as n→5

        if recent_scored is not None:
            if stats.goals_scored_pg > 0:
                stats.goals_scored_pg = round(
                    stats.goals_scored_pg * (1 - w) + recent_scored * w, 3)
            if stats.xg_pg > 0:
                stats.xg_pg = round(stats.xg_pg * (1 - w) + recent_scored * w, 3)

        if recent_conceded is not None:
            if stats.goals_conceded_pg > 0:
                stats.goals_conceded_pg = round(
                    stats.goals_conceded_pg * (1 - w) + recent_conceded * w, 3)
            if stats.xga_pg > 0:
                stats.xga_pg = round(stats.xga_pg * (1 - w) + recent_conceded * w, 3)

        if n_games > 0 and w > 0:
            logger.debug(
                "Form weighting %s: w=%.2f (n=%d) scored %.2f→%.2f conceded %.2f→%.2f",
                stats.team_name, w, n_games,
                recent_scored or 0, stats.goals_scored_pg,
                recent_conceded or 0, stats.goals_conceded_pg,
            )

    @staticmethod
    def _validate_fbref_stats(stats: TeamStats) -> None:
        """Log a warning when critical FBref fields are missing (zero = not scraped)."""
        gaps = [f for f in ("xg_pg", "xga_pg", "corners_pg", "yellow_cards_pg")
                if getattr(stats, f, 0.0) == 0.0]
        if gaps:
            logger.warning(
                "FBref data gap for %s — fields missing: %s. "
                "Poisson will use league-average fallback for these.",
                stats.team_name, ", ".join(gaps),
            )

    @staticmethod
    def _poisson_confidence(poisson, home_stats, away_stats, h2h) -> str:
        """
        Multi-factor confidence score.

        Factor 1 – Data completeness (max 0.40): awards points for xg_pg, xga_pg,
          corners_pg being non-zero and for sufficient games_played.
        Factor 2 – Outcome certainty (max 0.35): maps dominant outcome probability
          from 33.3% (uniform) to 70%+ linearly.
        Factor 3 – H2H depth (max 0.25): rewards meaningful head-to-head history.

        Thresholds: ≥0.65 = "high", 0.40–0.65 = "medium", <0.40 = "low".
        """
        score = 0.0

        # Factor 1: data completeness
        data_score = 0.0
        for stats in (home_stats, away_stats):
            if getattr(stats, "xg_pg", 0.0) > 0:
                data_score += 0.05
            if getattr(stats, "xga_pg", 0.0) > 0:
                data_score += 0.05
            if getattr(stats, "corners_pg", 0.0) > 0:
                data_score += 0.025
            gp = getattr(stats, "games_played", 0)
            if gp >= 10:
                data_score += 0.075
            elif gp >= 5:
                data_score += 0.05
            elif gp >= 3:
                data_score += 0.025
        score += min(data_score, 0.40)

        # Factor 2: outcome certainty
        dominant = max(poisson.home_win_pct, poisson.draw_pct, poisson.away_win_pct)
        certainty = max(0.0, (dominant - 33.3) / (70.0 - 33.3))
        score += min(certainty * 0.35, 0.35)

        # Factor 3: H2H depth
        if h2h:
            meetings = getattr(h2h, "total_meetings", 0)
            if meetings >= 5:
                score += 0.25
            elif meetings >= 3:
                score += 0.15
            elif meetings >= 1:
                score += 0.05

        if score >= 0.65:
            return "high"
        if score >= 0.40:
            return "medium"
        return "low"

    @staticmethod
    def _motivation_context(fixture: Fixture, home_standing, away_standing) -> str:
        """
        Informational flag derived from standings position and stage.
        Does not modify lambda — used as display context only.
        """
        parts = []

        stage = getattr(fixture, "stage", "") or ""
        if any(k in stage.lower() for k in ("final", "semi", "quarter", "knockout", "round of")):
            parts.append(f"Knockout: {stage}")

        if home_standing and away_standing:
            code = fixture.competition_code
            is_cup = code in ("CL", "ECL", "WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF")
            rel_thr   = 3  if is_cup else 16
            title_thr = 2  if is_cup else 3

            h_pos, a_pos = home_standing.position, away_standing.position

            if h_pos <= title_thr:
                parts.append(f"{home_standing.team_name} title race (#{h_pos})")
            if a_pos <= title_thr:
                parts.append(f"{away_standing.team_name} title race (#{a_pos})")

            elim_lbl = "elimination zone" if is_cup else "relegation battle"
            if h_pos >= rel_thr:
                parts.append(f"{home_standing.team_name} {elim_lbl} (#{h_pos})")
            if a_pos >= rel_thr:
                parts.append(f"{away_standing.team_name} {elim_lbl} (#{a_pos})")

            # Possible dead rubber: mid-table, large points gap, no other context
            if not parts:
                pts_gap  = abs(home_standing.points - away_standing.points)
                mid_pos  = (h_pos + a_pos) / 2
                if pts_gap >= 15 and 5 <= mid_pos <= 14:
                    parts.append("Low-stakes (large points gap, mid-table)")

        return "; ".join(parts) if parts else ""

    @staticmethod
    def _build_summary(fixture: Fixture, report, prediction) -> str:
        """
        Conversational tipster-style narrative built from model outputs.
        No technical notation — readable by anyone, not just analysts.
        """
        hs  = report.home_stats
        as_ = report.away_stats
        h2h = report.head_to_head
        hw_p = prediction.home_win_pct
        d_p  = prediction.draw_pct
        aw_p = prediction.away_win_pct
        o25  = prediction.over_2_5_pct
        btts = prediction.btts_yes_pct
        conf = prediction.confidence

        def _attack_desc(goals_pg: float, xg_pg: float) -> str:
            ref = xg_pg if xg_pg > 0 else goals_pg
            if ref >= 2.2:   return "a very clinical attack"
            if ref >= 1.7:   return "a solid attacking threat"
            if ref >= 1.2:   return "a decent but not overwhelming attack"
            if ref >= 0.8:   return "a modest attacking output"
            return "a struggling attack"

        def _defence_desc(ga_pg: float) -> str:
            if ga_pg == 0:   return ""
            if ga_pg < 0.8:  return "excellent defensively"
            if ga_pg < 1.2:  return "solid at the back"
            if ga_pg < 1.7:  return "occasionally leaky"
            return "very open at the back"

        parts: list[str] = []

        # ── Sentence 1: Team profiles ─────────────────────────────────────
        h_atk = _attack_desc(hs.goals_scored_pg, hs.xg_pg)
        a_atk = _attack_desc(as_.goals_scored_pg, as_.xg_pg)
        h_def = _defence_desc(hs.goals_conceded_pg)
        a_def = _defence_desc(as_.goals_conceded_pg)

        neutral_note = " at a neutral venue" if fixture.is_neutral else ""
        h_desc = f"{h_atk}" + (f" and {h_def}" if h_def else "")
        a_desc = f"{a_atk}" + (f" and {a_def}" if a_def else "")
        parts.append(
            f"{fixture.home_team} come into this{neutral_note} with {h_desc}, "
            f"while {fixture.away_team} bring {a_desc}."
        )

        # ── Sentence 2: H2H ───────────────────────────────────────────────
        if h2h and h2h.total_meetings >= 1:
            m = h2h.total_meetings
            hw, d, aw = h2h.home_wins, h2h.draws, h2h.away_wins
            if hw > aw + 1:
                h2h_bias = f"history clearly favours {fixture.home_team} ({hw}W-{d}D-{aw}L in {m} meetings)"
            elif aw > hw + 1:
                h2h_bias = f"history leans towards {fixture.away_team} ({hw}W-{d}D-{aw}L in {m} meetings)"
            elif d >= m // 2:
                h2h_bias = f"these two sides draw a lot — {d} draws in {m} meetings"
            else:
                h2h_bias = f"head-to-head is pretty even ({hw}W-{d}D-{aw}L in {m} meetings)"
            avg_note = f", typically producing around {h2h.avg_goals:.1f} goals" if h2h.avg_goals > 0 else ""
            parts.append(f"In terms of history, {h2h_bias}{avg_note}.")
        else:
            parts.append("There's no head-to-head history available to draw from.")

        # ── Sentence 3: Model verdict ─────────────────────────────────────
        if any((hw_p, d_p, aw_p)):
            dominant = max(hw_p, d_p, aw_p)
            if hw_p >= d_p and hw_p >= aw_p:
                result_line = (
                    f"The numbers point to a {fixture.home_team} win "
                    f"({'strong favourite' if dominant >= 60 else 'slight edge'} at {dominant:.0f}%)"
                )
            elif d_p >= aw_p:
                result_line = (
                    f"The model sees this as an even contest leaning towards a draw ({dominant:.0f}%)"
                )
            else:
                result_line = (
                    f"The numbers actually favour {fixture.away_team} here "
                    f"({'strong favourite' if dominant >= 60 else 'slight edge'} at {dominant:.0f}%)"
                )

            goal_notes: list[str] = []
            if o25 >= 65:
                goal_notes.append(f"goals look very likely (Over 2.5 at {o25:.0f}%)")
            elif o25 >= 55:
                goal_notes.append(f"a higher-scoring game seems probable (Over 2.5 at {o25:.0f}%)")
            elif o25 > 0 and o25 <= 45:
                goal_notes.append(f"a tight, low-scoring affair is more likely (Under 2.5 at {100 - o25:.0f}%)")

            if btts >= 60:
                goal_notes.append(f"both teams are likely to score ({btts:.0f}%)")
            elif btts > 0 and btts <= 40:
                goal_notes.append(f"at least one side could keep a clean sheet (BTTS only {btts:.0f}%)")

            verdict = result_line
            if goal_notes:
                verdict += " — " + " and ".join(goal_notes)
            parts.append(verdict + ".")

        # ── Sentence 4: Confidence caveat ─────────────────────────────────
        if conf == "low":
            thin = [
                name for name, st in (
                    (fixture.home_team, hs), (fixture.away_team, as_)
                )
                if st.xg_pg == 0.0 or st.games_played < 5
            ]
            if thin:
                parts.append(
                    f"Worth noting: data is limited for {' and '.join(thin)}, "
                    f"so take this one with a pinch of salt."
                )
            else:
                parts.append("The outcome is genuinely hard to call — proceed with caution.")
        elif conf == "medium" and (not h2h or h2h.total_meetings == 0):
            parts.append("No previous meetings to validate against, so confidence is moderate.")

        return " ".join(parts)

    def _referee_factor(self, fixture: Fixture) -> float:
        """
        Look up the assigned referee's yellow-cards-per-game rate and return
        a multiplier relative to the competition average.

        Returns 1.0 when no referee is known or no stats are available.
        """
        if not fixture.referee:
            return 1.0

        from predictor.poisson import LEAGUE_AVG_CARDS, _DEFAULT_AVG_CARDS
        ref_stats = self._safe(
            lambda: self.fbref.get_referee_stats(fixture.competition_code),
            default_factory=dict,
        ) or {}

        ref_yellow_pg = ref_stats.get(fixture.referee)
        if not ref_yellow_pg:
            # Partial name match (e.g. "C. Taylor" vs "Craig Taylor")
            ref_lower = fixture.referee.lower()
            for name, val in ref_stats.items():
                if any(part in name.lower() for part in ref_lower.split() if len(part) > 2):
                    ref_yellow_pg = val
                    break

        if not ref_yellow_pg:
            return 1.0

        league_avg = LEAGUE_AVG_CARDS.get(
            fixture.competition_code, _DEFAULT_AVG_CARDS
        )["yellow"]
        factor = round(ref_yellow_pg / league_avg, 3) if league_avg > 0 else 1.0
        logger.info(
            "Referee %s: %.2f Y/game vs league avg %.2f → factor %.3f",
            fixture.referee, ref_yellow_pg, league_avg, factor,
        )
        return factor


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run a betting prediction for a fixture")
    parser.add_argument("--fixture-id", type=int, help="football-data.org fixture ID")
    parser.add_argument("--list", action="store_true", help="List upcoming fixtures")
    args = parser.parse_args()

    orch = BettingOrchestrator()

    if args.list:
        fixtures = orch.get_upcoming_fixtures()
        print(f"\nUpcoming fixtures ({len(fixtures)} total):\n")
        for f in fixtures:
            print(f"  [{f.fixture_id}] {f.match_date.strftime('%d/%m %H:%M')} | "
                  f"{f.competition:20s} | {f.home_team} vs {f.away_team}")
        return

    if args.fixture_id:
        # Fetch fixture details
        fixtures = orch.get_upcoming_fixtures()
        match = next((f for f in fixtures if f.fixture_id == args.fixture_id), None)
        if not match:
            print(f"Fixture {args.fixture_id} not found. Use --list to see available fixtures.")
            sys.exit(1)
        pred = orch.predict_fixture(match)
        print("\n=== BETTING PREDICTIONS ===\n")
        print(json.dumps(pred.to_dict(), indent=2))
        return

    print("Use --list to see fixtures or --fixture-id <id> to predict a match.")


if __name__ == "__main__":
    main()
