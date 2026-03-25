"""
predictor/orchestrator.py

Main entry point: given a fixture, fetches all data, builds the seed,
runs MiroFish, and returns structured betting predictions.

Can be called:
  - Via HTTP API (see api_server.py)
  - Directly from CLI: python -m predictor.orchestrator --fixture-id 12345
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from football import FootballDataClient, FBrefScraper, MatchReport, Fixture
from football.models import TeamStats, InjuryReport
from seed import build_seed_document
from predictor.mirofish_client import MiroFishClient
from predictor.result_parser import ResultParser, BettingPrediction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BettingOrchestrator:
    """Coordinates the full data → prediction pipeline."""

    def __init__(self):
        self.fd_client = FootballDataClient()
        self.fbref = FBrefScraper()
        self.mf_client = MiroFishClient()
        self.parser = ResultParser()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_fixture(self, fixture: Fixture) -> BettingPrediction:
        """Run a full prediction for a single fixture."""
        logger.info("=== Predicting: %s vs %s (%s) ===",
                    fixture.home_team, fixture.away_team, fixture.competition)

        # 1. Gather all data
        report = self._build_match_report(fixture)

        # 2. Generate seed document
        seed_text, pred_prompt = build_seed_document(report)

        match_label = (
            f"{fixture.home_team} vs {fixture.away_team} — "
            f"{fixture.competition} {fixture.match_date.strftime('%d/%m/%Y')}"
        )

        # 3. Run MiroFish
        result = self.mf_client.run_match_prediction(
            seed_text=seed_text,
            prediction_prompt=pred_prompt,
            match_label=match_label,
            simulation_rounds=10,
        )

        if result["status"] != "success":
            raise RuntimeError(f"MiroFish pipeline failed: {result.get('error')}")

        # 4. Parse predictions
        prediction = self.parser.parse(result["report_markdown"])
        prediction.match = match_label
        prediction.competition = fixture.competition

        logger.info("Prediction complete: %s", json.dumps(prediction.to_dict(), indent=2))
        return prediction

    def get_upcoming_fixtures(self) -> list[Fixture]:
        """Return all upcoming fixtures across all supported competitions."""
        fixtures = []
        for code in ("SA", "CL", "WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF"):
            try:
                upcoming = self.fd_client.get_upcoming_fixtures(code, days_ahead=14)
                fixtures.extend(upcoming)
            except Exception as e:
                logger.warning("Could not fetch %s fixtures: %s", code, e)
        fixtures.sort(key=lambda f: f.match_date)
        return fixtures

    # ── Data gathering ────────────────────────────────────────────────────────

    def _build_match_report(self, fixture: Fixture) -> MatchReport:
        code = fixture.competition_code

        # Team forms (from football-data.org)
        home_form = self._safe(
            lambda: self.fd_client.get_team_form(
                fixture.home_team_id, fixture.home_team
            ),
            default_factory=lambda: self._empty_form(fixture.home_team),
        )
        away_form = self._safe(
            lambda: self.fd_client.get_team_form(
                fixture.away_team_id, fixture.away_team
            ),
            default_factory=lambda: self._empty_form(fixture.away_team),
        )

        # Standings
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

        # BTTS / clean sheet rates from FBref schedule
        btts_data = self._safe(
            lambda: self.fbref.get_btts_and_clean_sheets(code),
            default_factory=dict,
        )
        self._apply_btts(home_stats, fixture.home_team, btts_data)
        self._apply_btts(away_stats, fixture.away_team, btts_data)

        return MatchReport(
            fixture=fixture,
            home_form=home_form,
            away_form=away_form,
            home_stats=home_stats,
            away_stats=away_stats,
            home_standing=home_standing,
            away_standing=away_standing,
            head_to_head=h2h,
            home_injuries=InjuryReport(team_name=fixture.home_team),
            away_injuries=InjuryReport(team_name=fixture.away_team),
        )

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
