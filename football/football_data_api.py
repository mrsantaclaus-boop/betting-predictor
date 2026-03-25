"""
football_data_api.py — Client for football-data.org free API.

Covers:
  - Serie A (competition code: SA)
  - Champions League (competition code: CL)

Free tier: 10 requests/minute.
Register at https://www.football-data.org/client/register
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

from .models import Fixture, Standing, TeamForm, HeadToHead

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
COMPETITIONS = {
    "SA":   "Serie A",
    "CL":   "Champions League",
    "WC":   "FIFA World Cup",
    "WCQE": "WCQ Europe",
    "WCQA": "WCQ Americas",
    "WCQC": "WCQ CONCACAF",
    "WCQAS": "WCQ Asia",
    "WCQAF": "WCQ Africa",
}

# Polite rate limit — free tier allows 10 req/min
_MIN_INTERVAL = 6.5  # seconds between requests
_last_call: float = 0.0


class FootballDataClient:
    """Thin wrapper around the football-data.org v4 API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FOOTBALL_DATA_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "FOOTBALL_DATA_API_KEY not set — requests will be rate-limited to 10/min."
            )
        self.session = requests.Session()
        self.session.headers.update({
            "X-Auth-Token": self.api_key,
            "Accept": "application/json",
        })

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        global _last_call
        elapsed = time.time() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        url = f"{BASE_URL}/{path}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            _last_call = time.time()
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error("API error %s: %s", e.response.status_code, url)
            if e.response.status_code == 429:
                logger.info("Rate limited — waiting 65s")
                time.sleep(65)
                return self._get(path, params)
            return {}
        except requests.RequestException as e:
            logger.error("Request failed: %s", e)
            return {}

    # ── Fixtures ──────────────────────────────────────────────────────────────

    def get_upcoming_fixtures(self, competition_code: str,
                               days_ahead: int = 7) -> list[Fixture]:
        """Return fixtures scheduled in the next N days."""
        data = self._get(f"competitions/{competition_code}/matches",
                         params={"status": "SCHEDULED"})
        return self._parse_fixtures(data, competition_code, limit=20)

    def get_recent_matches(self, competition_code: str,
                            limit: int = 10) -> list[Fixture]:
        """Return most recently finished matches."""
        data = self._get(f"competitions/{competition_code}/matches",
                         params={"status": "FINISHED"})
        fixtures = self._parse_fixtures(data, competition_code)
        return fixtures[-limit:]

    def get_match_result(self, fixture_id: int) -> Optional[Fixture]:
        """
        Fetch a single match by ID and return it as a Fixture.
        Returns None if not found or not yet FINISHED.
        """
        data = self._get(f"matches/{fixture_id}")
        # v4 API wraps the object under a "match" key
        match = data.get("match") or data
        if not match or match.get("status") != "FINISHED":
            return None
        try:
            fixtures = self._parse_fixtures({"matches": [match]}, competition_code="")
            return fixtures[0] if fixtures else None
        except (IndexError, KeyError):
            return None

    def get_team_matches(self, team_id: int, limit: int = 10) -> list[Fixture]:
        """Return last N finished matches for a specific team."""
        data = self._get(f"teams/{team_id}/matches",
                         params={"status": "FINISHED", "limit": limit})
        return self._parse_fixtures(data, competition_code="")[:limit]

    def _parse_fixtures(self, data: dict, competition_code: str,
                         limit: Optional[int] = None) -> list[Fixture]:
        fixtures = []
        for m in data.get("matches", []):
            try:
                utc_date = m.get("utcDate", "")
                match_dt = (
                    datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
                    if utc_date else datetime.now(timezone.utc)
                )
                comp_code = (
                    competition_code
                    or m.get("competition", {}).get("code", "")
                )
                fixtures.append(Fixture(
                    fixture_id=m.get("id", 0),
                    competition=COMPETITIONS.get(comp_code, comp_code),
                    competition_code=comp_code,
                    home_team=m["homeTeam"]["name"],
                    home_team_id=m["homeTeam"]["id"],
                    away_team=m["awayTeam"]["name"],
                    away_team_id=m["awayTeam"]["id"],
                    match_date=match_dt,
                    status=m.get("status", "SCHEDULED"),
                    home_score=m.get("score", {}).get("fullTime", {}).get("home"),
                    away_score=m.get("score", {}).get("fullTime", {}).get("away"),
                    matchday=m.get("matchday"),
                    stage=m.get("stage"),
                ))
            except (KeyError, ValueError) as e:
                logger.debug("Skip malformed fixture: %s", e)
                continue
        if limit:
            return fixtures[:limit]
        return fixtures

    # ── Standings ─────────────────────────────────────────────────────────────

    def get_standings(self, competition_code: str) -> list[Standing]:
        """Return current league table (Serie A only — UCL uses group tables)."""
        data = self._get(f"competitions/{competition_code}/standings")
        standings = []
        for table in data.get("standings", []):
            if table.get("type") == "TOTAL":
                for row in table.get("table", []):
                    try:
                        standings.append(Standing(
                            position=row["position"],
                            team_name=row["team"]["name"],
                            team_id=row["team"]["id"],
                            played=row["playedGames"],
                            won=row["won"],
                            drawn=row["draw"],
                            lost=row["lost"],
                            goals_for=row["goalsFor"],
                            goals_against=row["goalsAgainst"],
                            goal_difference=row["goalDifference"],
                            points=row["points"],
                            form=row.get("form", ""),
                        ))
                    except KeyError:
                        continue
                break
        return standings

    # ── Team form ─────────────────────────────────────────────────────────────

    def get_team_form(self, team_id: int, team_name: str,
                       limit: int = 10) -> TeamForm:
        """
        Build TeamForm from the team's last N finished matches.
        """
        matches = self.get_team_matches(team_id, limit=limit)
        results, scored, conceded = [], [], []

        for m in matches:
            if m.status != "FINISHED":
                continue
            is_home = m.home_team_id == team_id
            gs = m.home_score if is_home else m.away_score
            gc = m.away_score if is_home else m.home_score
            if gs is None or gc is None:
                continue
            scored.append(gs)
            conceded.append(gc)
            if gs > gc:
                results.append("W")
            elif gs == gc:
                results.append("D")
            else:
                results.append("L")

        # Home / away records
        home_matches = [m for m in matches if m.home_team_id == team_id and m.status == "FINISHED"]
        away_matches = [m for m in matches if m.away_team_id == team_id and m.status == "FINISHED"]

        def record(lst, is_home: bool) -> str:
            w = d = l = 0
            for m in lst:
                gs = m.home_score if is_home else m.away_score
                gc = m.away_score if is_home else m.home_score
                if gs is None or gc is None:
                    continue
                if gs > gc:
                    w += 1
                elif gs == gc:
                    d += 1
                else:
                    l += 1
            return f"{w}W-{d}D-{l}L"

        return TeamForm(
            team_name=team_name,
            results=results,
            goals_scored=scored,
            goals_conceded=conceded,
            home_record=record(home_matches, True),
            away_record=record(away_matches, False),
        )

    # ── Head-to-Head ──────────────────────────────────────────────────────────

    def get_head_to_head(self, fixture_id: int,
                          home_team: str, away_team: str,
                          limit: int = 5) -> HeadToHead:
        """Fetch H2H from a fixture's head2head data."""
        data = self._get(f"matches/{fixture_id}/head2head",
                         params={"limit": limit})
        meetings = []
        hw = d = aw = 0
        total_goals = 0

        for m in data.get("matches", []):
            hs = m.get("score", {}).get("fullTime", {}).get("home")
            as_ = m.get("score", {}).get("fullTime", {}).get("away")
            if hs is None or as_ is None:
                continue
            dt = m.get("utcDate", "")[:10]
            meetings.append({
                "date": dt,
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_score": hs,
                "away_score": as_,
                "competition": m.get("competition", {}).get("name", ""),
            })
            total_goals += hs + as_
            if m["homeTeam"]["name"] == home_team:
                if hs > as_:
                    hw += 1
                elif hs == as_:
                    d += 1
                else:
                    aw += 1
            else:
                if as_ > hs:
                    hw += 1
                elif hs == as_:
                    d += 1
                else:
                    aw += 1

        total = len(meetings)
        return HeadToHead(
            home_team=home_team,
            away_team=away_team,
            meetings=meetings,
            home_wins=hw,
            draws=d,
            away_wins=aw,
            avg_goals=round(total_goals / total, 2) if total else 0.0,
        )

    # ── Scorers (for narrative) ───────────────────────────────────────────────

    def get_top_scorers(self, competition_code: str,
                         limit: int = 5) -> list[dict]:
        data = self._get(f"competitions/{competition_code}/scorers",
                         params={"limit": limit})
        scorers = []
        for s in data.get("scorers", [])[:limit]:
            scorers.append({
                "name": s.get("player", {}).get("name", ""),
                "team": s.get("team", {}).get("name", ""),
                "goals": s.get("goals", 0),
                "assists": s.get("assists", 0),
            })
        return scorers
