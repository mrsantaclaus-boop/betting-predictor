"""
football/api_football_client.py — Client for api-football.com (v3)

Covers World Cup Qualifier competitions not available on the football-data.org
free tier.

Free tier: 100 requests/day.
Register at https://dashboard.api-football.com/register

Set API_FOOTBALL_KEY in your .env file.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

from .models import Fixture, Standing, InjuryReport

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"

# Competition code → (league_id, competition_name)
# IDs are for the 2026 World Cup qualifying cycle
WCQ_LEAGUES: dict[str, tuple[int, str]] = {
    "WC":    (1,  "FIFA World Cup"),
    "WCQE":  (32, "WCQ Europe"),
    "WCQA":  (34,  "WCQ Americas"),
    "WCQC":  (30,  "WCQ CONCACAF"),
    "WCQAS": (36,  "WCQ Asia"),
    "WCQAF": (29,  "WCQ Africa"),
}

# Status short codes → our canonical status
_STATUS_MAP = {
    "NS": "SCHEDULED",
    "TBD": "SCHEDULED",
    "1H": "IN_PLAY", "HT": "IN_PLAY", "2H": "IN_PLAY",
    "ET": "IN_PLAY", "BT": "IN_PLAY", "P": "IN_PLAY", "LIVE": "IN_PLAY",
    "FT": "FINISHED", "AET": "FINISHED", "PEN": "FINISHED",
    "PST": "POSTPONED", "CANC": "CANCELLED", "ABD": "CANCELLED",
}

# Polite rate limiting — free tier: 10 req/min
_MIN_INTERVAL = 7.0
_last_call: float = 0.0


class ApiFootballClient:
    """Fetches fixture and standings data from api-football.com."""

    def __init__(self):
        self._key = os.getenv("API_FOOTBALL_KEY", "")
        if not self._key:
            logger.warning("API_FOOTBALL_KEY not set — WCQ fixtures unavailable")

    def _get(self, path: str, params: dict | None = None) -> dict:
        global _last_call
        if not self._key:
            return {}

        elapsed = time.time() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        url = f"{BASE_URL}/{path.lstrip('/')}"
        headers = {
            "x-apisports-key": self._key,
            "Accept": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, params=params or {}, timeout=20)
            _last_call = time.time()
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error("API-Football HTTP error %s: %s", e.response.status_code, url)
            return {}
        except requests.RequestException as e:
            logger.error("API-Football request failed: %s", e)
            return {}

    def get_upcoming_fixtures(self, competition_code: str,
                               days_ahead: int = 14) -> list[Fixture]:
        """Return fixtures for a WCQ competition scheduled in the next N days."""
        league_info = WCQ_LEAGUES.get(competition_code)
        if not league_info:
            return []
        league_id, comp_name = league_info

        today = datetime.now(timezone.utc).date()
        to_date = today + timedelta(days=days_ahead)

        data = self._get("fixtures", {
            "league": league_id,
            "season": today.year,
            "from": today.isoformat(),
            "to": to_date.isoformat(),
        })
        return self._parse_fixtures(data, competition_code, comp_name)

    def get_standings(self, competition_code: str,
                       season: int | None = None) -> list[Standing]:
        """Return group/table standings for a WCQ competition."""
        league_info = WCQ_LEAGUES.get(competition_code)
        if not league_info:
            return []
        league_id, _ = league_info

        year = season or datetime.now(timezone.utc).year
        data = self._get("standings", {"league": league_id, "season": year})
        return self._parse_standings(data)

    def get_injuries(
        self,
        fixture_id: int,
        home_team_name: str,
        away_team_name: str,
    ) -> tuple[InjuryReport, InjuryReport]:
        """
        Fetch injury/suspension list for both teams in a fixture.

        Uses the /injuries endpoint filtered by fixture ID.
        Players with type "Questionable" go to doubtful; all others to unavailable.
        Returns (home_injuries, away_injuries).
        """
        data = self._get("injuries", {"fixture": fixture_id})
        home_ir = InjuryReport(team_name=home_team_name)
        away_ir = InjuryReport(team_name=away_team_name)

        home_key = home_team_name.lower()
        away_key = away_team_name.lower()

        for entry in data.get("response", []):
            try:
                player_name = entry["player"]["name"]
                team_name   = entry["team"]["name"].lower()
                injury_type = entry.get("injury", {}).get("type", "").lower()

                is_home = home_key in team_name or team_name in home_key
                ir = home_ir if is_home else away_ir

                if "questionable" in injury_type:
                    ir.doubtful.append(player_name)
                else:
                    ir.unavailable.append(player_name)
            except (KeyError, TypeError):
                continue

        return home_ir, away_ir

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse_fixtures(self, data: dict, competition_code: str,
                         comp_name: str) -> list[Fixture]:
        fixtures = []
        for entry in data.get("response", []):
            try:
                fix = entry["fixture"]
                teams = entry["teams"]
                goals = entry.get("goals", {})

                date_str = fix.get("date", "")
                match_dt = (
                    datetime.fromisoformat(date_str)
                    if date_str else datetime.now(timezone.utc)
                )

                raw_status = fix.get("status", {}).get("short", "NS")
                status = _STATUS_MAP.get(raw_status, "SCHEDULED")

                round_str = entry.get("league", {}).get("round", "")

                fixtures.append(Fixture(
                    fixture_id=fix["id"],
                    competition=comp_name,
                    competition_code=competition_code,
                    home_team=teams["home"]["name"],
                    home_team_id=teams["home"]["id"],
                    away_team=teams["away"]["name"],
                    away_team_id=teams["away"]["id"],
                    match_date=match_dt,
                    status=status,
                    home_score=goals.get("home"),
                    away_score=goals.get("away"),
                    stage=round_str or None,
                ))
            except (KeyError, ValueError) as e:
                logger.debug("Skip malformed API-Football fixture: %s", e)
        return fixtures

    def _parse_standings(self, data: dict) -> list[Standing]:
        rows = []
        # API-Football nests standings: response[0].league.standings[][]
        try:
            groups = data["response"][0]["league"]["standings"]
        except (IndexError, KeyError):
            return []

        # Flatten all groups (each group is a list of team rows)
        position_counter = 0
        for group in groups:
            for entry in group:
                position_counter += 1
                team = entry.get("team", {})
                all_ = entry.get("all", {})
                goals = all_.get("goals", {})
                gf = goals.get("for", 0) or 0
                ga = goals.get("against", 0) or 0
                rows.append(Standing(
                    position=entry.get("rank", position_counter),
                    team_name=team.get("name", ""),
                    team_id=team.get("id", 0),
                    played=all_.get("played", 0) or 0,
                    won=all_.get("win", 0) or 0,
                    drawn=all_.get("draw", 0) or 0,
                    lost=all_.get("lose", 0) or 0,
                    goals_for=gf,
                    goals_against=ga,
                    goal_difference=gf - ga,
                    points=entry.get("points", 0) or 0,
                    form=entry.get("form", "") or "",
                ))
        return rows
