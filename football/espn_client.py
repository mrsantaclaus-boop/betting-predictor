"""
football/espn_client.py — Client for ESPN's public soccer API.

No API key required. Covers UEFA Europa Conference League.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import requests

from .models import Fixture

logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

ESPN_LEAGUES: dict[str, tuple[str, str]] = {
    "ECL":   ("UEFA.EUROPA.CONF",  "UEFA Conference League"),
    "BSA":   ("BRA.1",             "Brasileirao Serie A"),
    "WC":    ("FIFA.WORLD",        "FIFA World Cup"),
    "WCQA":  ("CONMEBOL.WORLD",    "WCQ CONMEBOL"),
    "WCQC":  ("CONCACAF.WORLD",    "WCQ CONCACAF"),
    "WCQAS": ("AFC.WORLD",         "WCQ Asia"),
    "WCQAF": ("CAF.WORLD",         "WCQ Africa"),
}

_STATUS_MAP = {
    "STATUS_SCHEDULED":   "SCHEDULED",
    "STATUS_IN_PROGRESS": "IN_PLAY",
    "STATUS_FINAL":       "FINISHED",
    "STATUS_FULL_TIME":   "FINISHED",
    "STATUS_POSTPONED":   "POSTPONED",
    "STATUS_CANCELED":    "CANCELLED",
}


class EspnClient:
    """Fetches fixture data from ESPN's public API (no key required)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def get_upcoming_fixtures(self, competition_code: str,
                               days_ahead: int = 14) -> list[Fixture]:
        league_info = ESPN_LEAGUES.get(competition_code)
        if not league_info:
            return []
        league_slug, comp_name = league_info

        today = datetime.now(timezone.utc).date()
        end_date = today + timedelta(days=days_ahead)
        date_range = f"{today.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

        try:
            url = f"{BASE_URL}/{league_slug}/scoreboard"
            resp = self.session.get(url, params={"dates": date_range}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_events(data, competition_code, comp_name)
        except Exception as e:
            logger.warning("ESPN fetch failed for %s: %s", competition_code, e)
            return []

    def _parse_events(self, data: dict, competition_code: str,
                       comp_name: str) -> list[Fixture]:
        fixtures = []
        for event in data.get("events", []):
            try:
                competition = event.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])

                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                date_str = event.get("date", "")
                match_dt = (
                    datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if date_str else datetime.now(timezone.utc)
                )

                status_name = competition.get("status", {}).get("type", {}).get("name", "")
                status = _STATUS_MAP.get(status_name, "SCHEDULED")

                home_score = home.get("score")
                away_score = away.get("score")

                notes = event.get("notes", [])
                stage = notes[0].get("text", "") if notes else None

                fixtures.append(Fixture(
                    fixture_id=int(event["id"]),
                    competition=comp_name,
                    competition_code=competition_code,
                    home_team=home["team"]["displayName"],
                    home_team_id=int(home["team"]["id"]),
                    away_team=away["team"]["displayName"],
                    away_team_id=int(away["team"]["id"]),
                    match_date=match_dt,
                    status=status,
                    home_score=int(home_score) if home_score is not None else None,
                    away_score=int(away_score) if away_score is not None else None,
                    stage=stage or None,
                ))
            except (KeyError, ValueError, IndexError) as e:
                logger.debug("Skip malformed ESPN event: %s", e)

        return fixtures
