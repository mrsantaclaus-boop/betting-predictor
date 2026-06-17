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
    "WCQE":  ("UEFA.WORLD",        "WCQ Europe"),
}

# Competitions where all matches are played at neutral venues
_NEUTRAL_VENUE_CODES = {"WC"}

def _safe_int(val) -> int | None:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val: str) -> float:
    try:
        return float(val.replace("%", "").replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _acc(stats: dict, team: str, scored: int, conceded: int, box: dict) -> None:
    """Accumulate one match worth of stats for a team."""
    if team not in stats:
        stats[team] = {"scored": [], "conceded": [], "shots": [],
                       "shots_ot": [], "corners": [], "yellow": [],
                       "red": [], "xg": [], "xga": []}
    s = stats[team]
    s["scored"].append(scored)
    s["conceded"].append(conceded)
    s["shots"].append(_safe_float(box.get("totalShots", "0")))
    s["shots_ot"].append(_safe_float(box.get("shotsOnGoal", "0")))
    s["corners"].append(_safe_float(box.get("cornerKicks", "0")))
    s["yellow"].append(_safe_float(box.get("yellowCards", "0")))
    s["red"].append(_safe_float(box.get("redCards", "0")))
    # ESPN sometimes exposes xG under these keys
    xg_val = _safe_float(box.get("expectedGoals", box.get("xG", "0")))
    s["xg"].append(xg_val)


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

    def get_wc_team_stats(self, competition_code: str = "WC") -> dict:
        """
        Aggregate per-team stats from all completed WC matches via ESPN summary API.
        Returns {team_name: {"scored": [...], "conceded": [...], "shots": [...],
                              "corners": [...], "yellow": [...], "red": [...]}}
        Caller converts lists to per-game averages.
        """
        league_info = ESPN_LEAGUES.get(competition_code)
        if not league_info:
            return {}
        league_slug = league_info[0]

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=60)
        date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

        try:
            url = f"{BASE_URL}/{league_slug}/scoreboard"
            resp = self.session.get(url, params={"dates": date_range}, timeout=15)
            resp.raise_for_status()
            events = resp.json().get("events", [])
        except Exception as e:
            logger.warning("ESPN WC scoreboard failed: %s", e)
            return {}

        stats: dict[str, dict] = {}

        for event in events:
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
                continue

            event_id = event.get("id")
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue

            home_score = _safe_int(home.get("score"))
            away_score = _safe_int(away.get("score"))
            if home_score is None or away_score is None:
                continue

            # Fetch detailed match stats
            try:
                import time as _time
                _time.sleep(1.5)  # polite rate limit
                sum_url = f"{BASE_URL}/{league_slug}/summary"
                sr = self.session.get(sum_url, params={"event": event_id}, timeout=15)
                sr.raise_for_status()
                summary = sr.json()
            except Exception as e:
                logger.debug("ESPN summary failed for event %s: %s", event_id, e)
                # Fall back to goals only
                _acc(stats, home["team"]["displayName"], home_score, away_score, {})
                _acc(stats, away["team"]["displayName"], away_score, home_score, {})
                continue

            box_teams = summary.get("boxscore", {}).get("teams", [])
            box_map: dict[str, dict] = {}
            for bt in box_teams:
                name = bt.get("team", {}).get("displayName", "")
                if name:
                    box_map[name] = {s["name"]: s.get("displayValue", "0")
                                     for s in bt.get("statistics", [])}

            home_name = home["team"]["displayName"]
            away_name = away["team"]["displayName"]
            _acc(stats, home_name, home_score, away_score, box_map.get(home_name, {}))
            _acc(stats, away_name, away_score, home_score, box_map.get(away_name, {}))

        return stats

    def get_competition_results(self, competition_code: str,
                                days_back: int = 365) -> list[Fixture]:
        """Return recently FINISHED fixtures going back N days."""
        league_info = ESPN_LEAGUES.get(competition_code)
        if not league_info:
            return []
        league_slug, comp_name = league_info

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

        try:
            url = f"{BASE_URL}/{league_slug}/scoreboard"
            resp = self.session.get(url, params={"dates": date_range}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            fixtures = self._parse_events(data, competition_code, comp_name)
            return [f for f in fixtures if f.status == "FINISHED"
                    and f.home_score is not None and f.away_score is not None]
        except Exception as e:
            logger.warning("ESPN results fetch failed for %s: %s", competition_code, e)
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
                    is_neutral=competition_code in _NEUTRAL_VENUE_CODES,
                ))
            except (KeyError, ValueError, IndexError) as e:
                logger.debug("Skip malformed ESPN event: %s", e)

        return fixtures
