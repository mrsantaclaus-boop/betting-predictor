"""
data/odds_api.py — The Odds API client (https://the-odds-api.com)

Free tier: 500 requests/month.
Register at https://the-odds-api.com/#get-access

Fetches live bookmaker odds for:
  - Serie A (soccer_italy_serie_a)
  - Champions League (soccer_uefa_champions_league)
  - FIFA World Cup (soccer_fifa_world_cup)
  - WCQ Europe (soccer_eu_world_cup_qualification)
  - WCQ Americas (soccer_conmebol_world_cup_qualifying)
  - WCQ CONCACAF (soccer_concacaf_world_cup_qualifying)
  - WCQ Asia (soccer_afc_world_cup_qualifying)
  - WCQ Africa (soccer_caf_world_cup_qualifying)

Markets covered:
  - h2h          → 1X2 match result
  - totals        → Over/Under goals
  - btts          → Both Teams To Score (where available)
  - team_totals   → corners / cards (where available)
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "SA":   "soccer_italy_serie_a",
    "CL":   "soccer_uefa_champions_league",
    "WC":   "soccer_fifa_world_cup",
    "WCQE": "soccer_eu_world_cup_qualification",
    "WCQA": "soccer_conmebol_world_cup_qualifying",
    "WCQC": "soccer_concacaf_world_cup_qualifying",
    "WCQAS": "soccer_afc_world_cup_qualifying",
    "WCQAF": "soccer_caf_world_cup_qualifying",
}

# Bookmakers to include (shown in the UI)
PREFERRED_BOOKS = [
    "bet365", "williamhill", "unibet", "bwin",
    "betfair", "pinnacle", "1xbet", "betway",
]

# Polite rate-limiting
_MIN_INTERVAL = 2.0
_last_call: float = 0.0


class OddsAPIClient:
    """Fetches live odds from The Odds API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            logger.warning("ODDS_API_KEY not set — odds fetch will fail.")
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"
        self._quota_remaining: Optional[int] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_fixture_odds(
        self,
        competition_code: str,
        home_team: str,
        away_team: str,
    ) -> dict:
        """
        Return aggregated odds for a specific fixture.
        Matches by team name (fuzzy).

        Returns:
        {
          "match": "Home vs Away",
          "bookmakers": {
            "bet365": {"home": 1.85, "draw": 3.50, "away": 4.20,
                       "over_2_5": 1.90, "under_2_5": 1.95,
                       "btts_yes": 1.75, "btts_no": 2.05},
            ...
          },
          "consensus": { ... best-odds across books ... },
          "quota_remaining": N,
        }
        """
        sport = SPORT_KEYS.get(competition_code)
        if not sport:
            return {"error": f"Unknown competition: {competition_code}"}

        odds_data = self._get_odds(sport, markets=["h2h", "totals"])
        if "error" in odds_data:
            return odds_data

        # Find the matching event
        event = self._find_event(odds_data, home_team, away_team)
        if not event:
            return {"error": f"Match {home_team} vs {away_team} not found in live odds"}

        return self._parse_event(event)

    def get_all_odds(self, competition_code: str) -> list[dict]:
        """Return odds for all upcoming fixtures in a competition."""
        sport = SPORT_KEYS.get(competition_code)
        if not sport:
            return []

        odds_data = self._get_odds(sport, markets=["h2h", "totals"])
        if isinstance(odds_data, dict) and "error" in odds_data:
            return []

        return [self._parse_event(e) for e in odds_data]

    @property
    def quota_remaining(self) -> Optional[int]:
        return self._quota_remaining

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_odds(self, sport: str, markets: list[str]) -> list | dict:
        global _last_call
        elapsed = time.time() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }
        url = f"{BASE_URL}/sports/{sport}/odds"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            _last_call = time.time()

            # Track quota
            remaining = resp.headers.get("x-requests-remaining")
            if remaining:
                self._quota_remaining = int(remaining)

            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            code = e.response.status_code
            if code == 401:
                return {"error": "Invalid ODDS_API_KEY"}
            if code == 422:
                return {"error": f"Sport {sport} not available on free tier"}
            logger.error("Odds API error %d for %s", code, sport)
            return {"error": f"HTTP {code}"}
        except requests.RequestException as e:
            logger.error("Odds API request failed: %s", e)
            return {"error": str(e)}

    def _find_event(self, events: list, home_team: str, away_team: str) -> Optional[dict]:
        home_kw = self._keywords(home_team)
        away_kw = self._keywords(away_team)
        for event in events:
            ht = self._keywords(event.get("home_team", ""))
            at = self._keywords(event.get("away_team", ""))
            if (home_kw & ht) and (away_kw & at):
                return event
            if (home_kw & at) and (away_kw & ht):
                return event  # reversed
        return None

    def _parse_event(self, event: dict) -> dict:
        result: dict = {
            "event_id": event.get("id", ""),
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "commence_time": event.get("commence_time", ""),
            "bookmakers": {},
            "consensus": {},
        }

        best_home = best_draw = best_away = 0.0
        best_over25 = best_under25 = 0.0
        best_btts_yes = best_btts_no = 0.0

        for book in event.get("bookmakers", []):
            bname = book.get("key", "")
            if PREFERRED_BOOKS and bname not in PREFERRED_BOOKS:
                continue

            book_odds: dict = {}
            for market in book.get("markets", []):
                mkey = market.get("key", "")
                outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

                if mkey == "h2h":
                    book_odds["home"] = outcomes.get(event["home_team"], 0.0)
                    book_odds["draw"] = outcomes.get("Draw", 0.0)
                    book_odds["away"] = outcomes.get(event["away_team"], 0.0)
                    best_home = max(best_home, book_odds.get("home", 0))
                    best_draw = max(best_draw, book_odds.get("draw", 0))
                    best_away = max(best_away, book_odds.get("away", 0))

                elif mkey == "totals":
                    for o in market.get("outcomes", []):
                        if o.get("name") == "Over" and abs(o.get("point", 0) - 2.5) < 0.01:
                            book_odds["over_2_5"] = o["price"]
                            best_over25 = max(best_over25, o["price"])
                        elif o.get("name") == "Under" and abs(o.get("point", 0) - 2.5) < 0.01:
                            book_odds["under_2_5"] = o["price"]
                            best_under25 = max(best_under25, o["price"])

                elif mkey == "btts":
                    book_odds["btts_yes"] = outcomes.get("Yes", 0.0)
                    book_odds["btts_no"] = outcomes.get("No", 0.0)
                    best_btts_yes = max(best_btts_yes, book_odds.get("btts_yes", 0))
                    best_btts_no = max(best_btts_no, book_odds.get("btts_no", 0))

            if book_odds:
                result["bookmakers"][bname] = book_odds

        result["consensus"] = {
            "home": round(best_home, 2),
            "draw": round(best_draw, 2),
            "away": round(best_away, 2),
            "over_2_5": round(best_over25, 2),
            "under_2_5": round(best_under25, 2),
            "btts_yes": round(best_btts_yes, 2) if best_btts_yes else None,
            "btts_no": round(best_btts_no, 2) if best_btts_no else None,
        }
        result["quota_remaining"] = self._quota_remaining
        return result

    @staticmethod
    def _keywords(name: str) -> set[str]:
        stopwords = {"fc", "ac", "as", "ss", "afc", "cf", "united",
                     "city", "sport", "club", "calcio"}
        import re
        words = re.sub(r"[^a-z0-9\s]", "", name.lower()).split()
        return {w for w in words if w not in stopwords and len(w) > 2}
