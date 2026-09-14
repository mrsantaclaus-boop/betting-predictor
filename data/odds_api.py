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
  - totals        → Over/Under 2.5 goals
  - btts          → Both Teams To Score, via a separate per-event request
                     (get_fixture_odds only — the batch /sports/{sport}/odds
                     endpoint rejects "btts" with INVALID_MARKET if bundled
                     with h2h/totals; confirmed live 2026-09-14). Best-effort:
                     null if the follow-up call fails, h2h/totals unaffected.

Corners/cards: no known market key on The Odds API for soccer (its
public docs don't list one); "team_totals" (each team's own goal total,
mentioned in an earlier version of this docstring) is a different
market and was never actually corners/cards data. Use
probe_event_markets() against a real event_id to test a market key
live before assuming either way.
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
    "SA":    "soccer_italy_serie_a",
    "SB":    "soccer_italy_serie_b",
    "CL":    "soccer_uefa_champs_league",
    "EL":    "soccer_uefa_europa_league",
    "ECL":   "soccer_uefa_europa_conference_league",
    "USC":   "soccer_uefa_super_cup",
    "WC":    "soccer_fifa_world_cup",
    "WCQE":  "soccer_fifa_world_cup_qualifiers_europe",
    "WCQA":  "soccer_fifa_world_cup_qualifiers_south_america",
    "WCQC":  "soccer_concacaf_world_cup_qualifying",
    "WCQAS": "soccer_afc_world_cup_qualifying",
    "WCQAF": "soccer_caf_world_cup_qualifying",
    "BSA":   "soccer_brazil_campeonato",
}
# NOTE: USC, WCQC, WCQAS, WCQAF have no confirmed match in The Odds API's
# sports catalog (checked via GET /api/odds/sports) — likely not covered by
# the provider at all, or only listed during an active qualifying window.
# Left as best-guess keys; they'll surface a clear "Unknown sport" error
# rather than silently returning no odds if still wrong.

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

        result = self._parse_event(event)

        # BTTS isn't offered by the batch /sports/{sport}/odds endpoint used
        # above (confirmed: INVALID_MARKET "not supported by this endpoint"),
        # but The Odds API does offer it per-event. Best-effort follow-up
        # call — on any failure we just keep btts_yes/btts_no null rather
        # than losing the h2h/totals odds we already have.
        event_id = event.get("id", "")
        if event_id:
            btts_event = self._get_event_odds(sport, event_id, markets=["btts"])
            if isinstance(btts_event, dict) and "error" not in btts_event:
                btts_parsed = self._parse_event(btts_event)
                btts_consensus = btts_parsed.get("consensus", {})
                if btts_consensus.get("btts_yes") or btts_consensus.get("btts_no"):
                    result["consensus"]["btts_yes"] = btts_consensus.get("btts_yes")
                    result["consensus"]["btts_no"] = btts_consensus.get("btts_no")
                    for bname, book_odds in btts_parsed.get("bookmakers", {}).items():
                        result["bookmakers"].setdefault(bname, {}).update(
                            {k: v for k, v in book_odds.items() if k in ("btts_yes", "btts_no")}
                        )

        return result

    def probe_event_markets(self, competition_code: str, event_id: str,
                             markets: list[str]) -> dict:
        """Diagnostic: raw per-event odds request for arbitrary market keys.

        Used to check, against the provider's real behavior rather than
        guesswork, whether a market (e.g. a corners/cards line) exists at
        all for this event — an INVALID_MARKET error names the rejected
        key, a real market returns actual bookmaker prices.
        """
        sport = SPORT_KEYS.get(competition_code)
        if not sport:
            return {"error": f"Unknown competition: {competition_code}"}
        return self._get_event_odds(sport, event_id, markets)

    def list_sports(self) -> list[dict] | dict:
        """Return the full list of sport keys The Odds API currently recognizes.

        Diagnostic helper — used to verify/correct entries in SPORT_KEYS
        rather than guessing at the provider's key naming.
        """
        url = f"{BASE_URL}/sports"
        try:
            resp = self.session.get(
                url, params={"apiKey": self.api_key, "all": "true"}, timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            return {"error": str(e)}

    def get_all_odds(self, competition_code: str) -> list[dict] | dict:
        """Return odds for all upcoming fixtures in a competition.

        Returns the error dict as-is on failure (same shape as
        get_fixture_odds) instead of silently returning an empty list —
        an empty list must mean "no events currently listed", not "the
        request failed", otherwise a real API error (bad key, quota,
        sport not yet activated) looks identical to a genuine market gap.
        """
        sport = SPORT_KEYS.get(competition_code)
        if not sport:
            return {"error": f"Unknown competition: {competition_code}"}

        odds_data = self._get_odds(sport, markets=["h2h", "totals"])
        if isinstance(odds_data, dict) and "error" in odds_data:
            return odds_data

        return [self._parse_event(e) for e in odds_data]

    @property
    def quota_remaining(self) -> Optional[int]:
        return self._quota_remaining

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_odds(self, sport: str, markets: list[str]) -> list | dict:
        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }
        return self._request(url, params)

    def _get_event_odds(self, sport: str, event_id: str, markets: list[str]) -> dict:
        """Per-event odds — some markets (e.g. btts) are rejected by the
        batch /sports/{sport}/odds endpoint but are available here."""
        url = f"{BASE_URL}/sports/{sport}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
        }
        return self._request(url, params)

    def _request(self, url: str, params: dict) -> list | dict:
        global _last_call
        elapsed = time.time() - _last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

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
            body = (e.response.text or "")[:300]
            markets = params.get("markets", "")
            if code == 401:
                return {"error": "Invalid ODDS_API_KEY", "detail": body}
            if code == 422:
                logger.error("Odds API 422 for %s (markets=%s): %s", url, markets, body)
                return {"error": "Not available on this plan/endpoint", "detail": body}
            logger.error("Odds API error %d for %s (markets=%s): %s", code, url, markets, body)
            return {"error": f"HTTP {code}", "detail": body}
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
        # Same team, different name on football-data.org vs The Odds API —
        # without this, the keyword-intersection match below finds zero
        # overlap and the fixture silently gets no odds (e.g. Inter's
        # matches never had a live price and were never proposed as bets).
        aliases = {"internazionale": "inter", "milano": "milan"}
        import re
        words = re.sub(r"[^a-z0-9\s]", "", name.lower()).split()
        return {aliases.get(w, w) for w in words if w not in stopwords and len(w) > 2}
