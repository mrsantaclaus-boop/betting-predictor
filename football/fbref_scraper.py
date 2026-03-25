"""
fbref_scraper.py — Scrapes FBref.com for detailed team statistics.

Provides: xG, shots, corners, yellow cards, red cards per game.
No API key required — public data with respectful rate limiting.

FBref competition IDs:
  - Serie A  : 11
  - Champions League: 8
"""

from __future__ import annotations

import time
import logging
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .models import TeamStats

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://fbref.com/",
}
_DELAY = 4.0   # seconds between requests (be polite)
_last_req: float = 0.0

FBREF_COMPETITIONS = {
    "SA":   {"id": "11",  "slug": "Serie-A"},
    "CL":   {"id": "8",   "slug": "Champions-League"},
    "WC":   {"id": "1",   "slug": "World-Cup"},
    "WCQE": {"id": "680", "slug": "UEFA-World-Cup-Qualifying-UEFA"},
    "WCQA": {"id": "22",  "slug": "CONMEBOL-World-Cup-Qualifying"},
    "WCQC": {"id": "30",  "slug": "CONCACAF-World-Cup-Qualifying"},
    "WCQAS": {"id": "36", "slug": "AFC-Asian-Qualifiers-World-Cup"},
    "WCQAF": {"id": "46", "slug": "African-World-Cup-Qualifying"},
}


def _get(url: str) -> Optional[BeautifulSoup]:
    global _last_req
    elapsed = time.time() - _last_req
    if elapsed < _DELAY:
        time.sleep(_DELAY - elapsed)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        _last_req = time.time()
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        logger.warning("FBref fetch failed for %s: %s", url, e)
        return None


class FBrefScraper:
    """Scrapes per-team statistics from FBref season pages."""

    def __init__(self):
        self._cache: dict[str, list[TeamStats]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_team_stats(self, competition_code: str,
                        season: str = "") -> list[TeamStats]:
        """
        Return a list of TeamStats for all teams in the competition.
        Results are cached per competition_code.
        """
        cache_key = competition_code
        if cache_key in self._cache:
            return self._cache[cache_key]

        comp = FBREF_COMPETITIONS.get(competition_code)
        if not comp:
            logger.warning("Unknown competition code: %s", competition_code)
            return []

        url = f"https://fbref.com/en/comps/{comp['id']}/{comp['slug']}-Stats"
        soup = _get(url)
        if not soup:
            return []

        stats = self._parse_squad_stats(soup, competition_code)
        self._cache[cache_key] = stats
        return stats

    def get_team_stats_by_name(self, team_name: str,
                                competition_code: str) -> Optional[TeamStats]:
        """Find a specific team's stats by name (fuzzy match)."""
        all_stats = self.get_team_stats(competition_code)
        name_lower = team_name.lower()

        # Exact match first
        for s in all_stats:
            if s.team_name.lower() == name_lower:
                return s

        # Partial / keyword match
        keywords = self._name_keywords(team_name)
        for s in all_stats:
            s_keywords = self._name_keywords(s.team_name)
            if keywords & s_keywords:
                return s

        logger.warning("Team '%s' not found in FBref %s stats", team_name, competition_code)
        return None

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_squad_stats(self, soup: BeautifulSoup,
                            competition_code: str) -> list[TeamStats]:
        """Parse the main squad stats table on the competition page."""
        stats_list = []

        # FBref has multiple stats tables; we need "Squad Standard Stats" and
        # "Squad Miscellaneous Stats" for cards/corners
        standard = self._parse_standard_table(soup, competition_code)
        misc = self._parse_misc_table(soup, competition_code)

        # Merge misc data into standard
        misc_map = {s.team_name: s for s in misc}
        for s in standard:
            m = misc_map.get(s.team_name)
            if m:
                s.yellow_cards_pg = m.yellow_cards_pg
                s.red_cards_pg = m.red_cards_pg
                s.fouls_committed_pg = m.fouls_committed_pg

        # Fetch corners from passing table
        corners = self._parse_corners(soup, competition_code)
        corners_map = {name: val for name, val in corners}
        for s in standard:
            if s.team_name in corners_map:
                s.corners_pg = corners_map[s.team_name]

        return standard

    def _parse_standard_table(self, soup: BeautifulSoup,
                               competition_code: str) -> list[TeamStats]:
        stats_list = []
        table = soup.find("table", {"id": re.compile(r"stats_squads_standard_for")})
        if not table:
            # Try alternative table ID patterns
            table = soup.find("table", id=lambda x: x and "standard" in x)
        if not table:
            logger.debug("Standard stats table not found")
            return []

        tbody = table.find("tbody")
        if not tbody:
            return []

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            cells = row.find_all(["td", "th"])
            if len(cells) < 10:
                continue

            team_cell = row.find("td", {"data-stat": "team"})
            if not team_cell:
                continue
            team_name = team_cell.get_text(strip=True)
            if not team_name:
                continue

            def cell(stat: str) -> float:
                td = row.find("td", {"data-stat": stat})
                if td:
                    txt = td.get_text(strip=True).replace(",", "")
                    try:
                        return float(txt)
                    except ValueError:
                        return 0.0
                return 0.0

            gp = int(cell("games") or cell("games_equiv") or 1)

            stats_list.append(TeamStats(
                team_name=team_name,
                competition=competition_code,
                games_played=gp,
                goals_scored_pg=round(cell("goals") / max(gp, 1), 2),
                goals_conceded_pg=0.0,  # filled from misc/defense
                xg_pg=round(cell("xg") / max(gp, 1), 2),
                shots_pg=round(cell("shots") / max(gp, 1), 2),
                shots_on_target_pg=round(cell("shots_on_target") / max(gp, 1), 2),
            ))

        return stats_list

    def _parse_misc_table(self, soup: BeautifulSoup,
                           competition_code: str) -> list[TeamStats]:
        """Parse yellow/red cards and fouls from misc stats table."""
        result = []
        table = soup.find("table", id=re.compile(r"stats_squads_misc_for"))
        if not table:
            return []

        tbody = table.find("tbody")
        if not tbody:
            return []

        for row in tbody.find_all("tr"):
            team_cell = row.find("td", {"data-stat": "team"})
            if not team_cell:
                continue
            team_name = team_cell.get_text(strip=True)
            if not team_name:
                continue

            def cell(stat: str) -> float:
                td = row.find("td", {"data-stat": stat})
                if td:
                    try:
                        return float(td.get_text(strip=True).replace(",", ""))
                    except ValueError:
                        return 0.0
                return 0.0

            gp = int(cell("games") or 1)
            result.append(TeamStats(
                team_name=team_name,
                competition=competition_code,
                games_played=gp,
                yellow_cards_pg=round(cell("cards_yellow") / max(gp, 1), 2),
                red_cards_pg=round(cell("cards_red") / max(gp, 1), 2),
                fouls_committed_pg=round(cell("fouls") / max(gp, 1), 2),
            ))

        return result

    def _parse_corners(self, soup: BeautifulSoup,
                        competition_code: str) -> list[tuple]:
        """Extract corner kicks per game from passing stats table."""
        corners = []
        table = soup.find("table", id=re.compile(r"stats_squads_passing_for"))
        if not table:
            return corners

        tbody = table.find("tbody")
        if not tbody:
            return corners

        for row in tbody.find_all("tr"):
            team_cell = row.find("td", {"data-stat": "team"})
            if not team_cell:
                continue
            team_name = team_cell.get_text(strip=True)

            gp_cell = row.find("td", {"data-stat": "games"})
            ck_cell = row.find("td", {"data-stat": "corner_kicks"})
            if not gp_cell or not ck_cell:
                continue
            try:
                gp = int(gp_cell.get_text(strip=True))
                ck = float(ck_cell.get_text(strip=True).replace(",", ""))
                corners.append((team_name, round(ck / max(gp, 1), 2)))
            except (ValueError, ZeroDivisionError):
                continue

        return corners

    # ── BTTS & clean sheets (from schedule page) ──────────────────────────────

    def get_btts_and_clean_sheets(self, competition_code: str) -> dict[str, dict]:
        """
        Scrape the full schedule to compute BTTS rate and clean sheets per team.
        Returns: {team_name: {"btts": int, "clean_sheets": int, "games": int}}
        """
        comp = FBREF_COMPETITIONS.get(competition_code)
        if not comp:
            return {}

        url = (
            f"https://fbref.com/en/comps/{comp['id']}/"
            f"schedule/{comp['slug']}-Scores-and-Fixtures"
        )
        soup = _get(url)
        if not soup:
            return {}

        table = soup.find("table", id=re.compile(r"sched_"))
        if not table:
            return {}

        team_data: dict[str, dict] = {}

        for row in table.find_all("tr"):
            score_cell = row.find("td", {"data-stat": "score"})
            if not score_cell:
                continue
            score_txt = score_cell.get_text(strip=True)
            m = re.match(r"(\d+)[–\-](\d+)", score_txt)
            if not m:
                continue

            hg, ag = int(m.group(1)), int(m.group(2))

            home_cell = row.find("td", {"data-stat": "home_team"})
            away_cell = row.find("td", {"data-stat": "away_team"})
            if not home_cell or not away_cell:
                continue

            home = home_cell.get_text(strip=True)
            away = away_cell.get_text(strip=True)
            btts = 1 if hg > 0 and ag > 0 else 0

            for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
                if team not in team_data:
                    team_data[team] = {"btts": 0, "clean_sheets": 0, "games": 0}
                team_data[team]["games"] += 1
                team_data[team]["btts"] += btts
                if ga == 0:
                    team_data[team]["clean_sheets"] += 1

        return team_data

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _name_keywords(name: str) -> set[str]:
        """Extract meaningful keywords from a team name for fuzzy matching."""
        stopwords = {"fc", "ac", "as", "ss", "afc", "cf", "inter",
                     "calcio", "sport", "club", "united", "city"}
        words = re.sub(r"[^a-z0-9\s]", "", name.lower()).split()
        return {w for w in words if w not in stopwords and len(w) > 2}
