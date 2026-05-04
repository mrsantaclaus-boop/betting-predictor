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

        # Enrich with home/away corner splits (2 extra requests, cached)
        try:
            homeaway = self._get_corners_homeaway(comp["id"], comp["slug"], competition_code)
            for s in stats:
                entry = homeaway.get(s.team_name)
                if not entry:
                    # Fuzzy match
                    kw = self._name_keywords(s.team_name)
                    for name, val in homeaway.items():
                        if kw & self._name_keywords(name):
                            entry = val
                            break
                if entry:
                    s.corners_home_pg, s.corners_away_pg = entry
        except Exception as e:
            logger.warning("Home/away corners fetch failed (non-fatal): %s", e)

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

    def _get_corners_homeaway(
        self, comp_id: str, slug: str, competition_code: str
    ) -> dict[str, tuple[float, float]]:
        """
        Fetch per-team corner averages split by venue from FBref home/away pages.
        Returns {team_name: (home_cpg, away_cpg)}.
        Pages: /comps/{id}/home/{slug}-Stats and /comps/{id}/away/{slug}-Stats
        """
        result: dict[str, tuple[float, float]] = {}

        home_url = f"https://fbref.com/en/comps/{comp_id}/home/{slug}-Stats"
        away_url = f"https://fbref.com/en/comps/{comp_id}/away/{slug}-Stats"

        home_soup = _get(home_url)
        home_map: dict[str, float] = {}
        if home_soup:
            for name, cpg in self._parse_corners(home_soup, competition_code):
                home_map[name] = cpg

        away_soup = _get(away_url)
        away_map: dict[str, float] = {}
        if away_soup:
            for name, cpg in self._parse_corners(away_soup, competition_code):
                away_map[name] = cpg

        for team in set(home_map) | set(away_map):
            result[team] = (home_map.get(team, 0.0), away_map.get(team, 0.0))

        logger.debug(
            "Home/away corners fetched for %s: %d teams", competition_code, len(result)
        )
        return result

    # ── Per-match corner counts ───────────────────────────────────────────────

    def get_match_corners(
        self,
        competition_code: str,
        home_team: str,
        away_team: str,
        date_str: str = "",
    ) -> dict[str, int]:
        """
        Scrape the FBref match report for a specific fixture to get actual corner counts.

        Strategy:
          1. Scrape the competition schedule page to find the match report URL.
          2. Scrape the match report page and parse corner kicks from team_stats_extra.

        Returns {"home": int, "away": int} or {} if not found / parsing fails.
        Uses 1–2 HTTP requests; respects the 4 s polite delay.
        """
        match_url = self._find_match_url(competition_code, home_team, away_team, date_str)
        if not match_url:
            logger.debug("FBref: no match URL found for %s vs %s", home_team, away_team)
            return {}
        return self._scrape_corners_from_report(match_url, home_team, away_team)

    def _find_match_url(
        self,
        competition_code: str,
        home_team: str,
        away_team: str,
        date_str: str = "",
    ) -> Optional[str]:
        """Return the FBref match report URL for the given fixture, or None."""
        comp = FBREF_COMPETITIONS.get(competition_code)
        if not comp:
            return None

        url = (
            f"https://fbref.com/en/comps/{comp['id']}/"
            f"schedule/{comp['slug']}-Scores-and-Fixtures"
        )
        soup = _get(url)
        if not soup:
            return None

        table = soup.find("table", id=re.compile(r"sched_"))
        if not table:
            return None

        home_kw = self._name_keywords(home_team)
        away_kw = self._name_keywords(away_team)

        for row in table.find_all("tr"):
            # Optional date pre-filter
            if date_str:
                date_cell = row.find("td", {"data-stat": "date"})
                if date_cell and date_str not in date_cell.get_text(strip=True):
                    continue

            home_cell  = row.find("td", {"data-stat": "home_team"})
            away_cell  = row.find("td", {"data-stat": "away_team"})
            score_cell = row.find("td", {"data-stat": "score"})

            if not all([home_cell, away_cell, score_cell]):
                continue

            h_kw = self._name_keywords(home_cell.get_text(strip=True))
            a_kw = self._name_keywords(away_cell.get_text(strip=True))

            if (home_kw & h_kw) and (away_kw & a_kw):
                link = score_cell.find("a")
                if link and link.get("href"):
                    href = link["href"]
                    if not href.startswith("http"):
                        href = "https://fbref.com" + href
                    return href

        return None

    def _scrape_corners_from_report(
        self, match_url: str, home_team: str = "", away_team: str = ""
    ) -> dict[str, int]:
        """
        Parse corner kicks from a FBref match report page.
        Tries team_stats_extra (preferred) then falls back to team_stats table.
        """
        soup = _get(match_url)
        if not soup:
            return {}

        # ── Approach 1: div#team_stats_extra ─────────────────────────────────
        # Structure: each <div> child has <p> label, <p> home_val, <p> away_val
        extra = soup.find("div", id="team_stats_extra")
        if extra:
            child_divs = extra.find_all("div", recursive=False)
            for div in child_divs:
                paras = div.find_all("p")
                if not paras:
                    continue
                label = paras[0].get_text(strip=True).lower()
                if "corner" in label and len(paras) >= 3:
                    try:
                        return {
                            "home": int(paras[1].get_text(strip=True)),
                            "away": int(paras[2].get_text(strip=True)),
                        }
                    except ValueError:
                        pass

        # ── Approach 2: div#team_stats table rows ────────────────────────────
        team_stats = soup.find("div", id="team_stats")
        if team_stats:
            for tr in team_stats.find_all("tr"):
                if "corner" not in tr.get_text().lower():
                    continue
                nums = []
                for td in tr.find_all("td"):
                    txt = td.get_text(strip=True).split("\n")[0].strip()
                    try:
                        nums.append(int(txt))
                    except ValueError:
                        continue
                if len(nums) >= 2:
                    return {"home": nums[0], "away": nums[1]}

        logger.debug("FBref: corners not found in match report %s", match_url)
        return {}

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
