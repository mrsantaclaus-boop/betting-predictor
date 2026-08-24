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
    "SA":    {"id": "11",  "slug": "Serie-A"},
    "SB":    {"id": "18",  "slug": "Serie-B"},
    "CL":    {"id": "8",   "slug": "Champions-League"},
    "EL":    {"id": "19",  "slug": "Europa-League"},
    "WC":    {"id": "1",   "slug": "World-Cup"},
    "WCQE":  {"id": "680", "slug": "UEFA-World-Cup-Qualifying-UEFA"},
    "WCQA":  {"id": "22",  "slug": "CONMEBOL-World-Cup-Qualifying"},
    "WCQC":  {"id": "30",  "slug": "CONCACAF-World-Cup-Qualifying"},
    "WCQAS": {"id": "36",  "slug": "AFC-Asian-Qualifiers-World-Cup"},
    "WCQAF": {"id": "46",  "slug": "African-World-Cup-Qualifying"},
    "BSA":   {"id": "24",  "slug": "Serie-A"},
    # USC (Super Cup) omitted — one-off match, no season team stats on FBref
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

        # Merge defensive stats (goals conceded, xGA) from the "against" table
        against = self._parse_against_table(soup, competition_code)
        against_map = {name: (gc_pg, xga_pg) for name, gc_pg, xga_pg in against}
        for s in standard:
            entry = against_map.get(s.team_name)
            if not entry:
                kw = self._name_keywords(s.team_name)
                for name, val in against_map.items():
                    if kw & self._name_keywords(name):
                        entry = val
                        break
            if entry:
                s.goals_conceded_pg, s.xga_pg = entry

        # Fetch corners from passing table
        corners = self._parse_corners(soup, competition_code)
        corners_map = {name: val for name, val in corners}
        for s in standard:
            if s.team_name in corners_map:
                s.corners_pg = corners_map[s.team_name]

        # Corners conceded — used for the attack/defense matchup adjustment
        # (a team's own corners_pg says nothing about how many corners its
        # opponents tend to win against it; this does).
        corners_against = self._parse_corners_against(soup, competition_code)
        corners_against_map = {name: val for name, val in corners_against}
        for s in standard:
            entry = corners_against_map.get(s.team_name)
            if entry is None:
                kw = self._name_keywords(s.team_name)
                for name, val in corners_against_map.items():
                    if kw & self._name_keywords(name):
                        entry = val
                        break
            if entry is not None:
                s.corners_against_pg = entry

        return standard

    def _parse_against_table(self, soup: BeautifulSoup,
                              competition_code: str) -> list[tuple]:
        """
        Parse goals conceded and xGA from the 'stats_squads_standard_against' table.
        Returns list of (team_name, goals_conceded_pg, xga_pg).
        """
        result = []
        table = soup.find("table", id=re.compile(r"stats_squads_standard_against"))
        if not table:
            return result
        tbody = table.find("tbody")
        if not tbody:
            return result
        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
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

            gp = int(cell("games") or 1)
            goals_conceded_pg = round(cell("goals") / max(gp, 1), 2)
            xga_pg = round(cell("xg") / max(gp, 1), 2)
            result.append((team_name, goals_conceded_pg, xga_pg))
        return result

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

    def _parse_corners_against(self, soup: BeautifulSoup,
                                competition_code: str) -> list[tuple]:
        """
        Extract corner kicks CONCEDED per game from the 'passing against' table
        (mirrors _parse_against_table's approach for goals conceded/xGA).
        Returns [] if the table/column isn't present — callers must treat this
        as optional data and fall back gracefully (same as every other
        FBref field in this scraper).
        """
        result = []
        table = soup.find("table", id=re.compile(r"stats_squads_passing_against"))
        if not table:
            return result
        tbody = table.find("tbody")
        if not tbody:
            return result

        for row in tbody.find_all("tr"):
            if row.get("class") and "thead" in row.get("class", []):
                continue
            team_cell = row.find("td", {"data-stat": "team"})
            if not team_cell:
                continue
            team_name = team_cell.get_text(strip=True)
            if not team_name:
                continue

            gp_cell = row.find("td", {"data-stat": "games"})
            ck_cell = row.find("td", {"data-stat": "corner_kicks"})
            if not gp_cell or not ck_cell:
                continue
            try:
                gp = int(gp_cell.get_text(strip=True))
                ck = float(ck_cell.get_text(strip=True).replace(",", ""))
                result.append((team_name, round(ck / max(gp, 1), 2)))
            except (ValueError, ZeroDivisionError):
                continue

        return result

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

    # ── Per-match corners + cards ──────────────────────────────────────────────

    def get_match_corners_and_cards(
        self,
        competition_code: str,
        home_team: str,
        away_team: str,
        date_str: str = "",
    ) -> tuple[dict[str, int], dict[str, int]]:
        """
        Scrape the FBref match report for a specific fixture to get actual
        corner and card counts, in a single page fetch.

        Strategy:
          1. Scrape the competition schedule page to find the match report URL.
          2. Scrape the match report page once and parse both corners
             (from team_stats_extra) and cards (summed from each team's
             player summary table).

        Returns (corners, cards):
          corners = {"home": int, "away": int} or {} if not found.
          cards   = {"home_yellow": int, "away_yellow": int,
                     "home_red": int, "away_red": int} or {} if not found.
        Uses 1–2 HTTP requests total; respects the 4 s polite delay.
        """
        match_url = self._find_match_url(competition_code, home_team, away_team, date_str)
        if not match_url:
            logger.warning("FBref: no match URL found for %s vs %s (%s, date=%s)",
                           home_team, away_team, competition_code, date_str)
            return {}, {}
        soup = _get(match_url)
        if not soup:
            return {}, {}
        return (self._parse_corners_from_soup(soup, match_url),
                self._parse_cards_from_soup(soup, match_url))

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
            logger.warning("FBref: no competition mapping for code '%s'", competition_code)
            return None

        url = (
            f"https://fbref.com/en/comps/{comp['id']}/"
            f"schedule/{comp['slug']}-Scores-and-Fixtures"
        )
        soup = _get(url)
        if not soup:
            return None  # _get() already logs the fetch failure

        table = soup.find("table", id=re.compile(r"sched_"))
        if not table:
            logger.warning("FBref: no schedule table found at %s", url)
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

    def _parse_corners_from_soup(self, soup, match_url: str = "") -> dict[str, int]:
        """
        Parse corner kicks from an already-fetched FBref match report page.
        Tries team_stats_extra (preferred) then falls back to team_stats table.
        """
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

        logger.warning("FBref: corners not found in match report %s", match_url)
        return {}

    def _parse_cards_from_soup(self, soup, match_url: str = "") -> dict[str, int]:
        """
        Parse total yellow/red cards per team from an already-fetched
        FBref match report page.

        Sums the 'cards_yellow' / 'cards_red' data-stat columns across each
        team's player summary table (id like "stats_<squad_id>_summary" —
        exactly two per match report, home team first, away team second,
        same data-stat convention already used for season-aggregate squad
        stats in _parse_misc_table above).
        """
        summary_tables = soup.find_all("table", id=re.compile(r"^stats_.+_summary$"))
        if len(summary_tables) < 2:
            logger.warning(
                "FBref: expected 2 player summary tables for cards, found %d in %s",
                len(summary_tables), match_url,
            )
            return {}

        totals = []
        for table in summary_tables[:2]:
            body = table.find("tbody")
            if not body:
                totals.append(None)
                continue
            yellow = red = 0
            for row in body.find_all("tr"):
                for stat, acc in (("cards_yellow", "yellow"), ("cards_red", "red")):
                    cell = row.find("td", {"data-stat": stat})
                    if cell is None:
                        continue
                    try:
                        n = int(cell.get_text(strip=True) or 0)
                    except ValueError:
                        continue
                    if acc == "yellow":
                        yellow += n
                    else:
                        red += n
            totals.append((yellow, red))

        if len(totals) != 2 or None in totals:
            logger.warning("FBref: could not parse both summary tables for cards in %s", match_url)
            return {}

        (h_yellow, h_red), (a_yellow, a_red) = totals
        return {
            "home_yellow": h_yellow, "away_yellow": a_yellow,
            "home_red": h_red, "away_red": a_red,
        }

    # ── Referee stats ─────────────────────────────────────────────────────────

    def get_referee_stats(self, competition_code: str) -> dict[str, float]:
        """
        Scrape per-referee yellow cards per game for a competition.
        Returns {referee_name: yellow_cards_pg}.
        Only available for competitions in FBREF_COMPETITIONS.
        """
        cache_key = f"referees:{competition_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        comp = FBREF_COMPETITIONS.get(competition_code)
        if not comp:
            return {}

        url = (
            f"https://fbref.com/en/comps/{comp['id']}/"
            f"referees/{comp['slug']}-Referees"
        )
        soup = _get(url)
        if not soup:
            return {}

        table = soup.find("table", id=re.compile(r"stats_referee"))
        if not table:
            logger.debug("FBref: no referee table found for %s", competition_code)
            return {}

        result: dict[str, float] = {}
        for row in table.find("tbody").find_all("tr"):
            name_cell = row.find("th", {"data-stat": "referee"})
            games_cell = row.find("td", {"data-stat": "games"})
            yellow_cell = row.find("td", {"data-stat": "cards_yellow"})
            if not name_cell or not games_cell or not yellow_cell:
                continue
            try:
                name = name_cell.get_text(strip=True)
                games = int(games_cell.get_text(strip=True) or 0)
                yellows = int(yellow_cell.get_text(strip=True) or 0)
                if games > 0 and name:
                    result[name] = round(yellows / games, 3)
            except (ValueError, AttributeError):
                continue

        self._cache[cache_key] = result  # type: ignore[assignment]
        logger.info("FBref referees loaded for %s: %d entries", competition_code, len(result))
        return result

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

    # ── WC qualifying fallback ────────────────────────────────────────────────

    _WCQ_CODES = ["WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF"]

    # Maps each likely WC 2026 participant to its confederation WCQ code so
    # get_wcq_stats() only needs ONE FBref request instead of up to five.
    _TEAM_CONFEDERATION: dict[str, str] = {
        # UEFA → WCQE
        **{t: "WCQE" for t in [
            "France", "England", "Germany", "Spain", "Portugal", "Netherlands",
            "Belgium", "Croatia", "Denmark", "Switzerland", "Austria", "Serbia",
            "Scotland", "Poland", "Turkey", "Hungary", "Romania", "Ukraine",
            "Czech Republic", "Slovakia", "Sweden", "Norway", "Finland", "Greece",
            "Albania", "Slovenia", "Montenegro", "Wales", "Ireland", "Georgia",
            "North Macedonia", "Israel", "Bosnia and Herzegovina", "Armenia",
            "Kosovo", "Luxembourg", "Iceland", "Northern Ireland", "Bulgaria",
            "Italy", "Russia",
        ]},
        # CONMEBOL → WCQA
        **{t: "WCQA" for t in [
            "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay",
            "Venezuela", "Chile", "Peru", "Bolivia",
        ]},
        # CONCACAF → WCQC
        **{t: "WCQC" for t in [
            "United States", "USA", "Mexico", "Canada", "Jamaica", "Panama",
            "Honduras", "Costa Rica", "El Salvador", "Trinidad and Tobago",
            "Haiti", "Cuba", "Guatemala", "Suriname", "Guyana", "Barbados",
            "Belize", "Antigua and Barbuda", "Saint Kitts and Nevis",
        ]},
        # AFC → WCQAS
        **{t: "WCQAS" for t in [
            "Japan", "South Korea", "Korea Republic", "Republic of Korea",
            "Iran", "IR Iran", "Australia", "Saudi Arabia", "Iraq", "Jordan",
            "Uzbekistan", "Qatar", "UAE", "United Arab Emirates", "China PR",
            "China", "Vietnam", "Indonesia", "Thailand", "Bahrain", "Oman",
            "Kuwait", "Syria", "Palestine", "New Zealand", "Kyrgyzstan", "Tajikistan",
        ]},
        # CAF → WCQAF
        **{t: "WCQAF" for t in [
            "Morocco", "Senegal", "Nigeria", "Cameroon", "Egypt", "Ivory Coast",
            "Côte d'Ivoire", "Tunisia", "Algeria", "DR Congo", "Mali", "Ghana",
            "South Africa", "Cape Verde", "Guinea", "Zambia", "Congo",
            "Gabon", "Tanzania", "Uganda", "Angola", "Mauritania", "Libya",
            "Zimbabwe", "Ethiopia", "Rwanda", "Comoros", "Burkina Faso",
            "Benin", "Togo", "Niger", "Gambia", "Malawi", "Equatorial Guinea",
            "Sudan", "Liberia", "Namibia", "Sierra Leone", "Mozambique",
        ]},
    }

    def get_wcq_stats(self, team_name: str) -> Optional[TeamStats]:
        """
        Search WCQ competitions for a team's stats.
        Uses a confederation map to make only one FBref request when the team
        is known, falling back to a full search for unrecognised names.
        Results are served from cache after the first fetch per competition.
        """
        kw = self._name_keywords(team_name)

        # Fast path: look up the team's confederation directly
        confederation = self._TEAM_CONFEDERATION.get(team_name)
        if not confederation:
            # Fuzzy lookup — use stripped keywords that exclude geographic generics
            # so "Korea Republic" doesn't incorrectly match "Czech Republic"
            _geo_stop = {"republic", "north", "south", "east", "west",
                         "central", "democratic", "island", "islands"}
            kw_strict = kw - _geo_stop
            for known_name, conf in self._TEAM_CONFEDERATION.items():
                known_strict = self._name_keywords(known_name) - _geo_stop
                if kw_strict and kw_strict & known_strict:
                    confederation = conf
                    break

        # Search the known confederation first (single FBref request if cached)
        search_order = (
            [confederation] + [c for c in self._WCQ_CODES if c != confederation]
            if confederation
            else self._WCQ_CODES
        )

        for code in search_order:
            try:
                stats_list = self.get_team_stats(code)
            except Exception:
                continue
            for s in stats_list:
                if s.team_name.lower() == team_name.lower():
                    return s
            for s in stats_list:
                if kw & self._name_keywords(s.team_name):
                    return s
            # If we found the confederation and it has teams but none matched,
            # stop — the team is unlikely to be in other confederations.
            if confederation and code == confederation and stats_list:
                break
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _name_keywords(name: str) -> set[str]:
        """Extract meaningful keywords from a team name for fuzzy matching."""
        stopwords = {"fc", "ac", "as", "ss", "afc", "cf", "inter",
                     "calcio", "sport", "club", "united", "city"}
        words = re.sub(r"[^a-z0-9\s]", "", name.lower()).split()
        return {w for w in words if w not in stopwords and len(w) > 2}
