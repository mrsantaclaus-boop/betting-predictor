"""
football/espn_client.py — Client for ESPN's public soccer API.

No API key required. Two roles:

1. Fixture/result source for competitions football-data.org does not cover
   (ESPN_LEAGUES: Conference League, Brasileirao, World Cup + qualifiers).
   For these, our fixture_id IS the ESPN event id.
2. Match-statistics source (corners, cards, referee) for EVERY competition,
   including the football-data.org ones (ESPN_LOOKUP_SLUGS). For those the
   event is found by kickoff date + team names, because the ids differ.

NOTE on User-Agent: ESPN returns 403 to browser-like User-Agents
("Mozilla/5.0", full Chrome strings) but accepts python-requests' default.
Do not override the session User-Agent.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

import requests

from .models import Fixture
from .team_names import match_score

logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Competitions whose fixtures come from ESPN (fixture_id == ESPN event id).
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

# Competitions whose fixtures come from football-data.org; ESPN is used only
# to look up match statistics / team stats by date + team names.
ESPN_LOOKUP_SLUGS: dict[str, str] = {
    "SA":  "ita.1",
    "SB":  "ita.2",
    "CL":  "uefa.champions",
    "EL":  "uefa.europa",
    "USC": "uefa.super_cup",
}

_FINISHED_STATUSES = ("STATUS_FINAL", "STATUS_FULL_TIME", "STATUS_FINAL_AET", "STATUS_FINAL_PEN")

# Competitions where all matches are played at neutral venues
_NEUTRAL_VENUE_CODES = {"WC"}

_POLITE_DELAY_S = 1.0          # between scoreboard lookups of a single match
_SUMMARY_LOOP_DELAY_S = 0.4    # inside bulk summary loops (team-stats aggregation)


def espn_slug(competition_code: str) -> str | None:
    """ESPN league slug for any supported competition code, or None."""
    league = ESPN_LEAGUES.get(competition_code)
    if league:
        return league[0]
    return ESPN_LOOKUP_SLUGS.get(competition_code)


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
    """Accumulate one match worth of stats for a team.

    `box` is {statistic name: displayValue} from a boxscore team entry.
    ESPN's stat names are `wonCorners`, `shotsOnTarget`, `totalShots`,
    `yellowCards`, `redCards`, `foulsCommitted` (verified 2026-09-27).
    """
    if team not in stats:
        stats[team] = {"scored": [], "conceded": [], "shots": [],
                       "shots_ot": [], "corners": [], "yellow": [],
                       "red": [], "xg": [], "xga": []}
    s = stats[team]
    s["scored"].append(scored)
    s["conceded"].append(conceded)
    s["shots"].append(_safe_float(box.get("totalShots", "0")))
    s["shots_ot"].append(_safe_float(box.get("shotsOnTarget", box.get("shotsOnGoal", "0"))))
    s["corners"].append(_safe_float(box.get("wonCorners", box.get("cornerKicks", "0"))))
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
    "STATUS_FINAL_AET":   "FINISHED",
    "STATUS_FINAL_PEN":   "FINISHED",
    "STATUS_POSTPONED":   "POSTPONED",
    "STATUS_CANCELED":    "CANCELLED",
}


def _referee_from_summary(summary: dict) -> str:
    """Referee display name from a summary payload ('' if absent)."""
    candidates = list(summary.get("gameInfo", {}).get("officials", []) or [])
    comps = summary.get("header", {}).get("competitions", [])
    if comps:
        candidates += list(comps[0].get("officials", []) or [])
    for off in candidates:
        pos = (off.get("position") or {}).get("name", "")
        if pos.lower() == "referee" or not pos:
            return off.get("displayName") or off.get("fullName") or ""
    return ""


def parse_match_stats(summary: dict) -> dict | None:
    """
    Extract final score, corners, cards and referee from an ESPN
    `/summary?event=` payload. Returns None when the payload has no header
    (unknown event). `status` tells the caller whether the match is finished.

    Boxscore teams are matched to home/away by team id, not display name.
    """
    header = summary.get("header", {})
    comps = header.get("competitions", [])
    if not comps:
        return None
    comp = comps[0]
    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    box_by_id: dict[str, dict] = {}
    for bt in summary.get("boxscore", {}).get("teams", []):
        tid = str(bt.get("team", {}).get("id", ""))
        box_by_id[tid] = {s.get("name"): s.get("displayValue", "0")
                          for s in bt.get("statistics", [])}
    hb = box_by_id.get(str(home.get("team", {}).get("id", "")), {})
    ab = box_by_id.get(str(away.get("team", {}).get("id", "")), {})

    def _stat(box: dict, key: str) -> int | None:
        if key not in box:
            return None
        return _safe_int(box.get(key))

    status = comp.get("status", {}).get("type", {}).get("name", "")
    return {
        "event_id":     str(comp.get("id") or header.get("id") or ""),
        "status":       status,
        "finished":     status in _FINISHED_STATUSES,
        "date":         comp.get("date", ""),
        "home_team":    home.get("team", {}).get("displayName", ""),
        "away_team":    away.get("team", {}).get("displayName", ""),
        "home_score":   _safe_int(home.get("score")),
        "away_score":   _safe_int(away.get("score")),
        "home_corners": _stat(hb, "wonCorners"),
        "away_corners": _stat(ab, "wonCorners"),
        "home_yellow":  _stat(hb, "yellowCards"),
        "away_yellow":  _stat(ab, "yellowCards"),
        "home_red":     _stat(hb, "redCards"),
        "away_red":     _stat(ab, "redCards"),
        "home_fouls":   _stat(hb, "foulsCommitted"),
        "away_fouls":   _stat(ab, "foulsCommitted"),
        "referee":      _referee_from_summary(summary),
    }


def stats_to_outcome_inputs(stats: dict) -> tuple[dict, dict]:
    """
    Convert parse_match_stats() output to the (corners, cards) dicts that
    api_server._compute_outcomes expects. Either may be {} when incomplete.
    """
    corners: dict = {}
    cards: dict = {}
    if stats.get("home_corners") is not None and stats.get("away_corners") is not None:
        corners = {"home": stats["home_corners"], "away": stats["away_corners"]}
    keys = ("home_yellow", "away_yellow", "home_red", "away_red")
    if all(stats.get(k) is not None for k in keys):
        cards = {k: stats[k] for k in keys}
    return corners, cards


class EspnClient:
    """Fetches fixture data from ESPN's public API (no key required)."""

    def __init__(self):
        self.session = requests.Session()
        # Deliberately no User-Agent override — see module docstring.

    # ── Scoreboard access ─────────────────────────────────────────────────

    def _events_between(self, league_slug: str, start, end) -> list[dict]:
        """
        All scoreboard events with start <= kickoff date <= end.

        ESPN rejects `dates=YYYYMMDD-YYYYMMDD` ranges with HTTP 400 (verified
        2026-09-27) but accepts a single day (`YYYYMMDD`) or a whole month
        (`YYYYMM`), so the window is covered month by month and filtered here.
        """
        events: list[dict] = []
        seen: set[str] = set()
        cursor = start.replace(day=1)
        while cursor <= end:
            month = cursor.strftime("%Y%m")
            try:
                r = self.session.get(f"{BASE_URL}/{league_slug}/scoreboard",
                                     params={"dates": month, "limit": 500}, timeout=15)
                r.raise_for_status()
                batch = r.json().get("events", [])
            except Exception as e:
                logger.warning("ESPN scoreboard %s %s failed: %s", league_slug, month, e)
                batch = []
            for ev in batch:
                try:
                    day = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).date()
                except (KeyError, ValueError):
                    continue
                if start <= day <= end and ev.get("id") not in seen:
                    seen.add(ev.get("id"))
                    events.append(ev)
            # next month
            cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        return events

    # ── Fixtures (ESPN-sourced competitions) ──────────────────────────────

    def get_upcoming_fixtures(self, competition_code: str,
                               days_ahead: int = 14) -> list[Fixture]:
        league_info = ESPN_LEAGUES.get(competition_code)
        if not league_info:
            return []
        league_slug, comp_name = league_info

        today = datetime.now(timezone.utc).date()
        end_date = today + timedelta(days=days_ahead)
        try:
            events = self._events_between(league_slug, today, end_date)
            return self._parse_events({"events": events}, competition_code, comp_name)
        except Exception as e:
            logger.warning("ESPN fetch failed for %s: %s", competition_code, e)
            return []

    # ── Team stats aggregated from match summaries ────────────────────────

    def get_team_stats_from_summaries(self, competition_code: str,
                                      days_back: int = 60) -> dict:
        """
        Aggregate per-team stats from completed matches of the last
        `days_back` days via the ESPN summary API (any supported competition).
        Returns {team_name: {"scored": [...], "conceded": [...], "shots": [...],
                              "corners": [...], "yellow": [...], "red": [...]}}
        Caller converts lists to per-game averages.
        """
        league_slug = espn_slug(competition_code)
        if not league_slug:
            return {}

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        events = self._events_between(league_slug, start_date, end_date)

        stats: dict[str, dict] = {}

        for event in events:
            comp = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            if status not in _FINISHED_STATUSES:
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

            home_name = home["team"]["displayName"]
            away_name = away["team"]["displayName"]

            # Fetch detailed match stats
            try:
                time.sleep(_SUMMARY_LOOP_DELAY_S)
                sum_url = f"{BASE_URL}/{league_slug}/summary"
                sr = self.session.get(sum_url, params={"event": event_id}, timeout=15)
                sr.raise_for_status()
                summary = sr.json()
            except Exception as e:
                logger.debug("ESPN summary failed for event %s: %s", event_id, e)
                # Fall back to goals only
                _acc(stats, home_name, home_score, away_score, {})
                _acc(stats, away_name, away_score, home_score, {})
                continue

            box_by_id: dict[str, dict] = {}
            for bt in summary.get("boxscore", {}).get("teams", []):
                tid = str(bt.get("team", {}).get("id", ""))
                box_by_id[tid] = {s["name"]: s.get("displayValue", "0")
                                  for s in bt.get("statistics", [])}
            _acc(stats, home_name, home_score, away_score,
                 box_by_id.get(str(home["team"].get("id", "")), {}))
            _acc(stats, away_name, away_score, home_score,
                 box_by_id.get(str(away["team"].get("id", "")), {}))

        return stats

    def get_wc_team_stats(self, competition_code: str = "WC") -> dict:
        """Backwards-compatible alias of get_team_stats_from_summaries()."""
        return self.get_team_stats_from_summaries(competition_code, days_back=60)

    # ── Results ───────────────────────────────────────────────────────────

    def get_competition_results(self, competition_code: str,
                                days_back: int = 365) -> list[Fixture]:
        """Return recently FINISHED fixtures going back N days."""
        league_info = ESPN_LEAGUES.get(competition_code)
        if not league_info:
            return []
        league_slug, comp_name = league_info

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days_back)
        try:
            events = self._events_between(league_slug, start_date, end_date)
            fixtures = self._parse_events({"events": events}, competition_code, comp_name)
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

                # Referee is sometimes published on the scoreboard before kickoff
                referee = None
                for off in competition.get("officials", []) or []:
                    pos = (off.get("position") or {}).get("name", "")
                    if pos.lower() == "referee" or not pos:
                        referee = off.get("displayName") or off.get("fullName") or None
                        break

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
                    referee=referee,
                ))
            except (KeyError, ValueError, IndexError) as e:
                logger.debug("Skip malformed ESPN event: %s", e)

        return fixtures

    def get_match_result(self, fixture_id: int, competition_code: str) -> "Fixture | None":
        """
        Fetch a single finished match from ESPN by event ID.
        Returns None if the match isn't finished yet or isn't found.
        """
        from .models import Fixture as _Fixture
        league_info = ESPN_LEAGUES.get(competition_code)
        if not league_info:
            return None
        league_slug, comp_name = league_info

        summary = self._get_summary(league_slug, fixture_id)
        if summary is None:
            return None
        stats = parse_match_stats(summary)
        if not stats or not stats["finished"]:
            return None
        if stats["home_score"] is None or stats["away_score"] is None:
            return None

        try:
            match_dt = datetime.fromisoformat(stats["date"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            match_dt = datetime.now(timezone.utc)

        comp = summary["header"]["competitions"][0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        return _Fixture(
            fixture_id=fixture_id,
            competition=comp_name,
            competition_code=competition_code,
            home_team=stats["home_team"],
            home_team_id=int(home.get("team", {}).get("id", 0)),
            away_team=stats["away_team"],
            away_team_id=int(away.get("team", {}).get("id", 0)),
            match_date=match_dt,
            status="FINISHED",
            home_score=stats["home_score"],
            away_score=stats["away_score"],
            is_neutral=competition_code in _NEUTRAL_VENUE_CODES,
            referee=stats["referee"] or None,
        )

    # ── Match statistics (corners, cards, referee) for ANY competition ────

    def find_event_id(self, competition_code: str, home_team: str,
                      away_team: str, kickoff: datetime) -> str | None:
        """
        Locate the ESPN event id of a match by kickoff date and team names.
        Used for football-data.org competitions whose fixture ids are not
        ESPN ids. Checks the kickoff date and the following day (time zones).
        """
        league_slug = espn_slug(competition_code)
        if not league_slug:
            return None
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)
        kickoff_utc = kickoff.astimezone(timezone.utc)

        best: tuple[int, str | None] = (0, None)
        for delta in (0, -1, 1):
            day = (kickoff_utc + timedelta(days=delta)).strftime("%Y%m%d")
            try:
                time.sleep(_POLITE_DELAY_S)
                r = self.session.get(f"{BASE_URL}/{league_slug}/scoreboard",
                                     params={"dates": day}, timeout=15)
                r.raise_for_status()
                events = r.json().get("events", [])
            except Exception as e:
                logger.debug("ESPN scoreboard %s %s failed: %s", league_slug, day, e)
                continue
            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                h = next((c for c in competitors if c.get("homeAway") == "home"), None)
                a = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not h or not a:
                    continue
                hs = match_score(home_team, h["team"]["displayName"])
                as_ = match_score(away_team, a["team"]["displayName"])
                if hs > 0 and as_ > 0 and hs + as_ > best[0]:
                    best = (hs + as_, str(ev.get("id")))
            if best[1]:
                break
        return best[1]

    def get_match_stats(self, competition_code: str, home_team: str,
                        away_team: str, kickoff: datetime | None = None,
                        espn_event_id: int | str | None = None) -> dict | None:
        """
        Final score + corners + cards + referee for one match, from the ESPN
        summary endpoint. Pass `espn_event_id` when known (ESPN-sourced
        competitions: the fixture id); otherwise the event is located by
        `kickoff` date and team names. Returns parse_match_stats() output
        (with "finished" flag) or None when the match cannot be found.
        """
        league_slug = espn_slug(competition_code)
        if not league_slug:
            return None
        event_id = espn_event_id
        if event_id is None:
            if kickoff is None:
                return None
            event_id = self.find_event_id(competition_code, home_team, away_team, kickoff)
            if event_id is None:
                logger.info("ESPN: no event found for %s vs %s (%s, %s)",
                            home_team, away_team, competition_code, kickoff.date())
                return None
        summary = self._get_summary(league_slug, event_id)
        if summary is None:
            return None
        stats = parse_match_stats(summary)
        if stats:
            stats["event_id"] = str(event_id)
        return stats

    def _get_summary(self, league_slug: str, event_id) -> dict | None:
        try:
            r = self.session.get(f"{BASE_URL}/{league_slug}/summary",
                                 params={"event": event_id}, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("ESPN summary failed for %s event %s: %s", league_slug, event_id, e)
            return None
