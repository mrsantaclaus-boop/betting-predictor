"""Fouls come from ESPN; missing data is said to be missing, never shown as 0.0 or "no absences".

Found 2026-11-19: the dossier printed "Fouls committed 0.0" (only the dead FBref scraper filled
the field) and "No confirmed absences" (no source of absences exists for Serie A).
"""
from football.espn_client import _acc
from football.models import TeamForm, TeamStats, InjuryReport
from predictor.orchestrator import BettingOrchestrator
from seed.generator import SeedGenerator


# ── ESPN aggregation ─────────────────────────────────────────────────────────

def test_acc_aggregates_fouls_committed():
    stats = {}
    _acc(stats, "Napoli", 2, 1, {"foulsCommitted": "11", "wonCorners": "4"})
    _acc(stats, "Napoli", 0, 0, {"foulsCommitted": "15", "wonCorners": "6"})
    assert stats["Napoli"]["fouls"] == [11.0, 15.0]


def test_acc_does_not_invent_zero_fouls_when_stat_missing():
    stats = {}
    _acc(stats, "Napoli", 2, 1, {})                      # summary fetch failed: goals only
    _acc(stats, "Napoli", 1, 1, {"foulsCommitted": "12"})
    assert stats["Napoli"]["fouls"] == [12.0]            # not [0.0, 12.0]


def _orchestrator_with_espn(raw):
    class Stub:
        def get_team_stats_from_summaries(self, code, days_back=60):
            return raw
    o = BettingOrchestrator.__new__(BettingOrchestrator)   # no network clients
    o._espn_team_stats_cache = {}
    o.espn_client = Stub()
    return o


def _raw(fouls):
    return {"Napoli": {"scored": [1, 2], "conceded": [0, 1], "shots": [10, 12], "shots_ot": [4, 5],
                       "shots_against": [8, 9], "sot_against": [3, 4], "corners": [5, 6],
                       "yellow": [1, 2], "red": [0, 0], "xg": [0, 0], "xga": [0, 0], "fouls": fouls}}


def test_team_stats_carry_average_fouls():
    ts = _orchestrator_with_espn(_raw([10.0, 14.0]))._get_espn_team_stats("SA")["Napoli"]
    assert ts.fouls_committed_pg == 12.0


def test_team_stats_without_fouls_stay_zero_meaning_unavailable():
    ts = _orchestrator_with_espn(_raw([]))._get_espn_team_stats("SA")["Napoli"]
    assert ts.fouls_committed_pg == 0.0


def test_fill_from_espn_adds_missing_fouls_without_touching_other_fields():
    o = _orchestrator_with_espn(_raw([10.0, 14.0]))
    mine = TeamStats(team_name="SSC Napoli", competition="SA", corners_pg=7.0, yellow_cards_pg=2.5,
                     xg_pg=1.4, xga_pg=1.0, games_played=11)
    o._fill_corners_cards_from_espn(mine, "SSC Napoli", "SA")
    assert mine.fouls_committed_pg == 12.0
    assert mine.corners_pg == 7.0 and mine.yellow_cards_pg == 2.5   # already known: untouched


# ── Dossier text ─────────────────────────────────────────────────────────────

def _profile(stats, injuries):
    form = TeamForm(team_name="SSC Napoli", results=["W", "D"], goals_scored=[2, 1], goals_conceded=[0, 1],
                    home_record="1W-0D-0L", away_record="0W-1D-0L")
    return SeedGenerator()._team_profile("SSC Napoli", form, stats, None, injuries, is_home=True)


def test_dossier_shows_real_fouls():
    text = _profile(TeamStats(team_name="SSC Napoli", fouls_committed_pg=12.5), InjuryReport(team_name="x"))
    assert "| Fouls committed | **12.5** |" in text


def test_dossier_says_na_when_fouls_unknown():
    text = _profile(TeamStats(team_name="SSC Napoli"), InjuryReport(team_name="x"))
    assert "| Fouls committed | **N/A** |" in text
    assert "**0.0**" not in text.split("Fouls committed")[1].split("\n")[0]


def test_dossier_says_absences_not_available_without_source():
    text = _profile(TeamStats(team_name="SSC Napoli"), InjuryReport(team_name="x", source_available=False))
    assert "Not available" in text
    assert "No confirmed absences" not in text


def test_no_confirmed_absences_only_when_a_source_exists():
    text = _profile(TeamStats(team_name="SSC Napoli"), InjuryReport(team_name="x"))   # source available, none listed
    assert "No confirmed absences" in text


def test_real_absences_still_listed_when_source_exists():
    text = _profile(TeamStats(team_name="SSC Napoli"),
                    InjuryReport(team_name="x", unavailable=["A. Player"], doubtful=["B. Player"]))
    assert "A. Player" in text and "B. Player" in text
