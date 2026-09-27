"""Parsing of ESPN summary payloads (shape captured from Milan-Lecce, 2026-09-20)."""
from football.espn_client import parse_match_stats, stats_to_outcome_inputs, _acc, espn_slug


def _summary(status="STATUS_FULL_TIME", with_box=True):
    def team(tid, name, stats):
        return {"team": {"id": tid, "displayName": name},
                "statistics": [{"name": k, "displayValue": v} for k, v in stats.items()]}

    payload = {
        "header": {"id": "401874944", "competitions": [{
            "id": "401874944",
            "date": "2026-09-20T18:45Z",
            "status": {"type": {"name": status}},
            "competitors": [
                {"homeAway": "home", "score": "3", "team": {"id": "103", "displayName": "AC Milan"}},
                {"homeAway": "away", "score": "0", "team": {"id": "4004", "displayName": "Lecce"}},
            ],
        }]},
        "gameInfo": {"officials": [{"displayName": "Federico La Penna",
                                    "position": {"name": "Referee"}}]},
    }
    if with_box:
        # boxscore order deliberately reversed vs competitors: matching is by id
        payload["boxscore"] = {"teams": [
            team("4004", "Lecce", {"wonCorners": "3", "yellowCards": "1", "redCards": "0",
                                   "foulsCommitted": "15", "totalShots": "6"}),
            team("103", "AC Milan", {"wonCorners": "2", "yellowCards": "0", "redCards": "0",
                                     "foulsCommitted": "9", "totalShots": "10"}),
        ]}
    return payload


def test_parse_full_time_match():
    s = parse_match_stats(_summary())
    assert s["finished"] is True
    assert (s["home_team"], s["away_team"]) == ("AC Milan", "Lecce")
    assert (s["home_score"], s["away_score"]) == (3, 0)
    assert (s["home_corners"], s["away_corners"]) == (2, 3)
    assert (s["home_yellow"], s["away_yellow"]) == (0, 1)
    assert (s["home_red"], s["away_red"]) == (0, 0)
    assert s["referee"] == "Federico La Penna"


def test_outcome_inputs_complete():
    corners, cards = stats_to_outcome_inputs(parse_match_stats(_summary()))
    assert corners == {"home": 2, "away": 3}
    assert cards == {"home_yellow": 0, "away_yellow": 1, "home_red": 0, "away_red": 0}


def test_missing_boxscore_yields_empty_inputs():
    s = parse_match_stats(_summary(with_box=False))
    assert s["home_corners"] is None
    corners, cards = stats_to_outcome_inputs(s)
    assert corners == {} and cards == {}


def test_unfinished_match_flagged():
    s = parse_match_stats(_summary(status="STATUS_IN_PROGRESS"))
    assert s["finished"] is False


def test_aet_counts_as_finished():
    assert parse_match_stats(_summary(status="STATUS_FINAL_AET"))["finished"] is True


def test_unknown_event_returns_none():
    assert parse_match_stats({"header": {}}) is None


def test_acc_reads_espn_stat_names():
    stats = {}
    _acc(stats, "AC Milan", 3, 0, {"wonCorners": "7", "shotsOnTarget": "5", "totalShots": "12",
                                   "yellowCards": "2", "redCards": "1"})
    assert stats["AC Milan"]["corners"] == [7.0]
    assert stats["AC Milan"]["shots_ot"] == [5.0]
    assert stats["AC Milan"]["yellow"] == [2.0]
    assert stats["AC Milan"]["red"] == [1.0]


def test_slugs_cover_both_source_families():
    assert espn_slug("WC") == "FIFA.WORLD"
    assert espn_slug("SA") == "ita.1"
    assert espn_slug("CL") == "uefa.champions"
    assert espn_slug("XX") is None
