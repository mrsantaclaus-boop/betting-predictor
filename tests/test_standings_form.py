from football.football_data_api import compute_form_from_matches


def _m(day, h, a, hs, as_, status="FINISHED"):
    return {"status": status, "utcDate": f"2026-11-{day:02d}T18:00:00Z",
            "homeTeam": {"id": h}, "awayTeam": {"id": a},
            "score": {"fullTime": {"home": hs, "away": as_}}}


def test_results_and_order_most_recent_last():
    ms = [_m(3, 1, 2, 0, 1), _m(1, 1, 3, 2, 0), _m(2, 4, 1, 1, 1)]  # unsorted on purpose
    f = compute_form_from_matches(ms)
    assert f[1] == "W,D,L"        # day1 W, day2 D, day3 L
    assert f[2] == "W"
    assert f[3] == "L"
    assert f[4] == "D"


def test_only_last_n_kept():
    ms = [_m(d, 1, 2, 1, 0) for d in range(1, 9)] + [_m(9, 1, 2, 0, 3)]
    assert compute_form_from_matches(ms, n=5)[1] == "W,W,W,W,L"


def test_unfinished_or_unscored_ignored_and_no_invented_form():
    ms = [_m(1, 1, 2, None, None, status="SCHEDULED"), _m(2, 5, 6, None, None)]
    assert compute_form_from_matches(ms) == {}
