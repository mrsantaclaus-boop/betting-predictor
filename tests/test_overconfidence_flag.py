from api_server import _overconfidence_flag, OVERCONFIDENCE_RATIO_THRESHOLD


def test_flags_extreme_result_market_ratio():
    # Aston Villa away win at Barcelona, 2026-11-03: model 40.6% vs market-implied
    # 11.2% (ratio 3.63x) -- lost (Barcelona won 3-0).
    flag = _overconfidence_flag("away_win", 40.6, 11.2)
    assert flag["model_market_ratio"] == 3.63
    assert flag["overconfidence_warning"] is True


def test_flags_right_at_threshold():
    flag = _overconfidence_flag("away_win", OVERCONFIDENCE_RATIO_THRESHOLD * 10, 10)
    assert flag["overconfidence_warning"] is True


def test_no_flag_below_threshold():
    # PSG away win at Villarreal, same slate: ratio 1.24x -- won.
    flag = _overconfidence_flag("away_win", 68.2, 54.8)
    assert flag["overconfidence_warning"] is False


def test_no_flag_for_goals_or_btts_markets():
    # The pattern is specific to 1X2 result markets; goals/BTTS markets
    # performed close to their historical norm on the same slate.
    assert _overconfidence_flag("over_2_5", 78.6, 10.0) == {}
    assert _overconfidence_flag("btts_yes", 90.0, 10.0) == {}


def test_no_flag_when_implied_pct_is_zero():
    assert _overconfidence_flag("away_win", 50.0, 0.0) == {}
