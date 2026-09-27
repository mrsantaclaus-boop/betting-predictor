from predictor.xg_proxy import xg_from_shots, SOT_TO_XG
from football.espn_client import _acc


def test_xg_proxy_scales_with_shots_on_target():
    assert xg_from_shots(0) == 0.0
    assert xg_from_shots(4.47) == round(SOT_TO_XG * 4.47, 2)   # league-average SOT -> ~1.39 xG
    assert 1.3 < xg_from_shots(4.47) < 1.5


def test_acc_records_shots_for_and_against():
    stats = {}
    home = {"totalShots": "15", "shotsOnTarget": "6"}
    away = {"totalShots": "7", "shotsOnTarget": "2"}
    _acc(stats, "Home FC", 2, 0, home, away)
    _acc(stats, "Away FC", 0, 2, away, home)
    assert stats["Home FC"]["shots_ot"] == [6.0]
    assert stats["Home FC"]["sot_against"] == [2.0]
    assert stats["Away FC"]["shots_ot"] == [2.0]
    assert stats["Away FC"]["sot_against"] == [6.0]


def test_acc_without_opponent_box_still_works():
    stats = {}
    _acc(stats, "Solo FC", 1, 1, {"shotsOnTarget": "3"})
    assert stats["Solo FC"]["sot_against"] == [0.0]
