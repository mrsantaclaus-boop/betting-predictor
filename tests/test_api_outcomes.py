import os

os.environ.setdefault(
    "PRED_DB_PATH",
    "/tmp/claude-0/-home-user-betting-predictor/5952ed34-7455-5553-a334-3c9350f0c1a4/scratchpad/test_predictions.db",
)

from api_server import _compute_outcomes  # noqa: E402


def test_outcomes_without_corners_or_cards():
    out = _compute_outcomes(2, 1)
    assert out["home_win"] is True
    assert "over_9_5_corners" not in out
    assert "over_3_5_cards" not in out
    assert "red_card" not in out


def test_outcomes_with_cards_over_3_5():
    cards = {"home_yellow": 2, "away_yellow": 2, "home_red": 0, "away_red": 0}
    out = _compute_outcomes(1, 1, cards=cards)
    assert out["over_3_5_cards"] is True
    assert out["under_3_5_cards"] is False
    assert out["red_card"] is False


def test_outcomes_with_cards_under_3_5_and_red_card():
    cards = {"home_yellow": 1, "away_yellow": 1, "home_red": 1, "away_red": 0}
    out = _compute_outcomes(0, 0, cards=cards)
    assert out["over_3_5_cards"] is False
    assert out["under_3_5_cards"] is True
    assert out["red_card"] is True


def test_outcomes_with_incomplete_cards_dict_is_ignored():
    out = _compute_outcomes(1, 0, cards={"home_yellow": 2})
    assert "over_3_5_cards" not in out
    assert "red_card" not in out
