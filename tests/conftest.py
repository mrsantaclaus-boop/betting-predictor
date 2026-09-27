import os
import tempfile

import pytest

# api_server.py initialises its prediction DB at import time. Point it (and the
# TTL cache) at a throwaway directory and make sure no real DATABASE_URL leaks
# into the tests. conftest.py is imported before any test module, so this runs
# before `from api_server import ...`.
_TEST_DIR = tempfile.mkdtemp(prefix="sai-tipster-tests-")
os.environ["PRED_DB_PATH"] = os.path.join(_TEST_DIR, "predictions.db")
os.environ["CACHE_DB_PATH"] = os.path.join(_TEST_DIR, "cache.db")
os.environ["DATABASE_URL"] = ""

from football.models import TeamStats  # noqa: E402


@pytest.fixture
def avg_team():
    """A typical Serie A team with full-season stats."""
    return TeamStats(
        team_name="Test FC",
        competition="SA",
        games_played=20,
        goals_scored_pg=1.35,
        goals_conceded_pg=1.35,
        xg_pg=1.35,
        xga_pg=1.35,
        corners_pg=5.1,
        yellow_cards_pg=2.2,
        red_cards_pg=0.08,
    )


@pytest.fixture
def early_season_team():
    """A team with only 4 games played — stats are noisy."""
    return TeamStats(
        team_name="Early FC",
        competition="SA",
        games_played=4,
        goals_scored_pg=3.0,
        goals_conceded_pg=0.5,
        xg_pg=2.8,
        xga_pg=0.6,
        corners_pg=8.0,
        yellow_cards_pg=4.0,
        red_cards_pg=0.5,
    )
