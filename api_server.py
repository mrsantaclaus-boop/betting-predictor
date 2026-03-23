"""
api_server.py — Betting Predictor API server.

Runs on port 5050 (locally) or as the main web service on Render.
Serves the built Vue.js frontend as static files in production.

Endpoints:
  GET  /api/health
  GET  /api/fixtures
  GET  /api/standings/<code>
  GET  /api/odds/<competition_code>
  GET  /api/odds/fixture/<fixture_id>
  GET  /api/news/<home_team>/<away_team>
  POST /api/predict
  GET  /api/predictions
  POST /api/cache/clear
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from football import FootballDataClient
from data.odds_api import OddsAPIClient
from data.news_fetcher import NewsFetcher
from data.cache import get_cache, TTL
from predictor.orchestrator import BettingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend_static", static_url_path="")
CORS(app, origins="*")

_orchestrator: BettingOrchestrator | None = None
_fd_client: FootballDataClient | None = None
_odds_client: OddsAPIClient | None = None
_news_fetcher: NewsFetcher | None = None
_cache = get_cache()

_PRED_DB = Path("data/predictions.db")


def get_orch() -> BettingOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = BettingOrchestrator()
    return _orchestrator


def get_fd() -> FootballDataClient:
    global _fd_client
    if _fd_client is None:
        _fd_client = FootballDataClient()
    return _fd_client


def get_odds() -> OddsAPIClient:
    global _odds_client
    if _odds_client is None:
        _odds_client = OddsAPIClient()
    return _odds_client


def get_news() -> NewsFetcher:
    global _news_fetcher
    if _news_fetcher is None:
        _news_fetcher = NewsFetcher()
    return _news_fetcher


# ── Predictions DB ────────────────────────────────────────────────────────────

def _init_pred_db():
    _PRED_DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(_PRED_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                fixture_id  INTEGER NOT NULL UNIQUE,
                match_label TEXT    NOT NULL,
                competition TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                data        TEXT    NOT NULL
            )
        """)


def _save_pred(fixture_id: int, match_label: str, competition: str, data: dict):
    with sqlite3.connect(_PRED_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(fixture_id, match_label, competition, created_at, data) VALUES (?,?,?,?,?)",
            (fixture_id, match_label, competition,
             datetime.now(timezone.utc).isoformat(), json.dumps(data)),
        )


def _load_preds() -> list[dict]:
    with sqlite3.connect(_PRED_DB) as conn:
        rows = conn.execute(
            "SELECT fixture_id, match_label, competition, created_at, data "
            "FROM predictions ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    out = []
    for fid, label, comp, ts, raw in rows:
        entry = json.loads(raw)
        entry.update(fixture_id=fid, match_label=label,
                     competition=comp, created_at=ts)
        out.append(entry)
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fixture_list():
    def fetch():
        orch = get_orch()
        fixtures = orch.get_upcoming_fixtures()
        return [
            {
                "fixture_id": f.fixture_id,
                "competition": f.competition,
                "competition_code": f.competition_code,
                "home_team": f.home_team,
                "home_team_id": f.home_team_id,
                "away_team": f.away_team,
                "away_team_id": f.away_team_id,
                "match_date": f.match_date.isoformat(),
                "matchday": f.matchday,
                "stage": f.stage,
                "status": f.status,
            }
            for f in fixtures
        ]
    return _cache.get_or_fetch("fixtures:all", fetch, TTL["fixtures"])


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "odds_quota": get_odds().quota_remaining,
    })


@app.route("/api/fixtures")
def fixtures():
    try:
        data = _fixture_list() or []
        return jsonify(data)
    except Exception as e:
        logger.exception("fixtures error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/standings/<competition_code>")
def standings(competition_code: str):
    if competition_code not in ("SA", "CL"):
        return jsonify({"error": "Use SA or CL"}), 400

    def fetch():
        rows = get_fd().get_standings(competition_code)
        return [r.model_dump() for r in rows]

    try:
        data = _cache.get_or_fetch(
            f"standings:{competition_code}", fetch, TTL["standings"]
        )
        return jsonify(data or [])
    except Exception as e:
        logger.exception("standings error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/odds/<competition_code>")
def odds_by_competition(competition_code: str):
    """All live odds for a competition."""
    if competition_code not in ("SA", "CL"):
        return jsonify({"error": "Use SA or CL"}), 400
    if not os.getenv("ODDS_API_KEY"):
        return jsonify({"error": "ODDS_API_KEY not configured"}), 503

    def fetch():
        return get_odds().get_all_odds(competition_code)

    try:
        data = _cache.get_or_fetch(
            f"odds:{competition_code}", fetch, TTL["odds"]
        )
        return jsonify(data or [])
    except Exception as e:
        logger.exception("odds error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/odds/fixture/<int:fixture_id>")
def odds_for_fixture(fixture_id: int):
    """Live odds for a specific fixture."""
    cache_key = f"odds:fixture:{fixture_id}"
    cached = _cache.get(cache_key)
    if cached:
        return jsonify(cached)

    # Find fixture
    fixture_data = next(
        (f for f in (_fixture_list() or []) if f["fixture_id"] == fixture_id),
        None,
    )
    if not fixture_data:
        return jsonify({"error": "Fixture not found"}), 404

    if not os.getenv("ODDS_API_KEY"):
        return jsonify({"error": "ODDS_API_KEY not configured"}), 503

    try:
        result = get_odds().get_fixture_odds(
            fixture_data["competition_code"],
            fixture_data["home_team"],
            fixture_data["away_team"],
        )
        if "error" not in result:
            _cache.set(cache_key, result, TTL["odds"])
        return jsonify(result)
    except Exception as e:
        logger.exception("fixture odds error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/news/<path:teams>")
def news(teams: str):
    """
    News for a match. URL: /api/news/HomeTeam/AwayTeam
    or /api/news/TeamName for single team.
    """
    parts = [p.strip() for p in teams.split("/") if p.strip()]
    home = parts[0] if parts else ""
    away = parts[1] if len(parts) > 1 else ""
    competition = request.args.get("competition", "")

    cache_key = f"news:{home}:{away}"
    cached = _cache.get(cache_key)
    if cached:
        return jsonify(cached)

    try:
        if away:
            articles = get_news().get_match_news(home, away, competition)
        else:
            articles = get_news().get_team_news(home)
        _cache.set(cache_key, articles, TTL["news"])
        return jsonify(articles)
    except Exception as e:
        logger.exception("news error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True) or {}
    fixture_id = body.get("fixture_id")
    if not fixture_id:
        return jsonify({"error": "fixture_id required"}), 400

    fixture_data = next(
        (f for f in (_fixture_list() or []) if f["fixture_id"] == fixture_id),
        None,
    )
    if not fixture_data:
        return jsonify({"error": f"Fixture {fixture_id} not found"}), 404

    try:
        from football.models import Fixture
        from datetime import datetime as dt
        fixture = Fixture(
            fixture_id=fixture_data["fixture_id"],
            competition=fixture_data["competition"],
            competition_code=fixture_data["competition_code"],
            home_team=fixture_data["home_team"],
            home_team_id=fixture_data["home_team_id"],
            away_team=fixture_data["away_team"],
            away_team_id=fixture_data["away_team_id"],
            match_date=dt.fromisoformat(fixture_data["match_date"]),
            matchday=fixture_data.get("matchday"),
            stage=fixture_data.get("stage"),
            status=fixture_data.get("status", "SCHEDULED"),
        )

        prediction = get_orch().predict_fixture(fixture)
        data = prediction.to_dict()
        _save_pred(
            fixture_id,
            f"{fixture.home_team} vs {fixture.away_team}",
            fixture.competition,
            data,
        )
        # Also try to fetch odds alongside (non-blocking)
        try:
            if os.getenv("ODDS_API_KEY"):
                odds = get_odds().get_fixture_odds(
                    fixture.competition_code,
                    fixture.home_team,
                    fixture.away_team,
                )
                data["live_odds"] = odds
        except Exception:
            pass
        return jsonify(data)
    except Exception as e:
        logger.exception("predict error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def list_predictions():
    try:
        return jsonify(_load_preds())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """Force-refresh all cached data."""
    body = request.get_json(force=True, silent=True) or {}
    prefix = body.get("prefix", "")
    if prefix:
        count = _cache.invalidate_prefix(prefix)
    else:
        count = _cache.clear_expired()
    return jsonify({"deleted": count})


# ── Serve Vue.js frontend (production) ────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    """Serve the built Vue.js app for all non-API routes."""
    static = Path("frontend_static")
  if path and (static / path).exists():
      return send_from_directory("frontend_static", path)
  return send_from_directory("frontend_static", "index.html")
        return send_from_directory("frontend_dist", "index.html")
    return jsonify({"message": "Frontend not built yet. Run: npm run build in mirofish/"}), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_pred_db()
    port = int(os.getenv("PORT", "5050"))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    logger.info("Betting API server starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
