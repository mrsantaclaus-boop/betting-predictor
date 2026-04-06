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
  POST /api/results/sync
  GET  /api/edge
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

from football import FootballDataClient, ApiFootballClient
from data.odds_api import OddsAPIClient
from data.news_fetcher import NewsFetcher
from data.cache import get_cache, TTL
from predictor.orchestrator import BettingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend_static", static_url_path="")
CORS(app, origins="*")

_orchestrator: BettingOrchestrator | None = None
_fd_client: FootballDataClient | None = None
_af_client: ApiFootballClient | None = None
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


_AF_CODES = {"WC", "WCQA", "WCQC", "WCQAS", "WCQAF"}


def get_af() -> ApiFootballClient:
    global _af_client
    if _af_client is None:
        _af_client = ApiFootballClient()
    return _af_client


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


# ── Predictions DB ──────────────────────────────────────────────────────────────────

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


# Initialise DB immediately at import time (works under Gunicorn/Render, not just __main__)
_init_pred_db()


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


def _update_pred_data(fixture_id: int, patch: dict) -> bool:
    """Merge patch dict into the existing data JSON blob for a prediction."""
    with sqlite3.connect(_PRED_DB) as conn:
        row = conn.execute(
            "SELECT data FROM predictions WHERE fixture_id = ?", (fixture_id,)
        ).fetchone()
        if not row:
            return False
        data = json.loads(row[0])
        data.update(patch)
        conn.execute(
            "UPDATE predictions SET data = ? WHERE fixture_id = ?",
            (json.dumps(data), fixture_id),
        )
    return True


def _load_resolved_preds() -> list[dict]:
    """Load predictions that have outcomes recorded."""
    with sqlite3.connect(_PRED_DB) as conn:
        try:
            rows = conn.execute(
                "SELECT fixture_id, data FROM predictions "
                "WHERE json_extract(data, '$.outcomes') IS NOT NULL"
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback for older SQLite without json_extract
            all_rows = conn.execute(
                "SELECT fixture_id, data FROM predictions"
            ).fetchall()
            rows = [(fid, raw) for fid, raw in all_rows
                    if "outcomes" in json.loads(raw)]
    out = []
    for fid, raw in rows:
        entry = json.loads(raw)
        entry["fixture_id"] = fid
        out.append(entry)
    return out


# ── Outcome helpers ────────────────────────────────────────────────────────────────────

def _compute_outcomes(hs: int, as_: int) -> dict:
    """Determine which betting markets hit based on final score."""
    total = hs + as_
    return {
        "home_win":  hs > as_,
        "draw":      hs == as_,
        "away_win":  hs < as_,
        "over_2_5":  total > 2,
        "under_2_5": total <= 2,
        "over_3_5":  total > 3,
        "under_3_5": total <= 3,
        "btts_yes":  hs > 0 and as_ > 0,
        "btts_no":   hs == 0 or as_ == 0,
        # corners and cards not trackable on free API tier
    }


# Maps market key → consensus odds field name in live_odds.consensus
_ODDS_KEY_MAP: dict[str, str] = {
    "home_win":         "home",
    "draw":             "draw",
    "away_win":         "away",
    "over_2_5":         "over_2_5",
    "under_2_5":        "under_2_5",
    "btts_yes":         "btts_yes",
    "btts_no":          "btts_no",
    # over_3_5/under_3_5, corners, cards have no odds counterpart — always use ai_pct
}

_COUNTERPART: dict[str, str] = {
    "over_2_5":           "under_2_5",    "under_2_5":          "over_2_5",
    "over_3_5":           "under_3_5",    "under_3_5":          "over_3_5",
    "btts_yes":           "btts_no",      "btts_no":            "btts_yes",
    "over_9_5_corners":   "under_9_5_corners", "under_9_5_corners": "over_9_5_corners",
    "over_3_5_cards":     "under_3_5_cards",   "under_3_5_cards":   "over_3_5_cards",
}


def _implied_pct(market_key: str, consensus: dict, ai_pct: float) -> float:
    """
    Convert decimal odds to an implied probability (de-vigged).
    Falls back to ai_pct if odds are absent or zero.
    """
    odds_key = _ODDS_KEY_MAP.get(market_key)
    if not odds_key or not consensus:
        return ai_pct

    price = consensus.get(odds_key, 0)
    if not price or price <= 1.0:
        return ai_pct

    # 1X2 — three-way de-vig
    if market_key in ("home_win", "draw", "away_win"):
        home_p = 1 / consensus["home"] if consensus.get("home") else 0
        draw_p = 1 / consensus["draw"] if consensus.get("draw") else 0
        away_p = 1 / consensus["away"] if consensus.get("away") else 0
        total_raw = home_p + draw_p + away_p
        if total_raw == 0:
            return ai_pct
        return round(1 / price / total_raw * 100, 1)

    # Paired markets — two-way de-vig
    counter_key = _COUNTERPART.get(market_key)
    if counter_key:
        counter_odds_key = _ODDS_KEY_MAP.get(counter_key, "")
        counter_price = consensus.get(counter_odds_key, 0)
        if counter_price and counter_price > 1.0:
            raw_this = 1 / price
            raw_counter = 1 / counter_price
            return round(raw_this / (raw_this + raw_counter) * 100, 1)

    return round(1 / price * 100, 1)


# ── Helpers ─────────────────────────────────────────────────────────────────────────────

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


# ── Routes ────────────────────────────────────────────────────────────────────────────

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
    VALID = {"SA", "CL", "WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF"}
    if competition_code not in VALID:
        return jsonify({"error": f"Unknown competition: {competition_code}"}), 400

    def fetch():
        client = get_af() if competition_code in _AF_CODES else get_fd()
        rows = client.get_standings(competition_code)
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
    VALID = {"SA", "CL", "WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF"}
    if competition_code not in VALID:
        return jsonify({"error": f"Unknown competition: {competition_code}"}), 400
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
        # Fetch odds first so they are persisted alongside the prediction
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
        _save_pred(
            fixture_id,
            f"{fixture.home_team} vs {fixture.away_team}",
            fixture.competition,
            data,
        )
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


@app.route("/api/results/sync", methods=["POST"])
def results_sync():
    """
    Fetch final scores for all unresolved predictions and record outcomes.
    Idempotent — already-resolved predictions are skipped.
    """
    preds = _load_preds()
    updated = []
    skipped = 0
    errors = []

    for p in preds:
        fid = p["fixture_id"]

        if p.get("result_fetched_at"):
            skipped += 1
            continue

        try:
            fixture = get_fd().get_match_result(fid)
        except Exception as e:
            logger.warning("Could not fetch result for %d: %s", fid, e)
            errors.append({"fixture_id": fid, "error": str(e)})
            continue

        if fixture is None:
            skipped += 1
            continue

        hs, as_ = fixture.home_score, fixture.away_score
        if hs is None or as_ is None:
            skipped += 1
            continue

        outcomes = _compute_outcomes(hs, as_)
        patch = {
            "actual_score": {"home": hs, "away": as_},
            "outcomes": outcomes,
            "result_fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        _update_pred_data(fid, patch)
        updated.append({"fixture_id": fid, "score": f"{hs}-{as_}"})

    return jsonify({
        "updated": updated,
        "skipped_count": skipped,
        "errors": errors,
    })


@app.route("/api/edge")
def edge_stats():
    """
    Returns per-market edge stats across all resolved predictions.
    Edge = actual_pct - implied_pct.
    implied_pct is derived from live_odds consensus (de-vigged), or AI prediction % if no odds.
    """
    resolved = _load_resolved_preds()

    # market_key → (ai_pred_field, outcome_key)
    MARKET_MAP = {
        "home_win":           ("home_win_pct",           "home_win"),
        "draw":               ("draw_pct",               "draw"),
        "away_win":           ("away_win_pct",           "away_win"),
        "over_2_5":           ("over_2_5_pct",           "over_2_5"),
        "under_2_5":          ("under_2_5_pct",          "under_2_5"),
        "over_3_5":           ("over_3_5_pct",           "over_3_5"),
        "under_3_5":          ("under_3_5_pct",          "under_3_5"),
        "btts_yes":           ("btts_yes_pct",           "btts_yes"),
        "btts_no":            ("btts_no_pct",            "btts_no"),
        "over_9_5_corners":   ("over_9_5_corners_pct",  "over_9_5_corners"),
        "under_9_5_corners":  ("under_9_5_corners_pct", "under_9_5_corners"),
        "over_3_5_cards":     ("over_3_5_cards_pct",    "over_3_5_cards"),
        "under_3_5_cards":    ("under_3_5_cards_pct",   "under_3_5_cards"),
        "red_card":           ("red_card_pct",           "red_card"),
    }

    # Accumulate per-market: list of (implied_pct, did_hit)
    accum: dict[str, list[tuple[float, bool]]] = {k: [] for k in MARKET_MAP}

    for p in resolved:
        outcomes = p.get("outcomes", {})
        consensus = p.get("live_odds", {}).get("consensus", {})

        for mkt_key, (pred_field, outcome_key) in MARKET_MAP.items():
            did_hit = outcomes.get(outcome_key)
            if did_hit is None:
                continue
            implied = _implied_pct(mkt_key, consensus, p.get(pred_field, 0.0))
            accum[mkt_key].append((implied, did_hit))

    result = {}
    for mkt_key, data_points in accum.items():
        n = len(data_points)
        if n == 0:
            result[mkt_key] = {
                "implied_pct": None, "actual_pct": None,
                "edge": None, "sample_size": 0,
            }
            continue
        avg_implied = round(sum(x[0] for x in data_points) / n, 1)
        actual = round(sum(1 for _, hit in data_points if hit) / n * 100, 1)
        result[mkt_key] = {
            "implied_pct": avg_implied,
            "actual_pct":  actual,
            "edge":        round(actual - avg_implied, 1),
            "sample_size": n,
        }

    return jsonify(result)


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


# ── Serve Vue.js frontend (production) ────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path: str):
    """Serve the static frontend for all non-API routes."""
    static = Path("frontend_static")
    if path and (static / path).exists():
        return send_from_directory("frontend_static", path)
    return send_from_directory("frontend_static", "index.html")


# ── Entry point ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_pred_db()
    port = int(os.getenv("PORT", "5050"))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    logger.info("Betting API server starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
