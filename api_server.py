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
  DELETE /api/predictions/unplayed
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# PostgreSQL support (Supabase)
_DATABASE_URL = os.getenv("DATABASE_URL")
if _DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
        _USE_PG = True
    except ImportError:
        _USE_PG = False
else:
    _USE_PG = False

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from football import FootballDataClient, ApiFootballClient, FBrefScraper
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
_fbref_client: FBrefScraper | None = None
_odds_client: OddsAPIClient | None = None
_news_fetcher: NewsFetcher | None = None
_cache = get_cache()

_PRED_DB = Path(os.getenv("PRED_DB_PATH", "data/predictions.db"))


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


_AF_CODES: set[str] = set()  # API Football key unavailable; WCQ/BSA now via ESPN


def get_af() -> ApiFootballClient:
    global _af_client
    if _af_client is None:
        _af_client = ApiFootballClient()
    return _af_client


def get_fbref() -> FBrefScraper:
    global _fbref_client
    if _fbref_client is None:
        _fbref_client = FBrefScraper()
    return _fbref_client


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

def _get_conn():
    """Return a DB connection — PostgreSQL if DATABASE_URL set, else SQLite."""
    if _USE_PG:
        try:
            return psycopg2.connect(_DATABASE_URL, connect_timeout=5)
        except Exception as e:
            logger.warning("PostgreSQL connection failed, falling back to SQLite: %s", e)
    _PRED_DB.parent.mkdir(exist_ok=True)
    return sqlite3.connect(_PRED_DB)


def _is_pg_conn(conn) -> bool:
    return _USE_PG and hasattr(conn, 'cursor')


def _init_pred_db():
    conn = _get_conn()
    if _is_pg_conn(conn):
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id          SERIAL PRIMARY KEY,
                        fixture_id  INTEGER NOT NULL UNIQUE,
                        match_label TEXT    NOT NULL,
                        competition TEXT    NOT NULL,
                        created_at  TEXT    NOT NULL,
                        data        TEXT    NOT NULL
                    )
                """)
    else:
        with conn:
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
    ts = datetime.now(timezone.utc).isoformat()
    raw = json.dumps(data)
    if _USE_PG:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (fixture_id, match_label, competition, created_at, data) "
                    "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (fixture_id) DO UPDATE "
                    "SET match_label=EXCLUDED.match_label, competition=EXCLUDED.competition, "
                    "created_at=EXCLUDED.created_at, data=EXCLUDED.data",
                    (fixture_id, match_label, competition, ts, raw),
                )
    else:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO predictions "
                "(fixture_id, match_label, competition, created_at, data) VALUES (?,?,?,?,?)",
                (fixture_id, match_label, competition, ts, raw),
            )


def _load_preds() -> list[dict]:
    if _USE_PG:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT fixture_id, match_label, competition, created_at, data "
                    "FROM predictions ORDER BY created_at DESC LIMIT 100"
                )
                rows = cur.fetchall()
    else:
        with _get_conn() as conn:
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
    if _USE_PG:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT data FROM predictions WHERE fixture_id = %s", (fixture_id,))
                row = cur.fetchone()
                if not row:
                    return False
                data = json.loads(row[0])
                data.update(patch)
                cur.execute("UPDATE predictions SET data = %s WHERE fixture_id = %s",
                            (json.dumps(data), fixture_id))
    else:
        with _get_conn() as conn:
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
    if _USE_PG:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT fixture_id, data FROM predictions "
                    "WHERE data::jsonb ? 'outcomes'"
                )
                rows = cur.fetchall()
    else:
        with _get_conn() as conn:
            try:
                rows = conn.execute(
                    "SELECT fixture_id, data FROM predictions "
                    "WHERE json_extract(data, '$.outcomes') IS NOT NULL"
                ).fetchall()
            except sqlite3.OperationalError:
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

def _compute_outcomes(hs: int, as_: int, corners: dict | None = None) -> dict:
    """
    Determine which betting markets hit based on final score.
    Pass corners={"home": N, "away": N} to also resolve corner markets.
    """
    total = hs + as_
    outcomes = {
        "home_win":  hs > as_,
        "draw":      hs == as_,
        "away_win":  hs < as_,
        "over_2_5":  total > 2,
        "under_2_5": total <= 2,
        "over_3_5":  total > 3,
        "under_3_5": total <= 3,
        "btts_yes":  hs > 0 and as_ > 0,
        "btts_no":   hs == 0 or as_ == 0,
    }
    if corners and "home" in corners and "away" in corners:
        total_corners = corners["home"] + corners["away"]
        outcomes["over_9_5_corners"]  = total_corners > 9
        outcomes["under_9_5_corners"] = total_corners <= 9
    return outcomes


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

MIN_EDGE_PCT = 3.0


def _odds_age_label(fetched_at: str | None) -> str:
    """Return human-readable age like '2h ago', empty string if unknown."""
    if not fetched_at:
        return ""
    try:
        dt = datetime.fromisoformat(fetched_at)
        age_s = (datetime.now(timezone.utc) - dt).total_seconds()
        if age_s < 3600:
            return f"{int(age_s // 60)}m ago"
        return f"{age_s / 3600:.1f}h ago"
    except (ValueError, TypeError):
        return ""


def _is_odds_stale(fetched_at: str | None, warn_hours: int = 12) -> bool:
    """Return True if odds are older than warn_hours or timestamp is absent."""
    if not fetched_at:
        return True
    try:
        dt = datetime.fromisoformat(fetched_at)
        return (datetime.now(timezone.utc) - dt).total_seconds() > warn_hours * 3600
    except (ValueError, TypeError):
        return True

_MARKET_LABELS: dict[str, str] = {
    "home_win": "Home Win", "draw": "Draw", "away_win": "Away Win",
    "over_2_5": "Over 2.5", "under_2_5": "Under 2.5",
    "over_3_5": "Over 3.5", "under_3_5": "Under 3.5",
    "btts_yes": "BTTS Yes", "btts_no": "BTTS No",
    "over_9_5_corners": "Corners +9.5", "under_9_5_corners": "Corners -9.5",
    "over_3_5_cards": "Cards +3.5", "under_3_5_cards": "Cards -3.5",
    "red_card": "Red Card",
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
                "is_neutral": f.is_neutral,
            }
            for f in fixtures
        ]
    return _cache.get_or_fetch("fixtures:all", fetch, TTL["fixtures"])


# ── Routes ────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    import os as _os
    # Count predictions with at least one valid xG value as a data quality proxy
    try:
        preds = _load_preds()
        total_preds = len(preds)
        resolved = sum(1 for p in preds if p.get("result_fetched_at"))
        with_odds = sum(1 for p in preds if p.get("live_odds", {}).get("consensus"))
    except Exception:
        total_preds = resolved = with_odds = 0

    return jsonify({
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "odds_quota": get_odds().quota_remaining,
        "llm_enabled": _os.getenv("USE_LLM", "false").lower() == "true",
        "predictions": {
            "total": total_preds,
            "resolved": resolved,
            "with_odds": with_odds,
        },
    })


@app.route("/api/fixtures")
def fixtures():
    try:
        data = _fixture_list() or []
        return jsonify(data)
    except Exception as e:
        logger.exception("fixtures error")
        return jsonify({"error": str(e)}), 500


_ALL_COMP_CODES = {"SA", "SB", "CL", "EL", "ECL", "USC", "WC", "WCQE", "WCQA", "WCQC", "WCQAS", "WCQAF", "BSA"}


@app.route("/api/standings/<competition_code>")
def standings(competition_code: str):
    VALID = _ALL_COMP_CODES
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
    VALID = _ALL_COMP_CODES
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
            is_neutral=fixture_data.get("is_neutral", False),
        )

        prediction = get_orch().predict_fixture(fixture)
        data = prediction.to_dict()
        data["competition_code"] = fixture.competition_code  # stored for per-competition analysis
        # Fetch odds first so they are persisted alongside the prediction
        try:
            if os.getenv("ODDS_API_KEY"):
                odds = get_odds().get_fixture_odds(
                    fixture.competition_code,
                    fixture.home_team,
                    fixture.away_team,
                )
                data["live_odds"] = odds
                data["odds_fetched_at"] = datetime.now(timezone.utc).isoformat()
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
    Fetch final scores (and corner counts if API_FOOTBALL_KEY is set)
    for all unresolved predictions and record outcomes.
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

        # ── Corner stats via FBref scraping (no API key needed) ─────────────
        corners: dict = {}
        try:
            comp_code = getattr(fixture, "competition_code", "") or ""
            date_str  = fixture.match_date.strftime("%Y-%m-%d")
            corners = get_fbref().get_match_corners(
                comp_code,
                fixture.home_team,
                fixture.away_team,
                date_str,
            )
        except Exception as e:
            logger.debug("FBref corner fetch skipped for %d: %s", fid, e)

        outcomes = _compute_outcomes(hs, as_, corners or None)
        patch = {
            "actual_score": {"home": hs, "away": as_},
            "outcomes": outcomes,
            "result_fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        if corners:
            patch["corner_stats"] = corners  # store raw counts for reference

        _update_pred_data(fid, patch)
        updated.append({
            "fixture_id": fid,
            "score": f"{hs}-{as_}",
            "corners": f"{corners.get('home','?')}-{corners.get('away','?')}" if corners else None,
        })

    return jsonify({
        "updated": updated,
        "skipped_count": skipped,
        "errors": errors,
    })


@app.route("/api/value-bets")
def value_bets():
    """
    Returns upcoming (unresolved) predictions with at least one market where
    model_pct - implied_pct >= MIN_EDGE_PCT (odds-backed markets only).
    Sorted by best edge descending.
    """
    preds = _load_preds()

    # Only markets where we can compare against real odds
    VALUE_MARKETS = {
        "home_win":  "home_win_pct",
        "draw":      "draw_pct",
        "away_win":  "away_win_pct",
        "over_2_5":  "over_2_5_pct",
        "under_2_5": "under_2_5_pct",
        "btts_yes":  "btts_yes_pct",
        "btts_no":   "btts_no_pct",
    }

    results = []
    for p in preds:
        if p.get("result_fetched_at"):
            continue
        consensus = p.get("live_odds", {}).get("consensus", {})
        if not consensus:
            continue

        bets = []
        for mkt_key, pred_field in VALUE_MARKETS.items():
            model_pct = p.get(pred_field, 0.0)
            if not model_pct:
                continue
            odds_key = _ODDS_KEY_MAP.get(mkt_key)
            if not odds_key:
                continue
            price = consensus.get(odds_key, 0)
            if not price or price <= 1.0:
                continue
            implied = _implied_pct(mkt_key, consensus, model_pct)
            edge = round(model_pct - implied, 1)
            if edge >= MIN_EDGE_PCT:
                bets.append({
                    "market_key": mkt_key,
                    "market":     _MARKET_LABELS.get(mkt_key, mkt_key),
                    "model_pct":  round(model_pct, 1),
                    "implied_pct": implied,
                    "edge":       edge,
                    "odds":       price,
                })

        if bets:
            bets.sort(key=lambda x: x["edge"], reverse=True)
            fetched_at = p.get("odds_fetched_at")
            results.append({
                "fixture_id":    p["fixture_id"],
                "match_label":   p.get("match_label", ""),
                "competition":   p.get("competition", ""),
                "odds_fetched_at": fetched_at,
                "odds_age":      _odds_age_label(fetched_at),
                "odds_stale":    _is_odds_stale(fetched_at),
                "bets":          bets,
            })

    results.sort(key=lambda x: max(b["edge"] for b in x["bets"]), reverse=True)
    return jsonify(results)


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

    from collections import defaultdict as _dd

    # comp_code from stored field; fall back to name→code mapping for older records
    _NAME_TO_CODE = {
        "Serie A": "SA", "Serie B": "SB", "Champions League": "CL",
        "UEFA Europa League": "EL", "UEFA Conference League": "ECL",
        "UEFA Super Cup": "USC", "FIFA World Cup": "WC",
        "WCQ Europe": "WCQE", "WCQ Americas": "WCQA", "WCQ CONCACAF": "WCQC",
        "WCQ Asia": "WCQAS", "WCQ Africa": "WCQAF", "Brasileirao Serie A": "BSA",
    }

    def _comp_code(p: dict) -> str:
        return (p.get("competition_code")
                or _NAME_TO_CODE.get(p.get("competition", ""), "?"))

    # Accumulate per-market overall + per-competition
    accum: dict[str, list] = {k: [] for k in MARKET_MAP}
    accum_by_comp: dict[str, dict[str, list]] = _dd(lambda: {k: [] for k in MARKET_MAP})

    for p in resolved:
        outcomes = p.get("outcomes", {})
        consensus = p.get("live_odds", {}).get("consensus", {})
        code = _comp_code(p)

        for mkt_key, (pred_field, outcome_key) in MARKET_MAP.items():
            did_hit = outcomes.get(outcome_key)
            if did_hit is None:
                continue
            implied = _implied_pct(mkt_key, consensus, p.get(pred_field, 0.0))
            accum[mkt_key].append((implied, did_hit))
            accum_by_comp[code][mkt_key].append((implied, did_hit))

    def _mkt_stats(data_points: list) -> dict:
        n = len(data_points)
        if n == 0:
            return {"implied_pct": None, "actual_pct": None,
                    "edge": None, "sample_size": 0, "calibration_buckets": []}
        avg_implied = round(sum(x[0] for x in data_points) / n, 1)
        actual = round(sum(1 for _, hit in data_points if hit) / n * 100, 1)
        buckets: dict[int, list] = _dd(list)
        for imp, hit in data_points:
            lo = min(int(imp // 10) * 10, 90)
            buckets[lo].append((imp, hit))
        cal_buckets = []
        for lo in sorted(buckets):
            pts = buckets[lo]
            cal_buckets.append({
                "range":      f"{lo}-{lo+10}%",
                "pred_avg":   round(sum(x[0] for x in pts) / len(pts), 1),
                "actual_pct": round(sum(1 for _, h in pts if h) / len(pts) * 100, 1),
                "n":          len(pts),
            })
        return {"implied_pct": avg_implied, "actual_pct": actual,
                "edge": round(actual - avg_implied, 1),
                "sample_size": n, "calibration_buckets": cal_buckets}

    result = {}
    for mkt_key in MARKET_MAP:
        result[mkt_key] = _mkt_stats(accum[mkt_key])

    # Add per-competition breakdown
    by_comp: dict[str, dict] = {}
    for code, mkt_accum in accum_by_comp.items():
        by_comp[code] = {mkt: _mkt_stats(pts) for mkt, pts in mkt_accum.items()}
    result["_by_competition"] = by_comp

    return jsonify(result)


@app.route("/api/model-calibration")
def model_calibration():
    """
    Per-market calibration curves for three prediction sources:
      - blend:   main fields (LLM+Poisson blend; pure Poisson for historical records)
      - llm:     llm_* fields (only predictions run with USE_LLM=true)
      - poisson: poisson_* fields, falling back to main fields for historical records
                 where llm_* are all zero (i.e. pre-blend era, Poisson-only)

    X-axis: model's own predicted probability (not bookmaker's implied odds).
    Y-axis: empirical hit rate within each 10% probability bucket.
    Also returns Brier score per source (lower = better, 0 = perfect).
    """
    resolved = _load_resolved_preds()

    MARKETS = {
        "home_win": ("home_win_pct",  "llm_home_win_pct", "poisson_home_win_pct", "home_win"),
        "draw":     ("draw_pct",      "llm_draw_pct",     "poisson_draw_pct",     "draw"),
        "away_win": ("away_win_pct",  "llm_away_win_pct", "poisson_away_win_pct", "away_win"),
        "over_2_5": ("over_2_5_pct", "llm_over_2_5_pct", "poisson_over_2_5_pct", "over_2_5"),
        "btts_yes": ("btts_yes_pct", "llm_btts_yes_pct", "poisson_btts_yes_pct", "btts_yes"),
    }

    def _stats(pts: list) -> dict | None:
        if not pts:
            return None
        n = len(pts)
        brier = round(sum((p / 100 - (1 if h else 0)) ** 2 for p, h in pts) / n, 4)
        from collections import defaultdict
        buckets: dict[int, list] = defaultdict(list)
        for p, h in pts:
            lo = min(int(p // 10) * 10, 90)
            buckets[lo].append((p, h))
        cal = []
        for lo in sorted(buckets):
            bp = buckets[lo]
            cal.append({
                "range":      f"{lo}-{lo+10}%",
                "pred_avg":   round(sum(x[0] for x in bp) / len(bp), 1),
                "actual_pct": round(sum(1 for _, h in bp if h) / len(bp) * 100, 1),
                "n":          len(bp),
            })
        return {"calibration_buckets": cal, "brier_score": brier, "sample_size": n}

    from collections import defaultdict as _dd2

    _NAME_TO_CODE2 = {
        "Serie A": "SA", "Serie B": "SB", "Champions League": "CL",
        "UEFA Conference League": "ECL", "FIFA World Cup": "WC",
        "WCQ Europe": "WCQE", "WCQ Americas": "WCQA", "WCQ CONCACAF": "WCQC",
        "WCQ Asia": "WCQAS", "WCQ Africa": "WCQAF", "Brasileirao Serie A": "BSA",
    }

    def _pcomp(p: dict) -> str:
        return (p.get("competition_code")
                or _NAME_TO_CODE2.get(p.get("competition", ""), "?"))

    # per-market overall + per-competition accumulators
    # structure: {mkt: {src: [pts]}}  and  {comp: {mkt: {src: [pts]}}}
    overall: dict[str, dict[str, list]] = {
        mkt: {"blend": [], "llm": [], "poisson": []} for mkt in MARKETS
    }
    by_comp: dict[str, dict[str, dict[str, list]]] = _dd2(
        lambda: {mkt: {"blend": [], "llm": [], "poisson": []} for mkt in MARKETS}
    )

    for p in resolved:
        code = _pcomp(p)
        for mkt_key, (blend_f, llm_f, poi_f, out_key) in MARKETS.items():
            did_hit = p.get("outcomes", {}).get(out_key)
            if did_hit is None:
                continue
            blend_pct   = p.get(blend_f, 0.0) or 0.0
            llm_pct     = p.get(llm_f,   0.0) or 0.0
            poisson_pct = p.get(poi_f,   0.0) or 0.0

            for store in (overall[mkt_key], by_comp[code][mkt_key]):
                if blend_pct > 0:
                    store["blend"].append((blend_pct, did_hit))
                if llm_pct > 0:
                    store["llm"].append((llm_pct, did_hit))
                if poisson_pct > 0:
                    store["poisson"].append((poisson_pct, did_hit))
                elif llm_pct == 0 and blend_pct > 0:
                    store["poisson"].append((blend_pct, did_hit))

    result = {}
    for mkt_key in MARKETS:
        result[mkt_key] = {src: _stats(pts) for src, pts in overall[mkt_key].items()}

    result["_by_competition"] = {
        code: {
            mkt: {src: _stats(pts) for src, pts in srcs.items()}
            for mkt, srcs in mkts.items()
        }
        for code, mkts in by_comp.items()
    }

    return jsonify(result)


def _delete_unplayed_preds() -> int:
    """Delete all predictions that have no recorded actual score (unplayed matches).
    Returns the number of rows deleted.
    """
    if _USE_PG:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM predictions "
                    "WHERE data::jsonb ->> 'actual_score' IS NULL "
                    "AND (data::jsonb -> 'outcomes') IS NULL"
                )
                return cur.rowcount
    else:
        with _get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM predictions "
                "WHERE json_extract(data, '$.actual_score') IS NULL "
                "AND json_extract(data, '$.outcomes') IS NULL"
            )
            return cur.rowcount


@app.route("/api/predictions/unplayed", methods=["DELETE"])
def delete_unplayed_predictions():
    """Delete stored predictions for matches that have not yet been played.
    Unplayed = no actual_score and no outcomes recorded.
    After calling this, predictions will be re-generated fresh on the next
    POST /api/predict request for each fixture.
    """
    try:
        deleted = _delete_unplayed_preds()
        return jsonify({"deleted": deleted})
    except Exception as e:
        logger.exception("delete unplayed predictions error")
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
