"""
data/odds_quota.py — The Odds API credit tracking, persisted across restarts.

The free plan has 500 credits/month and the TTL cache lives in SQLite on
Render's ephemeral disk, so it is wiped at every restart. This module keeps the
last known credit state in the predictions database (PostgreSQL via
DATABASE_URL, SQLite locally) and offers:

  - record() / record_exhausted(): update the state from API responses;
  - is_blocked(): circuit breaker — after OUT_OF_USAGE_CREDITS, no more calls
    until the monthly reset (ODDS_QUOTA_RESET_DAY, default 1);
  - is_low(): below ODDS_LOW_QUOTA credits, skip optional calls (BTTS);
  - summary(): data for GET /api/odds/quota, watched daily by Fedor.

Competitions in ODDS_DISABLED_CODES never request odds (Brazil and World Cup
qualifiers by default — decided 2026-10-25 to save credits).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LOW_QUOTA = int(os.getenv("ODDS_LOW_QUOTA", "75"))
RESET_DAY = int(os.getenv("ODDS_QUOTA_RESET_DAY", "1"))
DISABLED = frozenset(
    c.strip().upper()
    for c in os.getenv("ODDS_DISABLED_CODES", "BSA,WCQE,WCQA,WCQC,WCQAS,WCQAF").split(",")
    if c.strip()
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def next_reset(now: datetime, reset_day: int = RESET_DAY) -> date:
    """First reset date strictly after today (the reset day itself counts as next month)."""
    if now.day < reset_day:
        return date(now.year, now.month, reset_day)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return date(year, month, reset_day)


# ── Storage ──────────────────────────────────────────────────────────────────

class SqliteStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS odds_quota_state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")

    def get(self) -> dict:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT data FROM odds_quota_state WHERE id = 1").fetchone()
        return json.loads(row[0]) if row else {}

    def put(self, state: dict) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT OR REPLACE INTO odds_quota_state (id, data) VALUES (1, ?)", (json.dumps(state),))


class PgStore:
    def __init__(self, url: str):
        import psycopg2
        self._connect = lambda: psycopg2.connect(url, connect_timeout=5)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE TABLE IF NOT EXISTS odds_quota_state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")

    def get(self) -> dict:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT data FROM odds_quota_state WHERE id = 1")
            row = cur.fetchone()
        return json.loads(row[0]) if row else {}

    def put(self, state: dict) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO odds_quota_state (id, data) VALUES (1, %s) "
                "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data",
                (json.dumps(state),),
            )


_store = None


def set_store(store) -> None:
    """Override the store (tests); None resets to the default."""
    global _store
    _store = store


def get_store():
    global _store
    if _store is None:
        url = os.getenv("DATABASE_URL")
        try:
            _store = PgStore(url) if url else None
        except Exception as e:  # pragma: no cover - depends on the environment
            logger.warning("odds quota: PostgreSQL unavailable, using SQLite: %s", e)
        if _store is None:
            _store = SqliteStore(os.getenv("PRED_DB_PATH", "./data/predictions.db"))
    return _store


def _load() -> dict:
    try:
        return get_store().get()
    except Exception as e:
        logger.warning("odds quota: cannot read state: %s", e)
        return {}


def _save(state: dict) -> None:
    try:
        get_store().put(state)
    except Exception as e:
        logger.warning("odds quota: cannot save state: %s", e)


# ── State updates and checks ─────────────────────────────────────────────────

def record(remaining: Optional[int], used: Optional[int], now: Optional[datetime] = None) -> None:
    now = now or _now()
    state = _load()
    if remaining is not None:
        state["remaining"] = int(remaining)
    if used is not None:
        state["used"] = int(used)
    state["updated_at"] = now.isoformat()
    if remaining is not None and int(remaining) <= 0:
        state["exhausted_until"] = next_reset(now).isoformat()
    else:
        state.pop("exhausted_until", None)
    _save(state)


def record_exhausted(now: Optional[datetime] = None) -> None:
    now = now or _now()
    state = _load()
    state.update(remaining=0, updated_at=now.isoformat(), exhausted_until=next_reset(now).isoformat())
    _save(state)


def is_blocked(now: Optional[datetime] = None) -> bool:
    until = _load().get("exhausted_until")
    if not until:
        return False
    return (now or _now()).date() < date.fromisoformat(until)


def is_low() -> bool:
    remaining = _load().get("remaining")
    return remaining is not None and remaining < LOW_QUOTA


def summary(now: Optional[datetime] = None) -> dict:
    now = now or _now()
    state = _load()
    remaining = state.get("remaining")
    reset_on = next_reset(now)
    days = max(1, (reset_on - now.date()).days)
    if is_blocked(now):
        status = "exhausted"
    elif remaining is None:
        status = "unknown"
    elif remaining < LOW_QUOTA:
        status = "low"
    else:
        status = "ok"
    return {
        "status": status,
        "remaining": remaining,
        "used": state.get("used"),
        "updated_at": state.get("updated_at"),
        "exhausted_until": state.get("exhausted_until") if status == "exhausted" else None,
        "reset_on": reset_on.isoformat(),
        "days_to_reset": days,
        "daily_allowance": (remaining // days) if remaining is not None else None,
        "low_threshold": LOW_QUOTA,
        "disabled_competitions": sorted(DISABLED),
    }
