#!/usr/bin/env python3
"""
scripts/daily_agent.py — the autonomous daily run of SAI Tipster.

Runs from GitHub Actions (.github/workflows/daily.yml) twice a day:
  full : wake the service, sync results, backfill corner/card stats,
         predict every fixture kicking off within SAI_HORIZON_HOURS that has
         no prediction yet, then print today's picks.
  sync : wake the service and sync results only (evening run).

Environment:
  SAI_BASE_URL       e.g. https://betting-predictor-i9rt.onrender.com
  SAI_ADMIN_TOKEN    value of ADMIN_TOKEN on Render
  SAI_HORIZON_HOURS  default 36
Exit code is non-zero when the service cannot be reached or when most
prediction calls fail, so a failed workflow run triggers GitHub's e-mail.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

BASE = os.environ.get("SAI_BASE_URL", "").rstrip("/")
TOKEN = os.environ.get("SAI_ADMIN_TOKEN", "")
HORIZON_H = int(os.environ.get("SAI_HORIZON_HOURS", "36"))
MODE = (sys.argv[1] if len(sys.argv) > 1 else "full").strip().lower()

if not BASE or not TOKEN:
    print("SAI_BASE_URL and SAI_ADMIN_TOKEN are required", file=sys.stderr)
    sys.exit(2)

HEADERS = {"X-Admin-Token": TOKEN, "Accept": "application/json"}
summary_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    summary_lines.append(msg)


def get(path: str, timeout: int = 120):
    r = requests.get(f"{BASE}{path}", headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post(path: str, timeout: int = 300):
    r = requests.post(f"{BASE}{path}", headers=HEADERS, timeout=timeout)
    if r.status_code in (401, 503):
        raise SystemExit(f"{path} -> {r.status_code}: {r.text[:200]} (check SAI_ADMIN_TOKEN / ADMIN_TOKEN)")
    r.raise_for_status()
    return r.json()


def wake() -> dict:
    """Render free tier sleeps; the first request can take up to ~90 s."""
    last = None
    for attempt in range(8):
        try:
            r = requests.get(f"{BASE}/api/health", timeout=90)
            if r.ok:
                return r.json()
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        time.sleep(15)
    raise SystemExit(f"service unreachable after retries: {last}")


def main() -> int:
    health = wake()
    log(f"health: llm_enabled={health.get('llm_enabled')} odds_quota={health.get('odds_quota')} "
        f"predictions={health.get('predictions')}")
    if not health.get("admin_configured", True):
        raise SystemExit("ADMIN_TOKEN is not configured on the server")

    # 1. Results of played matches (+ corners/cards/referee)
    sync = post("/api/results/sync")
    log(f"sync: updated={len(sync.get('updated', []))} skipped={sync.get('skipped_count')} "
        f"errors={len(sync.get('errors', []))}")
    for u in sync.get("updated", [])[:20]:
        log(f"  resolved {u.get('fixture_id')}: {u.get('score')} corners={u.get('corners')} cards={u.get('cards')}")

    if MODE == "sync":
        return 0

    # 2. Historical corner/card outcomes still missing (batches of 40)
    bf = post("/api/results/backfill-stats?limit=40")
    log(f"backfill-stats: updated={len(bf.get('updated', []))} failed={len(bf.get('failed', []))} "
        f"remaining={bf.get('remaining')}")

    # 3. Predictions for fixtures within the horizon
    fixtures = get("/api/fixtures")
    existing = {p["fixture_id"] for p in get("/api/predictions?limit=500")}
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=HORIZON_H)
    todo = []
    for f in fixtures:
        try:
            ko = datetime.fromisoformat(f["match_date"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if now <= ko <= horizon and f["fixture_id"] not in existing:
            todo.append(f)
    log(f"fixtures within {HORIZON_H}h without prediction: {len(todo)}")

    ok = failed = 0
    for f in todo:
        label = f"{f['home_team']} vs {f['away_team']} ({f['competition_code']})"
        try:
            r = requests.post(f"{BASE}/api/predict", json={"fixture_id": f["fixture_id"]},
                              headers=HEADERS, timeout=300)
            if r.ok:
                p = r.json()
                ok += 1
                log(f"  predicted {label}: 1X2 {p.get('home_win_pct')}/{p.get('draw_pct')}/{p.get('away_win_pct')} "
                    f"conf={p.get('confidence')} llm={'yes' if p.get('llm_analysis') else 'no'}")
            else:
                failed += 1
                log(f"  FAILED {label}: HTTP {r.status_code} {r.text[:120]}")
        except requests.RequestException as e:
            failed += 1
            log(f"  FAILED {label}: {e}")
        time.sleep(3)

    # 4. Today's picks
    picks = get("/api/daily-picks?limit=8")
    log(f"daily picks: {len(picks)}")
    for c in picks:
        b = c["bet"]
        log(f"  {c['match_label']} — {b['market']} @ {b['odds']} (model {b['model_pct']}% vs implied "
            f"{b['implied_pct']}%, edge +{b['edge']}%, conf {c['confidence']}"
            f"{', STALE ODDS' if c.get('odds_stale') else ''})")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("## SAI Tipster daily run\n\n```\n" + "\n".join(summary_lines) + "\n```\n")

    if todo and failed > ok:
        print(f"most predictions failed ({failed}/{len(todo)})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
