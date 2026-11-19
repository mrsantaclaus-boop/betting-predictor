# SAI Tipster

Football match predictions with probabilities, built as a hobby at zero cost: a Poisson / Dixon-Coles model blended with a small panel of LLM "analysts" (MiroFish), plus a ledger that checks honestly whether the model beats the bookmakers.

Live: https://betting-predictor-i9rt.onrender.com (Render free tier: the first load after idle takes ~30 s).

---

## What it covers

**Competitions:** Serie A, La Liga, Champions League, Europa League, Conference League, UEFA Super Cup, World Cup and its qualifiers, Brasileirao.
(Serie B is not offered: football-data.org does not cover it.)

**Markets predicted for each match:** 1X2, Over/Under 2.5 and 3.5 goals, BTTS, corners, yellow cards, red card, most likely scorelines.

**Tabs:** Fixtures, Predictions, Standings (with last-5 form), Value Bets, Edge (the ledger: does the model earn its keep?).

**Odds:** value bets compare the model with bookmaker odds from The Odds API. Odds are switched off for competitions where credits are scarce (`ODDS_DISABLED_CODES`, default: La Liga, Brasileirao, World Cup qualifiers); predictions work without them.

---

## How a prediction is made

1. Data for the match: fixtures, standings and form from football-data.org; xG (proxy from shots on target), corners and cards from ESPN match summaries; H2H and news as context.
2. Per-team stats are shrunk toward league averages (`predictor/shrinkage.py`). League constants are in `predictor/poisson.py` and `predictor/shrinkage.py`; for La Liga they are measured from ESPN results, and `tests/test_poisson.py` pins them.
3. Poisson / Dixon-Coles gives the scoreline grid and every market probability (`predictor/poisson.py`).
4. A panel of LLM personas (MiroFish, via Groq) is blended in when `USE_LLM=true`.
5. The prediction is saved (Postgres, via `DATABASE_URL`). When the match ends, results are synced and every market is settled.
6. `/api/edge`, `/api/ledger` and `/api/model-calibration` measure the model against real results and, with true odds only, against the market.

Confidence (high / medium / low) drops when a team has missing stats; missing data is never invented, it falls back to the league average.

---

## Data sources

| Source | What it provides | Key |
|--------|------------------|-----|
| football-data.org | Fixtures, standings, results, team form (Serie A, La Liga, Champions, Europa, Super Cup...) | `FOOTBALL_DATA_API_KEY` |
| ESPN public API | Fixtures/results for ECL, Brasileirao, World Cup; match stats (corners, cards, shots) for all competitions | none |
| The Odds API | Bookmaker odds, value bets, closing-line value | `ODDS_API_KEY` (credit quota at `/api/odds/quota`) |
| Groq | LLM panel and judge | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` |
| NewsAPI / GNews | Match news | `NEWS_API_KEY`, `GNEWS_API_KEY` |
| FBref | Optional extra stats (scraped, often blocked) | none |

Constraint of the project: **zero cost**. No paid data sources.

---

## Run it locally

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                              # then fill in the keys
python api_server.py                                # http://localhost:5050
```

The front end is a single file, `frontend_static/index.html` (Vue 3 from a CDN, no build step), served by the same Flask app.

Without `DATABASE_URL` predictions are kept in a local SQLite file. The MiroFish engine is installed by `setup.bat` (see `apply_patches.bat`); the Docker image does the same.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

CI (`.github/workflows/ci.yml`) runs the tests on every pull request.

---

## Deploy and automation

- **Render** (Docker, free tier) runs `docker-start.sh` (Gunicorn). Details in [README-CLOUD.md](README-CLOUD.md); `render.yaml` documents the settings.
- **Daily agent** (`.github/workflows/daily.yml`, `scripts/daily_agent.py`): 07:00 UTC sync + predict + picks, 17:00 UTC closing odds for tonight's value bets, 22:00 UTC sync of the day's results. It can also be started by hand (`gh workflow run daily.yml -f mode=full`).
- **Admin routes** (backfill, delete, cache clear, odds diagnostics) need the `X-Admin-Token` header (`ADMIN_TOKEN`). Predicting and syncing results are public.
- **Shadow trial:** variants of a home/draw bias fix are stored next to the production predictions and compared at `/api/shadow-report` (trial runs until 23/12/2026; production is unchanged).

---

## Project structure

```
betting-predictor/
├── api_server.py            Flask API + serves the front end
├── frontend_static/         index.html (Vue 3 via CDN)
├── football/                data clients: football_data_api, espn_client, fbref_scraper, team_names
├── data/                    odds_api, odds_quota, news_fetcher, cache
├── predictor/               orchestrator, poisson, shrinkage, xg_proxy, shadow, mirofish_client
├── seed/                    match report generator (input for the LLM panel)
├── scripts/daily_agent.py   automation run by the GitHub workflow
├── tests/                   pytest suite
├── patches/ + Dockerfile    MiroFish customisations and image
└── render.yaml              Render settings (documentation)
```
