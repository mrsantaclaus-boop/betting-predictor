# SAI Tipster — Cloud Deployment Guide

Deploy to the cloud for free: always-on access, live odds, auto-refreshing news.

---

## Architecture

```
Browser (anywhere)
      ↓ HTTPS
Render.com (free)
  ├── Flask API (port 8080)    ← serves Vue.js + all API routes
  │     ├── /api/fixtures      ← football-data.org
  │     ├── /api/odds/*        ← The Odds API (live bookmaker odds)
  │     ├── /api/news/*        ← NewsAPI + GNews + BBC Sport
  │     ├── /api/predict       ← triggers MiroFish simulation
  │     └── /api/standings/*   ← league tables
  └── MiroFish backend (port 5001, internal)
        └── Groq API (LLM) ← gpt-oss-120b / gpt-oss-20b / qwen (free; llama-3.3-70b was retired by Groq in Aug 2026)
```

All data has TTL cache (1-hr odds, 90-min news) so API budgets are respected.

---

## Free Tier Limits

| Service | Free limit | How we use it |
|---------|-----------|---------------|
| Render | 750 hrs/month, sleeps after 15min idle | Wakes in ~30s |
| Groq | Generous free rate limit | ~1 simulation per day |
| football-data.org | 10 req/min | Cached 6 hrs |
| The Odds API | 500 credits/month, tracked at /api/odds/quota (some competitions are switched off to save credits) | Cached 1 hr |
| NewsAPI | 100 req/day | Cached 90 min |
| GNews | 100 req/day | Cached 90 min |

---

## Step-by-Step Deployment

### 1. Get your API keys

| Key | Where to get it | Free? |
|-----|----------------|-------|
| `LLM_API_KEY` | https://console.groq.com | ✅ Yes |
| `FOOTBALL_DATA_API_KEY` | https://www.football-data.org/client/register | ✅ Yes |
| `ODDS_API_KEY` | https://the-odds-api.com/#get-access | ✅ Yes (500/mo) |
| `NEWS_API_KEY` | https://newsapi.org/register | ✅ Yes (100/day) |
| `GNEWS_API_KEY` | https://gnews.io/register | ✅ Yes (100/day) |

### 2. Push code to GitHub

1. Create a new GitHub repository (public or private)
2. Push this entire `betting-predictor` folder to it:

```bash
git init
git add .
git commit -m "Initial SAI Tipster"
git remote add origin https://github.com/YOUR_USERNAME/sai-tipster
git push -u origin main
```

### 3. Deploy on Render

1. Go to https://render.com and sign up (free, GitHub login works)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repo
4. Render will auto-detect the `render.yaml` and configure everything
5. In **"Environment Variables"** section, add all your API keys:
   - `LLM_API_KEY`
   - `FOOTBALL_DATA_API_KEY`
   - `ODDS_API_KEY`
   - `NEWS_API_KEY`
   - `GNEWS_API_KEY`
6. Click **Deploy**

First deploy takes ~5-8 minutes (building Docker image).

### 4. Open your app

Render gives you a URL like: `https://sai-tipster.onrender.com`

Bookmark it. The app will work from any device, anywhere.

---

## About the sleep behaviour (free tier)

Render's free tier sleeps the container after 15 minutes of inactivity.

- **First visit after idle:** ~30 seconds to wake up
- **While active:** instant responses
- **Predictions:** take 3-5 minutes regardless (MiroFish simulation)

This is fine for personal use. If it bothers you, **upgrade to Render's
Starter plan ($7/month)** and the sleep is disabled.

---

## Updating the app

To redeploy after code changes:
```bash
git add .
git commit -m "update"
git push
```
Render auto-deploys on every push to `main`.

---

## Local development

Still works locally — run `start.bat` as before.
For cloud-specific local testing:
```bash
cp .env.cloud.example .env
python api_server.py
```
