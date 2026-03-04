# SharpEdge AI — Phase 3: Delivery Design

**Date:** 2026-03-04
**Author:** Bala + Claude
**Status:** Approved
**Depends on:** Phase 1 (Data Foundation), Phase 2 (ML Engine)

---

## 1. Goal

Connect Phases 1+2 to the real world: automate daily predictions, deliver picks via Telegram, expose a public API for track record, and resolve results to build a verifiable performance history.

## 2. Architecture

**Monolith FastAPI + separate Telegram bot process, shared PostgreSQL.**

- **FastAPI server** — REST API for predictions, picks, track record, health. Also hosts the pipeline trigger endpoints (API-key protected).
- **Telegram bot** — Separate long-polling process. Broadcasts picks to a public channel and handles DM commands from individual users.
- **GitHub Actions** — Triggers the daily pipeline (8am UTC), result resolution (11pm UTC), and weekly retrain (Sunday 3am UTC) via HTTP calls to the API.
- **PostgreSQL** — Local for development, Supabase free tier for production (later).

No user auth, no payment, no dashboard in Phase 3. Everything is free.

```
GitHub Actions (cron)
    │
    ▼
FastAPI Server ──────── PostgreSQL
    │                       ▲
    │                       │
    ▼                       │
Telegram Bot ───────────────┘
    │
    ▼
@SharpEdgePicks Channel + DM Users
```

---

## 3. Daily Prediction Pipeline

### Flow

```
POST /api/pipeline/run (GitHub Actions, 08:00 UTC)
    │
    ├── 1. Collect today's fixtures (football-data.org)
    ├── 2. Run collectors for latest data (odds, xG, ELO, predictions)
    ├── 3. Build 50-feature matrix via FeaturePipeline
    ├── 4. Load trained model → predict (XGBoost + Poisson → ensemble → calibrate)
    ├── 5. Run 5-stage banker filter
    ├── 6. Save predictions + picks to DB (PipelineRun, Prediction, DailyPick)
    └── 7. Mark picks as pending_broadcast → bot picks them up
```

### Design Decisions

- **HTTP-triggered** — GitHub Actions calls the endpoint. Can also trigger manually via curl.
- **Idempotent** — Re-running for the same date overwrites predictions (upsert by date + match).
- **Pipeline status tracked** — `pipeline_runs` table records every run with timing and error info.
- **Bot notification** — Pipeline writes picks with `broadcasted_at = NULL`. Bot polls for unbroadcasted picks.

---

## 4. New Database Models

### PipelineRun

Tracks each pipeline execution.

```
PipelineRun:
    id                  INTEGER PRIMARY KEY
    run_date            DATE NOT NULL
    run_type            VARCHAR(20)     -- "predict", "resolve", "retrain"
    status              VARCHAR(20)     -- "running", "success", "failed"
    started_at          DATETIME
    completed_at        DATETIME
    fixtures_count      INTEGER
    predictions_count   INTEGER
    picks_count         INTEGER
    error_message       TEXT
```

### Prediction

Full model output for every match.

```
Prediction:
    id                  INTEGER PRIMARY KEY
    pipeline_run_id     INTEGER FK → pipeline_runs.id
    match_id            INTEGER FK → matches.id
    market              VARCHAR(20)     -- "1x2", "over_under", "btts"
    prob_home           FLOAT
    prob_draw           FLOAT
    prob_away           FLOAT
    prob_over           FLOAT
    prob_under          FLOAT
    prob_btts_yes       FLOAT
    prob_btts_no        FLOAT
    xgboost_probs       JSON
    poisson_probs       JSON
    ensemble_weights    JSON
    created_at          DATETIME

    UNIQUE(pipeline_run_id, match_id, market)
```

### DailyPick

Filtered banker picks — the core deliverable.

```
DailyPick:
    id                  INTEGER PRIMARY KEY
    pipeline_run_id     INTEGER FK → pipeline_runs.id
    prediction_id       INTEGER FK → predictions.id
    match_id            INTEGER FK → matches.id
    pick_market         VARCHAR(20)     -- "1x2_home", "over_25", "btts_yes"
    pick_selection      VARCHAR(50)     -- "Home Win", "Over 2.5", "BTTS Yes"
    model_prob          FLOAT
    model_spread        FLOAT
    best_odds           FLOAT
    bookmaker           VARCHAR(100)
    implied_prob        FLOAT
    edge                FLOAT
    tier                VARCHAR(20)     -- "platinum", "gold", "silver"
    meta_agreement      INTEGER
    risk_flags          JSON
    stake_flat          FLOAT           -- flat stake in units
    stake_kelly         FLOAT           -- Kelly stake in units
    result              VARCHAR(10)     -- NULL → "win" / "loss" / "void"
    profit_loss         FLOAT           -- NULL until resolved
    resolved_at         DATETIME
    broadcasted_at      DATETIME        -- NULL until sent to Telegram

    UNIQUE(pipeline_run_id, match_id, pick_market)
```

---

## 5. FastAPI Backend

### Module Structure

```
src/sharpedge/api/
    __init__.py
    app.py              # FastAPI app factory + lifespan
    deps.py             # Dependency injection (DB session, settings)
    auth.py             # API key validation (header-based)
    routers/
        __init__.py
        pipeline.py     # POST /api/pipeline/run, /resolve, /retrain
        predictions.py  # GET /api/predictions/today, /{date}
        picks.py        # GET /api/picks/today, /history
        track_record.py # GET /api/track-record, /by-tier, /by-league
        health.py       # GET /api/health
```

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `POST` | `/api/pipeline/run` | API Key | Trigger daily prediction pipeline |
| `POST` | `/api/pipeline/resolve` | API Key | Resolve yesterday's picks |
| `POST` | `/api/pipeline/retrain` | API Key | Retrain model with latest data |
| `GET` | `/api/pipeline/status` | API Key | Last pipeline run status |
| `GET` | `/api/predictions/today` | Public | Today's match predictions |
| `GET` | `/api/predictions/{date}` | Public | Predictions for a given date |
| `GET` | `/api/picks/today` | Public | Today's banker picks |
| `GET` | `/api/picks/history` | Public | Pick history (paginated, filterable) |
| `GET` | `/api/track-record` | Public | Overall performance stats |
| `GET` | `/api/track-record/by-tier` | Public | Performance by tier |
| `GET` | `/api/track-record/by-league` | Public | Performance by league |
| `GET` | `/api/health` | Public | System health |

### Auth

Simple API key in `X-API-Key` header for pipeline endpoints only. Key stored in `.env` as `SHARPEDGE_API_KEY`. All read endpoints are public.

### Response Envelope

```json
{
    "status": "ok",
    "data": { ... },
    "meta": { "generated_at": "2026-03-04T08:15:00Z", "count": 3 }
}
```

---

## 6. Telegram Bot

### Two Modes

**Channel Broadcast** — Posts to `@SharpEdgePicks` (or configured channel ID).

**Interactive Bot** — Handles DM commands from individual users.

### Channel Message Format (Picks)

```
🎯 SharpEdge Banker Pick

⚽ Arsenal vs Chelsea
🏆 Premier League | Mar 4, 2026

📊 Pick: Home Win (1X2)
💰 Best Odds: 1.85 @ Bet365
📈 Edge: +8.2% vs market
🏅 Tier: PLATINUM

Model: 83.2% confidence
Meta: 3/4 sites agree
Risk Flags: None

#PremierLeague #Platinum
```

### Channel Message Format (Results)

```
📊 Yesterday's Results (Mar 3)

✅ Arsenal 2-1 Chelsea — HOME WIN @ 1.85 → +0.85u
✅ Bayern 3-0 Dortmund — HOME WIN @ 1.65 → +0.65u
❌ Barcelona 1-1 Real Madrid — HOME WIN @ 2.10 → -1.00u

Day: 2W 1L | +0.50u
Month: 18W 5L (78%) | +6.2u | ROI: +8.5%
All Time: 142W 38L (79%) | +42.1u
```

### Bot Commands (DM)

| Command | Response |
|---------|----------|
| `/start` | Welcome + available commands |
| `/today` | Today's picks (all tiers) |
| `/platinum` | Platinum picks only |
| `/gold` | Gold + Platinum picks |
| `/record` | Track record summary (last 30 days) |
| `/leagues` | Available leagues |
| `/help` | Command list + about |

### Module Structure

```
src/sharpedge/bot/
    __init__.py
    app.py              # Bot application setup + handlers
    commands.py         # Command handlers (/today, /record, etc.)
    broadcast.py        # Channel broadcast logic
    formatters.py       # Message formatting (picks, results, record)
```

### Bot Process

Runs via `python -m sharpedge.bot.app` as a separate process. Uses long polling (no webhook). Polls DB for unbroadcasted picks every 60 seconds.

---

## 7. Result Resolution

### Flow

```
POST /api/pipeline/resolve (GitHub Actions, 23:00 UTC)
    │
    ├── 1. Find all unresolved DailyPick rows (result = NULL)
    ├── 2. Fetch final scores for those matches
    ├── 3. Compare pick_selection against actual result
    ├── 4. Set result = "win"/"loss"/"void", calculate profit_loss
    ├── 5. Update DailyPick rows
    └── 6. Mark results as pending_broadcast → bot posts summary
```

### Profit/Loss Calculation

- **Win:** `profit_loss = (best_odds - 1) * stake`
- **Loss:** `profit_loss = -stake`
- **Void:** `profit_loss = 0`

Stake = 1 unit (flat staking for track record; Kelly shown as info only).

---

## 8. Track Record Calculator

Aggregates DailyPick history into performance metrics:

- **Overall:** total_picks, wins, losses, win_rate, total_profit, roi, max_drawdown, longest_win_streak, longest_loss_streak, avg_odds
- **By tier:** same metrics filtered by tier (platinum/gold/silver)
- **By league:** same metrics filtered by league
- **By month:** monthly breakdown
- **Rolling:** last 7/30/90 days

All calculations are real-time queries against the `daily_picks` table. No caching needed until we have thousands of picks.

---

## 9. GitHub Actions Workflows

### daily-pipeline.yml

```yaml
name: Daily Predictions
on:
  schedule:
    - cron: "0 8 * * *"    # 08:00 UTC daily
  workflow_dispatch:         # manual trigger
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - run: |
          curl -X POST "${{ secrets.API_URL }}/api/pipeline/run" \
            -H "X-API-Key: ${{ secrets.API_KEY }}" \
            -H "Content-Type: application/json"
```

### resolve-results.yml

```yaml
name: Resolve Results
on:
  schedule:
    - cron: "0 23 * * *"   # 23:00 UTC daily
  workflow_dispatch:
jobs:
  resolve:
    runs-on: ubuntu-latest
    steps:
      - run: |
          curl -X POST "${{ secrets.API_URL }}/api/pipeline/resolve" \
            -H "X-API-Key: ${{ secrets.API_KEY }}"
```

### weekly-retrain.yml

```yaml
name: Weekly Retrain
on:
  schedule:
    - cron: "0 3 * * 0"    # Sunday 03:00 UTC
  workflow_dispatch:
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - run: |
          curl -X POST "${{ secrets.API_URL }}/api/pipeline/retrain" \
            -H "X-API-Key: ${{ secrets.API_KEY }}"
```

---

## 10. Configuration Changes

### New Settings (config.py)

```python
# API settings
api_key: str = ""               # X-API-Key for pipeline endpoints
api_host: str = "0.0.0.0"
api_port: int = 8000

# Telegram bot settings (existing telegram_bot_token, telegram_alert_chat_id)
telegram_channel_id: str = ""   # @SharpEdgePicks channel
telegram_poll_interval: int = 60  # seconds

# Model settings
model_path: str = "models/latest.pkl"
retrain_interval_days: int = 7

# Pipeline settings
pipeline_api_url: str = ""      # for GitHub Actions
```

### New Dependencies

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.109",
    "uvicorn[standard]>=0.27",
]
```

---

## 11. What Phase 3 Does NOT Include

- No user authentication / accounts
- No payment / Stripe integration
- No Next.js dashboard (deferred)
- No real-time odds scraping
- No Sentry / APM monitoring
- No Redis caching
- No Docker / containerisation (local Python processes)

---

## 12. Success Criteria

Phase 3 is complete when:

1. `python -m sharpedge.api.app` starts the FastAPI server
2. `python -m sharpedge.bot.app` starts the Telegram bot
3. `POST /api/pipeline/run` produces predictions and picks for today's fixtures
4. Picks appear in the Telegram channel within 60 seconds
5. `POST /api/pipeline/resolve` correctly resolves picks against actual results
6. Results summary appears in the Telegram channel
7. `GET /api/track-record` returns accurate historical performance
8. All bot commands (`/today`, `/record`, `/platinum`, etc.) work
9. All tests pass
