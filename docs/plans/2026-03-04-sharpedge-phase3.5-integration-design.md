# SharpEdge AI — Phase 3.5: Integration & Deployment Readiness Design

**Date:** 2026-03-04
**Author:** Bala + Claude
**Status:** Approved
**Depends on:** Phase 1 (Data Foundation), Phase 2 (ML Engine), Phase 3 (Delivery)

---

## 1. Goal

Bridge the gap between "tests pass with mocks" and "app runs end-to-end with real data." Install PostgreSQL, create all database tables, backfill 5 seasons of historical data via all 10 collectors, train the model on real data, and validate the full pipeline locally.

## 2. Scope

**In scope:**
- PostgreSQL installation via Homebrew
- Alembic migration (all Phase 1 + Phase 3 models)
- `.env` configuration
- Historical data backfill (5 seasons, Big 5 leagues, all 10 collectors)
- Model training on real data → `models/latest.pkl`
- End-to-end smoke test via FastAPI API

**Out of scope:**
- Telegram bot setup (bot creation, channel, token config)
- Cloud deployment (Railway, VPS, Supabase)
- Dashboard / frontend
- Production monitoring

## 3. PostgreSQL + Alembic

### PostgreSQL Setup
- `brew install postgresql@16` + `brew services start postgresql@16`
- Create database: `createdb sharpedge`
- Verify: `psql sharpedge -c "SELECT 1"`

### Alembic Migration
- `alembic/env.py` already imports `Base` from models and reads `settings.database_url`
- Run `alembic revision --autogenerate -m "initial schema"` — generates migration for all models:
  - Phase 1: Team, League, Season, Match, MatchStats, MatchXG, MatchOdds, EloRating, CompetitorPrediction, MatchWeather, SourceHealth, RawStagingRecord
  - Phase 3: PipelineRun, Prediction, DailyPick
- Run `alembic upgrade head` to apply

### `.env` Configuration
```
DATABASE_URL=postgresql://localhost:5432/sharpedge
SHARPEDGE_API_KEY=dev-secret-key-change-in-prod
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALERT_CHAT_ID=
TELEGRAM_CHANNEL_ID=
MODEL_PATH=models/latest.pkl
```

## 4. Historical Data Backfill

### Parameters
- **Seasons:** 2020-21 through 2024-25 (5 seasons)
- **Leagues:** Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- **Method:** `run_backfill()` from `src/sharpedge/orchestrator.py`

### Collectors (10, ordered by reliability)
1. Football-Data.co.uk — CSV downloads (fast)
2. ClubELO — CSV API (fast)
3. football-data.org — REST API, 10 req/min
4. Open-Meteo — REST API, no key needed
5. FBref — HTML scraping via soccerdata (slower)
6. Understat — JSON scraping
7. Forebet — HTML scraping
8. PredictZ — HTML scraping
9. WinDrawWin — HTML scraping
10. FootyStats — HTML scraping

### Expectations
- CSV/API sources: minutes
- Scraping sources: hours (rate limiting x 5 seasons x 5 leagues)
- Some scrapers may fail for older seasons — graceful degradation built in
- Data passes through 4-layer validation before production tables

### Error Handling
If a collector fails, it logs the error and continues. Individual collectors can be re-run later for any gaps.

## 5. Model Training

- Use `ModelTrainer` from `src/sharpedge/ml/training/trainer.py`
- `trainer.train()`: FeaturePipeline → XGBoost + Poisson → Ensemble → Calibration
- Walk-forward temporal validation (no future leakage)
- Save to `models/latest.pkl`
- Expected: ~15,000 matches, 50 features each

## 6. End-to-End Smoke Test

1. Start FastAPI: `python -m sharpedge.api`
2. Trigger pipeline: `curl -X POST http://localhost:8000/api/pipeline/run -H "X-API-Key: dev-secret-key-change-in-prod"`
3. Verify predictions written to DB
4. Check API: `curl http://localhost:8000/api/picks/today`
5. Check track record: `curl http://localhost:8000/api/track-record`
6. Verify health: `curl http://localhost:8000/api/health`

### Success Criteria
- Pipeline produces predictions for today's fixtures
- API serves picks with real model probabilities
- No 500 errors on any endpoint
- Model file exists at `models/latest.pkl`
- Database has historical data across 5 seasons

## 7. What This Does NOT Include

- No Telegram bot configuration (deferred)
- No cloud deployment
- No production PostgreSQL (Supabase)
- No GitHub Actions connected to live API
- No dashboard
