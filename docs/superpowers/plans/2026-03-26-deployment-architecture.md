# SharpEdge Deployment Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan.

**Goal:** Deploy SharpEdge as a production platform across Railway (compute), Netlify (frontend), and Supabase (database) — with automated daily pipelines, agent training, and real-time predictions.

**Architecture:** Microservice deployment with Railway services, 1 Netlify site, and 1 Supabase project.

---

## Infrastructure Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │   NETLIFY     │    │         RAILWAY PROJECT               │   │
│  │   Frontend    │    │                                       │   │
│  │               │    │  ┌─────────────┐  ┌──────────────┐   │   │
│  │  Next.js SSG  │───▶│  │  API Service │  │ Worker Service│   │   │
│  │  Static +     │    │  │  (FastAPI)   │  │ (Agent Train) │   │   │
│  │  ISR          │    │  │  Port 8000   │  │ Cron-based    │   │   │
│  │               │    │  └──────┬──────┘  └──────┬───────┘   │   │
│  └──────────────┘    │         │                 │            │   │
│                       │         │    ┌────────────┘            │   │
│  ┌──────────────┐    │         ▼    ▼                         │   │
│  │  SUPABASE     │◀──│──── PostgreSQL (shared DB)              │   │
│  │  Database     │    │                                       │   │
│  │  + Auth       │    │  ┌──────────────┐  ┌──────────────┐   │   │
│  │  + Storage    │    │  │ Scheduler    │  │ Monitor      │   │   │
│  │               │    │  │ (Cron Jobs)  │  │ (Health)     │   │   │
│  └──────────────┘    │  └──────────────┘  └──────────────┘   │   │
│                       └──────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │  TELEGRAM     │    │  ODDS API     │                           │
│  │  Bot          │    │  (data feed)  │                           │
│  └──────────────┘    └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Service Breakdown

### 1. Railway: API Service (always-on)
**Purpose:** FastAPI backend serving predictions, picks, agent leaderboards, track record.

- Service name: sharpedge-api
- Runtime: Python 3.12
- Start command: `uvicorn sharpedge.api.app:create_app --host 0.0.0.0 --port $PORT --factory`
- Plan: Starter ($5/mo) — 512MB RAM, shared CPU
- Environment variables: DATABASE_URL, SUPABASE_URL, SUPABASE_ANON_KEY, TELEGRAM_BOT_TOKEN, API_KEY, ENVIRONMENT=production

### 2. Railway: Worker Service (scheduled)
**Purpose:** Runs daily prediction pipeline, model training, agent optimization.

- Service name: sharpedge-worker
- Runtime: Python 3.12
- Plan: Pro ($20/mo) — 2GB RAM, 2 vCPU (needed for model training)
- Cron schedule:
  - 06:00 UTC — Data collection (all 12 leagues)
  - 07:00 UTC — Feature building + agent predictions
  - 08:00 UTC — Arbiter combination + banker filter + staking
  - 09:00 UTC — Publish picks to DB + Telegram broadcast
  - 23:00 UTC — Resolve yesterday's picks + update agent weights
  - Monthly — Full model retrain (all agents)

### 3. Netlify: Frontend
**Purpose:** Next.js web dashboard — predictions, picks, agent leaderboard, track record.

- Site name: sharpedge-web
- Framework: Next.js 16
- Build command: `cd web && npm run build`
- Publish directory: `web/.next`
- Environment: NEXT_PUBLIC_API_URL = Railway API URL

### 4. Supabase: Database + Auth + Storage
**Purpose:** PostgreSQL database (already configured), plus Storage for model artifacts.

- Project: SharpEdge (under Carpool Network org)
- Already configured and connected
- New tables: Sport, Agent, AgentPrediction, AgentPerformance
- Storage bucket: model-artifacts (JSON serialized models)

---

## Deployment Files to Create

### railway.toml
```toml
[build]
builder = "NIXPACKS"
buildCommand = "pip install -e '.[ml,api]'"

[deploy]
startCommand = "uvicorn sharpedge.api.app:create_app --host 0.0.0.0 --port $PORT --factory"
healthcheckPath = "/api/health"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
```

### Procfile
```
web: uvicorn sharpedge.api.app:create_app --host 0.0.0.0 --port $PORT --factory
worker: python -m sharpedge.warroom.run_daily
```

### web/netlify.toml
```toml
[build]
  command = "npm run build"
  publish = ".next"

[build.environment]
  NODE_VERSION = "20"

[[redirects]]
  from = "/api/*"
  to = "https://sharpedge-api.up.railway.app/api/:splat"
  status = 200
  force = true

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
```

### Daily Pipeline Runner (src/sharpedge/warroom/run_daily.py)

Production daily pipeline that:
1. Runs health checks
2. Collects data for all sports
3. Builds features + runs agent predictions
4. Combines via arbiter + applies banker filter + staking
5. Publishes picks to DB + Telegram
6. Checks if retraining is due

---

## Task Breakdown

### Task 1: Create Railway project + deploy API service
- [ ] Create Railway project: `railway init`
- [ ] Create `railway.toml` and `Procfile`
- [ ] Set environment variables: DATABASE_URL, API_KEY, TELEGRAM tokens
- [ ] Deploy: `railway up`
- [ ] Verify: `curl <railway-url>/api/health`

### Task 2: Create Railway worker service
- [ ] Create `src/sharpedge/warroom/run_daily.py` pipeline runner
- [ ] Add Railway cron service for daily execution
- [ ] Deploy worker service
- [ ] Test: trigger pipeline manually

### Task 3: Deploy Supabase schema updates
- [ ] Run schema migration to create Sport, Agent, AgentPrediction, AgentPerformance tables
- [ ] Create model-artifacts storage bucket
- [ ] Seed: register 6 sports, 9 agents
- [ ] Verify tables exist in Supabase dashboard

### Task 4: Deploy Netlify frontend
- [ ] Create `web/netlify.toml`
- [ ] Set NEXT_PUBLIC_API_URL environment variable
- [ ] `cd web && netlify deploy --prod`
- [ ] Verify: site loads, API proxy works

### Task 5: Wire up Telegram alerts
- [ ] Ensure tokens set in Railway env
- [ ] Test alert from worker
- [ ] Add daily pick broadcast

### Task 6: End-to-end verification
- [ ] Trigger full pipeline from Railway
- [ ] Verify: data → features → agents → picks → API → frontend → Telegram
- [ ] Monitor for 24 hours

---

## Cost Estimate

| Service | Plan | Monthly Cost |
|---------|------|-------------|
| Railway API | Starter | $5 |
| Railway Worker | Pro (2GB) | $20 |
| Netlify | Free tier | $0 |
| Supabase | Free tier | $0 |
| The Odds API | Free (500 req/mo) | $0 |
| Telegram Bot | Free | $0 |
| **Total** | | **~$25/month** |
