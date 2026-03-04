# SharpEdge AI — Full-Scale Product Design

**Date:** 2026-03-04
**Author:** Bala + Claude
**Status:** Approved
**Version:** 2.0 (Improved)

---

## 1. Executive Summary

SharpEdge AI is an AI-powered football prediction platform that combines data from 26 free sources, ensemble ML models, and edge detection against bookmaker pricing to deliver verified value betting opportunities. Unlike competitors (NerdyTips, Forebet), SharpEdge fuses institutional-grade multi-source data, tracks Closing Line Value, and maintains a radically transparent public track record.

**Core constraint:** Zero API costs. All data gathered for free via web scraping, public datasets, and free-tier APIs.

---

## 2. Target Users

- **Primary:** Semi-serious football bettors wanting 2-3 high-confidence picks/week. Bankroll: £500-5,000.
- **Secondary:** Systematic bettors using staking strategies (martingale, Kelly) who need highest-probability selections.
- **Tertiary:** Data-literate punters who want transparent reasoning behind picks.

---

## 3. Free Data Arsenal (26 Sources)

### Layer 1: Core Match Data
| Source | Data | Method | Cost |
|--------|------|--------|------|
| Football-Data.co.uk | Match results + closing odds, 25+ leagues, since 1993 | Direct CSV download | Free |
| FBref | Full match stats, xG, xA (StatsBomb-powered) | HTML scraping via `soccerdata` | Free |
| Understat | Shot-level xG for Big 5 leagues since 2014 | JSON in `<script>` tags | Free |
| StatsBomb Open Data | Event-level data for select competitions | GitHub JSON download | Free |
| OpenFootball | Schedules and results, public domain | GitHub JSON download | Free |

### Layer 2: Advanced Statistics & Ratings
| Source | Data | Method | Cost |
|--------|------|--------|------|
| ClubELO | Daily ELO ratings since 1940s | CSV API | Free |
| FiveThirtyEight SPI | Historical SPI ratings (archived) | GitHub CSV | Free |
| SofaScore | Player ratings, momentum, shot maps | Undocumented REST API | Free |
| FotMob | xG, shot maps, player stats | Undocumented API via `soccerdata` | Free |
| WhoScored | Opta-powered ratings, tactical stats | Selenium via `soccerdata` | Free |

### Layer 3: Odds From Every Bookmaker
| Source | Data | Method | Cost |
|--------|------|--------|------|
| Football-Data.co.uk | Historical closing odds: Bet365, Pinnacle, etc. | CSV download | Free |
| OddsPortal | 80+ bookmakers, opening/closing odds, 10+ years | Selenium + OddsHarvester | Free |
| BetExplorer | Historical odds with movement | HTML/JS scraping | Free |
| Oddschecker | Real-time odds, 25+ UK bookmakers | JS scraping (post-MVP) | Free |
| The Odds API | Clean REST API, multiple bookmakers | 500 requests/month free | Free |
| Oddspedia | Real-time odds, surebets, value bets | JS scraping | Free |

### Layer 4: Competitor Predictions (Meta-Model)
| Source | Data | Method | Cost |
|--------|------|--------|------|
| Forebet | ML predictions: 1X2, CS, O/U, BTTS for 800+ leagues | HTML scraping | Free |
| PredictZ | Statistical predictions + form guides | HTML scraping | Free |
| WinDrawWin | Predictions + stats for 140+ leagues | HTML scraping | Free |
| BetClan | Daily predictions, H2H, previews | HTML scraping | Free |
| FootyStats | Predictions + deep stats: O/U, BTTS, corners, cards, refs | HTML scraping | Free |
| NerdyTips | AI predictions (NT Apex engine) | HTML scraping | Free |

### Layer 5: Contextual Data
| Source | Data | Method | Cost |
|--------|------|--------|------|
| Transfermarkt | Squad values, injuries, suspensions | HTML scraping + GitHub datasets | Free |
| FootyStats / Soccerbase | Referee stats: cards, fouls, home bias | HTML scraping | Free |
| Open-Meteo | Weather: temp, rain, wind. No API key needed | REST API, 10K req/day | Free |
| football-data.org | Fixtures, lineups, standings via clean API | REST API, 10 req/min | Free |

---

## 4. Scraping Engine Architecture

### Design Principles
- **Self-healing:** Structural change detection with fallback selectors
- **Validated:** Raw data → staging zone → validation → production (never direct to prod)
- **Resilient:** Exponential backoff (5 attempts), graceful degradation chain
- **Observable:** Source health dashboard, Telegram alerts on failures
- **Stealthy:** User-agent rotation, request jitter, session management

### Collector Base Class
Every scraper inherits from `BaseCollector` which enforces:
- Rate limiting (per-source configurable delays)
- Retry with exponential backoff (5 attempts)
- Local + DB caching
- Health reporting
- Schema validation
- Team name normalisation
- Structural fingerprint checking
- Telegram alerting
- Metrics logging
- Raw staging (never writes directly to production tables)

### Validation Engine (4 layers)
1. **Schema Validator:** Column types, required fields, no nulls in critical columns
2. **Statistical Validator:** Values in expected ranges, match counts correct, no duplicates
3. **Cross-Source Validator:** Scores match across sources, xG values within tolerance
4. **Freshness Validator:** Data not stale, all matchdays present

### Team Name Normalisation
Master registry with canonical names + per-source aliases. Fuzzy matching (thefuzz library, >90% threshold) for new/unknown names.

### Graceful Degradation
FULL POWER (26 sources) → STRONG (25) → CAPABLE (24) → CORE (23) → EMERGENCY (alert)

---

## 5. ML Architecture

### Phase 1 (MVP): Two-Model Core
- **XGBoost:** P(Home/Draw/Away), P(Over/Under), P(BTTS)
- **Poisson Bivariate:** P(goals) → derives all markets including correct score
- **Ensemble:** Weighted combination, weights learned via validation
- **Calibration:** Platt scaling to ensure probability accuracy

### Feature Engineering (50 Curated Features)
| Group | Count | Source |
|-------|-------|--------|
| Form Features | 12 | FBref + Understat |
| ELO Features | 6 | ClubELO |
| xG Performance | 8 | Understat + FBref |
| Head-to-Head | 6 | Football-Data.co.uk |
| Market Features | 8 | Football-Data + OddsPortal |
| Contextual | 6 | Various |
| Meta-Predictions | 4 | Forebet + PredictZ + etc. |

### Feature Addition Rule
New features must improve RPS (Ranked Probability Score) by ≥ 0.002 on walk-forward validation to be included.

### Training
- **Data:** 5 seasons (2020-2025), ~15,000 matches across Big 5 leagues
- **Validation:** Temporal walk-forward ONLY (never random CV)
- **Metric:** Ranked Probability Score (RPS), not raw accuracy
- **Retraining:** Full retrain every 4 weeks

### Phase 2: Add CatBoost + LightGBM (only if they improve ensemble)
### Phase 3: Add Neural Network + Meta-Learner

---

## 6. Banker Tip Engine

### 5-Stage Filter Pipeline
1. **Minimum Confidence:** Model probability ≥ 70%
2. **Model Agreement:** All ensemble models agree, spread ≤ 8%
3. **Value Edge:** Model prob > best bookmaker implied prob by ≥ 5%
4. **Meta-Model Confirmation:** ≥ 2 of 4 competitor sites agree
5. **Risk Flag Check:** No injuries, fatigue, derby, weather, dead rubber

### Confidence Tiers
- **Platinum (85%+ win rate target):** Prob ≥ 85%, Edge ≥ 10%, 0 risk flags
- **Gold (78%+ win rate target):** Prob ≥ 78%, Edge ≥ 7%, ≤ 1 minor flag
- **Silver (70%+ win rate target):** Prob ≥ 70%, Edge ≥ 5%

---

## 7. Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| Language | Python 3.11+ (backend), TypeScript (frontend) | Free |
| Database | PostgreSQL 16 (Supabase free tier) | Free |
| ORM | SQLAlchemy 2.0 + Alembic | Free |
| Scraping | soccerdata, BeautifulSoup4, Playwright | Free |
| ML | XGBoost, scipy, scikit-learn, Optuna, SHAP, MLflow | Free |
| API | FastAPI + Pydantic v2 + Uvicorn | Free |
| Frontend | Next.js 14 + Tailwind + shadcn/ui + Recharts | Free |
| Delivery | python-telegram-bot, Resend, Web Push | Free |
| Hosting | Railway ($5/mo), Supabase (free), Vercel (free) | $5/mo |
| Scheduling | GitHub Actions (2000 min/mo free) | Free |
| Cache | Upstash Redis (free tier) | Free |
| Monitoring | Sentry (free tier) + custom dashboard | Free |

**Total monthly cost: $5-10**

---

## 8. Timeline (17 Weeks)

### Phase 1: Data Foundation (Weeks 1-4)
- Week 1: Project setup, PostgreSQL schema, team name registry
- Week 2: Tier 1 collectors (Football-Data, ClubELO, StatsBomb, OpenFootball)
- Week 3: Tier 2 collectors (FBref, Understat via soccerdata)
- Week 4: Prediction scrapers + validation engine + 5-season backfill

### Phase 2: ML Engine (Weeks 5-8)
- Week 5: Feature engineering pipeline (50 features)
- Week 6: XGBoost model — train + walk-forward validation
- Week 7: Poisson bivariate + ensemble + calibration
- Week 8: Banker filter + backtesting (target: ≥80% Platinum on 2 seasons)

### Phase 3: Delivery (Weeks 9-12)
- Week 9: FastAPI prediction service + daily pipeline
- Week 10: Telegram bot
- Week 11: Next.js dashboard (track record, pick history)
- Week 12: Landing page + free/premium auth

### Phase 4: Shadow Run (Weeks 13-16)
- Weeks 13-14: Daily predictions, not published. Validate on live matches.
- Weeks 15-16: Soft launch free Telegram channel. Build 100+ pick track record.

### Phase 5: Full Launch (Week 17+)
- Week 17: Public launch with proven record
- Week 18+: OddsPortal, Transfermarkt, referee data, additional models

---

## 9. Monetisation

- **Free tier:** Delayed picks (24h lag), basic dashboard
- **Pro (£19.99/mo):** Instant picks, edge alerts, staking calculator, full odds comparison
- **Affiliate revenue:** Odds comparison with bookmaker affiliate links (£2-5 per referred customer)

---

## 10. Leagues Covered (Launch)

Big 5 European leagues:
1. Premier League (England)
2. La Liga (Spain)
3. Bundesliga (Germany)
4. Serie A (Italy)
5. Ligue 1 (France)

---

## 11. Key Improvements Over Original Plan

1. **26 free sources** instead of 5 vague mentions
2. **Bulletproof scraping engine** with self-healing, 4-layer validation, graceful degradation
3. **Meta-model** using competitor predictions as features
4. **50 curated features** instead of 130+ (prevents overfitting)
5. **Phased model complexity** — earn it through validation scores
6. **Realistic 17-week timeline** instead of 8-10 weeks
7. **Team name normalisation** as a first-class system
8. **Removed blockchain verification** (unnecessary complexity for MVP)
9. **$5-10/month total cost** vs unspecified in original
