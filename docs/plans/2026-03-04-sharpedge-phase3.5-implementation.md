# SharpEdge Phase 3.5: Integration & Deployment Readiness — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Get the app running end-to-end locally: PostgreSQL database with all tables, 5 seasons of real data from all 10 collectors, a trained model, and a working API that serves real predictions.

**Architecture:** Install PostgreSQL → Alembic migration → .env config → backfill collectors → train model → smoke test the full pipeline via FastAPI.

**Tech Stack:** PostgreSQL 16 (Homebrew), Alembic, SQLAlchemy 2.0, all 10 collectors, ModelTrainer, FastAPI + uvicorn

**Design Doc:** `docs/plans/2026-03-04-sharpedge-phase3.5-integration-design.md`

---

## Existing Codebase Context

**Key files you'll touch:**
- `src/sharpedge/config.py` — `Settings` class, reads `.env`, has `database_url` default `postgresql://localhost:5432/sharpedge`
- `src/sharpedge/db/engine.py` — `create_engine(settings.database_url)`, `get_session()`
- `src/sharpedge/db/models.py` — 15 models: Team, League, Season, Match, MatchStats, MatchXG, MatchOdds, EloRating, CompetitorPrediction, MatchWeather, SourceHealth, RawStagingRecord, PipelineRun, Prediction, DailyPick
- `alembic/env.py` — already imports `Base` from models and overrides `sqlalchemy.url` from `settings.database_url`
- `alembic/versions/` — currently empty (no migrations exist yet)
- `src/sharpedge/orchestrator.py` — `run_backfill(seasons)` runs FootballDataUK, ClubELO, Understat for each season; `run_daily_collection()` runs all 10
- `src/sharpedge/ml/training/trainer.py` — `ModelTrainer.train(matches_df, elo_df, xg_df, predictions_df)`, `trainer.save(path)`, `ModelTrainer.load(path)`
- `src/sharpedge/pipeline/daily.py` — `DailyPipeline` with `load_model()`, `predict()`, `filter_picks()`
- `src/sharpedge/api/routers/pipeline.py` — `POST /api/pipeline/run` (has `TODO: Replace with actual fixture collection`)
- `.gitignore` — has `.env`, `data/cache/`, `*.db` but NOT `models/`

**Important:** The `run_backfill()` function currently only runs 3 collectors (FootballDataUK, ClubELO, Understat). We need to expand it to run all 10 for a full backfill.

---

### Task 1: Install PostgreSQL + Create Database

**Step 1: Install PostgreSQL 16 via Homebrew**

Run:
```bash
brew install postgresql@16
```

**Step 2: Start the PostgreSQL service**

Run:
```bash
brew services start postgresql@16
```

**Step 3: Add PostgreSQL to PATH (if needed)**

If `psql` is not found after install, add to your shell profile:
```bash
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Step 4: Create the sharpedge database**

Run:
```bash
createdb sharpedge
```

**Step 5: Verify the database connection**

Run:
```bash
psql sharpedge -c "SELECT 1 AS connection_ok;"
```
Expected: Table showing `connection_ok = 1`

**Step 6: Verify SQLAlchemy can connect**

Run:
```bash
cd "/Users/balabollineni/Sharp Edge"
.venv/bin/python -c "
from sharpedge.db.engine import engine
with engine.connect() as conn:
    result = conn.execute(__import__('sqlalchemy').text('SELECT 1'))
    print('SQLAlchemy connection OK:', result.scalar())
"
```
Expected: `SQLAlchemy connection OK: 1`

---

### Task 2: Create `.env` File + Add `models/` to `.gitignore`

**Step 1: Create the `.env` file**

Create `.env` in the project root:

```
DATABASE_URL=postgresql://localhost:5432/sharpedge
SHARPEDGE_API_KEY=dev-secret-key-change-in-prod
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALERT_CHAT_ID=
TELEGRAM_CHANNEL_ID=
MODEL_PATH=models/latest.pkl
```

**Step 2: Verify settings load from `.env`**

Run:
```bash
.venv/bin/python -c "
from sharpedge.config import Settings
s = Settings()
print('database_url:', s.database_url)
print('api_key:', s.api_key)
print('model_path:', s.model_path)
"
```
Expected:
```
database_url: postgresql://localhost:5432/sharpedge
api_key: dev-secret-key-change-in-prod
model_path: models/latest.pkl
```

**Step 3: Add `models/` to `.gitignore`**

Append to `.gitignore`:
```
models/
```

This prevents the trained model binary (~50MB+) from being committed to git.

**Step 4: Commit the `.gitignore` change**

```bash
git add .gitignore
git commit -m "chore: add models/ to .gitignore"
```

Note: `.env` is already in `.gitignore` so it won't be committed.

---

### Task 3: Generate + Run Alembic Migration

**Step 1: Generate the initial migration**

Run:
```bash
cd "/Users/balabollineni/Sharp Edge"
.venv/bin/python -m alembic revision --autogenerate -m "initial schema — all Phase 1 + Phase 3 models"
```
Expected: Creates a new file in `alembic/versions/` like `xxxx_initial_schema_all_phase_1_phase_3_models.py`

**Step 2: Review the generated migration**

Open the generated file in `alembic/versions/` and verify it creates tables for all 15 models:
- teams, leagues, seasons, matches, match_stats, match_xg, match_odds
- elo_ratings, competitor_predictions, match_weather
- source_health, raw_staging_records
- pipeline_runs, predictions, daily_picks

Check that indexes and unique constraints are included.

**Step 3: Run the migration**

Run:
```bash
.venv/bin/python -m alembic upgrade head
```
Expected: `INFO [alembic.runtime.migration] Running upgrade -> xxxx, initial schema...`

**Step 4: Verify all tables exist**

Run:
```bash
psql sharpedge -c "\dt"
```
Expected: All 15 tables plus `alembic_version` listed.

**Step 5: Verify via SQLAlchemy**

Run:
```bash
.venv/bin/python -c "
from sharpedge.db.engine import engine
from sqlalchemy import inspect
tables = inspect(engine).get_table_names()
print(f'{len(tables)} tables created:')
for t in sorted(tables):
    print(f'  {t}')
"
```
Expected: 16 tables (15 models + alembic_version)

**Step 6: Commit the migration**

```bash
git add alembic/versions/
git commit -m "feat: initial Alembic migration — all 15 database tables"
```

---

### Task 4: Expand `run_backfill()` to Run All 10 Collectors

**Files:**
- Modify: `src/sharpedge/orchestrator.py`

**Context:** Currently `run_backfill()` only runs 3 collectors per season (FootballDataUK, ClubELO, Understat). We need it to run all 10 collectors to get the full dataset for model training.

**Step 1: Write the test**

Create `tests/test_integration/test_backfill.py`:

```python
"""Integration test for expanded backfill — verifies all 10 collectors are called."""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from sharpedge.orchestrator import run_backfill


@pytest.fixture
def mock_all_collectors():
    """Patch all 10 collector classes to return mock DataFrames."""
    collectors = [
        "sharpedge.orchestrator.FootballDataUKCollector",
        "sharpedge.orchestrator.ClubELOCollector",
        "sharpedge.orchestrator.UnderstatCollector",
        "sharpedge.orchestrator.FBrefCollector",
        "sharpedge.orchestrator.ForebetCollector",
        "sharpedge.orchestrator.PredictZCollector",
        "sharpedge.orchestrator.WinDrawWinCollector",
        "sharpedge.orchestrator.FootyStatsCollector",
        "sharpedge.orchestrator.FootballDataOrgCollector",
        "sharpedge.orchestrator.OpenMeteoCollector",
    ]
    mocks = {}
    patches = []
    for collector_path in collectors:
        name = collector_path.split(".")[-1]
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.source_name = name.lower()
        mock_instance.collect.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_cls.return_value = mock_instance
        p = patch(collector_path, mock_cls)
        p.start()
        patches.append(p)
        mocks[name] = mock_instance
    yield mocks
    for p in patches:
        p.stop()


def test_backfill_calls_all_10_collectors(mock_all_collectors):
    """Backfill for 1 season should invoke all 10 collectors."""
    results = run_backfill(seasons=["2024-25"])
    assert len(results) == 10  # one result per collector per season
    for mock in mock_all_collectors.values():
        mock.collect.assert_called()


def test_backfill_multiple_seasons_calls_all_each_time(mock_all_collectors):
    """Backfill for 2 seasons should invoke each collector twice."""
    results = run_backfill(seasons=["2023-24", "2024-25"])
    assert len(results) == 20  # 10 collectors x 2 seasons
```

**Step 2: Run the test to see it fail**

Run:
```bash
.venv/bin/python -m pytest tests/test_integration/test_backfill.py -v
```
Expected: FAIL — `assert len(results) == 10` fails (currently returns 3 per season)

**Step 3: Update `run_backfill()` to use all 10 collectors**

Replace the `run_backfill()` function in `src/sharpedge/orchestrator.py`. The new version runs all 10 collectors per season, passing `season=season` to collectors that support it and no kwargs to collectors that don't:

```python
def run_backfill(
    seasons: Optional[list[str]] = None,
) -> list[CollectionResult]:
    """Backfill historical data for specified seasons.

    Runs all 10 collectors for each season. Collectors that don't support
    season-specific collection are run once per season anyway (they'll
    return their default data range).

    Parameters
    ----------
    seasons : list[str], optional
        Seasons to backfill (e.g. ``["2020-21", "2021-22"]``).
        Defaults to all 5 recent seasons.

    Returns
    -------
    list[CollectionResult]
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    results: list[CollectionResult] = []

    # Collectors that accept season= kwarg
    season_aware = [
        FootballDataUKCollector,
        UnderstatCollector,
        FBrefCollector,
        ForebetCollector,
        PredictZCollector,
        WinDrawWinCollector,
        FootyStatsCollector,
    ]
    # Collectors that don't filter by season (run once per iteration)
    season_agnostic = [
        ClubELOCollector,
        FootballDataOrgCollector,
        OpenMeteoCollector,
    ]

    for season in seasons:
        logger.info(f"=== Backfilling season {season} ===")

        for collector_cls in season_aware:
            result = _run_collector(collector_cls(), season=season)
            results.append(result)

        for collector_cls in season_agnostic:
            result = _run_collector(collector_cls())
            results.append(result)

    return results
```

**Step 4: Run the test to verify it passes**

Run:
```bash
.venv/bin/python -m pytest tests/test_integration/test_backfill.py -v
```
Expected: 2 passed

**Step 5: Run existing tests to check no regressions**

Run:
```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_integration/ -x
```
Expected: All existing tests still pass.

**Step 6: Commit**

```bash
git add src/sharpedge/orchestrator.py tests/test_integration/
git commit -m "feat: expand run_backfill() to use all 10 collectors per season"
```

---

### Task 5: Run the Historical Data Backfill

**Important:** This task hits real websites and APIs. It will take time (potentially hours for scraping sources). Run it in a tmux/screen session or be prepared to wait.

**Step 1: Create the backfill script**

Create `scripts/run_backfill.py`:

```python
"""Run historical backfill — 5 seasons, all 10 collectors."""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backfill.log"),
    ],
)

from sharpedge.orchestrator import run_backfill

if __name__ == "__main__":
    seasons = sys.argv[1:] if len(sys.argv) > 1 else None
    results = run_backfill(seasons=seasons)

    print("\n=== BACKFILL SUMMARY ===")
    total_rows = 0
    for r in results:
        status_icon = "✅" if r.status == "healthy" else "⚠️" if r.status == "degraded" else "❌"
        print(f"  {status_icon} {r.source}: {r.rows} rows ({r.status})")
        if r.errors:
            for e in r.errors:
                print(f"     Error: {e}")
        total_rows += r.rows

    healthy = sum(1 for r in results if r.status == "healthy")
    degraded = sum(1 for r in results if r.status == "degraded")
    down = sum(1 for r in results if r.status == "down")
    print(f"\nTotal: {total_rows} rows | {healthy} healthy, {degraded} degraded, {down} down")
```

**Step 2: Run the backfill**

Run:
```bash
.venv/bin/python scripts/run_backfill.py
```

This will run all 10 collectors for 5 seasons (2020-21 through 2024-25). Watch the output for errors.

Expected output pattern:
```
=== Backfilling season 2020-21 ===
[football_data_uk] Starting collection...
[football_data_uk] Collection complete — 380 rows in 2.1s
[club_elo] Starting collection...
...
```

**Step 3: Check the database has data**

Run:
```bash
psql sharpedge -c "
SELECT
    (SELECT COUNT(*) FROM matches) AS matches,
    (SELECT COUNT(*) FROM match_stats) AS stats,
    (SELECT COUNT(*) FROM match_xg) AS xg,
    (SELECT COUNT(*) FROM match_odds) AS odds,
    (SELECT COUNT(*) FROM elo_ratings) AS elo,
    (SELECT COUNT(*) FROM competitor_predictions) AS predictions,
    (SELECT COUNT(*) FROM teams) AS teams,
    (SELECT COUNT(*) FROM leagues) AS leagues;
"
```
Expected: Non-zero counts across tables. Target: ~9,500+ matches (5 leagues x 380 matches/season x 5 seasons).

**Step 4: If any collectors failed, re-run individually**

If specific collectors failed, you can re-run them. The backfill script accepts season arguments:
```bash
.venv/bin/python scripts/run_backfill.py 2024-25
```

**Step 5: Commit the backfill script**

```bash
git add scripts/
git commit -m "feat: add backfill script with logging and summary"
```

---

### Task 6: Train the Model on Real Data

**Step 1: Create the training script**

Create `scripts/train_model.py`:

```python
"""Train the model on real data from the database."""
import logging
import sys

import pandas as pd
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)

from sharpedge.db.engine import get_session
from sharpedge.ml.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def load_training_data():
    """Load matches + supporting data from the database."""
    session = get_session()

    # Load matches with goals and team IDs
    matches_query = text("""
        SELECT
            m.id, m.match_date, m.home_team_id, m.away_team_id,
            m.home_goals AS "FTHG", m.away_goals AS "FTAG",
            CASE
                WHEN m.home_goals > m.away_goals THEN 'H'
                WHEN m.home_goals = m.away_goals THEN 'D'
                ELSE 'A'
            END AS "FTR",
            ht.canonical_name AS home_team_name,
            at.canonical_name AS away_team_name,
            l.name AS league,
            s.label AS season
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN seasons s ON m.season_id = s.id
        JOIN leagues l ON s.league_id = l.id
        WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.match_date
    """)
    matches_df = pd.read_sql(matches_query, session.bind)

    # Load ELO ratings
    elo_query = text("""
        SELECT team_id, rating_date, elo, source
        FROM elo_ratings
        ORDER BY rating_date
    """)
    elo_df = pd.read_sql(elo_query, session.bind)

    # Load xG data
    xg_query = text("""
        SELECT match_id, source, home_xg, away_xg
        FROM match_xg
    """)
    xg_df = pd.read_sql(xg_query, session.bind)

    # Load competitor predictions
    pred_query = text("""
        SELECT match_id, source, predicted_result, prob_home, prob_draw, prob_away
        FROM competitor_predictions
    """)
    predictions_df = pd.read_sql(pred_query, session.bind)

    session.close()
    return matches_df, elo_df, xg_df, predictions_df


if __name__ == "__main__":
    logger.info("Loading training data from database...")
    matches_df, elo_df, xg_df, predictions_df = load_training_data()

    logger.info(f"Loaded {len(matches_df)} matches, {len(elo_df)} ELO records, "
                f"{len(xg_df)} xG records, {len(predictions_df)} competitor predictions")

    if len(matches_df) < 100:
        logger.error("Not enough matches to train. Run the backfill first.")
        sys.exit(1)

    logger.info("Starting model training...")
    trainer = ModelTrainer(n_splits=2, min_train_seasons=3)
    result = trainer.train(
        matches_df,
        elo_df=elo_df if len(elo_df) > 0 else None,
        xg_df=xg_df if len(xg_df) > 0 else None,
        predictions_df=predictions_df if len(predictions_df) > 0 else None,
    )

    logger.info(f"Training complete!")
    logger.info(f"  Mean RPS: {result.aggregate_metrics.get('mean_rps', 'N/A')}")
    logger.info(f"  Mean Accuracy: {result.aggregate_metrics.get('mean_accuracy', 'N/A')}")
    logger.info(f"  Ensemble weights: {result.ensemble_weights}")
    logger.info(f"  Features: {len(result.feature_names)}")

    # Save model
    save_path = "models/latest.pkl"
    trainer.save(save_path)
    logger.info(f"Model saved to {save_path}/model.pkl")

    # Print fold details
    for fold in result.fold_metrics:
        logger.info(f"  Fold {fold['fold']}: RPS={fold['rps']:.4f}, Acc={fold['accuracy']:.3f}, "
                     f"train={fold['train_size']}, val={fold['val_size']}")
```

**Step 2: Run the training**

Run:
```bash
.venv/bin/python scripts/train_model.py
```

Expected output:
```
Loading training data from database...
Loaded XXXX matches, XXX ELO records, XXX xG records, XXX competitor predictions
Starting model training...
Fold 1: train=XXXX, val=XXX
  Fold 1: RPS=0.XXXX, Acc=0.XXX
Fold 2: train=XXXX, val=XXX
  Fold 2: RPS=0.XXXX, Acc=0.XXX
Retraining final models on all data...
Training complete!
  Mean RPS: 0.XXXX
  Mean Accuracy: 0.XXX
Model saved to models/latest.pkl/model.pkl
```

**Step 3: Verify the model file exists**

Run:
```bash
ls -la models/latest.pkl/model.pkl
```
Expected: File exists, size > 0

**Step 4: Verify the model loads correctly**

Run:
```bash
.venv/bin/python -c "
from sharpedge.ml.training.trainer import ModelTrainer
trainer = ModelTrainer.load('models/latest.pkl')
print('XGB model:', type(trainer.xgb_model).__name__)
print('Poisson model:', type(trainer.poisson_model).__name__)
print('Ensemble:', type(trainer.ensemble).__name__)
print('Calibrator:', type(trainer.calibrator).__name__)
print('Model loaded successfully!')
"
```
Expected: All 4 components loaded.

**Step 5: Commit the training script**

```bash
git add scripts/train_model.py
git commit -m "feat: add model training script with database data loading"
```

---

### Task 7: End-to-End Smoke Test via FastAPI

**Step 1: Start the FastAPI server**

In a terminal:
```bash
cd "/Users/balabollineni/Sharp Edge"
.venv/bin/python -m sharpedge.api
```
Expected: `Uvicorn running on http://0.0.0.0:8000`

**Step 2: Test the health endpoint**

In another terminal:
```bash
curl -s http://localhost:8000/api/health | python3 -m json.tool
```
Expected:
```json
{
    "status": "ok",
    "data": { ... }
}
```

**Step 3: Trigger the prediction pipeline**

```bash
curl -s -X POST http://localhost:8000/api/pipeline/run \
  -H "X-API-Key: dev-secret-key-change-in-prod" \
  -H "Content-Type: application/json" | python3 -m json.tool
```
Expected: Response with `"status": "ok"` or `"status": "error"` — either way, no 500 error.

Note: The pipeline currently has `fixtures = []` (TODO in the code). The pipeline will "succeed" with 0 predictions and 0 picks until we wire up real fixture collection. This is expected.

**Step 4: Check pipeline status**

```bash
curl -s http://localhost:8000/api/pipeline/status \
  -H "X-API-Key: dev-secret-key-change-in-prod" | python3 -m json.tool
```
Expected: Shows the pipeline run we just triggered.

**Step 5: Check track record (will be empty)**

```bash
curl -s http://localhost:8000/api/track-record | python3 -m json.tool
```
Expected: Returns track record structure with zero picks.

**Step 6: Check predictions endpoint**

```bash
curl -s http://localhost:8000/api/predictions/today | python3 -m json.tool
```
Expected: Empty list (no predictions for today yet since fixtures aren't wired up).

**Step 7: Verify API docs are accessible**

Open in browser: `http://localhost:8000/docs`
Expected: Swagger UI showing all endpoints.

**Step 8: Stop the server (Ctrl+C) and run all tests one final time**

Run:
```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_integration/ -x
```
Expected: All tests pass (127+ from Phase 2 + Phase 3).

---

## Summary

| Task | What | Key Commands |
|------|------|-------------|
| 1 | Install PostgreSQL | `brew install postgresql@16`, `createdb sharpedge` |
| 2 | Create `.env` + gitignore models/ | Create `.env` file, append to `.gitignore` |
| 3 | Alembic migration | `alembic revision --autogenerate`, `alembic upgrade head` |
| 4 | Expand backfill to all 10 collectors | Modify `orchestrator.py`, add test |
| 5 | Run the backfill | `python scripts/run_backfill.py` |
| 6 | Train the model | `python scripts/train_model.py` |
| 7 | End-to-end smoke test | Start API, curl all endpoints |

**New files created:** 4 (test, backfill script, training script, migration)
**Files modified:** 2 (orchestrator.py, .gitignore)
**New tests:** 2 (backfill integration tests)
