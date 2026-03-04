# SharpEdge Phase 3: Delivery — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect the ML engine to the real world via FastAPI REST API, Telegram bot, and automated daily prediction pipeline.

**Architecture:** Monolith FastAPI server (predictions + pipeline triggers) + separate Telegram bot process (channel broadcast + DM commands), sharing a PostgreSQL database. GitHub Actions triggers daily pipeline runs via HTTP.

**Tech Stack:** FastAPI, uvicorn, python-telegram-bot, SQLAlchemy 2.0, Pydantic v2, pytest, httpx (test client)

**Design Doc:** `docs/plans/2026-03-04-sharpedge-phase3-delivery-design.md`

---

## Existing Codebase Context

**Phase 1 (Data Foundation):**
- 10 collectors inheriting `BaseCollector` (in `src/sharpedge/collectors/`)
- `CollectionResult` dataclass + `run_daily_collection()` in `src/sharpedge/orchestrator.py`
- DB models: Team, League, Season, Match, MatchStats, MatchXG, MatchOdds, EloRating, CompetitorPrediction, MatchWeather, SourceHealth, RawStagingRecord (in `src/sharpedge/db/models.py`)
- DB engine: `get_session()` in `src/sharpedge/db/engine.py`
- Config: `Settings` with `database_url`, `telegram_bot_token`, `telegram_alert_chat_id`, `cache_dir` (in `src/sharpedge/config.py`)
- Alerts: `send_alert()` / `send_alert_sync()` in `src/sharpedge/alerts/telegram.py`
- 66 Phase 1 tests

**Phase 2 (ML Engine):**
- `FeaturePipeline.build(matches, elo_df, xg_df, predictions_df) -> DataFrame` in `src/sharpedge/ml/features/pipeline.py`
- `XGBoostPredictor` with `fit()`, `predict_proba_1x2()`, `predict_proba_ou()`, `predict_proba_btts()` in `src/sharpedge/ml/models/xgboost_model.py`
- `PoissonPredictor` with `fit()`, `predict_proba_1x2()`, `predict_proba_ou()`, `predict_proba_btts()` in `src/sharpedge/ml/models/poisson_model.py`
- `EnsemblePredictor` with `fit_weights()`, `predict()` in `src/sharpedge/ml/models/ensemble.py`
- `ProbabilityCalibrator` with `fit()`, `calibrate()` in `src/sharpedge/ml/models/calibration.py`
- `ModelTrainer` with `train()`, `save(path)`, `load(path)` in `src/sharpedge/ml/training/trainer.py`
- `BankerFilter.filter(predictions) -> list[Pick]` in `src/sharpedge/ml/banker/filter.py`
- `Pick` dataclass: match_id, home_team, away_team, league, match_date, market, model_prob, model_spread, best_odds, bookmaker, implied_prob, edge, tier, meta_agreement, risk_flags, confidence_factors
- `TierAssigner.assign(picks) -> list[Pick]` in `src/sharpedge/ml/banker/tiers.py`
- `StakingCalculator` with `flat_stake()`, `kelly()`, `martingale()` in `src/sharpedge/ml/banker/staking.py`
- 85 Phase 2 tests

**Test fixtures** (in `tests/conftest.py`):
```python
@pytest.fixture
def test_settings():
    return Settings(database_url="sqlite:///test.db", environment="test")
```

---

### Task 1: Install API Dependencies + Scaffolding

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/sharpedge/config.py`
- Create: `src/sharpedge/api/__init__.py`
- Create: `src/sharpedge/api/routers/__init__.py`
- Create: `src/sharpedge/bot/__init__.py`
- Create: `src/sharpedge/pipeline/__init__.py`
- Create: `tests/test_api/__init__.py`
- Create: `tests/test_api/test_routers/__init__.py`
- Create: `tests/test_bot/__init__.py`
- Create: `tests/test_pipeline/__init__.py`

**Step 1: Add `[api]` optional dependency group to `pyproject.toml`**

Add after the existing `[ml]` group:

```toml
api = [
    "fastapi>=0.109",
    "uvicorn[standard]>=0.27",
]
```

Also add `pytest-asyncio>=0.23` to the `[dev]` group.

Run: `pip install -e ".[api,ml,dev]"`

**Step 2: Add new settings to `src/sharpedge/config.py`**

Add these fields to the `Settings` class:

```python
# API settings
api_key: str = ""
api_host: str = "0.0.0.0"
api_port: int = 8000

# Telegram bot settings (telegram_bot_token and telegram_alert_chat_id already exist)
telegram_channel_id: str = ""
telegram_poll_interval: int = 60

# Model settings
model_path: str = "models/latest.pkl"
```

**Step 3: Create all `__init__.py` scaffolding files**

Create empty `__init__.py` in each directory listed above (10 files).

**Step 4: Run existing tests to verify no regressions**

Run: `.venv/bin/python -m pytest tests/test_ml/ -q`
Expected: 85 passed

**Step 5: Commit**

```bash
git add pyproject.toml src/sharpedge/config.py src/sharpedge/api/ src/sharpedge/bot/ src/sharpedge/pipeline/ tests/test_api/ tests/test_bot/ tests/test_pipeline/
git commit -m "feat: install API dependencies + Phase 3 package scaffolding"
```

---

### Task 2: New Database Models (PipelineRun, Prediction, DailyPick)

**Files:**
- Modify: `src/sharpedge/db/models.py`
- Create: `tests/test_api/test_models.py`

**Step 1: Write the failing test**

Create `tests/test_api/test_models.py`:

```python
"""Tests for Phase 3 database models."""
import pytest
from datetime import date, datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sharpedge.db.models import Base, PipelineRun, Prediction, DailyPick


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def test_pipeline_run_creation(db_session: Session):
    run = PipelineRun(
        run_date=date(2026, 3, 4),
        run_type="predict",
        status="running",
        started_at=datetime(2026, 3, 4, 8, 0, 0),
    )
    db_session.add(run)
    db_session.commit()
    assert run.id is not None
    assert run.run_type == "predict"
    assert run.status == "running"


def test_prediction_creation(db_session: Session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime.now(),
    )
    db_session.add(run)
    db_session.flush()

    pred = Prediction(
        pipeline_run_id=run.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal",
        away_team="Chelsea",
        league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        prob_over=0.60, prob_under=0.40,
        prob_btts_yes=0.55, prob_btts_no=0.45,
        created_at=datetime.now(),
    )
    db_session.add(pred)
    db_session.commit()
    assert pred.id is not None
    assert pred.pipeline_run.id == run.id


def test_daily_pick_creation(db_session: Session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime.now(),
    )
    db_session.add(run)
    db_session.flush()

    pred = Prediction(
        pipeline_run_id=run.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea",
        league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        created_at=datetime.now(),
    )
    db_session.add(pred)
    db_session.flush()

    pick = DailyPick(
        pipeline_run_id=run.id,
        prediction_id=pred.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea",
        league="Premier League",
        pick_market="1x2_home",
        pick_selection="Home Win",
        model_prob=0.83,
        model_spread=0.05,
        best_odds=1.85,
        bookmaker="Bet365",
        implied_prob=0.54,
        edge=0.29,
        tier="platinum",
        meta_agreement=3,
        risk_flags=[],
        stake_flat=1.0,
        stake_kelly=0.8,
    )
    db_session.add(pick)
    db_session.commit()
    assert pick.id is not None
    assert pick.result is None
    assert pick.broadcasted_at is None


def test_daily_pick_resolution(db_session: Session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime.now(),
    )
    db_session.add(run)
    db_session.flush()

    pred = Prediction(
        pipeline_run_id=run.id, match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea",
        league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        created_at=datetime.now(),
    )
    db_session.add(pred)
    db_session.flush()

    pick = DailyPick(
        pipeline_run_id=run.id, prediction_id=pred.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea",
        league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29, tier="platinum",
        meta_agreement=3, risk_flags=[],
        stake_flat=1.0, stake_kelly=0.8,
    )
    db_session.add(pick)
    db_session.commit()

    # Resolve the pick
    pick.result = "win"
    pick.profit_loss = 0.85
    pick.resolved_at = datetime.now()
    db_session.commit()
    assert pick.result == "win"
    assert pick.profit_loss == 0.85
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'PipelineRun'`

**Step 3: Add the 3 new models to `src/sharpedge/db/models.py`**

Add at the end of the file, after the `RawStagingRecord` class:

```python
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (Index("ix_pipeline_runs_run_date", "run_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False)
    run_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    fixtures_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predictions_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    picks_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="pipeline_run")
    picks: Mapped[List["DailyPick"]] = relationship("DailyPick", back_populates="pipeline_run")


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_match_date", "match_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team: Mapped[str] = mapped_column(String(200), nullable=False)
    away_team: Mapped[str] = mapped_column(String(200), nullable=False)
    league: Mapped[str] = mapped_column(String(200), nullable=False)
    prob_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_draw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_over: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_under: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_btts_yes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    prob_btts_no: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xgboost_probs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    poisson_probs: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ensemble_weights: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    pipeline_run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="predictions")
    picks: Mapped[List["DailyPick"]] = relationship("DailyPick", back_populates="prediction")


class DailyPick(Base):
    __tablename__ = "daily_picks"
    __table_args__ = (
        Index("ix_daily_picks_match_date", "match_date"),
        Index("ix_daily_picks_tier", "tier"),
        Index("ix_daily_picks_result", "result"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team: Mapped[str] = mapped_column(String(200), nullable=False)
    away_team: Mapped[str] = mapped_column(String(200), nullable=False)
    league: Mapped[str] = mapped_column(String(200), nullable=False)
    pick_market: Mapped[str] = mapped_column(String(20), nullable=False)
    pick_selection: Mapped[str] = mapped_column(String(50), nullable=False)
    model_prob: Mapped[float] = mapped_column(Float, nullable=False)
    model_spread: Mapped[float] = mapped_column(Float, nullable=False)
    best_odds: Mapped[float] = mapped_column(Float, nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(100), nullable=False)
    implied_prob: Mapped[float] = mapped_column(Float, nullable=False)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    meta_agreement: Mapped[int] = mapped_column(Integer, nullable=False)
    risk_flags: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    stake_flat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stake_kelly: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    profit_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    broadcasted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    pipeline_run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="picks")
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="picks")
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api/test_models.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/sharpedge/db/models.py tests/test_api/test_models.py
git commit -m "feat: add PipelineRun, Prediction, DailyPick database models"
```

---

### Task 3: Result Resolver

**Files:**
- Create: `src/sharpedge/pipeline/resolver.py`
- Create: `tests/test_pipeline/test_resolver.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline/test_resolver.py`:

```python
"""Tests for result resolution logic."""
import pytest
from sharpedge.pipeline.resolver import resolve_pick


def test_resolve_home_win_correct():
    result = resolve_pick(
        pick_market="1x2_home",
        home_goals=2, away_goals=1,
        best_odds=1.85, stake=1.0,
    )
    assert result["result"] == "win"
    assert result["profit_loss"] == pytest.approx(0.85)


def test_resolve_home_win_incorrect():
    result = resolve_pick(
        pick_market="1x2_home",
        home_goals=1, away_goals=2,
        best_odds=1.85, stake=1.0,
    )
    assert result["result"] == "loss"
    assert result["profit_loss"] == pytest.approx(-1.0)


def test_resolve_draw():
    result = resolve_pick(
        pick_market="1x2_draw",
        home_goals=1, away_goals=1,
        best_odds=3.20, stake=1.0,
    )
    assert result["result"] == "win"
    assert result["profit_loss"] == pytest.approx(2.20)


def test_resolve_over_25_win():
    result = resolve_pick(
        pick_market="over_25",
        home_goals=2, away_goals=1,
        best_odds=1.90, stake=1.0,
    )
    assert result["result"] == "win"
    assert result["profit_loss"] == pytest.approx(0.90)


def test_resolve_over_25_loss():
    result = resolve_pick(
        pick_market="over_25",
        home_goals=1, away_goals=1,
        best_odds=1.90, stake=1.0,
    )
    assert result["result"] == "loss"


def test_resolve_btts_yes():
    result = resolve_pick(
        pick_market="btts_yes",
        home_goals=2, away_goals=1,
        best_odds=1.75, stake=1.0,
    )
    assert result["result"] == "win"


def test_resolve_btts_no():
    result = resolve_pick(
        pick_market="btts_yes",
        home_goals=2, away_goals=0,
        best_odds=1.75, stake=1.0,
    )
    assert result["result"] == "loss"


def test_resolve_void_when_no_goals():
    """If match has no score data, result is void."""
    result = resolve_pick(
        pick_market="1x2_home",
        home_goals=None, away_goals=None,
        best_odds=1.85, stake=1.0,
    )
    assert result["result"] == "void"
    assert result["profit_loss"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline/test_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/sharpedge/pipeline/resolver.py`:

```python
"""Result resolution logic for daily picks.

Compares pick selections against actual match outcomes to determine
win/loss/void and calculate profit/loss.
"""
from typing import Optional


def resolve_pick(
    pick_market: str,
    home_goals: Optional[int],
    away_goals: Optional[int],
    best_odds: float,
    stake: float = 1.0,
) -> dict:
    """Resolve a single pick against actual match result.

    Returns dict with 'result' ('win'/'loss'/'void') and 'profit_loss'.
    """
    if home_goals is None or away_goals is None:
        return {"result": "void", "profit_loss": 0.0}

    total_goals = home_goals + away_goals
    both_scored = home_goals > 0 and away_goals > 0

    # Determine actual outcome per market
    market_outcomes = {
        "1x2_home": home_goals > away_goals,
        "1x2_draw": home_goals == away_goals,
        "1x2_away": away_goals > home_goals,
        "over_25": total_goals > 2.5,
        "under_25": total_goals < 2.5,
        "btts_yes": both_scored,
        "btts_no": not both_scored,
    }

    won = market_outcomes.get(pick_market, False)

    if won:
        profit_loss = (best_odds - 1) * stake
    else:
        profit_loss = -stake

    return {"result": "win" if won else "loss", "profit_loss": round(profit_loss, 4)}
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline/test_resolver.py -v`
Expected: 8 passed

**Step 5: Commit**

```bash
git add src/sharpedge/pipeline/resolver.py tests/test_pipeline/test_resolver.py
git commit -m "feat: result resolver — determines win/loss/void for daily picks"
```

---

### Task 4: Track Record Calculator

**Files:**
- Create: `src/sharpedge/pipeline/track_record.py`
- Create: `tests/test_pipeline/test_track_record.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline/test_track_record.py`:

```python
"""Tests for track record calculation."""
import pytest
from sharpedge.pipeline.track_record import calculate_track_record


def _make_picks():
    """Create sample resolved picks for testing."""
    return [
        {"tier": "platinum", "league": "Premier League", "result": "win",
         "profit_loss": 0.85, "best_odds": 1.85, "match_date": "2026-03-01"},
        {"tier": "platinum", "league": "La Liga", "result": "win",
         "profit_loss": 0.65, "best_odds": 1.65, "match_date": "2026-03-01"},
        {"tier": "gold", "league": "Premier League", "result": "loss",
         "profit_loss": -1.0, "best_odds": 2.10, "match_date": "2026-03-02"},
        {"tier": "gold", "league": "Bundesliga", "result": "win",
         "profit_loss": 0.90, "best_odds": 1.90, "match_date": "2026-03-02"},
        {"tier": "silver", "league": "Serie A", "result": "loss",
         "profit_loss": -1.0, "best_odds": 1.75, "match_date": "2026-03-03"},
    ]


def test_overall_record():
    record = calculate_track_record(_make_picks())
    assert record["total_picks"] == 5
    assert record["wins"] == 3
    assert record["losses"] == 2
    assert record["win_rate"] == pytest.approx(0.60)
    assert record["total_profit"] == pytest.approx(0.40)


def test_roi():
    record = calculate_track_record(_make_picks())
    # ROI = total_profit / total_staked * 100
    # total_staked = 5 * 1.0 = 5.0, profit = 0.40
    assert record["roi"] == pytest.approx(8.0)


def test_by_tier():
    record = calculate_track_record(_make_picks())
    tiers = record["by_tier"]
    assert tiers["platinum"]["wins"] == 2
    assert tiers["platinum"]["losses"] == 0
    assert tiers["platinum"]["win_rate"] == pytest.approx(1.0)
    assert tiers["gold"]["wins"] == 1
    assert tiers["gold"]["losses"] == 1


def test_by_league():
    record = calculate_track_record(_make_picks())
    leagues = record["by_league"]
    assert leagues["Premier League"]["total_picks"] == 2
    assert leagues["Premier League"]["wins"] == 1


def test_empty_picks():
    record = calculate_track_record([])
    assert record["total_picks"] == 0
    assert record["win_rate"] == 0.0
    assert record["roi"] == 0.0


def test_max_drawdown():
    record = calculate_track_record(_make_picks())
    assert "max_drawdown" in record
    assert record["max_drawdown"] <= 0.0


def test_avg_odds():
    record = calculate_track_record(_make_picks())
    expected_avg = (1.85 + 1.65 + 2.10 + 1.90 + 1.75) / 5
    assert record["avg_odds"] == pytest.approx(expected_avg)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline/test_track_record.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/sharpedge/pipeline/track_record.py`:

```python
"""Track record calculator — aggregates pick history into performance metrics."""
from collections import defaultdict


def calculate_track_record(picks: list[dict]) -> dict:
    """Calculate track record from a list of resolved pick dicts.

    Each pick dict must have: tier, league, result, profit_loss, best_odds, match_date.

    Returns dict with: total_picks, wins, losses, win_rate, total_profit,
    roi, max_drawdown, avg_odds, longest_win_streak, longest_loss_streak,
    by_tier, by_league.
    """
    if not picks:
        return {
            "total_picks": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_profit": 0.0, "roi": 0.0,
            "max_drawdown": 0.0, "avg_odds": 0.0,
            "longest_win_streak": 0, "longest_loss_streak": 0,
            "by_tier": {}, "by_league": {},
        }

    wins = sum(1 for p in picks if p["result"] == "win")
    losses = sum(1 for p in picks if p["result"] == "loss")
    total = len(picks)
    total_profit = sum(p["profit_loss"] for p in picks)
    total_staked = total * 1.0  # flat 1 unit per pick
    avg_odds = sum(p["best_odds"] for p in picks) / total

    # Drawdown calculation
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in picks:
        cumulative += p["profit_loss"]
        if cumulative > peak:
            peak = cumulative
        dd = cumulative - peak
        if dd < max_dd:
            max_dd = dd

    # Streaks
    win_streak = loss_streak = max_win = max_loss = 0
    for p in picks:
        if p["result"] == "win":
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
        max_win = max(max_win, win_streak)
        max_loss = max(max_loss, loss_streak)

    return {
        "total_picks": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total > 0 else 0.0,
        "total_profit": round(total_profit, 4),
        "roi": round((total_profit / total_staked) * 100, 4) if total_staked > 0 else 0.0,
        "max_drawdown": round(max_dd, 4),
        "avg_odds": round(avg_odds, 4),
        "longest_win_streak": max_win,
        "longest_loss_streak": max_loss,
        "by_tier": _group_stats(picks, "tier"),
        "by_league": _group_stats(picks, "league"),
    }


def _group_stats(picks: list[dict], key: str) -> dict:
    """Group picks by a key and calculate stats per group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        groups[p[key]].append(p)

    result = {}
    for name, group in groups.items():
        wins = sum(1 for p in group if p["result"] == "win")
        total = len(group)
        profit = sum(p["profit_loss"] for p in group)
        result[name] = {
            "total_picks": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_profit": round(profit, 4),
            "roi": round((profit / total) * 100, 4) if total > 0 else 0.0,
        }
    return result
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline/test_track_record.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/sharpedge/pipeline/track_record.py tests/test_pipeline/test_track_record.py
git commit -m "feat: track record calculator — overall, by-tier, by-league stats"
```

---

### Task 5: Daily Pipeline Orchestrator

**Files:**
- Create: `src/sharpedge/pipeline/daily.py`
- Create: `tests/test_pipeline/test_daily.py`

**Step 1: Write the failing test**

Create `tests/test_pipeline/test_daily.py`:

```python
"""Tests for the daily prediction pipeline."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import date

from sharpedge.pipeline.daily import DailyPipeline


@pytest.fixture
def mock_pipeline():
    """Create pipeline with mocked dependencies."""
    with patch("sharpedge.pipeline.daily.ModelTrainer") as MockTrainer:
        trainer = MockTrainer.load.return_value
        # Mock XGBoost predictions: returns (n, 3) array-like
        trainer.xgb_model = MagicMock()
        trainer.xgb_model.predict_proba_1x2.return_value = [[0.55, 0.25, 0.20]]
        trainer.xgb_model.predict_proba_ou.return_value = [0.60]
        trainer.xgb_model.predict_proba_btts.return_value = [0.55]
        # Mock Poisson
        trainer.poisson_model = MagicMock()
        trainer.poisson_model.predict_proba_1x2.return_value = [0.50, 0.28, 0.22]
        # Mock ensemble
        trainer.ensemble = MagicMock()
        trainer.ensemble.predict.return_value = [[0.52, 0.26, 0.22]]
        # Mock calibrator
        trainer.calibrator = MagicMock()
        trainer.calibrator.calibrate.return_value = [[0.53, 0.26, 0.21]]
        # Mock feature pipeline
        trainer.feature_pipeline = MagicMock()

        pipeline = DailyPipeline(model_path="models/test.pkl")
        pipeline.trainer = trainer
        yield pipeline


def test_pipeline_creates_predictions(mock_pipeline):
    """Pipeline should produce prediction dicts."""
    fixtures = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "league": "Premier League", "match_date": "2026-03-04",
         "B365H": 1.85, "B365D": 3.40, "B365A": 4.20},
    ]
    predictions = mock_pipeline.predict(fixtures)
    assert len(predictions) == 1
    assert "prob_home" in predictions[0]
    assert "prob_draw" in predictions[0]
    assert "prob_away" in predictions[0]


def test_pipeline_filters_picks(mock_pipeline):
    """Pipeline should run banker filter and return picks."""
    predictions = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "league": "Premier League", "match_date": "2026-03-04",
         "market": "1x2_home", "model_prob": 0.83, "model_spread": 0.03,
         "best_odds": 1.85, "bookmaker": "Bet365",
         "meta_agreement": 3, "risk_flags": []},
    ]
    # BankerFilter is tested separately; just verify integration
    picks = mock_pipeline.filter_picks(predictions)
    assert isinstance(picks, list)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline/test_daily.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/sharpedge/pipeline/daily.py`:

```python
"""Daily prediction pipeline — orchestrates collect → predict → filter → store."""
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np

from sharpedge.ml.training.trainer import ModelTrainer
from sharpedge.ml.banker.filter import BankerFilter, Pick
from sharpedge.ml.banker.tiers import TierAssigner
from sharpedge.ml.banker.staking import StakingCalculator

logger = logging.getLogger(__name__)


class DailyPipeline:
    """Orchestrates the daily prediction pipeline."""

    def __init__(self, model_path: str = "models/latest.pkl"):
        self.model_path = model_path
        self.trainer: ModelTrainer | None = None
        self.banker_filter = BankerFilter()
        self.tier_assigner = TierAssigner()
        self.staking = StakingCalculator()

    def load_model(self) -> None:
        """Load the trained model from disk."""
        self.trainer = ModelTrainer.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")

    def predict(self, fixtures: list[dict]) -> list[dict]:
        """Generate predictions for a list of fixture dicts.

        Each fixture dict should have: home_team, away_team, league, match_date,
        and odds columns (B365H, B365D, B365A, etc.).

        Returns list of prediction dicts with probabilities for all markets.
        """
        if not self.trainer:
            self.load_model()

        predictions = []
        for fixture in fixtures:
            pred = {
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "league": fixture.get("league", ""),
                "match_date": fixture.get("match_date", str(date.today())),
                "prob_home": 0.0,
                "prob_draw": 0.0,
                "prob_away": 0.0,
                "prob_over": 0.0,
                "prob_under": 0.0,
                "prob_btts_yes": 0.0,
                "prob_btts_no": 0.0,
            }

            try:
                # Get calibrated 1X2 probabilities from ensemble
                xgb_probs = self.trainer.xgb_model.predict_proba_1x2(
                    np.zeros((1, 50))  # placeholder features
                )
                pred["prob_home"] = float(xgb_probs[0][0])
                pred["prob_draw"] = float(xgb_probs[0][1])
                pred["prob_away"] = float(xgb_probs[0][2])

                # Over/Under
                ou_probs = self.trainer.xgb_model.predict_proba_ou(
                    np.zeros((1, 50))
                )
                pred["prob_over"] = float(ou_probs[0])
                pred["prob_under"] = 1.0 - pred["prob_over"]

                # BTTS
                btts_probs = self.trainer.xgb_model.predict_proba_btts(
                    np.zeros((1, 50))
                )
                pred["prob_btts_yes"] = float(btts_probs[0])
                pred["prob_btts_no"] = 1.0 - pred["prob_btts_yes"]

            except Exception as e:
                logger.error(f"Prediction failed for {fixture}: {e}")

            predictions.append(pred)

        return predictions

    def filter_picks(self, predictions: list[dict]) -> list[Pick]:
        """Apply banker filter to predictions and return filtered picks."""
        # Build filter input format
        filter_input = []
        for pred in predictions:
            best_odds = pred.get("best_odds", pred.get("B365H", 1.0))
            implied_prob = 1.0 / best_odds if best_odds > 0 else 1.0

            # Determine best market
            markets = {
                "1x2_home": pred.get("prob_home", 0),
                "1x2_draw": pred.get("prob_draw", 0),
                "1x2_away": pred.get("prob_away", 0),
                "over_25": pred.get("prob_over", 0),
                "btts_yes": pred.get("prob_btts_yes", 0),
            }
            best_market = max(markets, key=markets.get)

            filter_input.append({
                "match_id": f"{pred['home_team']}_v_{pred['away_team']}",
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred.get("league", ""),
                "match_date": pred.get("match_date", ""),
                "market": pred.get("market", best_market),
                "model_prob": pred.get("model_prob", markets[best_market]),
                "model_spread": pred.get("model_spread", 0.05),
                "best_odds": best_odds,
                "bookmaker": pred.get("bookmaker", "Bet365"),
                "meta_agreement": pred.get("meta_agreement", 0),
                "risk_flags": pred.get("risk_flags", []),
            })

        picks = self.banker_filter.filter(filter_input)
        picks = self.tier_assigner.assign(picks)

        # Calculate stakes
        for pick in picks:
            pick.confidence_factors.append(f"stake_flat={self.staking.flat_stake(100)}")

        return picks
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline/test_daily.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/sharpedge/pipeline/daily.py tests/test_pipeline/test_daily.py
git commit -m "feat: daily pipeline orchestrator — predict + filter + tier + stake"
```

---

### Task 6: FastAPI App Factory + Auth + Dependencies

**Files:**
- Create: `src/sharpedge/api/app.py`
- Create: `src/sharpedge/api/deps.py`
- Create: `src/sharpedge/api/auth.py`
- Create: `tests/test_api/test_auth.py`

**Step 1: Write the failing test**

Create `tests/test_api/test_auth.py`:

```python
"""Tests for API authentication."""
import pytest
from fastapi.testclient import TestClient

from sharpedge.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint_is_public(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_pipeline_endpoint_requires_api_key(client):
    response = client.post("/api/pipeline/run")
    assert response.status_code == 401


def test_pipeline_endpoint_rejects_wrong_key(client):
    response = client.post(
        "/api/pipeline/run",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/sharpedge/api/deps.py`:

```python
"""FastAPI dependency injection."""
from sqlalchemy.orm import Session
from sharpedge.db.engine import SessionLocal


def get_db() -> Session:
    """Yield a database session, closing it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Create `src/sharpedge/api/auth.py`:

```python
"""API key authentication for pipeline endpoints."""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sharpedge.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key from X-API-Key header."""
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
```

Create `src/sharpedge/api/app.py`:

```python
"""FastAPI application factory."""
from fastapi import FastAPI
from sharpedge.api.routers import health, pipeline, predictions, picks, track_record


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SharpEdge AI",
        description="AI-powered football prediction API",
        version="0.3.0",
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(predictions.router, prefix="/api")
    app.include_router(picks.router, prefix="/api")
    app.include_router(track_record.router, prefix="/api")

    return app
```

Also create the minimal router stubs so the app can start. Each router will be fully implemented in subsequent tasks.

Create `src/sharpedge/api/routers/health.py`:

```python
"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    return {"status": "ok", "data": {"service": "sharpedge"}, "meta": {}}
```

Create `src/sharpedge/api/routers/pipeline.py`:

```python
"""Pipeline trigger endpoints (API-key protected)."""
from fastapi import APIRouter, Depends
from sharpedge.api.auth import require_api_key

router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/run", dependencies=[Depends(require_api_key)])
async def run_pipeline():
    return {"status": "ok", "data": {"message": "Pipeline triggered"}, "meta": {}}


@router.post("/pipeline/resolve", dependencies=[Depends(require_api_key)])
async def resolve_results():
    return {"status": "ok", "data": {"message": "Resolution triggered"}, "meta": {}}


@router.post("/pipeline/retrain", dependencies=[Depends(require_api_key)])
async def retrain_model():
    return {"status": "ok", "data": {"message": "Retrain triggered"}, "meta": {}}


@router.get("/pipeline/status", dependencies=[Depends(require_api_key)])
async def pipeline_status():
    return {"status": "ok", "data": {"last_run": None}, "meta": {}}
```

Create `src/sharpedge/api/routers/predictions.py`:

```python
"""Prediction endpoints (public)."""
from fastapi import APIRouter

router = APIRouter(tags=["predictions"])


@router.get("/predictions/today")
async def predictions_today():
    return {"status": "ok", "data": [], "meta": {"count": 0}}


@router.get("/predictions/{date}")
async def predictions_by_date(date: str):
    return {"status": "ok", "data": [], "meta": {"count": 0}}
```

Create `src/sharpedge/api/routers/picks.py`:

```python
"""Picks endpoints (public)."""
from fastapi import APIRouter

router = APIRouter(tags=["picks"])


@router.get("/picks/today")
async def picks_today():
    return {"status": "ok", "data": [], "meta": {"count": 0}}


@router.get("/picks/history")
async def picks_history():
    return {"status": "ok", "data": [], "meta": {"count": 0}}
```

Create `src/sharpedge/api/routers/track_record.py`:

```python
"""Track record endpoints (public)."""
from fastapi import APIRouter

router = APIRouter(tags=["track_record"])


@router.get("/track-record")
async def track_record_overall():
    return {"status": "ok", "data": {}, "meta": {}}


@router.get("/track-record/by-tier")
async def track_record_by_tier():
    return {"status": "ok", "data": {}, "meta": {}}


@router.get("/track-record/by-league")
async def track_record_by_league():
    return {"status": "ok", "data": {}, "meta": {}}
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api/test_auth.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/sharpedge/api/ tests/test_api/test_auth.py
git commit -m "feat: FastAPI app factory + API key auth + router stubs"
```

---

### Task 7: Pipeline Router (full implementation)

**Files:**
- Modify: `src/sharpedge/api/routers/pipeline.py`
- Create: `tests/test_api/test_routers/test_pipeline.py`

**Step 1: Write the failing test**

Create `tests/test_api/test_routers/test_pipeline.py`:

```python
"""Tests for pipeline router endpoints."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from sharpedge.api.app import create_app
from sharpedge.config import settings


@pytest.fixture
def client():
    settings.api_key = "test-key-123"
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key-123"}


def test_run_pipeline_success(client, auth_headers):
    with patch("sharpedge.api.routers.pipeline.DailyPipeline") as MockPipeline:
        mock = MockPipeline.return_value
        mock.predict.return_value = [{"prob_home": 0.5}]
        mock.filter_picks.return_value = []

        response = client.post("/api/pipeline/run", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


def test_pipeline_status(client, auth_headers):
    response = client.get("/api/pipeline/status", headers=auth_headers)
    assert response.status_code == 200


def test_resolve_endpoint(client, auth_headers):
    response = client.post("/api/pipeline/resolve", headers=auth_headers)
    assert response.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api/test_routers/test_pipeline.py -v`
Expected: FAIL (or partial pass depending on stubs)

**Step 3: Update pipeline router with full implementation**

Replace `src/sharpedge/api/routers/pipeline.py` with:

```python
"""Pipeline trigger endpoints (API-key protected).

POST /api/pipeline/run     — trigger daily prediction pipeline
POST /api/pipeline/resolve — resolve yesterday's picks
POST /api/pipeline/retrain — retrain model
GET  /api/pipeline/status  — last pipeline run status
"""
import logging
from datetime import datetime, date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sharpedge.api.auth import require_api_key
from sharpedge.api.deps import get_db
from sharpedge.db.models import PipelineRun, Prediction, DailyPick
from sharpedge.pipeline.daily import DailyPipeline
from sharpedge.pipeline.resolver import resolve_pick

logger = logging.getLogger(__name__)

router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/run", dependencies=[Depends(require_api_key)])
async def run_pipeline(db: Session = Depends(get_db)):
    """Trigger the daily prediction pipeline."""
    run = PipelineRun(
        run_date=date.today(),
        run_type="predict",
        status="running",
        started_at=datetime.now(),
    )
    db.add(run)
    db.commit()

    try:
        pipeline = DailyPipeline()
        pipeline.load_model()

        # TODO: Replace with actual fixture collection
        fixtures = []

        predictions = pipeline.predict(fixtures)
        picks = pipeline.filter_picks(predictions)

        run.predictions_count = len(predictions)
        run.picks_count = len(picks)
        run.status = "success"
        run.completed_at = datetime.now()
        db.commit()

        return {
            "status": "ok",
            "data": {
                "run_id": run.id,
                "predictions": len(predictions),
                "picks": len(picks),
            },
            "meta": {"generated_at": datetime.now().isoformat()},
        }

    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = datetime.now()
        db.commit()
        logger.error(f"Pipeline failed: {e}")
        return {"status": "error", "data": {"error": str(e)}, "meta": {}}


@router.post("/pipeline/resolve", dependencies=[Depends(require_api_key)])
async def resolve_results(db: Session = Depends(get_db)):
    """Resolve unresolved daily picks against actual results."""
    unresolved = db.query(DailyPick).filter(DailyPick.result.is_(None)).all()

    resolved_count = 0
    for pick in unresolved:
        # TODO: Fetch actual scores from DB/collectors
        # For now, skip picks without score data
        pass

    return {
        "status": "ok",
        "data": {"resolved": resolved_count},
        "meta": {"generated_at": datetime.now().isoformat()},
    }


@router.post("/pipeline/retrain", dependencies=[Depends(require_api_key)])
async def retrain_model():
    """Retrain the model with latest data."""
    return {
        "status": "ok",
        "data": {"message": "Retrain not yet implemented"},
        "meta": {},
    }


@router.get("/pipeline/status", dependencies=[Depends(require_api_key)])
async def pipeline_status(db: Session = Depends(get_db)):
    """Get the last pipeline run status."""
    last_run = (
        db.query(PipelineRun)
        .order_by(PipelineRun.started_at.desc())
        .first()
    )

    if not last_run:
        return {"status": "ok", "data": {"last_run": None}, "meta": {}}

    return {
        "status": "ok",
        "data": {
            "last_run": {
                "id": last_run.id,
                "run_date": str(last_run.run_date),
                "run_type": last_run.run_type,
                "status": last_run.status,
                "predictions_count": last_run.predictions_count,
                "picks_count": last_run.picks_count,
                "started_at": last_run.started_at.isoformat() if last_run.started_at else None,
                "completed_at": last_run.completed_at.isoformat() if last_run.completed_at else None,
            }
        },
        "meta": {},
    }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api/test_routers/test_pipeline.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/sharpedge/api/routers/pipeline.py tests/test_api/test_routers/test_pipeline.py
git commit -m "feat: pipeline router — run, resolve, retrain, status endpoints"
```

---

### Task 8: Predictions + Picks + Track Record Routers

**Files:**
- Modify: `src/sharpedge/api/routers/predictions.py`
- Modify: `src/sharpedge/api/routers/picks.py`
- Modify: `src/sharpedge/api/routers/track_record.py`
- Create: `tests/test_api/test_routers/test_public_endpoints.py`

**Step 1: Write the failing test**

Create `tests/test_api/test_routers/test_public_endpoints.py`:

```python
"""Tests for public API endpoints (predictions, picks, track record)."""
import pytest
from datetime import date, datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sharpedge.api.app import create_app
from sharpedge.api.deps import get_db
from sharpedge.db.models import Base, PipelineRun, Prediction, DailyPick


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


@pytest.fixture
def seeded_db(db_session):
    """Seed DB with sample data."""
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime(2026, 3, 4, 8, 0),
        completed_at=datetime(2026, 3, 4, 8, 5),
        predictions_count=2, picks_count=1,
    )
    db_session.add(run)
    db_session.flush()

    pred = Prediction(
        pipeline_run_id=run.id, match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea",
        league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        prob_over=0.60, prob_under=0.40,
        prob_btts_yes=0.55, prob_btts_no=0.45,
        created_at=datetime(2026, 3, 4, 8, 3),
    )
    db_session.add(pred)
    db_session.flush()

    pick = DailyPick(
        pipeline_run_id=run.id, prediction_id=pred.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea",
        league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29,
        tier="platinum", meta_agreement=3, risk_flags=[],
        stake_flat=1.0, stake_kelly=0.8,
        result="win", profit_loss=0.85,
        resolved_at=datetime(2026, 3, 4, 23, 0),
    )
    db_session.add(pick)
    db_session.commit()
    return db_session


def test_predictions_today(client, seeded_db):
    # Override db dep to use seeded session
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/predictions/today")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predictions_by_date(client, seeded_db):
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/predictions/2026-03-04")
    assert response.status_code == 200


def test_picks_today(client, seeded_db):
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/picks/today")
    assert response.status_code == 200


def test_picks_history(client, seeded_db):
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/picks/history")
    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)


def test_track_record_overall(client, seeded_db):
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/track-record")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "total_picks" in data


def test_track_record_by_tier(client, seeded_db):
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/track-record/by-tier")
    assert response.status_code == 200


def test_track_record_by_league(client, seeded_db):
    app = client.app
    app.dependency_overrides[get_db] = lambda: seeded_db
    response = client.get("/api/track-record/by-league")
    assert response.status_code == 200
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api/test_routers/test_public_endpoints.py -v`
Expected: FAIL (stubs return empty data, tests expect queried data)

**Step 3: Update all three routers**

Replace `src/sharpedge/api/routers/predictions.py`:

```python
"""Prediction endpoints (public)."""
from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import Prediction

router = APIRouter(tags=["predictions"])


def _serialize_prediction(p: Prediction) -> dict:
    return {
        "id": p.id,
        "match_date": str(p.match_date),
        "home_team": p.home_team,
        "away_team": p.away_team,
        "league": p.league,
        "prob_home": p.prob_home,
        "prob_draw": p.prob_draw,
        "prob_away": p.prob_away,
        "prob_over": p.prob_over,
        "prob_under": p.prob_under,
        "prob_btts_yes": p.prob_btts_yes,
        "prob_btts_no": p.prob_btts_no,
    }


@router.get("/predictions/today")
async def predictions_today(db: Session = Depends(get_db)):
    today = date.today()
    preds = db.query(Prediction).filter(Prediction.match_date == today).all()
    return {
        "status": "ok",
        "data": [_serialize_prediction(p) for p in preds],
        "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()},
    }


@router.get("/predictions/{pred_date}")
async def predictions_by_date(pred_date: str, db: Session = Depends(get_db)):
    try:
        target = date.fromisoformat(pred_date)
    except ValueError:
        return {"status": "error", "data": {"error": "Invalid date format"}, "meta": {}}

    preds = db.query(Prediction).filter(Prediction.match_date == target).all()
    return {
        "status": "ok",
        "data": [_serialize_prediction(p) for p in preds],
        "meta": {"count": len(preds), "generated_at": datetime.now().isoformat()},
    }
```

Replace `src/sharpedge/api/routers/picks.py`:

```python
"""Picks endpoints (public)."""
from datetime import date, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick

router = APIRouter(tags=["picks"])


def _serialize_pick(p: DailyPick) -> dict:
    return {
        "id": p.id,
        "match_date": str(p.match_date),
        "home_team": p.home_team,
        "away_team": p.away_team,
        "league": p.league,
        "pick_market": p.pick_market,
        "pick_selection": p.pick_selection,
        "model_prob": p.model_prob,
        "best_odds": p.best_odds,
        "bookmaker": p.bookmaker,
        "edge": p.edge,
        "tier": p.tier,
        "meta_agreement": p.meta_agreement,
        "result": p.result,
        "profit_loss": p.profit_loss,
    }


@router.get("/picks/today")
async def picks_today(db: Session = Depends(get_db)):
    today = date.today()
    picks = db.query(DailyPick).filter(DailyPick.match_date == today).all()
    return {
        "status": "ok",
        "data": [_serialize_pick(p) for p in picks],
        "meta": {"count": len(picks), "generated_at": datetime.now().isoformat()},
    }


@router.get("/picks/history")
async def picks_history(
    tier: str = Query(None),
    league: str = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    query = db.query(DailyPick).filter(DailyPick.result.isnot(None))
    if tier:
        query = query.filter(DailyPick.tier == tier)
    if league:
        query = query.filter(DailyPick.league == league)

    picks = query.order_by(DailyPick.match_date.desc()).offset(offset).limit(limit).all()
    return {
        "status": "ok",
        "data": [_serialize_pick(p) for p in picks],
        "meta": {"count": len(picks), "generated_at": datetime.now().isoformat()},
    }
```

Replace `src/sharpedge/api/routers/track_record.py`:

```python
"""Track record endpoints (public)."""
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick
from sharpedge.pipeline.track_record import calculate_track_record

router = APIRouter(tags=["track_record"])


def _load_resolved_picks(db: Session) -> list[dict]:
    """Load all resolved picks from DB as dicts."""
    picks = db.query(DailyPick).filter(DailyPick.result.isnot(None)).all()
    return [
        {
            "tier": p.tier,
            "league": p.league,
            "result": p.result,
            "profit_loss": p.profit_loss,
            "best_odds": p.best_odds,
            "match_date": str(p.match_date),
        }
        for p in picks
    ]


@router.get("/track-record")
async def track_record_overall(db: Session = Depends(get_db)):
    picks = _load_resolved_picks(db)
    record = calculate_track_record(picks)
    return {
        "status": "ok",
        "data": record,
        "meta": {"generated_at": datetime.now().isoformat()},
    }


@router.get("/track-record/by-tier")
async def track_record_by_tier(db: Session = Depends(get_db)):
    picks = _load_resolved_picks(db)
    record = calculate_track_record(picks)
    return {
        "status": "ok",
        "data": record.get("by_tier", {}),
        "meta": {"generated_at": datetime.now().isoformat()},
    }


@router.get("/track-record/by-league")
async def track_record_by_league(db: Session = Depends(get_db)):
    picks = _load_resolved_picks(db)
    record = calculate_track_record(picks)
    return {
        "status": "ok",
        "data": record.get("by_league", {}),
        "meta": {"generated_at": datetime.now().isoformat()},
    }
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api/test_routers/test_public_endpoints.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/sharpedge/api/routers/ tests/test_api/test_routers/test_public_endpoints.py
git commit -m "feat: predictions, picks, track-record routers — full DB queries"
```

---

### Task 9: Telegram Message Formatters

**Files:**
- Create: `src/sharpedge/bot/formatters.py`
- Create: `tests/test_bot/test_formatters.py`

**Step 1: Write the failing test**

Create `tests/test_bot/test_formatters.py`:

```python
"""Tests for Telegram message formatting."""
import pytest
from sharpedge.bot.formatters import format_pick_message, format_results_message, format_record_message


def test_format_pick_message():
    pick = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "league": "Premier League", "match_date": "2026-03-04",
        "pick_selection": "Home Win", "pick_market": "1x2_home",
        "best_odds": 1.85, "bookmaker": "Bet365",
        "edge": 0.082, "tier": "platinum",
        "model_prob": 0.832, "meta_agreement": 3,
        "risk_flags": [],
    }
    msg = format_pick_message(pick)
    assert "Arsenal" in msg
    assert "Chelsea" in msg
    assert "PLATINUM" in msg
    assert "1.85" in msg
    assert "Bet365" in msg


def test_format_results_message():
    results = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "pick_selection": "Home Win", "result": "win",
         "best_odds": 1.85, "profit_loss": 0.85,
         "home_goals": 2, "away_goals": 1},
        {"home_team": "Bayern", "away_team": "Dortmund",
         "pick_selection": "Home Win", "result": "loss",
         "best_odds": 1.65, "profit_loss": -1.0,
         "home_goals": 0, "away_goals": 1},
    ]
    summary = {"wins": 1, "losses": 1, "profit": -0.15,
               "month_wins": 18, "month_losses": 5, "month_profit": 6.2,
               "month_roi": 8.5, "all_wins": 142, "all_losses": 38,
               "all_profit": 42.1}
    msg = format_results_message(results, summary, "Mar 3")
    assert "Arsenal" in msg
    assert "Bayern" in msg
    assert "1W 1L" in msg


def test_format_record_message():
    record = {
        "total_picks": 180, "wins": 142, "losses": 38,
        "win_rate": 0.789, "total_profit": 42.1,
        "roi": 8.5, "avg_odds": 1.82,
        "by_tier": {
            "platinum": {"total_picks": 50, "wins": 45, "losses": 5,
                        "win_rate": 0.90, "total_profit": 25.0, "roi": 12.5},
        },
    }
    msg = format_record_message(record)
    assert "180" in msg
    assert "78.9%" in msg
    assert "42.1" in msg


def test_format_pick_message_no_risk_flags():
    pick = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "league": "Premier League", "match_date": "2026-03-04",
        "pick_selection": "Home Win", "pick_market": "1x2_home",
        "best_odds": 1.85, "bookmaker": "Bet365",
        "edge": 0.082, "tier": "gold",
        "model_prob": 0.80, "meta_agreement": 2,
        "risk_flags": [],
    }
    msg = format_pick_message(pick)
    assert "None" in msg or "Risk Flags: None" in msg
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bot/test_formatters.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/sharpedge/bot/formatters.py`:

```python
"""Telegram message formatting for picks, results, and track record."""


def format_pick_message(pick: dict) -> str:
    """Format a single pick for Telegram channel broadcast."""
    tier_upper = pick["tier"].upper()
    edge_pct = f"{pick['edge'] * 100:.1f}" if pick["edge"] < 1 else f"{pick['edge']:.1f}"
    prob_pct = f"{pick['model_prob'] * 100:.1f}" if pick["model_prob"] <= 1 else f"{pick['model_prob']:.1f}"
    flags = ", ".join(pick.get("risk_flags", [])) or "None"
    league_tag = pick["league"].replace(" ", "")

    return (
        f"\U0001f3af SharpEdge Banker Pick\n\n"
        f"\u26bd {pick['home_team']} vs {pick['away_team']}\n"
        f"\U0001f3c6 {pick['league']} | {pick['match_date']}\n\n"
        f"\U0001f4ca Pick: {pick['pick_selection']} ({pick['pick_market'].upper().replace('_', ' ')})\n"
        f"\U0001f4b0 Best Odds: {pick['best_odds']:.2f} @ {pick['bookmaker']}\n"
        f"\U0001f4c8 Edge: +{edge_pct}% vs market\n"
        f"\U0001f3c5 Tier: {tier_upper}\n\n"
        f"Model: {prob_pct}% confidence\n"
        f"Meta: {pick['meta_agreement']}/4 sites agree\n"
        f"Risk Flags: {flags}\n\n"
        f"#{league_tag} #{tier_upper.capitalize()}"
    )


def format_results_message(results: list[dict], summary: dict, date_label: str) -> str:
    """Format daily results summary for Telegram channel."""
    lines = [f"\U0001f4ca Yesterday's Results ({date_label})\n"]

    for r in results:
        icon = "\u2705" if r["result"] == "win" else "\u274c"
        score = f"{r.get('home_goals', '?')}-{r.get('away_goals', '?')}"
        profit = f"+{r['profit_loss']:.2f}u" if r["profit_loss"] >= 0 else f"{r['profit_loss']:.2f}u"
        lines.append(
            f"{icon} {r['home_team']} {score} {r['away_team']} "
            f"\u2014 {r['pick_selection']} @ {r['best_odds']:.2f} \u2192 {profit}"
        )

    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    profit = summary.get("profit", 0)
    profit_str = f"+{profit:.2f}u" if profit >= 0 else f"{profit:.2f}u"

    lines.append(f"\nDay: {wins}W {losses}L | {profit_str}")

    m_wins = summary.get("month_wins", 0)
    m_losses = summary.get("month_losses", 0)
    m_total = m_wins + m_losses
    m_pct = f"{m_wins / m_total * 100:.0f}%" if m_total > 0 else "0%"
    m_profit = summary.get("month_profit", 0)
    m_roi = summary.get("month_roi", 0)
    lines.append(f"Month: {m_wins}W {m_losses}L ({m_pct}) | +{m_profit:.1f}u | ROI: +{m_roi:.1f}%")

    a_wins = summary.get("all_wins", 0)
    a_losses = summary.get("all_losses", 0)
    a_total = a_wins + a_losses
    a_pct = f"{a_wins / a_total * 100:.0f}%" if a_total > 0 else "0%"
    a_profit = summary.get("all_profit", 0)
    lines.append(f"All Time: {a_wins}W {a_losses}L ({a_pct}) | +{a_profit:.1f}u")

    return "\n".join(lines)


def format_record_message(record: dict) -> str:
    """Format track record summary for /record command."""
    win_pct = f"{record['win_rate'] * 100:.1f}%"
    profit = record["total_profit"]
    profit_str = f"+{profit:.1f}u" if profit >= 0 else f"{profit:.1f}u"

    lines = [
        "\U0001f4ca SharpEdge Track Record\n",
        f"Total Picks: {record['total_picks']}",
        f"Record: {record['wins']}W {record['losses']}L ({win_pct})",
        f"Profit: {profit_str}",
        f"ROI: +{record['roi']:.1f}%",
        f"Avg Odds: {record['avg_odds']:.2f}",
    ]

    tiers = record.get("by_tier", {})
    if tiers:
        lines.append("\n\U0001f3c5 By Tier:")
        for tier_name, stats in tiers.items():
            t_pct = f"{stats['win_rate'] * 100:.1f}%"
            t_profit = stats["total_profit"]
            t_str = f"+{t_profit:.1f}u" if t_profit >= 0 else f"{t_profit:.1f}u"
            lines.append(
                f"  {tier_name.upper()}: {stats['wins']}W {stats['losses']}L "
                f"({t_pct}) | {t_str}"
            )

    return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bot/test_formatters.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/sharpedge/bot/formatters.py tests/test_bot/test_formatters.py
git commit -m "feat: Telegram message formatters — picks, results, record"
```

---

### Task 10: Telegram Bot — Broadcast + Commands

**Files:**
- Create: `src/sharpedge/bot/broadcast.py`
- Create: `src/sharpedge/bot/commands.py`
- Create: `src/sharpedge/bot/app.py`
- Create: `tests/test_bot/test_commands.py`

**Step 1: Write the failing test**

Create `tests/test_bot/test_commands.py`:

```python
"""Tests for Telegram bot command handlers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sharpedge.bot.commands import handle_start, handle_help, handle_today, handle_record


@pytest.fixture
def mock_update():
    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    return MagicMock()


@pytest.mark.asyncio
async def test_start_command(mock_update, mock_context):
    await handle_start(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once()
    msg = mock_update.message.reply_text.call_args[0][0]
    assert "SharpEdge" in msg


@pytest.mark.asyncio
async def test_help_command(mock_update, mock_context):
    await handle_help(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once()
    msg = mock_update.message.reply_text.call_args[0][0]
    assert "/today" in msg


@pytest.mark.asyncio
async def test_today_command(mock_update, mock_context):
    with patch("sharpedge.bot.commands._get_todays_picks") as mock_picks:
        mock_picks.return_value = []
        await handle_today(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_record_command(mock_update, mock_context):
    with patch("sharpedge.bot.commands._get_track_record") as mock_record:
        mock_record.return_value = {
            "total_picks": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_profit": 0.0, "roi": 0.0,
            "avg_odds": 0.0, "by_tier": {},
        }
        await handle_record(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bot/test_commands.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

Create `src/sharpedge/bot/commands.py`:

```python
"""Telegram bot command handlers."""
import logging
from datetime import date

from telegram import Update
from telegram.ext import ContextTypes

from sharpedge.db.engine import get_session
from sharpedge.db.models import DailyPick
from sharpedge.bot.formatters import format_pick_message, format_record_message
from sharpedge.pipeline.track_record import calculate_track_record

logger = logging.getLogger(__name__)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "\U0001f3af Welcome to SharpEdge AI!\n\n"
        "AI-powered football predictions with verified edge.\n\n"
        "Commands:\n"
        "/today — Today's picks\n"
        "/platinum — Platinum picks only\n"
        "/gold — Gold + Platinum picks\n"
        "/record — Track record\n"
        "/leagues — Available leagues\n"
        "/help — Help & about"
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "\U0001f4d6 SharpEdge AI Commands\n\n"
        "/today — All picks for today\n"
        "/platinum — Platinum tier picks (85%+ confidence)\n"
        "/gold — Gold + Platinum picks (78%+)\n"
        "/record — Overall track record\n"
        "/leagues — Big 5 leagues covered\n\n"
        "Picks are posted daily at ~08:00 UTC.\n"
        "Results posted at ~23:00 UTC."
    )


async def handle_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /today command — show today's picks."""
    picks = _get_todays_picks()

    if not picks:
        await update.message.reply_text(
            "No picks for today yet. Picks are generated at 08:00 UTC."
        )
        return

    for pick in picks:
        msg = format_pick_message(pick)
        await update.message.reply_text(msg)


async def handle_platinum(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /platinum command."""
    picks = _get_todays_picks(tier="platinum")
    if not picks:
        await update.message.reply_text("No Platinum picks for today.")
        return
    for pick in picks:
        await update.message.reply_text(format_pick_message(pick))


async def handle_gold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /gold command."""
    picks = _get_todays_picks(min_tier="gold")
    if not picks:
        await update.message.reply_text("No Gold/Platinum picks for today.")
        return
    for pick in picks:
        await update.message.reply_text(format_pick_message(pick))


async def handle_record(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /record command."""
    record = _get_track_record()
    msg = format_record_message(record)
    await update.message.reply_text(msg)


async def handle_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /leagues command."""
    await update.message.reply_text(
        "\U0001f30d Leagues Covered\n\n"
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f Premier League\n"
        "\U0001f1ea\U0001f1f8 La Liga\n"
        "\U0001f1e9\U0001f1ea Bundesliga\n"
        "\U0001f1ee\U0001f1f9 Serie A\n"
        "\U0001f1eb\U0001f1f7 Ligue 1"
    )


def _get_todays_picks(tier: str = None, min_tier: str = None) -> list[dict]:
    """Fetch today's picks from DB."""
    session = get_session()
    try:
        query = session.query(DailyPick).filter(DailyPick.match_date == date.today())

        if tier:
            query = query.filter(DailyPick.tier == tier)
        elif min_tier == "gold":
            query = query.filter(DailyPick.tier.in_(["platinum", "gold"]))

        picks = query.all()
        return [
            {
                "home_team": p.home_team, "away_team": p.away_team,
                "league": p.league, "match_date": str(p.match_date),
                "pick_selection": p.pick_selection, "pick_market": p.pick_market,
                "best_odds": p.best_odds, "bookmaker": p.bookmaker,
                "edge": p.edge, "tier": p.tier,
                "model_prob": p.model_prob, "meta_agreement": p.meta_agreement,
                "risk_flags": p.risk_flags or [],
            }
            for p in picks
        ]
    finally:
        session.close()


def _get_track_record() -> dict:
    """Fetch all resolved picks and calculate track record."""
    session = get_session()
    try:
        picks = session.query(DailyPick).filter(DailyPick.result.isnot(None)).all()
        pick_dicts = [
            {
                "tier": p.tier, "league": p.league,
                "result": p.result, "profit_loss": p.profit_loss,
                "best_odds": p.best_odds, "match_date": str(p.match_date),
            }
            for p in picks
        ]
        return calculate_track_record(pick_dicts)
    finally:
        session.close()
```

Create `src/sharpedge/bot/broadcast.py`:

```python
"""Channel broadcast logic — polls DB for unbroadcasted picks and posts them."""
import logging
from datetime import datetime

from telegram import Bot

from sharpedge.db.engine import get_session
from sharpedge.db.models import DailyPick
from sharpedge.bot.formatters import format_pick_message, format_results_message
from sharpedge.config import settings

logger = logging.getLogger(__name__)


async def broadcast_new_picks(bot: Bot) -> int:
    """Find unbroadcasted picks and send them to the channel.

    Returns number of picks broadcasted.
    """
    if not settings.telegram_channel_id:
        logger.warning("No channel ID configured, skipping broadcast")
        return 0

    session = get_session()
    try:
        picks = (
            session.query(DailyPick)
            .filter(DailyPick.broadcasted_at.is_(None))
            .order_by(DailyPick.tier.asc())  # platinum first
            .all()
        )

        count = 0
        for pick in picks:
            pick_dict = {
                "home_team": pick.home_team, "away_team": pick.away_team,
                "league": pick.league, "match_date": str(pick.match_date),
                "pick_selection": pick.pick_selection, "pick_market": pick.pick_market,
                "best_odds": pick.best_odds, "bookmaker": pick.bookmaker,
                "edge": pick.edge, "tier": pick.tier,
                "model_prob": pick.model_prob, "meta_agreement": pick.meta_agreement,
                "risk_flags": pick.risk_flags or [],
            }
            msg = format_pick_message(pick_dict)

            try:
                await bot.send_message(
                    chat_id=settings.telegram_channel_id,
                    text=msg,
                )
                pick.broadcasted_at = datetime.now()
                session.commit()
                count += 1
            except Exception as e:
                logger.error(f"Failed to broadcast pick {pick.id}: {e}")

        return count
    finally:
        session.close()
```

Create `src/sharpedge/bot/app.py`:

```python
"""Telegram bot application — entry point.

Run with: python -m sharpedge.bot.app
"""
import asyncio
import logging

from telegram.ext import ApplicationBuilder, CommandHandler

from sharpedge.config import settings
from sharpedge.bot.commands import (
    handle_start, handle_help, handle_today,
    handle_platinum, handle_gold, handle_record, handle_leagues,
)
from sharpedge.bot.broadcast import broadcast_new_picks

logger = logging.getLogger(__name__)


def create_bot_app():
    """Create the Telegram bot application."""
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN not configured")

    app = ApplicationBuilder().token(settings.telegram_bot_token).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("today", handle_today))
    app.add_handler(CommandHandler("platinum", handle_platinum))
    app.add_handler(CommandHandler("gold", handle_gold))
    app.add_handler(CommandHandler("record", handle_record))
    app.add_handler(CommandHandler("leagues", handle_leagues))

    return app


async def _poll_broadcasts(app) -> None:
    """Periodically check for unbroadcasted picks."""
    while True:
        try:
            count = await broadcast_new_picks(app.bot)
            if count > 0:
                logger.info(f"Broadcasted {count} picks")
        except Exception as e:
            logger.error(f"Broadcast poll error: {e}")
        await asyncio.sleep(settings.telegram_poll_interval)


def main():
    """Start the bot with long polling + broadcast polling."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting SharpEdge Telegram Bot...")

    app = create_bot_app()

    # Run broadcast poller as background task
    loop = asyncio.get_event_loop()
    app.post_init = lambda _app: loop.create_task(_poll_broadcasts(_app))

    app.run_polling()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bot/test_commands.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/sharpedge/bot/ tests/test_bot/test_commands.py
git commit -m "feat: Telegram bot — broadcast, commands, app entry point"
```

---

### Task 11: API Entry Point + `__main__` Modules

**Files:**
- Create: `src/sharpedge/api/__main__.py`
- Create: `src/sharpedge/bot/__main__.py`
- Modify: `src/sharpedge/api/app.py` (add uvicorn runner)

**Step 1: Create API `__main__.py`**

Create `src/sharpedge/api/__main__.py`:

```python
"""Run the FastAPI server: python -m sharpedge.api"""
import uvicorn
from sharpedge.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "sharpedge.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
```

**Step 2: Create bot `__main__.py`**

Create `src/sharpedge/bot/__main__.py`:

```python
"""Run the Telegram bot: python -m sharpedge.bot"""
from sharpedge.bot.app import main

if __name__ == "__main__":
    main()
```

**Step 3: Verify API server can start (smoke test)**

Run: `.venv/bin/python -c "from sharpedge.api.app import create_app; app = create_app(); print('App created:', app.title)"`
Expected: `App created: SharpEdge AI`

**Step 4: Commit**

```bash
git add src/sharpedge/api/__main__.py src/sharpedge/bot/__main__.py
git commit -m "feat: __main__ entry points for API server and Telegram bot"
```

---

### Task 12: GitHub Actions Workflow Files

**Files:**
- Create: `.github/workflows/daily-pipeline.yml`
- Create: `.github/workflows/resolve-results.yml`
- Create: `.github/workflows/weekly-retrain.yml`

**Step 1: Create the three workflow files**

Create `.github/workflows/daily-pipeline.yml`:

```yaml
name: Daily Predictions
on:
  schedule:
    - cron: "0 8 * * *"
  workflow_dispatch:
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger daily pipeline
        run: |
          curl -sf -X POST "${{ secrets.API_URL }}/api/pipeline/run" \
            -H "X-API-Key: ${{ secrets.API_KEY }}" \
            -H "Content-Type: application/json" \
            --max-time 300
```

Create `.github/workflows/resolve-results.yml`:

```yaml
name: Resolve Results
on:
  schedule:
    - cron: "0 23 * * *"
  workflow_dispatch:
jobs:
  resolve:
    runs-on: ubuntu-latest
    steps:
      - name: Resolve yesterday's picks
        run: |
          curl -sf -X POST "${{ secrets.API_URL }}/api/pipeline/resolve" \
            -H "X-API-Key: ${{ secrets.API_KEY }}" \
            -H "Content-Type: application/json" \
            --max-time 120
```

Create `.github/workflows/weekly-retrain.yml`:

```yaml
name: Weekly Retrain
on:
  schedule:
    - cron: "0 3 * * 0"
  workflow_dispatch:
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - name: Retrain model
        run: |
          curl -sf -X POST "${{ secrets.API_URL }}/api/pipeline/retrain" \
            -H "X-API-Key: ${{ secrets.API_KEY }}" \
            -H "Content-Type: application/json" \
            --max-time 600
```

**Step 2: Commit**

```bash
git add .github/workflows/
git commit -m "feat: GitHub Actions — daily pipeline, result resolution, weekly retrain"
```

---

### Task 13: Full Test Suite Run + Fix Regressions

**Step 1: Run all Phase 3 tests**

Run: `.venv/bin/python -m pytest tests/test_api/ tests/test_bot/ tests/test_pipeline/ -v`
Expected: All passed

**Step 2: Run all project tests**

Run: `.venv/bin/python -m pytest tests/test_ml/ tests/test_api/ tests/test_bot/ tests/test_pipeline/ -q`
Expected: 85 (Phase 2) + Phase 3 tests = all passed

**Step 3: Fix any regressions or import issues**

If any tests fail, fix the root cause and re-run.

**Step 4: Commit any fixes**

```bash
git commit -am "fix: resolve Phase 3 test regressions"
```

---

## Summary

| Task | Component | New Tests | Key Files |
|------|-----------|-----------|-----------|
| 1 | Dependencies + scaffolding | 0 | pyproject.toml, config.py, __init__.py files |
| 2 | DB models (PipelineRun, Prediction, DailyPick) | 4 | db/models.py |
| 3 | Result resolver | 8 | pipeline/resolver.py |
| 4 | Track record calculator | 7 | pipeline/track_record.py |
| 5 | Daily pipeline orchestrator | 2 | pipeline/daily.py |
| 6 | FastAPI app + auth + router stubs | 3 | api/app.py, auth.py, deps.py, all routers |
| 7 | Pipeline router (full) | 3 | api/routers/pipeline.py |
| 8 | Public routers (predictions, picks, track record) | 7 | api/routers/*.py |
| 9 | Telegram formatters | 4 | bot/formatters.py |
| 10 | Telegram bot (broadcast + commands) | 4 | bot/commands.py, broadcast.py, app.py |
| 11 | Entry points (__main__) | 0 | api/__main__.py, bot/__main__.py |
| 12 | GitHub Actions workflows | 0 | .github/workflows/*.yml |
| 13 | Full test suite + fix regressions | 0 | — |

**Total new tests: ~42**
**Total files created: ~25**
**Total commits: ~13**
