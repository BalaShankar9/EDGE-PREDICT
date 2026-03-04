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


def test_pipeline_run_creation(db_session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="running", started_at=datetime(2026, 3, 4, 8, 0, 0),
    )
    db_session.add(run)
    db_session.commit()
    assert run.id is not None
    assert run.run_type == "predict"
    assert run.status == "running"


def test_prediction_creation(db_session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime.now(),
    )
    db_session.add(run)
    db_session.flush()
    pred = Prediction(
        pipeline_run_id=run.id, match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        prob_over=0.60, prob_under=0.40,
        prob_btts_yes=0.55, prob_btts_no=0.45,
        created_at=datetime.now(),
    )
    db_session.add(pred)
    db_session.commit()
    assert pred.id is not None
    assert pred.pipeline_run.id == run.id


def test_daily_pick_creation(db_session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime.now(),
    )
    db_session.add(run)
    db_session.flush()
    pred = Prediction(
        pipeline_run_id=run.id, match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        created_at=datetime.now(),
    )
    db_session.add(pred)
    db_session.flush()
    pick = DailyPick(
        pipeline_run_id=run.id, prediction_id=pred.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29, tier="platinum",
        meta_agreement=3, risk_flags=[], stake_flat=1.0, stake_kelly=0.8,
    )
    db_session.add(pick)
    db_session.commit()
    assert pick.id is not None
    assert pick.result is None
    assert pick.broadcasted_at is None


def test_daily_pick_resolution(db_session):
    run = PipelineRun(
        run_date=date(2026, 3, 4), run_type="predict",
        status="success", started_at=datetime.now(),
    )
    db_session.add(run)
    db_session.flush()
    pred = Prediction(
        pipeline_run_id=run.id, match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        created_at=datetime.now(),
    )
    db_session.add(pred)
    db_session.flush()
    pick = DailyPick(
        pipeline_run_id=run.id, prediction_id=pred.id,
        match_date=date(2026, 3, 4),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29, tier="platinum",
        meta_agreement=3, risk_flags=[], stake_flat=1.0, stake_kelly=0.8,
    )
    db_session.add(pick)
    db_session.commit()
    pick.result = "win"
    pick.profit_loss = 0.85
    pick.resolved_at = datetime.now()
    db_session.commit()
    assert pick.result == "win"
    assert pick.profit_loss == 0.85
