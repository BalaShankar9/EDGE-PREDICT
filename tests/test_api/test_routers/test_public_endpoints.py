"""Tests for public API endpoints."""
import pytest
from datetime import date, datetime
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sharpedge.api.app import create_app
from sharpedge.api.deps import get_db
from sharpedge.db.models import Base, PipelineRun, Prediction, DailyPick


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def seeded_db(db_session):
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
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
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
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29, tier="platinum",
        meta_agreement=3, risk_flags=[], stake_flat=1.0, stake_kelly=0.8,
        result="win", profit_loss=0.85,
        resolved_at=datetime(2026, 3, 4, 23, 0),
    )
    db_session.add(pick)
    db_session.commit()
    return db_session


@pytest.fixture
def client(seeded_db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: seeded_db
    return TestClient(app)


def test_predictions_today(client):
    response = client.get("/api/predictions/today")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predictions_by_date(client):
    response = client.get("/api/predictions/2026-03-04")
    assert response.status_code == 200


def test_picks_today(client):
    response = client.get("/api/picks/today")
    assert response.status_code == 200


def test_picks_history(client):
    response = client.get("/api/picks/history")
    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)


def test_track_record_overall(client):
    response = client.get("/api/track-record")
    assert response.status_code == 200
    data = response.json()["data"]
    assert "total_picks" in data


def test_track_record_by_tier(client):
    response = client.get("/api/track-record/by-tier")
    assert response.status_code == 200


def test_track_record_by_league(client):
    response = client.get("/api/track-record/by-league")
    assert response.status_code == 200
