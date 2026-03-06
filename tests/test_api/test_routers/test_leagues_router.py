"""Tests for leagues API endpoints."""
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
        run_date=date.today(), run_type="predict",
        status="success", started_at=datetime(2026, 3, 6, 8, 0),
        completed_at=datetime(2026, 3, 6, 8, 5),
        predictions_count=2, picks_count=1,
    )
    db_session.add(run)
    db_session.flush()

    pred1 = Prediction(
        pipeline_run_id=run.id, match_date=date.today(),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        prob_over=0.60, prob_under=0.40,
        prob_btts_yes=0.55, prob_btts_no=0.45,
        created_at=datetime(2026, 3, 6, 8, 3),
    )
    pred2 = Prediction(
        pipeline_run_id=run.id, match_date=date.today(),
        home_team="Barcelona", away_team="Real Madrid", league="La Liga",
        prob_home=0.45, prob_draw=0.30, prob_away=0.25,
        prob_over=0.55, prob_under=0.45,
        prob_btts_yes=0.50, prob_btts_no=0.50,
        created_at=datetime(2026, 3, 6, 8, 3),
    )
    db_session.add_all([pred1, pred2])
    db_session.flush()

    pick = DailyPick(
        pipeline_run_id=run.id, prediction_id=pred1.id,
        match_date=date.today(),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29, tier="platinum",
        meta_agreement=3, risk_flags=[], stake_flat=1.0, stake_kelly=0.8,
    )
    db_session.add(pick)
    db_session.commit()
    return db_session


@pytest.fixture
def client(seeded_db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: seeded_db
    return TestClient(app)


def test_list_leagues(client):
    response = client.get("/api/leagues")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["meta"]["count"] == 2
    names = {lg["name"] for lg in body["data"]}
    assert "Premier League" in names
    assert "La Liga" in names


def test_list_leagues_pick_counts(client):
    response = client.get("/api/leagues")
    body = response.json()
    leagues_by_name = {lg["name"]: lg for lg in body["data"]}
    assert leagues_by_name["Premier League"]["pick_count"] == 1
    assert leagues_by_name["La Liga"]["pick_count"] == 0


def test_list_leagues_sorted_by_pick_count(client):
    response = client.get("/api/leagues")
    body = response.json()
    # Premier League has picks, should be first
    assert body["data"][0]["name"] == "Premier League"


def test_league_predictions(client):
    response = client.get("/api/league/premier-league/predictions")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["data"]["league"] == "Premier League"
    assert len(body["data"]["predictions"]) == 1
    pred = body["data"]["predictions"][0]
    assert pred["home_team"] == "Arsenal"
    assert "pick" in pred
    assert pred["pick"]["selection"] == "Home Win"


def test_league_predictions_no_pick(client):
    response = client.get("/api/league/la-liga/predictions")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    pred = body["data"]["predictions"][0]
    assert "pick" not in pred


def test_league_predictions_not_found(client):
    response = client.get("/api/league/nonexistent/predictions")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
