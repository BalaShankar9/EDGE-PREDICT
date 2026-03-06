"""Tests for match detail API endpoint."""
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
        run_date=date(2026, 3, 6), run_type="predict",
        status="success", started_at=datetime(2026, 3, 6, 8, 0),
        completed_at=datetime(2026, 3, 6, 8, 5),
        predictions_count=1, picks_count=1,
    )
    db_session.add(run)
    db_session.flush()

    pred = Prediction(
        pipeline_run_id=run.id, match_date=date(2026, 3, 6),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        prob_home=0.55, prob_draw=0.25, prob_away=0.20,
        prob_over=0.60, prob_under=0.40,
        prob_btts_yes=0.55, prob_btts_no=0.45,
        xgboost_probs={"home": 0.52, "draw": 0.28, "away": 0.20},
        poisson_probs={"home": 0.50, "draw": 0.27, "away": 0.23},
        ensemble_weights={"xgboost": 0.6, "poisson": 0.4},
        created_at=datetime(2026, 3, 6, 8, 3),
    )
    db_session.add(pred)
    db_session.flush()

    pick = DailyPick(
        pipeline_run_id=run.id, prediction_id=pred.id,
        match_date=date(2026, 3, 6),
        home_team="Arsenal", away_team="Chelsea", league="Premier League",
        pick_market="1x2_home", pick_selection="Home Win",
        model_prob=0.83, model_spread=0.05,
        best_odds=1.85, bookmaker="Bet365",
        implied_prob=0.54, edge=0.29, tier="platinum",
        meta_agreement=3, risk_flags=["key_player_doubt"],
        stake_flat=1.0, stake_kelly=0.8,
        result="win", profit_loss=0.85,
        resolved_at=datetime(2026, 3, 6, 23, 0),
    )
    db_session.add(pick)
    db_session.commit()
    return db_session


@pytest.fixture
def client(seeded_db):
    app = create_app()
    app.dependency_overrides[get_db] = lambda: seeded_db
    return TestClient(app)


def test_match_detail(client):
    response = client.get("/api/match/Arsenal/Chelsea/2026-03-06")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    data = body["data"]
    assert data["home_team"] == "Arsenal"
    assert data["away_team"] == "Chelsea"
    assert data["league"] == "Premier League"
    assert data["probabilities"]["home"] == 0.55
    assert data["xgboost_probs"]["home"] == 0.52


def test_match_detail_with_pick(client):
    response = client.get("/api/match/Arsenal/Chelsea/2026-03-06")
    body = response.json()
    pick = body["data"]["pick"]
    assert pick["market"] == "1x2_home"
    assert pick["selection"] == "Home Win"
    assert pick["tier"] == "platinum"
    assert pick["edge"] == 0.29
    assert pick["result"] == "win"
    assert pick["risk_flags"] == ["key_player_doubt"]


def test_match_detail_not_found(client):
    response = client.get("/api/match/Nonexistent/Team/2026-01-01")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert "Match not found" in body["data"]["error"]


def test_track_record_monthly(client):
    """Test the monthly track record endpoint with seeded resolved pick."""
    response = client.get("/api/track-record/monthly")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["data"], list)
    assert len(body["data"]) >= 1
    month = body["data"][0]
    assert "month" in month
    assert "picks" in month
    assert "wins" in month
    assert "profit" in month
    assert "roi" in month
    assert "cumulative_profit" in month
    assert month["picks"] >= 1
    assert month["wins"] >= 1
