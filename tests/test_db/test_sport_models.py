"""Tests for Sport, Agent, AgentPrediction, and AgentPerformance tables."""

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sharpedge.db.models import (
    Base,
    Sport,
    Agent,
    AgentPrediction,
    AgentPerformance,
    League,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Sport table
# ---------------------------------------------------------------------------


def test_create_sport(db_session):
    sport = Sport(
        name="Football",
        slug="football",
        config={"markets": ["1x2", "ou25", "btts"]},
        active=True,
    )
    db_session.add(sport)
    db_session.commit()
    db_session.refresh(sport)

    assert sport.id is not None
    assert sport.name == "Football"
    assert sport.slug == "football"
    assert sport.config["markets"] == ["1x2", "ou25", "btts"]
    assert sport.active is True


def test_sport_slug_unique(db_session):
    db_session.add(Sport(name="Football", slug="football"))
    db_session.commit()

    db_session.add(Sport(name="Soccer", slug="football"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_sport_name_unique(db_session):
    db_session.add(Sport(name="Football", slug="football"))
    db_session.commit()

    db_session.add(Sport(name="Football", slug="soccer"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_league_sport_fk(db_session):
    sport = Sport(name="Football", slug="football")
    db_session.add(sport)
    db_session.commit()

    league = League(name="Premier League", country="England", sport_id=sport.id)
    db_session.add(league)
    db_session.commit()
    db_session.refresh(league)

    assert league.sport_id == sport.id
    assert league.sport.slug == "football"
    assert len(sport.leagues) == 1


def test_league_without_sport(db_session):
    """Existing leagues without sport_id should still work (nullable FK)."""
    league = League(name="La Liga", country="Spain")
    db_session.add(league)
    db_session.commit()
    db_session.refresh(league)

    assert league.sport_id is None
    assert league.sport is None


# ---------------------------------------------------------------------------
# Agent table
# ---------------------------------------------------------------------------


def test_create_agent(db_session):
    agent = Agent(
        name="gradient_agent",
        agent_type="ml",
        description="XGBoost-based agent",
        config={"model": "xgboost", "version": "1.0"},
    )
    db_session.add(agent)
    db_session.commit()
    db_session.refresh(agent)

    assert agent.id is not None
    assert agent.name == "gradient_agent"
    assert agent.agent_type == "ml"
    assert agent.active is True
    assert agent.config["model"] == "xgboost"


def test_agent_name_unique(db_session):
    db_session.add(Agent(name="gradient_agent", agent_type="ml"))
    db_session.commit()

    db_session.add(Agent(name="gradient_agent", agent_type="statistical"))
    with pytest.raises(IntegrityError):
        db_session.commit()


# ---------------------------------------------------------------------------
# AgentPrediction table
# ---------------------------------------------------------------------------


def test_create_agent_prediction(db_session):
    agent = Agent(name="poisson_agent", agent_type="statistical")
    db_session.add(agent)
    db_session.commit()

    pred = AgentPrediction(
        agent_id=agent.id,
        sport_slug="football",
        match_date=date(2026, 3, 25),
        home_team="Arsenal",
        away_team="Chelsea",
        market="1x2",
        predicted_outcome="H",
        probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
        confidence=0.72,
        reasoning="Strong home form, xG advantage",
    )
    db_session.add(pred)
    db_session.commit()
    db_session.refresh(pred)

    assert pred.id is not None
    assert pred.probabilities["home"] == 0.55
    assert pred.confidence == 0.72
    assert pred.actual_outcome is None
    assert pred.correct is None
    assert pred.agent.name == "poisson_agent"


def test_agent_prediction_post_match_update(db_session):
    agent = Agent(name="market_agent", agent_type="market")
    db_session.add(agent)
    db_session.commit()

    pred = AgentPrediction(
        agent_id=agent.id,
        sport_slug="football",
        match_date=date(2026, 3, 20),
        home_team="Liverpool",
        away_team="Man City",
        market="1x2",
        predicted_outcome="H",
        probabilities={"home": 0.50, "draw": 0.28, "away": 0.22},
        confidence=0.65,
    )
    db_session.add(pred)
    db_session.commit()

    # Simulate post-match result update
    pred.actual_outcome = "H"
    pred.correct = True
    db_session.commit()
    db_session.refresh(pred)

    assert pred.actual_outcome == "H"
    assert pred.correct is True


# ---------------------------------------------------------------------------
# AgentPerformance table
# ---------------------------------------------------------------------------


def test_create_agent_performance(db_session):
    agent = Agent(name="ensemble_agent", agent_type="ml")
    db_session.add(agent)
    db_session.commit()

    perf = AgentPerformance(
        agent_id=agent.id,
        sport_slug="football",
        date=date(2026, 3, 25),
        window_days=30,
        total_bets=120,
        wins=68,
        roi_pct=4.5,
        clv_pct=2.1,
        avg_confidence=0.62,
        brier_score=0.21,
    )
    db_session.add(perf)
    db_session.commit()
    db_session.refresh(perf)

    assert perf.id is not None
    assert perf.total_bets == 120
    assert perf.wins == 68
    assert perf.roi_pct == 4.5
    assert perf.brier_score == 0.21
    assert perf.agent.name == "ensemble_agent"


def test_multiple_performance_windows(db_session):
    agent = Agent(name="window_agent", agent_type="ml")
    db_session.add(agent)
    db_session.commit()

    for window in [7, 30, 90]:
        perf = AgentPerformance(
            agent_id=agent.id,
            sport_slug="football",
            date=date(2026, 3, 25),
            window_days=window,
            total_bets=window * 4,
            wins=window * 2,
        )
        db_session.add(perf)
    db_session.commit()

    snapshots = (
        db_session.query(AgentPerformance)
        .filter_by(agent_id=agent.id)
        .order_by(AgentPerformance.window_days)
        .all()
    )
    assert len(snapshots) == 3
    assert [s.window_days for s in snapshots] == [7, 30, 90]
