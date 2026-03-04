from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sharpedge.db.models import (
    Base,
    Team,
    League,
    Season,
    Match,
    MatchOdds,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_create_team(db_session):
    team = Team(
        canonical_name="Arsenal",
        country="England",
        aliases={"fbref": "Arsenal", "understat": "Arsenal", "fd_uk": "Arsenal"},
    )
    db_session.add(team)
    db_session.commit()
    db_session.refresh(team)

    assert team.id is not None
    assert team.canonical_name == "Arsenal"
    assert team.aliases["fbref"] == "Arsenal"
    assert team.aliases["understat"] == "Arsenal"


def test_create_match_with_teams(db_session):
    home = Team(canonical_name="Liverpool", country="England")
    away = Team(canonical_name="Man City", country="England")
    league = League(name="Premier League", country="England")
    db_session.add_all([home, away, league])
    db_session.commit()

    season = Season(league_id=league.id, label="2024-25")
    db_session.add(season)
    db_session.commit()

    match = Match(
        season_id=season.id,
        match_date=date(2025, 1, 15),
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=2,
        away_goals=1,
        result="H",
        status="finished",
    )
    db_session.add(match)
    db_session.commit()
    db_session.refresh(match)

    assert match.id is not None
    assert match.result == "H"
    assert match.home_team.canonical_name == "Liverpool"
    assert match.away_team.canonical_name == "Man City"
    assert match.season.label == "2024-25"


def test_match_odds(db_session):
    home = Team(canonical_name="Chelsea", country="England")
    away = Team(canonical_name="Tottenham", country="England")
    league = League(name="Premier League", country="England")
    db_session.add_all([home, away, league])
    db_session.commit()

    season = Season(league_id=league.id, label="2024-25")
    db_session.add(season)
    db_session.commit()

    match = Match(
        season_id=season.id,
        match_date=date(2025, 2, 10),
        home_team_id=home.id,
        away_team_id=away.id,
        status="scheduled",
    )
    db_session.add(match)
    db_session.commit()

    odds = MatchOdds(
        match_id=match.id,
        bookmaker="Bet365",
        market="1x2",
        odds_type="opening",
        odds_home=1.85,
        odds_draw=3.50,
        odds_away=4.20,
        captured_at=datetime(2025, 2, 9, 12, 0, 0),
    )
    db_session.add(odds)
    db_session.commit()
    db_session.refresh(odds)

    assert odds.id is not None
    assert odds.bookmaker == "Bet365"
    assert odds.odds_home == 1.85
    assert odds.match.home_team.canonical_name == "Chelsea"
