"""Tests for football-data.org collector."""

from sharpedge.collectors.football_data_org import (
    FootballDataOrgCollector,
    FOOTBALL_DATA_ORG_LEAGUES,
)


def test_football_data_org_leagues():
    """All 5 major European leagues should be present with correct IDs."""
    assert len(FOOTBALL_DATA_ORG_LEAGUES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(FOOTBALL_DATA_ORG_LEAGUES.keys()) == expected

    # Verify known competition IDs
    assert FOOTBALL_DATA_ORG_LEAGUES["Premier League"] == 2021
    assert FOOTBALL_DATA_ORG_LEAGUES["La Liga"] == 2014
    assert FOOTBALL_DATA_ORG_LEAGUES["Bundesliga"] == 2002
    assert FOOTBALL_DATA_ORG_LEAGUES["Serie A"] == 2019
    assert FOOTBALL_DATA_ORG_LEAGUES["Ligue 1"] == 2015


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = FootballDataOrgCollector()
    assert collector.source_name == "football_data_org"
    assert collector.base_url == "https://api.football-data.org/v4"
    assert collector.request_delay == 6.0


def test_parse_fixture():
    """Parse a fixture from the API response format."""
    match = {
        "homeTeam": {"name": "Arsenal FC"},
        "awayTeam": {"name": "Chelsea FC"},
        "utcDate": "2025-03-15T15:00:00Z",
        "matchday": 28,
    }
    result = FootballDataOrgCollector._parse_fixture(match, "Premier League")
    assert result is not None
    assert result["home_team"] == "Arsenal FC"
    assert result["away_team"] == "Chelsea FC"
    assert result["league"] == "Premier League"
    assert result["matchday"] == 28
    assert result["source"] == "football_data_org"
    assert result["data_type"] == "fixture"


def test_parse_fixture_missing_teams():
    """Parsing should return None when team names are missing."""
    match = {"homeTeam": {}, "awayTeam": {"name": "Chelsea FC"}}
    result = FootballDataOrgCollector._parse_fixture(match, "Premier League")
    assert result is None


def test_parse_standing():
    """Parse a standing entry from the API response format."""
    entry = {
        "team": {"name": "Arsenal FC"},
        "position": 1,
        "playedGames": 28,
        "won": 20,
        "draw": 5,
        "lost": 3,
        "goalsFor": 60,
        "goalsAgainst": 25,
        "goalDifference": 35,
        "points": 65,
    }
    result = FootballDataOrgCollector._parse_standing(entry, "Premier League")
    assert result is not None
    assert result["team"] == "Arsenal FC"
    assert result["position"] == 1
    assert result["points"] == 65
    assert result["league"] == "Premier League"
    assert result["source"] == "football_data_org"
    assert result["data_type"] == "standing"
