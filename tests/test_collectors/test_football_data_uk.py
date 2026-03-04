"""Tests for Football-Data.co.uk collector."""

from sharpedge.collectors.football_data_uk import (
    FootballDataUKCollector,
    LEAGUE_CODES,
    SEASON_CODES,
)


def test_league_codes_cover_big_5():
    """All 5 major European leagues should be present."""
    assert len(LEAGUE_CODES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(LEAGUE_CODES.keys()) == expected


def test_season_labels_cover_5_seasons():
    """Five seasons should be available."""
    assert len(SEASON_CODES) == 5
    assert "2024-25" in SEASON_CODES
    assert "2020-21" in SEASON_CODES


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = FootballDataUKCollector()
    assert collector.source_name == "football_data_uk"
    assert collector.base_url == "https://www.football-data.co.uk"
    assert collector.request_delay == 1.0
