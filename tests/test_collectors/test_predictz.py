"""Tests for PredictZ collector."""

from sharpedge.collectors.predictz import PredictZCollector, PREDICTZ_LEAGUES


def test_predictz_leagues():
    """All 5 major European leagues should be present."""
    assert len(PREDICTZ_LEAGUES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(PREDICTZ_LEAGUES.keys()) == expected


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = PredictZCollector()
    assert collector.source_name == "predictz"
    assert collector.base_url == "https://www.predictz.com"
    assert collector.request_delay == 3.0


def test_league_paths():
    """League paths should be valid URL segments."""
    for league_name, path in PREDICTZ_LEAGUES.items():
        assert "/" not in path, f"Path for {league_name} should not contain '/'"
        assert path.islower() or "-" in path, f"Path for {league_name} should be lowercase"
