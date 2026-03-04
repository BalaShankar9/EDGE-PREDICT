"""Tests for FootyStats collector."""

from sharpedge.collectors.footystats import FootyStatsCollector, FOOTYSTATS_LEAGUES


def test_footystats_leagues():
    """All 5 major European leagues should be present."""
    assert len(FOOTYSTATS_LEAGUES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(FOOTYSTATS_LEAGUES.keys()) == expected


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = FootyStatsCollector()
    assert collector.source_name == "footystats"
    assert collector.base_url == "https://footystats.org"
    assert collector.request_delay == 3.0


def test_league_paths():
    """League paths should follow country/league format."""
    for league_name, path in FOOTYSTATS_LEAGUES.items():
        assert "/" in path, f"Path for {league_name} should contain country prefix"
        parts = path.split("/")
        assert len(parts) == 2, f"Path for {league_name} should have country/league format"
