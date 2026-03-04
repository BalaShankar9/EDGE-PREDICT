"""Tests for WinDrawWin collector."""

from sharpedge.collectors.windrawwin import WinDrawWinCollector, WINDRAWWIN_LEAGUES


def test_windrawwin_leagues():
    """All 5 major European leagues should be present."""
    assert len(WINDRAWWIN_LEAGUES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(WINDRAWWIN_LEAGUES.keys()) == expected


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = WinDrawWinCollector()
    assert collector.source_name == "windrawwin"
    assert collector.base_url == "https://www.windrawwin.com"
    assert collector.request_delay == 3.0


def test_league_paths():
    """League paths should be valid URL segments."""
    for league_name, path in WINDRAWWIN_LEAGUES.items():
        assert "/" not in path, f"Path for {league_name} should not contain '/'"
        assert path.islower() or "-" in path, f"Path for {league_name} should be lowercase"
