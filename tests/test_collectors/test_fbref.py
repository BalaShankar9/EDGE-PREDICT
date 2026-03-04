"""Tests for FBref collector."""

from sharpedge.collectors.fbref import FBrefCollector, FBREF_LEAGUES


def test_fbref_leagues():
    """All 5 major European leagues should be present."""
    assert len(FBREF_LEAGUES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(FBREF_LEAGUES.keys()) == expected


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = FBrefCollector()
    assert collector.source_name == "fbref"
    assert collector.base_url == "https://fbref.com"
    assert collector.request_delay == 4.0
