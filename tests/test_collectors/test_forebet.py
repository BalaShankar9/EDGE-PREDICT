"""Tests for Forebet collector."""

from sharpedge.collectors.forebet import ForebetCollector, FOREBET_LEAGUES


def test_forebet_leagues():
    """All 5 major European leagues should be present."""
    assert len(FOREBET_LEAGUES) == 5
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(FOREBET_LEAGUES.keys()) == expected


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = ForebetCollector()
    assert collector.source_name == "forebet"
    assert collector.base_url == "https://www.forebet.com"
    assert collector.request_delay == 3.0
