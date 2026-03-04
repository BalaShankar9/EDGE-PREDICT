"""Tests for Understat collector."""

from sharpedge.collectors.understat import (
    UnderstatCollector,
    LEAGUE_CODES,
)


def test_league_codes():
    """All 5 major European leagues should be present, with correct EPL mapping."""
    assert len(LEAGUE_CODES) == 5
    assert LEAGUE_CODES["Premier League"] == "EPL"
    expected = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    assert set(LEAGUE_CODES.keys()) == expected


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = UnderstatCollector()
    assert collector.source_name == "understat"
    assert collector.base_url == "https://understat.com"
    assert collector.request_delay == 3.0
