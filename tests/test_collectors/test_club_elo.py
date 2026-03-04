"""Tests for ClubELO collector."""

from sharpedge.collectors.club_elo import ClubELOCollector


def test_collector_attributes():
    """Collector should have correct source_name and base_url."""
    collector = ClubELOCollector()
    assert collector.source_name == "club_elo"
    assert collector.base_url == "http://api.clubelo.com"
    assert collector.request_delay == 1.0
