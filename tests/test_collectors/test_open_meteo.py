"""Tests for Open-Meteo weather collector."""

from sharpedge.collectors.open_meteo import OpenMeteoCollector


def test_collector_attributes():
    """Collector should have correct source_name, base_url, and request_delay."""
    collector = OpenMeteoCollector()
    assert collector.source_name == "open_meteo"
    assert collector.base_url == "https://api.open-meteo.com"
    assert collector.request_delay == 0.5


def test_unknown_venue_returns_empty():
    """Collecting for an unknown venue should return an empty DataFrame."""
    collector = OpenMeteoCollector()
    df = collector.collect(venue="Nonexistent Stadium XYZ")
    assert df.empty
