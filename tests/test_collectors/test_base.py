"""Tests for BaseCollector."""

import pandas as pd
import pytest

from sharpedge.collectors.base import BaseCollector


# ------------------------------------------------------------------ #
#  Test fixtures / mock collectors                                    #
# ------------------------------------------------------------------ #


class MockCollector(BaseCollector):
    """A minimal collector that returns a fixed DataFrame."""

    source_name = "fbref"
    base_url = "https://fbref.com"
    request_delay = 0.0  # fast tests

    def _collect(self, **kwargs):
        return pd.DataFrame({"team": ["Arsenal", "Chelsea"], "goals": [3, 1]})


class FailingCollector(BaseCollector):
    """A collector whose _collect always raises."""

    source_name = "failing_source"
    base_url = "https://example.com"
    request_delay = 0.0

    def _collect(self, **kwargs):
        raise ValueError("Simulated scraping failure")


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #


def test_collect_returns_dataframe():
    """MockCollector.collect() should return a 2-row DataFrame."""
    collector = MockCollector()
    df = collector.collect()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["team", "goals"]


def test_collect_handles_failure_gracefully():
    """FailingCollector.collect() should return an empty DataFrame, not raise."""
    collector = FailingCollector()
    df = collector.collect()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_cache_roundtrip(tmp_path, monkeypatch):
    """Set cache then get cache — data should match."""
    monkeypatch.setattr("sharpedge.config.settings.cache_dir", str(tmp_path / "cache"))
    collector = MockCollector()
    test_data = {"matches": [{"home": "Arsenal", "away": "Chelsea"}]}
    collector._set_cache("test_key", test_data)
    result = collector._get_cached("test_key")
    assert result == test_data


def test_cache_miss(tmp_path, monkeypatch):
    """Getting a nonexistent cache key should return None."""
    monkeypatch.setattr("sharpedge.config.settings.cache_dir", str(tmp_path / "cache"))
    collector = MockCollector()
    result = collector._get_cached("nonexistent_key")
    assert result is None


def test_structure_check_passes():
    """HTML containing all expected markers should pass."""
    collector = MockCollector()
    html = '<div class="table_wrapper"><table id="results">data</table></div>'
    assert collector._check_structure(html, ["table_wrapper", "results"]) is True


def test_structure_check_fails():
    """HTML missing expected markers should fail."""
    collector = MockCollector()
    html = "<div>nothing useful here</div>"
    assert collector._check_structure(html, ["table_wrapper", "results"]) is False


def test_normalise_team():
    """MockCollector (source=fbref) should normalise 'Arsenal' to 'arsenal'."""
    collector = MockCollector()
    result = collector.normalise_team("Arsenal")
    assert result == "arsenal"
