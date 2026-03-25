"""Tests for tennis data collectors."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from sharpedge.sports.tennis.collectors.tennis_data_uk import (
    TennisDataUKCollector,
    _COLUMN_MAP,
)


class TestTennisDataUKCollector:
    """Tests for TennisDataUKCollector."""

    def test_source_name(self):
        collector = TennisDataUKCollector()
        assert collector.source_name == "tennis_data_uk"

    def test_base_url(self):
        collector = TennisDataUKCollector()
        assert "tennis-data.co.uk" in collector.base_url

    def test_required_columns_count(self):
        collector = TennisDataUKCollector()
        assert len(collector.REQUIRED_COLUMNS) >= 10

    def test_required_columns_include_essentials(self):
        collector = TennisDataUKCollector()
        for col in ("date", "winner", "loser", "surface", "b365_winner"):
            assert col in collector.REQUIRED_COLUMNS

    def test_build_urls_atp(self):
        collector = TennisDataUKCollector()
        urls = collector._build_urls("atp", 2024)
        assert len(urls) == 3
        assert any("2024/2024.xlsx" in u for u in urls)

    def test_build_urls_wta(self):
        collector = TennisDataUKCollector()
        urls = collector._build_urls("wta", 2024)
        assert any("2024w/" in u for u in urls)

    def test_standardise_adds_missing_columns(self):
        """Standardise should add missing required columns as None."""
        collector = TennisDataUKCollector()
        raw = pd.DataFrame({
            "Date": ["2024-01-15"],
            "Winner": ["Djokovic N."],
            "Loser": ["Nadal R."],
        })
        result = collector._standardise(raw)
        for col in collector.REQUIRED_COLUMNS:
            assert col in result.columns

    def test_standardise_renames_columns(self):
        collector = TennisDataUKCollector()
        raw = pd.DataFrame({
            "Date": ["2024-06-01"],
            "Tournament": ["Wimbledon"],
            "Surface": ["Grass"],
            "Round": ["Final"],
            "Winner": ["Alcaraz C."],
            "Loser": ["Djokovic N."],
            "WRank": [2],
            "LRank": [1],
            "B365W": [1.50],
            "B365L": [2.75],
            "PSW": [1.48],
            "PSL": [2.80],
        })
        result = collector._standardise(raw)
        assert result["winner"].iloc[0] == "Alcaraz C."
        assert result["loser"].iloc[0] == "Djokovic N."
        assert result["surface"].iloc[0] == "Grass"
        assert result["b365_winner"].iloc[0] == 1.50

    def test_standardise_builds_score_from_sets(self):
        collector = TennisDataUKCollector()
        raw = pd.DataFrame({
            "Date": ["2024-01-15"],
            "Winner": ["Player A"],
            "Loser": ["Player B"],
            "W1": [6], "L1": [3],
            "W2": [7], "L2": [5],
        })
        result = collector._standardise(raw)
        assert "score" in result.columns
        assert "6-3" in result["score"].iloc[0]

    def test_standardise_parses_date(self):
        collector = TennisDataUKCollector()
        raw = pd.DataFrame({
            "Date": ["2024-06-01"],
            "Winner": ["A"],
            "Loser": ["B"],
        })
        result = collector._standardise(raw)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_collect_returns_empty_on_failure(self):
        """When all URLs fail, _collect returns empty DataFrame with correct schema."""
        collector = TennisDataUKCollector()
        with patch.object(collector, "_fetch", side_effect=Exception("Network error")):
            with patch.object(collector, "_get_cached", return_value=None):
                df = collector._collect(tour="atp", year=2020)
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 0
                for col in collector.REQUIRED_COLUMNS:
                    assert col in df.columns

    def test_collect_uses_cache(self):
        """When cache has data, _collect should return it without fetching."""
        collector = TennisDataUKCollector()
        cached_data = [{"date": "2024-01-01", "winner": "Test", "loser": "Test2"}]
        with patch.object(collector, "_get_cached", return_value=cached_data):
            with patch.object(collector, "_fetch") as mock_fetch:
                df = collector._collect(tour="atp", year=2024)
                mock_fetch.assert_not_called()
                assert isinstance(df, pd.DataFrame)
                assert len(df) == 1

    def test_column_map_has_key_mappings(self):
        assert "Date" in _COLUMN_MAP
        assert "Winner" in _COLUMN_MAP
        assert "Loser" in _COLUMN_MAP
        assert "B365W" in _COLUMN_MAP
        assert _COLUMN_MAP["B365W"] == "b365_winner"
