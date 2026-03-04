"""Tests for the database ingestion layer."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from sharpedge.db.ingest import (
    _parse_date,
    _safe_int,
    _safe_float,
    ingest_dataframe,
)


class TestHelpers:
    def test_parse_date_dd_mm_yyyy(self):
        result = _parse_date("25/12/2023")
        assert result.year == 2023
        assert result.month == 12
        assert result.day == 25

    def test_parse_date_dd_mm_yy(self):
        result = _parse_date("25/12/23")
        assert result.year == 2023

    def test_parse_date_iso(self):
        result = _parse_date("2023-12-25")
        assert result.year == 2023

    def test_parse_date_invalid(self):
        with pytest.raises(ValueError):
            _parse_date("not-a-date")

    def test_safe_int_valid(self):
        assert _safe_int(42) == 42
        assert _safe_int(42.0) == 42
        assert _safe_int("42") == 42

    def test_safe_int_none(self):
        assert _safe_int(None) is None
        assert _safe_int(float("nan")) is None

    def test_safe_float_valid(self):
        assert _safe_float(1.5) == 1.5
        assert _safe_float("1.5") == 1.5

    def test_safe_float_none(self):
        assert _safe_float(None) is None
        assert _safe_float(float("nan")) is None


class TestIngestDispatch:
    def test_empty_df_returns_zero(self):
        assert ingest_dataframe(pd.DataFrame(), "football_data_uk") == 0

    @patch("sharpedge.db.ingest.ingest_football_data_uk")
    def test_routes_football_data_uk(self, mock_fn):
        mock_fn.return_value = 5
        df = pd.DataFrame({"x": [1]})
        result = ingest_dataframe(df, "football_data_uk")
        mock_fn.assert_called_once_with(df)
        assert result == 5

    @patch("sharpedge.db.ingest.ingest_club_elo")
    def test_routes_club_elo(self, mock_fn):
        mock_fn.return_value = 10
        df = pd.DataFrame({"x": [1]})
        result = ingest_dataframe(df, "club_elo")
        mock_fn.assert_called_once_with(df)
        assert result == 10

    @patch("sharpedge.db.ingest.ingest_understat")
    def test_routes_understat(self, mock_fn):
        mock_fn.return_value = 3
        df = pd.DataFrame({"x": [1]})
        result = ingest_dataframe(df, "understat")
        mock_fn.assert_called_once_with(df)
        assert result == 3

    @patch("sharpedge.db.ingest.ingest_predictions")
    def test_routes_forebet(self, mock_fn):
        mock_fn.return_value = 7
        df = pd.DataFrame({"x": [1]})
        result = ingest_dataframe(df, "forebet")
        mock_fn.assert_called_once_with(df, "forebet")
        assert result == 7

    @patch("sharpedge.db.ingest.ingest_predictions")
    def test_routes_predictz(self, mock_fn):
        mock_fn.return_value = 2
        df = pd.DataFrame({"x": [1]})
        result = ingest_dataframe(df, "predictz")
        mock_fn.assert_called_once_with(df, "predictz")
        assert result == 2

    def test_unknown_source_returns_zero(self):
        df = pd.DataFrame({"x": [1]})
        assert ingest_dataframe(df, "totally_unknown") == 0

    def test_skipped_sources_return_zero(self):
        df = pd.DataFrame({"x": [1]})
        assert ingest_dataframe(df, "fbref") == 0
        assert ingest_dataframe(df, "open_meteo") == 0
