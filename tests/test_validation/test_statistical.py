"""Tests for the statistical validation layer."""

import pandas as pd
import pytest

from sharpedge.validation.statistical import validate_statistical


class TestValidateStatistical:
    """Statistical validation tests."""

    def test_valid_ranges(self):
        """All values within expected ranges passes validation."""
        df = pd.DataFrame(
            {
                "match_date": ["2024-01-01", "2024-01-02"],
                "home_team_id": ["ARS", "CHE"],
                "away_team_id": ["LIV", "MCI"],
                "home_goals": [2, 1],
                "away_goals": [1, 3],
                "home_xg": [1.5, 0.8],
                "away_xg": [1.2, 2.1],
                "elo": [1800, 1750],
            }
        )
        result = validate_statistical(df)
        assert result.passed is True
        assert result.errors == []

    def test_out_of_range_error(self):
        """More than 5% of values out of range triggers an error."""
        # All 10 rows have elo = 3000 (above max 2200) -> 100% out of range
        df = pd.DataFrame(
            {
                "elo": [3000] * 10,
                "club": ["Team"] * 10,
            }
        )
        result = validate_statistical(df)
        assert result.passed is False
        assert any("outside expected range" in e for e in result.errors)

    def test_out_of_range_warning(self):
        """A small fraction out of range triggers a warning, not an error."""
        # 1 out of 100 rows has elo out of range (1%) -> warning
        df = pd.DataFrame(
            {
                "elo": [1500] * 99 + [3000],
                "club": ["Team"] * 100,
            }
        )
        result = validate_statistical(df)
        assert result.passed is True
        assert any("outside expected range" in w for w in result.warnings)

    def test_duplicate_detection(self):
        """Duplicate matches on date + team IDs are flagged as errors."""
        df = pd.DataFrame(
            {
                "match_date": ["2024-01-01", "2024-01-01"],
                "home_team_id": ["ARS", "ARS"],
                "away_team_id": ["LIV", "LIV"],
                "home_goals": [2, 2],
                "away_goals": [1, 1],
            }
        )
        result = validate_statistical(df)
        assert result.passed is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_empty_dataframe(self):
        """Empty DataFrame passes without errors."""
        df = pd.DataFrame()
        result = validate_statistical(df)
        assert result.passed is True

    def test_probability_range(self):
        """Probability values outside [0, 1] are flagged."""
        df = pd.DataFrame(
            {
                "prob_home": [1.5] * 10,
                "prob_draw": [0.3] * 10,
                "prob_away": [0.2] * 10,
            }
        )
        result = validate_statistical(df)
        assert result.passed is False
        assert any("prob_home" in e for e in result.errors)
