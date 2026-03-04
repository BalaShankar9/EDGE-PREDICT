"""Tests for the schema validation layer."""

import pandas as pd
import pytest

from sharpedge.validation.schema import validate_schema


class TestValidateSchema:
    """Schema validation tests."""

    def test_valid_match_schema(self):
        """DataFrame with all required columns passes validation."""
        df = pd.DataFrame(
            {
                "match_date": ["2024-01-01", "2024-01-02"],
                "home_team_id": ["ARS", "CHE"],
                "away_team_id": ["LIV", "MCI"],
                "league": ["E0", "E0"],
                "season": ["2023-24", "2023-24"],
                "home_goals": [2, 1],
                "away_goals": [1, 3],
            }
        )
        result = validate_schema(df, "match")
        assert result.passed is True
        assert result.errors == []

    def test_missing_required_column(self):
        """DataFrame missing a required column fails validation."""
        df = pd.DataFrame(
            {
                "match_date": ["2024-01-01"],
                "home_team_id": ["ARS"],
                # away_team_id is missing
                "league": ["E0"],
                "season": ["2023-24"],
            }
        )
        result = validate_schema(df, "match")
        assert result.passed is False
        assert any("Missing required columns" in e for e in result.errors)

    def test_excessive_nulls_error(self):
        """More than 10% nulls in a required column triggers an error."""
        # Create 10 rows with 2 nulls in home_team_id (20%)
        df = pd.DataFrame(
            {
                "match_date": [f"2024-01-{i:02d}" for i in range(1, 11)],
                "home_team_id": ["ARS"] * 8 + [None, None],
                "away_team_id": ["LIV"] * 10,
                "league": ["E0"] * 10,
                "season": ["2023-24"] * 10,
            }
        )
        result = validate_schema(df, "match")
        assert result.passed is False
        assert any("nulls" in e for e in result.errors)

    def test_no_schema_defined(self):
        """Unknown record type passes with a warning."""
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = validate_schema(df, "unknown_type")
        assert result.passed is True
        assert any("No schema defined" in w for w in result.warnings)

    def test_minor_nulls_warning(self):
        """A small fraction of nulls triggers a warning, not an error."""
        # 1 null out of 20 rows = 5%
        df = pd.DataFrame(
            {
                "match_date": [f"2024-01-{i:02d}" for i in range(1, 21)],
                "home_team_id": ["ARS"] * 19 + [None],
                "away_team_id": ["LIV"] * 20,
                "league": ["E0"] * 20,
                "season": ["2023-24"] * 20,
            }
        )
        result = validate_schema(df, "match")
        assert result.passed is True
        assert any("nulls" in w for w in result.warnings)

    def test_elo_schema(self):
        """ELO schema validates correctly."""
        df = pd.DataFrame({"club": ["Arsenal", "Chelsea"], "elo": [1800, 1750]})
        result = validate_schema(df, "elo")
        assert result.passed is True
