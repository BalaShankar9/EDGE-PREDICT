"""
Integration tests for the full SharpEdge data pipeline.

These tests actually hit real APIs/websites and should be run separately:
    pytest tests/test_integration/ -m integration -v

They are excluded from normal test runs by the integration marker.
"""

import pytest
import pandas as pd


@pytest.mark.integration
class TestFullPipeline:
    def test_football_data_uk_pipeline(self):
        """Download real CSV, validate schema and stats."""
        from sharpedge.collectors.football_data_uk import FootballDataUKCollector
        from sharpedge.validation.schema import validate_schema
        from sharpedge.validation.statistical import validate_statistical

        collector = FootballDataUKCollector()
        df = collector.collect(league="Premier League", season="2023-24")
        assert len(df) > 300

        schema = validate_schema(df, "match")
        assert schema.passed, f"Schema errors: {schema.errors}"

        stats = validate_statistical(df)
        assert stats.passed, f"Statistical errors: {stats.errors}"

    def test_club_elo_pipeline(self):
        """Fetch real ELO ratings for a date."""
        from sharpedge.collectors.club_elo import ClubELOCollector
        from sharpedge.validation.schema import validate_schema

        collector = ClubELOCollector()
        df = collector.collect(date_str="2025-01-01")
        assert len(df) > 400

        schema = validate_schema(df, "elo")
        assert schema.passed

    def test_understat_pipeline(self):
        """Fetch real xG data."""
        from sharpedge.collectors.understat import UnderstatCollector
        from sharpedge.validation.schema import validate_schema

        collector = UnderstatCollector()
        df = collector.collect(league="Premier League", season=2023)
        assert len(df) > 300
        assert "home_xg" in df.columns

    def test_open_meteo_pipeline(self):
        """Fetch real weather data."""
        from sharpedge.collectors.open_meteo import OpenMeteoCollector

        collector = OpenMeteoCollector()
        df = collector.collect(
            venue="Emirates Stadium", date_str="2025-01-15", hour=15
        )
        assert len(df) == 1
        assert "temperature_c" in df.columns
