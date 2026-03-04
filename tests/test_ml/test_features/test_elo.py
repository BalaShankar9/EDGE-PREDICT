import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.features.elo import EloFeatures


@pytest.fixture
def sample_elo_data():
    """Create ELO ratings for 4 teams over multiple dates."""
    teams = ["arsenal", "chelsea", "liverpool", "man_city"]
    dates = pd.date_range("2024-06-01", periods=20, freq="W")
    rows = []
    base_elos = {"arsenal": 1800, "chelsea": 1750, "liverpool": 1850, "man_city": 1900}
    for team in teams:
        for i, d in enumerate(dates):
            rows.append({
                "team_id": team,
                "date": d,
                "elo": base_elos[team] + i * 5,  # Gradual improvement
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_matches():
    return pd.DataFrame({
        "match_date": pd.to_datetime(["2024-10-01", "2024-10-08"]),
        "home_team_id": ["arsenal", "man_city"],
        "away_team_id": ["chelsea", "liverpool"],
    })


def test_elo_features_shape(sample_matches, sample_elo_data):
    ef = EloFeatures()
    result = ef.compute(sample_matches, elo_df=sample_elo_data)
    assert len(result) == 2
    assert result.shape[1] == 6


def test_elo_features_columns(sample_matches, sample_elo_data):
    ef = EloFeatures()
    result = ef.compute(sample_matches, elo_df=sample_elo_data)
    assert list(result.columns) == ef.get_feature_names()


def test_elo_diff_positive_when_home_stronger(sample_elo_data):
    """Man City (1900+) vs Chelsea (1750+) should have positive diff."""
    matches = pd.DataFrame({
        "match_date": pd.to_datetime(["2024-10-01"]),
        "home_team_id": ["man_city"],
        "away_team_id": ["chelsea"],
    })
    ef = EloFeatures()
    result = ef.compute(matches, elo_df=sample_elo_data)
    assert result.iloc[0]["elo_diff"] > 0


def test_elo_handles_no_data():
    """Should return NaN-filled DataFrame when no ELO data."""
    matches = pd.DataFrame({
        "match_date": pd.to_datetime(["2024-10-01"]),
        "home_team_id": ["arsenal"],
        "away_team_id": ["chelsea"],
    })
    ef = EloFeatures()
    result = ef.compute(matches, elo_df=None)
    assert result.iloc[0].isna().all()


def test_elo_feature_metadata():
    ef = EloFeatures()
    assert ef.feature_count == 6
    assert ef.name == "elo"
