# tests/test_ml/test_features/test_xg_perf.py
import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.features.xg_perf import XGPerformanceFeatures


@pytest.fixture
def sample_xg_data():
    """Create xG data for testing."""
    rng = np.random.default_rng(42)
    teams = ["arsenal", "chelsea", "liverpool", "man_city"]
    n = 40
    dates = pd.date_range("2024-08-01", periods=n, freq="4D")
    rows = []
    for i in range(n):
        h = teams[i % 4]
        a = teams[(i + 1) % 4]
        rows.append({
            "match_date": dates[i],
            "home_team_id": h,
            "away_team_id": a,
            "home_xg": round(rng.uniform(0.5, 2.5), 2),
            "away_xg": round(rng.uniform(0.3, 2.0), 2),
            "home_shots": rng.integers(5, 20),
            "away_shots": rng.integers(3, 15),
            "FTHG": rng.integers(0, 4),
            "FTAG": rng.integers(0, 3),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_matches():
    return pd.DataFrame({
        "match_date": pd.to_datetime(["2025-01-01", "2025-01-08"]),
        "home_team_id": ["arsenal", "man_city"],
        "away_team_id": ["chelsea", "liverpool"],
    })


def test_xg_features_shape(sample_matches, sample_xg_data):
    xgf = XGPerformanceFeatures()
    result = xgf.compute(sample_matches, xg_df=sample_xg_data)
    assert len(result) == 2
    assert result.shape[1] == 8


def test_xg_features_columns(sample_matches, sample_xg_data):
    xgf = XGPerformanceFeatures()
    result = xgf.compute(sample_matches, xg_df=sample_xg_data)
    assert list(result.columns) == xgf.get_feature_names()


def test_xg_variance_non_negative(sample_matches, sample_xg_data):
    xgf = XGPerformanceFeatures()
    result = xgf.compute(sample_matches, xg_df=sample_xg_data)
    variance_vals = result["xg_home_variance"].dropna()
    assert (variance_vals >= 0).all()


def test_xg_handles_no_data():
    matches = pd.DataFrame({
        "match_date": pd.to_datetime(["2025-01-01"]),
        "home_team_id": ["arsenal"],
        "away_team_id": ["chelsea"],
    })
    xgf = XGPerformanceFeatures()
    result = xgf.compute(matches, xg_df=None)
    assert result.iloc[0].isna().all()


def test_xg_feature_metadata():
    xgf = XGPerformanceFeatures()
    assert xgf.feature_count == 8
    assert xgf.name == "xg_perf"
