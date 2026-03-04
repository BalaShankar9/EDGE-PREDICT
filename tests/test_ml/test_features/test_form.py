# tests/test_ml/test_features/test_form.py
import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.features.form import FormFeatures


@pytest.fixture
def sample_matches():
    """Create 20 synthetic matches for testing form features."""
    rng = np.random.default_rng(42)
    n = 20
    dates = pd.date_range("2025-01-01", periods=n, freq="3D")
    teams = ["arsenal", "chelsea", "liverpool", "man_city"]
    rows = []
    for i in range(n):
        h = teams[i % 4]
        a = teams[(i + 1) % 4]
        hg = rng.integers(0, 4)
        ag = rng.integers(0, 3)
        result = "H" if hg > ag else ("A" if ag > hg else "D")
        rows.append({
            "match_date": dates[i],
            "home_team_id": h,
            "away_team_id": a,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": result,
        })
    return pd.DataFrame(rows)


def test_form_features_shape(sample_matches):
    ff = FormFeatures()
    result = ff.compute(sample_matches)
    assert len(result) == len(sample_matches)
    assert result.shape[1] == 12


def test_form_features_columns(sample_matches):
    ff = FormFeatures()
    result = ff.compute(sample_matches)
    expected = ff.get_feature_names()
    assert list(result.columns) == expected


def test_early_matches_have_nans(sample_matches):
    """First few matches should have NaN (not enough history)."""
    ff = FormFeatures()
    result = ff.compute(sample_matches)
    assert pd.isna(result.iloc[0]["form_home_goals_scored_5"])


def test_later_matches_have_values(sample_matches):
    """Later matches should have computed form values."""
    ff = FormFeatures()
    result = ff.compute(sample_matches)
    last_row = result.iloc[-1]
    assert last_row.notna().sum() > 6


def test_form_feature_count():
    ff = FormFeatures()
    assert ff.feature_count == 12
    assert ff.name == "form"
