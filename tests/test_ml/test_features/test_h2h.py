import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.features.h2h import H2HFeatures


@pytest.fixture
def sample_matches():
    """Matches with repeated H2H pairs."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=15, freq="7D")
    teams = ["arsenal", "chelsea"]
    for i in range(15):
        h = teams[i % 2]
        a = teams[(i + 1) % 2]
        rows.append({
            "match_date": dates[i],
            "home_team_id": h,
            "away_team_id": a,
            "FTHG": np.random.default_rng(i).integers(0, 4),
            "FTAG": np.random.default_rng(i + 100).integers(0, 3),
            "FTR": "H",
        })
    return pd.DataFrame(rows)


def test_h2h_features_shape(sample_matches):
    hf = H2HFeatures()
    result = hf.compute(sample_matches)
    assert len(result) == len(sample_matches)
    assert result.shape[1] == 6


def test_h2h_early_nans(sample_matches):
    hf = H2HFeatures()
    result = hf.compute(sample_matches)
    assert pd.isna(result.iloc[0]["h2h_home_win_rate_5"])


def test_h2h_feature_metadata():
    hf = H2HFeatures()
    assert hf.feature_count == 6
    assert hf.name == "h2h"
