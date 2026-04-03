import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.features.market import MarketFeatures


@pytest.fixture
def sample_odds_matches():
    return pd.DataFrame({
        "match_date": pd.to_datetime(["2025-01-01", "2025-01-08"]),
        "home_team_id": ["arsenal", "chelsea"],
        "away_team_id": ["chelsea", "arsenal"],
        "B365H": [1.8, 2.1],
        "B365D": [3.5, 3.2],
        "B365A": [4.5, 3.8],
        "PSH": [1.85, 2.05],
        "PSD": [3.6, 3.3],
        "PSA": [4.2, 3.9],
        "WHH": [1.75, 2.0],
        "WHD": [3.4, 3.1],
        "WHA": [4.6, 3.7],
    })


def test_market_features_shape(sample_odds_matches):
    mf = MarketFeatures()
    result = mf.compute(sample_odds_matches)
    assert len(result) == 2
    assert result.shape[1] == mf.feature_count


def test_market_overround(sample_odds_matches):
    mf = MarketFeatures()
    result = mf.compute(sample_odds_matches)
    overround = result["mkt_overround"].dropna()
    assert (overround > 100).all()  # Bookmakers always have >100% overround


def test_market_best_odds(sample_odds_matches):
    mf = MarketFeatures()
    result = mf.compute(sample_odds_matches)
    assert result.iloc[0]["mkt_best_odds_home"] == pytest.approx(1.85)


def test_market_feature_metadata():
    mf = MarketFeatures()
    assert mf.feature_count == 20  # 11 original + 6 smart money + 3 goto
    assert mf.name == "market"
