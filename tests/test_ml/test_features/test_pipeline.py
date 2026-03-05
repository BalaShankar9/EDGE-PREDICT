import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.features.form import FormFeatures
from sharpedge.ml.features.market import MarketFeatures


@pytest.fixture
def sample_matches():
    """Minimal match data with odds columns."""
    rng = np.random.default_rng(42)
    n = 20
    dates = pd.date_range("2024-08-01", periods=n, freq="4D")
    teams = ["arsenal", "chelsea", "liverpool", "man_city"]
    rows = []
    for i in range(n):
        h = teams[i % 4]
        a = teams[(i + 1) % 4]
        hg = int(rng.integers(0, 4))
        ag = int(rng.integers(0, 3))
        result = "H" if hg > ag else ("A" if ag > hg else "D")
        rows.append({
            "match_date": dates[i],
            "home_team_id": h,
            "away_team_id": a,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": result,
            "league": "Premier League",
            "B365H": round(rng.uniform(1.3, 3.5), 2),
            "B365D": round(rng.uniform(3.0, 4.0), 2),
            "B365A": round(rng.uniform(2.0, 6.0), 2),
            "PSH": round(rng.uniform(1.3, 3.5), 2),
            "PSD": round(rng.uniform(3.0, 4.0), 2),
            "PSA": round(rng.uniform(2.0, 6.0), 2),
            "WHH": round(rng.uniform(1.3, 3.5), 2),
            "WHD": round(rng.uniform(3.0, 4.0), 2),
            "WHA": round(rng.uniform(2.0, 6.0), 2),
            "HS": int(rng.integers(5, 20)),
            "AS": int(rng.integers(3, 18)),
            "HST": int(rng.integers(2, 10)),
            "AST": int(rng.integers(1, 8)),
        })
    return pd.DataFrame(rows)


def test_pipeline_total_features():
    pipe = FeaturePipeline()
    assert pipe.total_features == 46


def test_pipeline_builds_all_columns(sample_matches):
    pipe = FeaturePipeline()
    result = pipe.build(sample_matches)
    assert result.shape[1] == 46
    assert len(result) == len(sample_matches)


def test_pipeline_with_subset_groups(sample_matches):
    pipe = FeaturePipeline(groups=[FormFeatures, MarketFeatures])
    result = pipe.build(sample_matches)
    assert result.shape[1] == 20  # 12 + 8


def test_pipeline_handles_missing_context(sample_matches):
    """Pipeline should NaN-fill for groups that fail due to missing context."""
    pipe = FeaturePipeline()
    result = pipe.build(sample_matches, elo_df=None, xg_df=None, predictions_df=None)
    assert result.shape[1] == 46
    # ELO and xG features should be NaN (no data), but form/market should have values
    assert result["mkt_best_odds_home"].notna().any()


def test_pipeline_feature_names():
    pipe = FeaturePipeline()
    names = pipe.get_all_feature_names()
    assert len(names) == 46
    assert names[0].startswith("form_")
    assert names[-1].startswith("shot_")
