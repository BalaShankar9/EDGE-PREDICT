"""Tests for ice hockey feature engineering modules."""

import pandas as pd
import numpy as np
import pytest

from sharpedge.sports.ice_hockey.features.corsi import CorsiFeatures
from sharpedge.sports.ice_hockey.features.goaltending import GoaltendingFeatures
from sharpedge.sports.ice_hockey.features.special_teams import SpecialTeamsFeatures
from sharpedge.sports.ice_hockey.features.rest import HockeyRestFeatures
from sharpedge.sports.ice_hockey.features.market import HockeyMarketFeatures
from sharpedge.sports.ice_hockey.features.pipeline import HockeyFeaturePipeline


def _make_hockey_df() -> pd.DataFrame:
    """Create a synthetic hockey DataFrame with ~30 games."""
    np.random.seed(42)
    teams = ["BOS", "TOR", "NYR", "TBL", "EDM", "COL"]
    data = []
    base_date = pd.Timestamp("2025-01-01")

    matchups = [
        ("BOS", "TOR", 0), ("NYR", "TBL", 1), ("EDM", "COL", 2),
        ("TOR", "NYR", 3), ("BOS", "TBL", 4), ("COL", "EDM", 5),
        ("TBL", "TOR", 7), ("BOS", "NYR", 8), ("EDM", "TBL", 9),
        ("COL", "BOS", 10), ("TOR", "EDM", 12), ("NYR", "COL", 13),
        ("BOS", "EDM", 14), ("TOR", "COL", 15), ("TBL", "NYR", 16),
        ("EDM", "BOS", 18), ("COL", "TOR", 19), ("NYR", "EDM", 20),
        ("BOS", "TOR", 22), ("TBL", "COL", 23), ("NYR", "BOS", 24),
        ("TOR", "TBL", 26), ("EDM", "NYR", 27), ("COL", "BOS", 28),
        ("BOS", "TBL", 30), ("TOR", "NYR", 31), ("EDM", "COL", 32),
        ("TBL", "EDM", 34), ("NYR", "TOR", 35), ("COL", "TBL", 36),
    ]

    for home, away, offset in matchups:
        home_score = np.random.randint(1, 6)
        away_score = np.random.randint(0, 5)
        data.append({
            "game_date": base_date + pd.Timedelta(days=offset),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
            "season": "2024-25",
            "odds_home": round(1.3 + np.random.random() * 0.5, 2),
            "odds_away": round(1.5 + np.random.random() * 0.7, 2),
            "b365_home": round(1.3 + np.random.random() * 0.5, 2),
            "b365_away": round(1.5 + np.random.random() * 0.7, 2),
            "total_line": round(5.0 + np.random.randn() * 0.5, 1),
        })

    return pd.DataFrame(data)


@pytest.fixture
def hockey_df():
    return _make_hockey_df()


class TestCorsiFeatures:
    def test_feature_count(self, hockey_df):
        fg = CorsiFeatures()
        result = fg.compute(hockey_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, hockey_df):
        fg = CorsiFeatures()
        result = fg.compute(hockey_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, hockey_df):
        fg = CorsiFeatures()
        result = fg.compute(hockey_df)
        assert result.iloc[0].isna().all()

    def test_later_games_have_data(self, hockey_df):
        fg = CorsiFeatures()
        result = fg.compute(hockey_df)
        assert not result.iloc[-1].isna().all()


class TestGoaltendingFeatures:
    def test_feature_count(self, hockey_df):
        fg = GoaltendingFeatures()
        result = fg.compute(hockey_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, hockey_df):
        fg = GoaltendingFeatures()
        result = fg.compute(hockey_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, hockey_df):
        fg = GoaltendingFeatures()
        result = fg.compute(hockey_df)
        assert result.iloc[0].isna().all()

    def test_save_pct_bounded(self, hockey_df):
        fg = GoaltendingFeatures()
        result = fg.compute(hockey_df)
        sv = result["nhl_save_pct_home"].dropna()
        assert (sv > 0).all()
        assert (sv <= 1).all()


class TestSpecialTeamsFeatures:
    def test_feature_count(self, hockey_df):
        fg = SpecialTeamsFeatures()
        result = fg.compute(hockey_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, hockey_df):
        fg = SpecialTeamsFeatures()
        result = fg.compute(hockey_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, hockey_df):
        fg = SpecialTeamsFeatures()
        result = fg.compute(hockey_df)
        assert result.iloc[0].isna().all()


class TestHockeyRestFeatures:
    def test_feature_count(self, hockey_df):
        fg = HockeyRestFeatures()
        result = fg.compute(hockey_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, hockey_df):
        fg = HockeyRestFeatures()
        result = fg.compute(hockey_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, hockey_df):
        fg = HockeyRestFeatures()
        result = fg.compute(hockey_df)
        assert pd.isna(result.loc[0, "nhl_rest_days_home"])

    def test_rest_days_positive(self, hockey_df):
        fg = HockeyRestFeatures()
        result = fg.compute(hockey_df)
        rest = result["nhl_rest_days_home"].dropna()
        assert (rest >= 0).all()

    def test_back_to_back_binary(self, hockey_df):
        fg = HockeyRestFeatures()
        result = fg.compute(hockey_df)
        b2b = result["nhl_back_to_back_home"].dropna()
        assert set(b2b.unique()).issubset({0.0, 1.0})


class TestHockeyMarketFeatures:
    def test_feature_count(self, hockey_df):
        fg = HockeyMarketFeatures()
        result = fg.compute(hockey_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, hockey_df):
        fg = HockeyMarketFeatures()
        result = fg.compute(hockey_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_implied_probabilities_sum_to_one(self, hockey_df):
        fg = HockeyMarketFeatures()
        result = fg.compute(hockey_df)
        for i in result.index:
            p_h = result.loc[i, "nhl_mkt_implied_home"]
            p_a = result.loc[i, "nhl_mkt_implied_away"]
            if pd.notna(p_h) and pd.notna(p_a):
                assert abs(p_h + p_a - 1.0) < 0.01

    def test_overround_positive(self, hockey_df):
        fg = HockeyMarketFeatures()
        result = fg.compute(hockey_df)
        overround = result["nhl_mkt_overround"].dropna()
        assert (overround > 0).all()
        assert len(overround) > 0


class TestHockeyFeaturePipeline:
    def test_total_features(self):
        pipeline = HockeyFeaturePipeline()
        assert pipeline.total_features == 32

    def test_all_feature_names(self):
        pipeline = HockeyFeaturePipeline()
        names = pipeline.get_all_feature_names()
        assert len(names) == 32
        assert len(set(names)) == 32  # No duplicates

    def test_build_produces_correct_shape(self, hockey_df):
        pipeline = HockeyFeaturePipeline()
        result = pipeline.build(hockey_df)
        assert result.shape == (len(hockey_df), 32)

    def test_build_same_index(self, hockey_df):
        pipeline = HockeyFeaturePipeline()
        result = pipeline.build(hockey_df)
        assert list(result.index) == list(hockey_df.index)

    def test_build_column_names_match(self, hockey_df):
        pipeline = HockeyFeaturePipeline()
        result = pipeline.build(hockey_df)
        assert list(result.columns) == pipeline.get_all_feature_names()

    def test_no_look_ahead_first_game(self, hockey_df):
        pipeline = HockeyFeaturePipeline()
        result = pipeline.build(hockey_df)
        first = result.iloc[0]
        assert pd.isna(first["nhl_corsi_for_pct"])
        assert pd.isna(first["nhl_save_pct_home"])

    def test_missing_data_handling(self):
        minimal_df = pd.DataFrame({
            "game_date": pd.date_range("2025-01-01", periods=5),
            "home_team": ["A", "B", "A", "C", "B"],
            "away_team": ["B", "C", "C", "A", "A"],
            "home_score": [3, 2, 4, 1, 3],
            "away_score": [2, 3, 1, 4, 2],
            "home_win": [1, 0, 1, 0, 1],
            "season": ["2024-25"] * 5,
        })
        pipeline = HockeyFeaturePipeline()
        result = pipeline.build(minimal_df)
        assert result.shape == (5, 32)
