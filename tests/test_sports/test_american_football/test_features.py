"""Tests for NFL feature engineering modules."""

import pandas as pd
import numpy as np
import pytest

from sharpedge.sports.american_football.features.efficiency import NFLEfficiencyFeatures
from sharpedge.sports.american_football.features.situational import SituationalFeatures
from sharpedge.sports.american_football.features.turnover import TurnoverFeatures
from sharpedge.sports.american_football.features.rest import NFLRestFeatures
from sharpedge.sports.american_football.features.market import NFLMarketFeatures
from sharpedge.sports.american_football.features.pipeline import NFLFeaturePipeline


def _make_nfl_df() -> pd.DataFrame:
    """Create a synthetic NFL DataFrame with ~30 games."""
    np.random.seed(42)
    teams = ["KC", "BUF", "SF", "DAL", "PHI", "BAL"]
    data = []
    base_date = pd.Timestamp("2025-09-07")

    matchups = [
        ("KC", "BUF", 0), ("SF", "DAL", 0), ("PHI", "BAL", 0),
        ("BUF", "SF", 7), ("KC", "DAL", 7), ("BAL", "PHI", 7),
        ("DAL", "BUF", 14), ("KC", "SF", 14), ("PHI", "BAL", 14),
        ("SF", "PHI", 21), ("BUF", "KC", 21), ("DAL", "BAL", 21),
        ("BAL", "KC", 28), ("SF", "BUF", 28), ("PHI", "DAL", 28),
        ("KC", "PHI", 35), ("BUF", "DAL", 35), ("SF", "BAL", 35),
        ("DAL", "KC", 42), ("BAL", "BUF", 42), ("PHI", "SF", 42),
        ("KC", "BAL", 49), ("BUF", "PHI", 49), ("SF", "DAL", 49),
        ("DAL", "SF", 56), ("PHI", "KC", 56), ("BAL", "BUF", 56),
        ("KC", "SF", 63), ("BUF", "BAL", 63), ("DAL", "PHI", 63),
    ]

    for home, away, offset in matchups:
        home_score = np.random.randint(10, 42)
        away_score = np.random.randint(7, 38)
        data.append({
            "game_date": base_date + pd.Timedelta(days=offset),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
            "season": "2025",
            "odds_home": round(1.3 + np.random.random() * 0.5, 2),
            "odds_away": round(1.5 + np.random.random() * 0.7, 2),
            "b365_home": round(1.3 + np.random.random() * 0.5, 2),
            "b365_away": round(1.5 + np.random.random() * 0.7, 2),
            "spread_line": round(-3 + np.random.randn() * 7, 1),
            "total_line": round(44 + np.random.randn() * 5, 1),
        })

    return pd.DataFrame(data)


@pytest.fixture
def nfl_df():
    return _make_nfl_df()


class TestNFLEfficiencyFeatures:
    def test_feature_count(self, nfl_df):
        fg = NFLEfficiencyFeatures()
        result = fg.compute(nfl_df)
        assert result.shape[1] == fg.feature_count == 10

    def test_feature_names_match(self, nfl_df):
        fg = NFLEfficiencyFeatures()
        result = fg.compute(nfl_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, nfl_df):
        fg = NFLEfficiencyFeatures()
        result = fg.compute(nfl_df)
        assert result.iloc[0].isna().all()

    def test_later_games_have_data(self, nfl_df):
        fg = NFLEfficiencyFeatures()
        result = fg.compute(nfl_df)
        assert not result.iloc[-1].isna().all()


class TestSituationalFeatures:
    def test_feature_count(self, nfl_df):
        fg = SituationalFeatures()
        result = fg.compute(nfl_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, nfl_df):
        fg = SituationalFeatures()
        result = fg.compute(nfl_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_dome_flag_binary(self, nfl_df):
        fg = SituationalFeatures()
        result = fg.compute(nfl_df)
        dome = result["nfl_home_dome"].dropna()
        assert set(dome.unique()).issubset({0.0, 1.0})

    def test_divisional_flag_binary(self, nfl_df):
        fg = SituationalFeatures()
        result = fg.compute(nfl_df)
        div = result["nfl_divisional"].dropna()
        assert set(div.unique()).issubset({0.0, 1.0})


class TestTurnoverFeatures:
    def test_feature_count(self, nfl_df):
        fg = TurnoverFeatures()
        result = fg.compute(nfl_df)
        assert result.shape[1] == fg.feature_count == 4

    def test_feature_names_match(self, nfl_df):
        fg = TurnoverFeatures()
        result = fg.compute(nfl_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, nfl_df):
        fg = TurnoverFeatures()
        result = fg.compute(nfl_df)
        assert result.iloc[0].isna().all()


class TestNFLRestFeatures:
    def test_feature_count(self, nfl_df):
        fg = NFLRestFeatures()
        result = fg.compute(nfl_df)
        assert result.shape[1] == fg.feature_count == 4

    def test_feature_names_match(self, nfl_df):
        fg = NFLRestFeatures()
        result = fg.compute(nfl_df)
        assert list(result.columns) == fg.get_feature_names()


class TestNFLMarketFeatures:
    def test_feature_count(self, nfl_df):
        fg = NFLMarketFeatures()
        result = fg.compute(nfl_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, nfl_df):
        fg = NFLMarketFeatures()
        result = fg.compute(nfl_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_implied_probabilities_sum_to_one(self, nfl_df):
        fg = NFLMarketFeatures()
        result = fg.compute(nfl_df)
        for i in result.index:
            p_h = result.loc[i, "nfl_mkt_implied_home"]
            p_a = result.loc[i, "nfl_mkt_implied_away"]
            if pd.notna(p_h) and pd.notna(p_a):
                assert abs(p_h + p_a - 1.0) < 0.01

    def test_overround_positive(self, nfl_df):
        fg = NFLMarketFeatures()
        result = fg.compute(nfl_df)
        overround = result["nfl_mkt_overround"].dropna()
        assert (overround > 0).all()
        assert len(overround) > 0


class TestNFLFeaturePipeline:
    def test_total_features(self):
        pipeline = NFLFeaturePipeline()
        assert pipeline.total_features == 34

    def test_all_feature_names(self):
        pipeline = NFLFeaturePipeline()
        names = pipeline.get_all_feature_names()
        assert len(names) == 34
        assert len(set(names)) == 34

    def test_build_produces_correct_shape(self, nfl_df):
        pipeline = NFLFeaturePipeline()
        result = pipeline.build(nfl_df)
        assert result.shape == (len(nfl_df), 34)

    def test_build_same_index(self, nfl_df):
        pipeline = NFLFeaturePipeline()
        result = pipeline.build(nfl_df)
        assert list(result.index) == list(nfl_df.index)

    def test_build_column_names_match(self, nfl_df):
        pipeline = NFLFeaturePipeline()
        result = pipeline.build(nfl_df)
        assert list(result.columns) == pipeline.get_all_feature_names()

    def test_no_look_ahead_first_game(self, nfl_df):
        pipeline = NFLFeaturePipeline()
        result = pipeline.build(nfl_df)
        first = result.iloc[0]
        assert pd.isna(first["nfl_epa_per_play_off"])

    def test_missing_data_handling(self):
        minimal_df = pd.DataFrame({
            "game_date": pd.date_range("2025-09-07", periods=5, freq="7D"),
            "home_team": ["KC", "BUF", "KC", "SF", "BUF"],
            "away_team": ["BUF", "SF", "SF", "KC", "KC"],
            "home_score": [27, 21, 31, 17, 24],
            "away_score": [24, 28, 14, 30, 21],
            "home_win": [1, 0, 1, 0, 1],
            "season": ["2025"] * 5,
        })
        pipeline = NFLFeaturePipeline()
        result = pipeline.build(minimal_df)
        assert result.shape == (5, 34)
