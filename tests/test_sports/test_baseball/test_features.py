"""Tests for baseball feature engineering modules."""

import pandas as pd
import numpy as np
import pytest

from sharpedge.sports.baseball.features.pitching import PitchingFeatures
from sharpedge.sports.baseball.features.batting import BattingFeatures
from sharpedge.sports.baseball.features.park_factor import ParkFactorFeatures
from sharpedge.sports.baseball.features.bullpen import BullpenFeatures
from sharpedge.sports.baseball.features.market import BaseballMarketFeatures
from sharpedge.sports.baseball.features.pipeline import BaseballFeaturePipeline


def _make_baseball_df() -> pd.DataFrame:
    """Create a synthetic baseball DataFrame with ~30 games."""
    np.random.seed(42)
    teams = ["NYY", "BOS", "LAD", "HOU", "ATL", "SF"]
    data = []
    base_date = pd.Timestamp("2025-04-01")

    matchups = [
        ("NYY", "BOS", 0), ("LAD", "HOU", 0), ("ATL", "SF", 0),
        ("BOS", "LAD", 1), ("NYY", "HOU", 1), ("SF", "ATL", 1),
        ("HOU", "BOS", 2), ("NYY", "LAD", 2), ("ATL", "SF", 2),
        ("SF", "NYY", 4), ("BOS", "ATL", 4), ("LAD", "HOU", 4),
        ("NYY", "ATL", 5), ("BOS", "SF", 5), ("HOU", "LAD", 5),
        ("ATL", "NYY", 7), ("SF", "BOS", 7), ("LAD", "HOU", 7),
        ("NYY", "BOS", 8), ("HOU", "SF", 8), ("LAD", "ATL", 8),
        ("BOS", "HOU", 10), ("ATL", "LAD", 10), ("SF", "NYY", 10),
        ("NYY", "HOU", 11), ("BOS", "LAD", 11), ("ATL", "SF", 11),
        ("HOU", "ATL", 13), ("LAD", "BOS", 13), ("SF", "NYY", 13),
    ]

    for home, away, offset in matchups:
        home_score = np.random.randint(1, 10)
        away_score = np.random.randint(0, 8)
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
            "total_line": round(8.0 + np.random.randn() * 1.0, 1),
        })

    return pd.DataFrame(data)


@pytest.fixture
def baseball_df():
    return _make_baseball_df()


class TestPitchingFeatures:
    def test_feature_count(self, baseball_df):
        fg = PitchingFeatures()
        result = fg.compute(baseball_df)
        assert result.shape[1] == fg.feature_count == 12

    def test_feature_names_match(self, baseball_df):
        fg = PitchingFeatures()
        result = fg.compute(baseball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, baseball_df):
        fg = PitchingFeatures()
        result = fg.compute(baseball_df)
        # First game should have NaN for rolling features
        assert pd.isna(result.iloc[0]["mlb_era_home"])

    def test_later_games_have_data(self, baseball_df):
        fg = PitchingFeatures()
        result = fg.compute(baseball_df)
        assert not result.iloc[-1].isna().all()


class TestBattingFeatures:
    def test_feature_count(self, baseball_df):
        fg = BattingFeatures()
        result = fg.compute(baseball_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, baseball_df):
        fg = BattingFeatures()
        result = fg.compute(baseball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, baseball_df):
        fg = BattingFeatures()
        result = fg.compute(baseball_df)
        assert result.iloc[0].isna().all()

    def test_wrc_plus_non_negative(self, baseball_df):
        fg = BattingFeatures()
        result = fg.compute(baseball_df)
        wrc = result["mlb_wrc_plus"].dropna()
        assert (wrc >= 0).all()


class TestParkFactorFeatures:
    def test_feature_count(self, baseball_df):
        fg = ParkFactorFeatures()
        result = fg.compute(baseball_df)
        assert result.shape[1] == fg.feature_count == 4

    def test_feature_names_match(self, baseball_df):
        fg = ParkFactorFeatures()
        result = fg.compute(baseball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_park_factor_positive(self, baseball_df):
        fg = ParkFactorFeatures()
        result = fg.compute(baseball_df)
        pf = result["mlb_park_run_factor"].dropna()
        assert (pf > 0).all()


class TestBullpenFeatures:
    def test_feature_count(self, baseball_df):
        fg = BullpenFeatures()
        result = fg.compute(baseball_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, baseball_df):
        fg = BullpenFeatures()
        result = fg.compute(baseball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, baseball_df):
        fg = BullpenFeatures()
        result = fg.compute(baseball_df)
        assert pd.isna(result.iloc[0]["mlb_bullpen_era"])

    def test_closer_available_binary(self, baseball_df):
        fg = BullpenFeatures()
        result = fg.compute(baseball_df)
        ca = result["mlb_closer_available"].dropna()
        assert set(ca.unique()).issubset({0.0, 1.0})


class TestBaseballMarketFeatures:
    def test_feature_count(self, baseball_df):
        fg = BaseballMarketFeatures()
        result = fg.compute(baseball_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, baseball_df):
        fg = BaseballMarketFeatures()
        result = fg.compute(baseball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_implied_probabilities_sum_to_one(self, baseball_df):
        fg = BaseballMarketFeatures()
        result = fg.compute(baseball_df)
        for i in result.index:
            p_h = result.loc[i, "mlb_mkt_implied_home"]
            p_a = result.loc[i, "mlb_mkt_implied_away"]
            if pd.notna(p_h) and pd.notna(p_a):
                assert abs(p_h + p_a - 1.0) < 0.01

    def test_overround_positive(self, baseball_df):
        fg = BaseballMarketFeatures()
        result = fg.compute(baseball_df)
        overround = result["mlb_mkt_overround"].dropna()
        assert (overround > 0).all()
        assert len(overround) > 0


class TestBaseballFeaturePipeline:
    def test_total_features(self):
        pipeline = BaseballFeaturePipeline()
        assert pipeline.total_features == 38

    def test_all_feature_names(self):
        pipeline = BaseballFeaturePipeline()
        names = pipeline.get_all_feature_names()
        assert len(names) == 38
        assert len(set(names)) == 38

    def test_build_produces_correct_shape(self, baseball_df):
        pipeline = BaseballFeaturePipeline()
        result = pipeline.build(baseball_df)
        assert result.shape == (len(baseball_df), 38)

    def test_build_same_index(self, baseball_df):
        pipeline = BaseballFeaturePipeline()
        result = pipeline.build(baseball_df)
        assert list(result.index) == list(baseball_df.index)

    def test_build_column_names_match(self, baseball_df):
        pipeline = BaseballFeaturePipeline()
        result = pipeline.build(baseball_df)
        assert list(result.columns) == pipeline.get_all_feature_names()

    def test_no_look_ahead_first_game(self, baseball_df):
        pipeline = BaseballFeaturePipeline()
        result = pipeline.build(baseball_df)
        first = result.iloc[0]
        assert pd.isna(first["mlb_era_home"])

    def test_missing_data_handling(self):
        minimal_df = pd.DataFrame({
            "game_date": pd.date_range("2025-04-01", periods=5),
            "home_team": ["A", "B", "A", "C", "B"],
            "away_team": ["B", "C", "C", "A", "A"],
            "home_score": [5, 3, 7, 2, 4],
            "away_score": [3, 5, 2, 6, 3],
            "home_win": [1, 0, 1, 0, 1],
            "season": ["2025"] * 5,
        })
        pipeline = BaseballFeaturePipeline()
        result = pipeline.build(minimal_df)
        assert result.shape == (5, 38)
