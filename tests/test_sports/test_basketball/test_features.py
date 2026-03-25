"""Tests for basketball feature engineering modules."""

import pandas as pd
import numpy as np
import pytest

from sharpedge.sports.basketball.features.pace import PaceFeatures
from sharpedge.sports.basketball.features.efficiency import EfficiencyFeatures
from sharpedge.sports.basketball.features.rest import RestFeatures
from sharpedge.sports.basketball.features.roster import RosterFeatures
from sharpedge.sports.basketball.features.matchup import MatchupFeatures
from sharpedge.sports.basketball.features.market import BasketballMarketFeatures
from sharpedge.sports.basketball.features.pipeline import BasketballFeaturePipeline


def _make_basketball_df() -> pd.DataFrame:
    """Create a synthetic basketball DataFrame with ~30 games."""
    np.random.seed(42)
    teams = ["BOS", "LAL", "GSW", "MIL", "DEN", "PHX"]
    data = []
    base_date = pd.Timestamp("2025-01-01")

    matchups = [
        ("BOS", "LAL", 0), ("GSW", "MIL", 1), ("DEN", "PHX", 2),
        ("LAL", "GSW", 3), ("BOS", "MIL", 4), ("PHX", "DEN", 5),
        ("MIL", "LAL", 7), ("BOS", "GSW", 8), ("DEN", "MIL", 9),
        ("PHX", "BOS", 10), ("LAL", "DEN", 12), ("GSW", "PHX", 13),
        ("BOS", "DEN", 14), ("LAL", "PHX", 15), ("MIL", "GSW", 16),
        ("DEN", "BOS", 18), ("PHX", "LAL", 19), ("GSW", "DEN", 20),
        ("BOS", "LAL", 22), ("MIL", "PHX", 23), ("GSW", "BOS", 24),
        ("LAL", "MIL", 26), ("DEN", "GSW", 27), ("PHX", "BOS", 28),
        ("BOS", "MIL", 30), ("LAL", "GSW", 31), ("DEN", "PHX", 32),
        ("MIL", "DEN", 34), ("GSW", "LAL", 35), ("PHX", "MIL", 36),
    ]

    for home, away, offset in matchups:
        home_score = np.random.randint(95, 130)
        away_score = np.random.randint(90, 125)
        data.append({
            "game_date": base_date + pd.Timedelta(days=offset),
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
            "season": "2024-25",
            # Odds columns — realistic bookmaker odds with overround
            # 1/h + 1/a > 1 always holds when h in [1.3,1.8] and a in [1.5,2.2]
            "odds_home": round(1.3 + np.random.random() * 0.5, 2),
            "odds_away": round(1.5 + np.random.random() * 0.7, 2),
            "b365_home": round(1.3 + np.random.random() * 0.5, 2),
            "b365_away": round(1.5 + np.random.random() * 0.7, 2),
            "spread_line": round(-3 + np.random.randn() * 5, 1),
            "total_line": round(215 + np.random.randn() * 10, 1),
        })

    return pd.DataFrame(data)


@pytest.fixture
def basketball_df():
    return _make_basketball_df()


class TestPaceFeatures:
    def test_feature_count(self, basketball_df):
        fg = PaceFeatures()
        result = fg.compute(basketball_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, basketball_df):
        fg = PaceFeatures()
        result = fg.compute(basketball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, basketball_df):
        """First game should have all NaN pace features."""
        fg = PaceFeatures()
        result = fg.compute(basketball_df)
        assert result.iloc[0].isna().all()

    def test_later_games_have_data(self, basketball_df):
        fg = PaceFeatures()
        result = fg.compute(basketball_df)
        # Games after the first few should have pace data
        assert not result.iloc[-1].isna().all()


class TestEfficiencyFeatures:
    def test_feature_count(self, basketball_df):
        fg = EfficiencyFeatures()
        result = fg.compute(basketball_df)
        assert result.shape[1] == fg.feature_count == 10

    def test_feature_names_match(self, basketball_df):
        fg = EfficiencyFeatures()
        result = fg.compute(basketball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, basketball_df):
        fg = EfficiencyFeatures()
        result = fg.compute(basketball_df)
        assert result.iloc[0].isna().all()

    def test_off_rating_positive(self, basketball_df):
        fg = EfficiencyFeatures()
        result = fg.compute(basketball_df)
        off = result["bke_off_rating_home"].dropna()
        assert (off > 0).all()


class TestRestFeatures:
    def test_feature_count(self, basketball_df):
        fg = RestFeatures()
        result = fg.compute(basketball_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, basketball_df):
        fg = RestFeatures()
        result = fg.compute(basketball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, basketball_df):
        """First game should have NaN rest days but travel flag can be set."""
        fg = RestFeatures()
        result = fg.compute(basketball_df)
        assert pd.isna(result.loc[0, "bkr_rest_days_home"])

    def test_rest_days_positive(self, basketball_df):
        fg = RestFeatures()
        result = fg.compute(basketball_df)
        rest = result["bkr_rest_days_home"].dropna()
        assert (rest >= 0).all()

    def test_back_to_back_binary(self, basketball_df):
        fg = RestFeatures()
        result = fg.compute(basketball_df)
        b2b = result["bkr_back_to_back_home"].dropna()
        assert set(b2b.unique()).issubset({0.0, 1.0})


class TestRosterFeatures:
    def test_feature_count(self, basketball_df):
        fg = RosterFeatures()
        result = fg.compute(basketball_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, basketball_df):
        fg = RosterFeatures()
        result = fg.compute(basketball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_roster_data_returns_nan(self, basketball_df):
        """Without roster context, all features should be NaN."""
        fg = RosterFeatures()
        result = fg.compute(basketball_df)
        assert result.isna().all().all()


class TestMatchupFeatures:
    def test_feature_count(self, basketball_df):
        fg = MatchupFeatures()
        result = fg.compute(basketball_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, basketball_df):
        fg = MatchupFeatures()
        result = fg.compute(basketball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_divisional_flag_binary(self, basketball_df):
        fg = MatchupFeatures()
        result = fg.compute(basketball_df)
        div = result["bkm_divisional_flag"].dropna()
        assert set(div.unique()).issubset({0.0, 1.0})

    def test_h2h_margin_exists_for_rematches(self, basketball_df):
        fg = MatchupFeatures()
        result = fg.compute(basketball_df)
        # Last BOS vs LAL should have H2H data since they played earlier
        h2h = result["bkm_historical_margin"].dropna()
        assert len(h2h) > 0


class TestBasketballMarketFeatures:
    def test_feature_count(self, basketball_df):
        fg = BasketballMarketFeatures()
        result = fg.compute(basketball_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, basketball_df):
        fg = BasketballMarketFeatures()
        result = fg.compute(basketball_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_implied_probabilities_sum_to_one(self, basketball_df):
        fg = BasketballMarketFeatures()
        result = fg.compute(basketball_df)
        for i in result.index:
            p_h = result.loc[i, "bkmkt_implied_home"]
            p_a = result.loc[i, "bkmkt_implied_away"]
            if pd.notna(p_h) and pd.notna(p_a):
                assert abs(p_h + p_a - 1.0) < 0.01

    def test_overround_positive(self, basketball_df):
        fg = BasketballMarketFeatures()
        result = fg.compute(basketball_df)
        overround = result["bkmkt_overround"].dropna()
        assert (overround > 0).all()
        assert len(overround) > 0

    def test_spread_line_present(self, basketball_df):
        fg = BasketballMarketFeatures()
        result = fg.compute(basketball_df)
        spread = result["bkmkt_spread_line"].dropna()
        assert len(spread) > 0


class TestBasketballFeaturePipeline:
    def test_total_features(self):
        pipeline = BasketballFeaturePipeline()
        assert pipeline.total_features == 44

    def test_all_feature_names(self):
        pipeline = BasketballFeaturePipeline()
        names = pipeline.get_all_feature_names()
        assert len(names) == 44
        assert len(set(names)) == 44  # No duplicates

    def test_build_produces_correct_shape(self, basketball_df):
        pipeline = BasketballFeaturePipeline()
        result = pipeline.build(basketball_df)
        assert result.shape == (len(basketball_df), 44)

    def test_build_same_index(self, basketball_df):
        pipeline = BasketballFeaturePipeline()
        result = pipeline.build(basketball_df)
        assert list(result.index) == list(basketball_df.index)

    def test_build_column_names_match(self, basketball_df):
        pipeline = BasketballFeaturePipeline()
        result = pipeline.build(basketball_df)
        assert list(result.columns) == pipeline.get_all_feature_names()

    def test_no_look_ahead_first_game(self, basketball_df):
        """First game should have mostly NaN features."""
        pipeline = BasketballFeaturePipeline()
        result = pipeline.build(basketball_df)
        first = result.iloc[0]
        # Rolling/historical features should be NaN
        assert pd.isna(first["bkp_pace_home"])
        assert pd.isna(first["bke_off_rating_home"])

    def test_missing_data_handling(self):
        """Pipeline should handle DataFrame with minimal columns gracefully."""
        minimal_df = pd.DataFrame({
            "game_date": pd.date_range("2025-01-01", periods=5),
            "home_team": ["A", "B", "A", "C", "B"],
            "away_team": ["B", "C", "C", "A", "A"],
            "home_score": [110, 105, 115, 98, 108],
            "away_score": [105, 110, 100, 112, 102],
            "home_win": [1, 0, 1, 0, 1],
            "season": ["2024-25"] * 5,
        })
        pipeline = BasketballFeaturePipeline()
        result = pipeline.build(minimal_df)
        assert result.shape == (5, 44)
