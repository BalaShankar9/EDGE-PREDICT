"""Tests for tennis feature engineering modules."""

import pandas as pd
import numpy as np
import pytest

from sharpedge.sports.tennis.features.ranking import RankingFeatures
from sharpedge.sports.tennis.features.surface import SurfaceFeatures
from sharpedge.sports.tennis.features.fatigue import FatigueFeatures
from sharpedge.sports.tennis.features.h2h import H2HFeatures
from sharpedge.sports.tennis.features.serve import ServeFeatures
from sharpedge.sports.tennis.features.market import TennisMarketFeatures
from sharpedge.sports.tennis.features.pipeline import TennisFeaturePipeline


def _make_tennis_df() -> pd.DataFrame:
    """Create a synthetic tennis DataFrame with ~30 matches."""
    data = []
    players = ["Djokovic", "Nadal", "Federer", "Medvedev", "Zverev", "Alcaraz"]
    surfaces = ["Hard", "Clay", "Grass", "Hard", "Clay", "Hard"]
    base_date = pd.Timestamp("2025-01-01")

    matchups = [
        # (p1, p2, winner, surface, w_rank, l_rank, w_pts, l_pts, score, day_offset)
        ("Djokovic", "Nadal", "Djokovic", "Hard", 1, 3, 9000, 7000, "6-3 6-4", 0),
        ("Federer", "Medvedev", "Medvedev", "Hard", 8, 4, 4000, 6500, "3-6 6-3 6-4", 1),
        ("Zverev", "Alcaraz", "Alcaraz", "Clay", 6, 2, 5000, 8000, "6-7 6-3 6-2", 2),
        ("Nadal", "Federer", "Nadal", "Clay", 3, 8, 7000, 4000, "6-2 6-1", 3),
        ("Djokovic", "Medvedev", "Djokovic", "Hard", 1, 4, 9200, 6500, "7-5 6-3", 4),
        ("Alcaraz", "Djokovic", "Djokovic", "Clay", 2, 1, 8000, 9200, "4-6 6-3 3-6", 5),
        ("Nadal", "Zverev", "Nadal", "Clay", 3, 6, 7200, 5000, "6-4 6-2", 6),
        ("Federer", "Alcaraz", "Alcaraz", "Grass", 8, 2, 3800, 8200, "6-7 3-6", 7),
        ("Djokovic", "Nadal", "Nadal", "Clay", 1, 3, 9200, 7500, "3-6 6-3 4-6", 8),
        ("Medvedev", "Zverev", "Medvedev", "Hard", 4, 6, 6800, 5000, "6-4 7-5", 10),
        ("Djokovic", "Federer", "Djokovic", "Hard", 1, 9, 9500, 3500, "6-3 6-2", 12),
        ("Nadal", "Alcaraz", "Alcaraz", "Clay", 3, 2, 7500, 8500, "6-7 6-4 4-6", 14),
        ("Zverev", "Federer", "Zverev", "Hard", 5, 9, 5200, 3500, "6-4 6-3", 15),
        ("Djokovic", "Zverev", "Djokovic", "Hard", 1, 5, 9800, 5200, "6-2 6-4", 16),
        ("Nadal", "Medvedev", "Nadal", "Clay", 3, 4, 7800, 6800, "6-3 6-1", 17),
        ("Alcaraz", "Federer", "Alcaraz", "Hard", 2, 10, 8800, 3200, "6-2 6-3", 18),
        ("Djokovic", "Alcaraz", "Alcaraz", "Hard", 1, 2, 9800, 9000, "6-7 6-4 7-6", 20),
        ("Nadal", "Federer", "Nadal", "Clay", 3, 10, 8000, 3000, "6-1 6-2", 22),
        ("Medvedev", "Alcaraz", "Alcaraz", "Hard", 4, 2, 6500, 9200, "4-6 3-6", 24),
        ("Djokovic", "Nadal", "Djokovic", "Hard", 1, 3, 10000, 8000, "6-4 7-5", 26),
        ("Zverev", "Nadal", "Nadal", "Clay", 5, 3, 5000, 8200, "4-6 6-7", 27),
        ("Federer", "Zverev", "Federer", "Grass", 10, 5, 3000, 5000, "7-6 6-4", 28),
        ("Djokovic", "Medvedev", "Djokovic", "Hard", 1, 4, 10200, 6500, "6-3 6-4", 30),
        ("Alcaraz", "Nadal", "Alcaraz", "Hard", 2, 3, 9500, 8200, "7-6 6-3", 32),
        ("Federer", "Medvedev", "Medvedev", "Hard", 11, 4, 2800, 6800, "3-6 4-6", 33),
        ("Djokovic", "Alcaraz", "Djokovic", "Clay", 1, 2, 10500, 9500, "6-4 3-6 6-3", 35),
        ("Nadal", "Zverev", "Zverev", "Hard", 3, 5, 8200, 5500, "6-7 4-6", 36),
        ("Medvedev", "Nadal", "Nadal", "Clay", 4, 3, 6800, 8500, "4-6 6-3 4-6", 38),
        ("Alcaraz", "Zverev", "Alcaraz", "Hard", 2, 5, 9800, 5500, "6-3 6-4", 40),
        ("Djokovic", "Federer", "Djokovic", "Grass", 1, 12, 10800, 2500, "6-4 6-3", 42),
    ]

    for p1, p2, winner, surface, w_rank, l_rank, w_pts, l_pts, score, offset in matchups:
        loser = p2 if winner == p1 else p1
        data.append({
            "date": base_date + pd.Timedelta(days=offset),
            "player1": p1,
            "player2": p2,
            "winner": winner,
            "loser": loser,
            "surface": surface,
            "winner_rank": w_rank,
            "loser_rank": l_rank,
            "winner_points": w_pts,
            "loser_points": l_pts,
            "score": score,
            "b365_winner": round(1.4 + np.random.random() * 0.5, 2),
            "b365_loser": round(2.5 + np.random.random() * 1.5, 2),
            "ps_winner": round(1.4 + np.random.random() * 0.5, 2),
            "ps_loser": round(2.5 + np.random.random() * 1.5, 2),
        })

    return pd.DataFrame(data)


@pytest.fixture
def tennis_df():
    np.random.seed(42)
    return _make_tennis_df()


class TestRankingFeatures:
    def test_feature_count(self, tennis_df):
        fg = RankingFeatures()
        result = fg.compute(tennis_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, tennis_df):
        fg = RankingFeatures()
        result = fg.compute(tennis_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, tennis_df):
        """First match should have no momentum (no prior data)."""
        fg = RankingFeatures()
        result = fg.compute(tennis_df)
        assert pd.isna(result.loc[0, "tnk_rank_momentum_p1"])
        assert pd.isna(result.loc[0, "tnk_rank_momentum_p2"])

    def test_rank_values(self, tennis_df):
        fg = RankingFeatures()
        result = fg.compute(tennis_df)
        # First match: Djokovic (rank 1, winner) vs Nadal (rank 3, loser)
        # P1 = Djokovic (player1), who is the winner -> p1_rank = winner_rank = 1
        assert result.loc[0, "tnk_rank_p1"] == 1
        assert result.loc[0, "tnk_rank_p2"] == 3
        assert result.loc[0, "tnk_rank_diff"] == -2  # 1 - 3
        assert result.loc[0, "tnk_top10_flag"] == 1.0

    def test_top10_flag(self, tennis_df):
        fg = RankingFeatures()
        result = fg.compute(tennis_df)
        # All matches in our dataset have at least one top-10 player
        assert (result["tnk_top10_flag"].dropna() == 1.0).all()


class TestSurfaceFeatures:
    def test_feature_count(self, tennis_df):
        fg = SurfaceFeatures()
        result = fg.compute(tennis_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, tennis_df):
        fg = SurfaceFeatures()
        result = fg.compute(tennis_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, tennis_df):
        """First match should have NaN surface stats (no prior data)."""
        fg = SurfaceFeatures()
        result = fg.compute(tennis_df)
        assert pd.isna(result.loc[0, "tns_surface_win_rate_p1"])

    def test_surface_experience_increases(self, tennis_df):
        """Surface match count should grow over time for active players."""
        fg = SurfaceFeatures()
        result = fg.compute(tennis_df)
        # Get Djokovic's hard court experience at different points
        djok_hard = result.loc[
            (tennis_df["player1"] == "Djokovic") & (tennis_df["surface"] == "Hard"),
            "tns_surface_matches_p1"
        ].dropna()
        if len(djok_hard) >= 2:
            assert djok_hard.iloc[-1] >= djok_hard.iloc[0]


class TestFatigueFeatures:
    def test_feature_count(self, tennis_df):
        fg = FatigueFeatures()
        result = fg.compute(tennis_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, tennis_df):
        fg = FatigueFeatures()
        result = fg.compute(tennis_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, tennis_df):
        """First match should have 0 matches in last 7d and NaN days_since_last."""
        fg = FatigueFeatures()
        result = fg.compute(tennis_df)
        assert result.loc[0, "tnf_matches_7d_p1"] == 0
        assert pd.isna(result.loc[0, "tnf_days_since_last_p1"])

    def test_match_counts_reasonable(self, tennis_df):
        fg = FatigueFeatures()
        result = fg.compute(tennis_df)
        # 7-day counts should be <= 28-day counts
        for i in result.index:
            m7 = result.loc[i, "tnf_matches_7d_p1"]
            m28 = result.loc[i, "tnf_matches_28d_p1"]
            if pd.notna(m7) and pd.notna(m28):
                assert m7 <= m28


class TestH2HFeatures:
    def test_feature_count(self, tennis_df):
        fg = H2HFeatures()
        result = fg.compute(tennis_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, tennis_df):
        fg = H2HFeatures()
        result = fg.compute(tennis_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_no_look_ahead(self, tennis_df):
        """First match between any pair should have no H2H data."""
        fg = H2HFeatures()
        result = fg.compute(tennis_df)
        assert pd.isna(result.loc[0, "tnh_h2h_win_rate_p1"])

    def test_h2h_count_grows(self, tennis_df):
        """H2H count should increase for repeated matchups."""
        fg = H2HFeatures()
        result = fg.compute(tennis_df)
        # Djokovic vs Nadal appears multiple times
        djok_nadal = result.loc[
            ((tennis_df["player1"] == "Djokovic") & (tennis_df["player2"] == "Nadal"))
            | ((tennis_df["player1"] == "Nadal") & (tennis_df["player2"] == "Djokovic")),
            "tnh_h2h_count"
        ].dropna()
        if len(djok_nadal) >= 2:
            assert djok_nadal.iloc[-1] >= djok_nadal.iloc[0]

    def test_win_rate_bounded(self, tennis_df):
        fg = H2HFeatures()
        result = fg.compute(tennis_df)
        wr = result["tnh_h2h_win_rate_p1"].dropna()
        assert (wr >= 0).all() and (wr <= 1).all()


class TestServeFeatures:
    def test_feature_count(self, tennis_df):
        fg = ServeFeatures()
        result = fg.compute(tennis_df)
        assert result.shape[1] == fg.feature_count == 8

    def test_feature_names_match(self, tennis_df):
        fg = ServeFeatures()
        result = fg.compute(tennis_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_missing_serve_columns_returns_nan(self, tennis_df):
        """Without serve stat columns, all features should be NaN."""
        fg = ServeFeatures()
        result = fg.compute(tennis_df)
        # Our test data doesn't have w_ace etc., so all should be NaN
        assert result.isna().all().all()

    def test_with_serve_data(self, tennis_df):
        """When serve columns exist, features should be computed."""
        df = tennis_df.copy()
        df["w_ace"] = np.random.randint(3, 15, size=len(df))
        df["l_ace"] = np.random.randint(1, 10, size=len(df))
        df["w_df"] = np.random.randint(1, 5, size=len(df))
        df["l_df"] = np.random.randint(1, 8, size=len(df))
        df["w_1stIn"] = np.random.randint(40, 70, size=len(df))
        df["w_svpt"] = np.random.randint(80, 120, size=len(df))
        df["l_1stIn"] = np.random.randint(35, 65, size=len(df))
        df["l_svpt"] = np.random.randint(80, 120, size=len(df))
        df["w_bpSaved"] = np.random.randint(2, 8, size=len(df))
        df["w_bpFaced"] = np.random.randint(5, 15, size=len(df))
        df["l_bpSaved"] = np.random.randint(1, 6, size=len(df))
        df["l_bpFaced"] = np.random.randint(5, 15, size=len(df))

        fg = ServeFeatures()
        result = fg.compute(df)
        # Later matches should have non-NaN serve features
        assert not result.iloc[-1].isna().all()


class TestTennisMarketFeatures:
    def test_feature_count(self, tennis_df):
        fg = TennisMarketFeatures()
        result = fg.compute(tennis_df)
        assert result.shape[1] == fg.feature_count == 6

    def test_feature_names_match(self, tennis_df):
        fg = TennisMarketFeatures()
        result = fg.compute(tennis_df)
        assert list(result.columns) == fg.get_feature_names()

    def test_implied_probabilities_sum_to_one(self, tennis_df):
        fg = TennisMarketFeatures()
        result = fg.compute(tennis_df)
        for i in result.index:
            p1 = result.loc[i, "tnm_implied_p1"]
            p2 = result.loc[i, "tnm_implied_p2"]
            if pd.notna(p1) and pd.notna(p2):
                assert abs(p1 + p2 - 1.0) < 0.01

    def test_overround_positive(self, tennis_df):
        fg = TennisMarketFeatures()
        result = fg.compute(tennis_df)
        overround = result["tnm_overround"].dropna()
        # Overround should always be positive (sum of implied probs * 100)
        assert (overround > 0).all()
        assert len(overround) > 0

    def test_odds_positive(self, tennis_df):
        fg = TennisMarketFeatures()
        result = fg.compute(tennis_df)
        for col in ["tnm_best_odds_p1", "tnm_best_odds_p2"]:
            vals = result[col].dropna()
            assert (vals > 0).all()


class TestTennisFeaturePipeline:
    def test_total_features(self):
        pipeline = TennisFeaturePipeline()
        assert pipeline.total_features == 40

    def test_all_feature_names(self):
        pipeline = TennisFeaturePipeline()
        names = pipeline.get_all_feature_names()
        assert len(names) == 40
        assert len(set(names)) == 40  # No duplicates

    def test_build_produces_correct_shape(self, tennis_df):
        pipeline = TennisFeaturePipeline()
        result = pipeline.build(tennis_df)
        assert result.shape == (len(tennis_df), 40)

    def test_build_same_index(self, tennis_df):
        pipeline = TennisFeaturePipeline()
        result = pipeline.build(tennis_df)
        assert list(result.index) == list(tennis_df.index)

    def test_build_column_names_match(self, tennis_df):
        pipeline = TennisFeaturePipeline()
        result = pipeline.build(tennis_df)
        assert list(result.columns) == pipeline.get_all_feature_names()

    def test_no_look_ahead_first_match(self, tennis_df):
        """First match should have mostly NaN features (no prior history)."""
        pipeline = TennisFeaturePipeline()
        result = pipeline.build(tennis_df)
        first = result.iloc[0]
        # Ranking features based on current match data should be present
        assert pd.notna(first["tnk_rank_p1"])
        # But rolling/historical features should be NaN
        assert pd.isna(first["tns_surface_win_rate_p1"])
        assert pd.isna(first["tnh_h2h_win_rate_p1"])

    def test_missing_data_handling(self):
        """Pipeline should handle DataFrame with missing columns gracefully."""
        minimal_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=5),
            "player1": ["A", "B", "A", "C", "B"],
            "player2": ["B", "C", "C", "A", "A"],
            "winner": ["A", "B", "A", "C", "A"],
            "loser": ["B", "C", "C", "A", "B"],
            "surface": ["Hard"] * 5,
            "winner_rank": [1, 5, 1, 10, 5],
            "loser_rank": [5, 10, 10, 1, 1],
            "winner_points": [9000, 5000, 9000, 2000, 5000],
            "loser_points": [5000, 2000, 2000, 9000, 9000],
            "score": ["6-3 6-4"] * 5,
            "b365_winner": [1.5, 1.8, 1.4, 2.0, 1.6],
            "b365_loser": [2.5, 2.0, 3.0, 1.8, 2.4],
            "ps_winner": [1.5, 1.8, 1.4, 2.0, 1.6],
            "ps_loser": [2.5, 2.0, 3.0, 1.8, 2.4],
        })
        pipeline = TennisFeaturePipeline()
        result = pipeline.build(minimal_df)
        assert result.shape == (5, 40)
