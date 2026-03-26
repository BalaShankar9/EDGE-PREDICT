"""Tests for AutoFeatureGenerator (tsfresh wrapper)."""
import numpy as np
import pandas as pd
import pytest

from sharpedge.intelligence.auto_features import AutoFeatureGenerator


@pytest.fixture
def sample_matches():
    """Create a small synthetic match DataFrame."""
    rng = np.random.RandomState(42)
    n = 40
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    data = {
        "home_team_id": rng.choice(teams, n),
        "FTHG": rng.randint(0, 4, n),
        "FTAG": rng.randint(0, 3, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def generator():
    return AutoFeatureGenerator(window=10, n_jobs=1)


class TestAutoFeatureGenerator:
    def test_init_defaults(self):
        gen = AutoFeatureGenerator()
        assert gen.window == 10
        assert gen.n_jobs == 1

    def test_init_custom(self):
        gen = AutoFeatureGenerator(window=5, n_jobs=2)
        assert gen.window == 5
        assert gen.n_jobs == 2

    def test_generate_produces_features(self, generator, sample_matches):
        result = generator.generate(sample_matches, team_col="home_team_id", value_col="FTHG")
        assert not result.empty
        assert result.shape[0] > 0  # at least one team
        assert result.shape[1] > 0  # at least one feature

    def test_generate_correct_prefix(self, generator, sample_matches):
        result = generator.generate(sample_matches, team_col="home_team_id", value_col="FTHG")
        for col in result.columns:
            assert col.startswith("tsf_FTHG_"), f"Column {col} missing prefix"

    def test_generate_multi_merges_columns(self, generator, sample_matches):
        result = generator.generate_multi(sample_matches, value_cols=["FTHG", "FTAG"])
        assert not result.empty
        has_fthg = any(c.startswith("tsf_FTHG_") for c in result.columns)
        has_ftag = any(c.startswith("tsf_FTAG_") for c in result.columns)
        assert has_fthg, "Missing FTHG features"
        assert has_ftag, "Missing FTAG features"

    def test_generate_multi_default_cols(self, generator, sample_matches):
        """Default value_cols should use FTHG and FTAG if present."""
        result = generator.generate_multi(sample_matches)
        assert not result.empty

    def test_handles_missing_column(self, generator, sample_matches):
        """Requesting a non-existent value column should not crash."""
        result = generator.generate(
            sample_matches, team_col="home_team_id", value_col="nonexistent"
        )
        # tsfresh should still produce features (value defaults to 0)
        # The generate method uses row.get(value_col, 0) so it won't crash
        assert isinstance(result, pd.DataFrame)

    def test_handles_empty_dataframe(self, generator):
        empty_df = pd.DataFrame(columns=["home_team_id", "FTHG"])
        result = generator.generate(empty_df, team_col="home_team_id", value_col="FTHG")
        assert result.empty

    def test_generate_multi_missing_cols(self, generator):
        """generate_multi with no matching columns returns empty."""
        df = pd.DataFrame({"home_team_id": ["A", "B"], "other": [1, 2]})
        result = generator.generate_multi(df)
        assert result.empty

    def test_no_nan_in_output(self, generator, sample_matches):
        """tsfresh impute should remove NaNs."""
        result = generator.generate(sample_matches, team_col="home_team_id", value_col="FTHG")
        if not result.empty:
            assert not result.isna().any().any(), "Output contains NaN values"
