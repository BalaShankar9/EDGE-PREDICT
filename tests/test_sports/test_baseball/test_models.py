"""Tests for baseball models: RunModel."""

import numpy as np
import pytest

from sharpedge.sports.baseball.models.run_model import RunModel


@pytest.fixture
def trained_gbm_model():
    """Return a RunModel fitted on synthetic data (GBM mode)."""
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y_margin = np.random.randn(200) * 3  # baseball margins are smaller
    model = RunModel()
    model.fit(X=X, y_margin=y_margin)
    return model, X


@pytest.fixture
def trained_poisson_model():
    """Return a RunModel fitted on synthetic match data (Poisson mode)."""
    matches = []
    teams = ["NYY", "BOS", "LAD", "HOU"]
    np.random.seed(42)
    for _ in range(100):
        h, a = np.random.choice(teams, 2, replace=False)
        matches.append({
            "home_team": h,
            "away_team": a,
            "home_score": np.random.randint(1, 10),
            "away_score": np.random.randint(0, 8),
        })
    model = RunModel()
    model.fit(matches=matches)
    return model


class TestRunModel:
    def test_name(self):
        assert RunModel().name == "baseball_run_model"

    def test_gbm_fit_sets_fitted_flag(self, trained_gbm_model):
        model, _ = trained_gbm_model
        assert model._fitted is True

    def test_gbm_predict_proba_shape(self, trained_gbm_model):
        model, X = trained_gbm_model
        probs = model.predict_proba(X=X[:5])
        assert probs.shape == (5, 2)

    def test_gbm_probabilities_sum_to_one(self, trained_gbm_model):
        model, X = trained_gbm_model
        probs = model.predict_proba(X=X[:20])
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_gbm_probabilities_bounded(self, trained_gbm_model):
        model, X = trained_gbm_model
        probs = model.predict_proba(X=X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_poisson_fit_sets_fitted_flag(self, trained_poisson_model):
        assert trained_poisson_model._fitted is True

    def test_poisson_predict_proba_shape(self, trained_poisson_model):
        probs = trained_poisson_model.predict_proba(home_team="NYY", away_team="BOS")
        assert probs.shape == (1, 2)

    def test_poisson_probabilities_sum_to_one(self, trained_poisson_model):
        probs = trained_poisson_model.predict_proba(home_team="NYY", away_team="BOS")
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_poisson_probabilities_bounded(self, trained_poisson_model):
        probs = trained_poisson_model.predict_proba(home_team="NYY", away_team="BOS")
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_total_shape(self, trained_gbm_model):
        model, X = trained_gbm_model
        probs = model.predict_total(X=X[:5])
        assert probs.shape == (5, 2)

    def test_predict_total_sum_to_one(self, trained_gbm_model):
        model, X = trained_gbm_model
        probs = model.predict_total(X=X[:20])
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_higher_total_line_lowers_over_prob(self, trained_gbm_model):
        model, X = trained_gbm_model
        probs_normal = model.predict_total(X=X[:10], total_line=8.5)
        probs_high = model.predict_total(X=X[:10], total_line=14.5)
        assert (probs_high[:, 0] <= probs_normal[:, 0] + 0.01).all()

    def test_two_outcome_only(self, trained_poisson_model):
        """Baseball model should predict 2 outcomes (home/away)."""
        probs = trained_poisson_model.predict_proba(home_team="NYY", away_team="BOS")
        assert probs.shape[1] == 2
