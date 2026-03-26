"""Tests for ice hockey models: HockeyGoalModel."""

import numpy as np
import pytest

from sharpedge.sports.ice_hockey.models.goal_model import HockeyGoalModel


@pytest.fixture
def trained_gbm_model():
    """Return a HockeyGoalModel fitted on synthetic data (GBM mode)."""
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y_margin = np.random.randn(200) * 2  # hockey margins are smaller
    model = HockeyGoalModel()
    model.fit(X=X, y_margin=y_margin)
    return model, X


@pytest.fixture
def trained_poisson_model():
    """Return a HockeyGoalModel fitted on synthetic match data (Poisson mode)."""
    matches = []
    teams = ["BOS", "TOR", "NYR", "TBL"]
    np.random.seed(42)
    for _ in range(100):
        h, a = np.random.choice(teams, 2, replace=False)
        matches.append({
            "home_team": h,
            "away_team": a,
            "home_score": np.random.randint(1, 6),
            "away_score": np.random.randint(0, 5),
        })
    model = HockeyGoalModel()
    model.fit(matches=matches)
    return model


class TestHockeyGoalModel:
    def test_name(self):
        assert HockeyGoalModel().name == "hockey_goal_model"

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
        probs = trained_poisson_model.predict_proba(home_team="BOS", away_team="TOR")
        assert probs.shape == (1, 2)

    def test_poisson_probabilities_sum_to_one(self, trained_poisson_model):
        probs = trained_poisson_model.predict_proba(home_team="BOS", away_team="TOR")
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_poisson_probabilities_bounded(self, trained_poisson_model):
        probs = trained_poisson_model.predict_proba(home_team="BOS", away_team="TOR")
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
        probs_normal = model.predict_total(X=X[:10], total_line=5.5)
        probs_high = model.predict_total(X=X[:10], total_line=8.5)
        assert (probs_high[:, 0] <= probs_normal[:, 0] + 0.01).all()

    def test_no_draws_in_hockey(self, trained_poisson_model):
        """Hockey model should never predict a draw — always home or away."""
        probs = trained_poisson_model.predict_proba(home_team="BOS", away_team="TOR")
        assert probs.shape[1] == 2  # Only home and away, no draw column
