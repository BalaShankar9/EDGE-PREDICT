"""Tests for NFL models: NFLSpreadModel."""

import numpy as np
import pytest

from sharpedge.sports.american_football.models.spread_model import NFLSpreadModel


@pytest.fixture
def trained_model():
    """Return an NFLSpreadModel fitted on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y_margin = np.random.randn(200) * 14  # NFL margins have higher variance
    model = NFLSpreadModel()
    model.fit(X, y_margin)
    return model, X


class TestNFLSpreadModel:
    def test_name(self):
        assert NFLSpreadModel().name == "nfl_spread_model"

    def test_fit_sets_fitted_flag(self, trained_model):
        model, _ = trained_model
        assert model._fitted is True

    def test_predict_proba_shape(self, trained_model):
        model, X = trained_model
        probs = model.predict_proba(X[:5])
        assert probs.shape == (5, 2)

    def test_probabilities_sum_to_one(self, trained_model):
        model, X = trained_model
        probs = model.predict_proba(X[:20])
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_probabilities_bounded(self, trained_model):
        model, X = trained_model
        probs = model.predict_proba(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_spread_line_shifts_probabilities(self, trained_model):
        model, X = trained_model
        probs_flat = model.predict_proba(X[:10], spread_line=0.0)
        probs_spread = model.predict_proba(X[:10], spread_line=-10.0)
        assert (probs_spread[:, 0] >= probs_flat[:, 0] - 0.01).all()

    def test_predict_total_shape(self, trained_model):
        model, X = trained_model
        probs = model.predict_total(X[:5])
        assert probs.shape == (5, 2)

    def test_predict_total_sum_to_one(self, trained_model):
        model, X = trained_model
        probs = model.predict_total(X[:20])
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_predict_total_bounded(self, trained_model):
        model, X = trained_model
        probs = model.predict_total(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_higher_total_line_lowers_over_prob(self, trained_model):
        model, X = trained_model
        probs_normal = model.predict_total(X[:10], total_line=45.0)
        probs_high = model.predict_total(X[:10], total_line=60.0)
        assert (probs_high[:, 0] <= probs_normal[:, 0] + 0.01).all()

    def test_predict_margin_shape(self, trained_model):
        model, X = trained_model
        margins = model.predict_margin(X[:5])
        assert margins.shape == (5,)

    def test_residual_std_reasonable(self, trained_model):
        model, _ = trained_model
        assert 5.0 < model._residual_std < 25.0
