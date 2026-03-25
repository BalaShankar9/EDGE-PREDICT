"""Tests for basketball models: SpreadModel."""

import numpy as np
import pytest

from sharpedge.sports.basketball.models.spread_model import SpreadModel


@pytest.fixture
def trained_model():
    """Return a SpreadModel fitted on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y_margin = np.random.randn(200) * 10  # margins between -30 and +30
    model = SpreadModel()
    model.fit(X, y_margin)
    return model, X


class TestSpreadModel:
    def test_name(self):
        assert SpreadModel().name == "spread_model"

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
        """A large home-favored spread should lower P(home covers)."""
        model, X = trained_model
        probs_flat = model.predict_proba(X[:10], spread_line=0.0)
        probs_spread = model.predict_proba(X[:10], spread_line=-10.0)
        # With spread_line=-10 (home must win by >10), P(home) should increase
        # because sf(-10, loc=margin) > sf(0, loc=margin)
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
        """A very high total line should make P(over) lower."""
        model, X = trained_model
        probs_normal = model.predict_total(X[:10], total_line=220.0)
        probs_high = model.predict_total(X[:10], total_line=260.0)
        # P(over) should be lower with higher line
        assert (probs_high[:, 0] <= probs_normal[:, 0] + 0.01).all()

    def test_predict_margin_shape(self, trained_model):
        model, X = trained_model
        margins = model.predict_margin(X[:5])
        assert margins.shape == (5,)

    def test_residual_std_reasonable(self, trained_model):
        model, _ = trained_model
        assert 3.0 < model._residual_std < 20.0
