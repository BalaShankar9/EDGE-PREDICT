"""Tests for BasketballTrainer."""

import numpy as np
import pytest

from sharpedge.sports.basketball.training.trainer import BasketballTrainer


@pytest.fixture
def trained_trainer():
    """Return a BasketballTrainer fitted on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y_margin = np.random.randn(200) * 10
    trainer = BasketballTrainer()
    trainer.train(X, y_margin)
    return trainer, X


class TestBasketballTrainer:
    def test_train_fits_model(self, trained_trainer):
        trainer, _ = trained_trainer
        assert trainer.spread_model is not None
        assert trainer.spread_model._fitted

    def test_predict_returns_all_markets(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:5])
        assert "moneyline_probs" in result
        assert "spread_probs" in result
        assert "total_probs" in result
        assert "predicted_margin" in result

    def test_moneyline_probs_shape(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:5])
        assert result["moneyline_probs"].shape == (5, 2)

    def test_spread_probs_shape(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:5], spread_line=-5.0)
        assert result["spread_probs"].shape == (5, 2)

    def test_total_probs_shape(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:5], total_line=215.0)
        assert result["total_probs"].shape == (5, 2)

    def test_predicted_margin_shape(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:5])
        assert result["predicted_margin"].shape == (5,)

    def test_all_probs_sum_to_one(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:10])
        for key in ["moneyline_probs", "spread_probs", "total_probs"]:
            sums = result[key].sum(axis=1)
            np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_all_probs_bounded(self, trained_trainer):
        trainer, X = trained_trainer
        result = trainer.predict(X[:10])
        for key in ["moneyline_probs", "spread_probs", "total_probs"]:
            assert (result[key] >= 0).all()
            assert (result[key] <= 1).all()

    def test_predict_without_train_raises(self):
        trainer = BasketballTrainer()
        X = np.random.randn(5, 10)
        with pytest.raises(RuntimeError, match="not trained"):
            trainer.predict(X)

    def test_different_spread_lines(self, trained_trainer):
        trainer, X = trained_trainer
        r1 = trainer.predict(X[:5], spread_line=0.0)
        r2 = trainer.predict(X[:5], spread_line=-7.5)
        # With a home-favored spread, moneyline should differ from spread probs
        assert not np.allclose(r1["spread_probs"], r2["spread_probs"])
