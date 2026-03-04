import numpy as np
import pytest
from sharpedge.ml.models.ensemble import EnsemblePredictor


@pytest.fixture
def two_model_predictions():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 3, size=n)
    # Model A: decent predictions
    model_a = rng.dirichlet([2, 1, 1], size=n)
    # Model B: slightly worse
    model_b = rng.dirichlet([1, 1, 1], size=n)
    return {"model_a": model_a, "model_b": model_b}, y_true


def test_ensemble_weights_sum_to_one(two_model_predictions):
    preds, y_true = two_model_predictions
    ens = EnsemblePredictor()
    weights = ens.fit_weights(preds, y_true)
    assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0


def test_ensemble_predict_shape(two_model_predictions):
    preds, y_true = two_model_predictions
    ens = EnsemblePredictor()
    ens.fit_weights(preds, y_true)
    result = ens.predict(preds)
    assert result.shape == (100, 3)


def test_ensemble_probabilities_sum_to_one(two_model_predictions):
    preds, y_true = two_model_predictions
    ens = EnsemblePredictor()
    ens.fit_weights(preds, y_true)
    result = ens.predict(preds)
    assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)


def test_ensemble_single_model():
    rng = np.random.default_rng(42)
    preds = {"only_model": rng.dirichlet([1, 1, 1], size=50)}
    y_true = rng.integers(0, 3, size=50)
    ens = EnsemblePredictor()
    weights = ens.fit_weights(preds, y_true)
    assert weights["only_model"] == 1.0
