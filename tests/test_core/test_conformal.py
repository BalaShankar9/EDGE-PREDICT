"""Tests for ConformalPredictor (MAPIE wrapper)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from sharpedge.core.conformal import ConformalPredictor


@pytest.fixture
def synthetic_data():
    """Generate synthetic 3-class classification data split into train/cal."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5)
    y = rng.choice(3, 300)
    # Split: first 150 for training base model, last 150 for calibration
    return {
        "X_train": X[:150],
        "y_train": y[:150],
        "X_cal": X[150:],
        "y_cal": y[150:],
    }


@pytest.fixture
def fitted_base_model(synthetic_data):
    """A fitted LogisticRegression to use as base_model."""
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(synthetic_data["X_train"], synthetic_data["y_train"])
    return model


@pytest.fixture
def fitted_conformal(synthetic_data, fitted_base_model):
    """A fitted ConformalPredictor (calibrated on calibration set)."""
    cp = ConformalPredictor(confidence_level=0.90)
    cp.fit(synthetic_data["X_cal"], synthetic_data["y_cal"], base_model=fitted_base_model)
    return cp


class TestConformalPredictor:
    def test_init_defaults(self):
        cp = ConformalPredictor()
        assert cp.confidence_level == 0.90
        assert cp._fitted is False

    def test_init_custom_level(self):
        cp = ConformalPredictor(confidence_level=0.95)
        assert cp.confidence_level == 0.95

    def test_fit_marks_fitted(self, fitted_conformal):
        assert fitted_conformal._fitted is True
        assert fitted_conformal._mapie is not None

    def test_predict_sets_shape(self, fitted_conformal, synthetic_data):
        X = synthetic_data["X_cal"]
        n = len(X)
        y_pred, pred_sets = fitted_conformal.predict_sets(X)
        assert y_pred.shape == (n,)
        assert pred_sets.shape == (n, 3)

    def test_predict_sets_boolean(self, fitted_conformal, synthetic_data):
        X = synthetic_data["X_cal"]
        _, pred_sets = fitted_conformal.predict_sets(X)
        assert pred_sets.dtype == bool

    def test_set_sizes_range(self, fitted_conformal, synthetic_data):
        X = synthetic_data["X_cal"]
        sizes = fitted_conformal.get_set_sizes(X)
        assert sizes.shape == (len(X),)
        assert np.all(sizes >= 1)
        assert np.all(sizes <= 3)

    def test_confidence_scores_range(self, fitted_conformal, synthetic_data):
        X = synthetic_data["X_cal"]
        scores = fitted_conformal.get_confidence_score(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_confidence_inversely_related_to_set_size(
        self, fitted_conformal, synthetic_data
    ):
        X = synthetic_data["X_cal"]
        sizes = fitted_conformal.get_set_sizes(X)
        scores = fitted_conformal.get_confidence_score(X)
        np.testing.assert_array_almost_equal(scores, 1.0 / sizes)

    def test_not_fitted_raises(self, synthetic_data):
        X = synthetic_data["X_cal"]
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="Not fitted"):
            cp.predict_sets(X)

    def test_predict_sets_small_batch(self, fitted_conformal):
        X_small = np.random.randn(5, 5)
        y_pred, pred_sets = fitted_conformal.predict_sets(X_small)
        assert y_pred.shape == (5,)
        assert pred_sets.shape == (5, 3)
