import numpy as np
import pytest
from sharpedge.ml.models.xgboost_model import XGBoostPredictor


@pytest.fixture
def synthetic_data():
    """Generate synthetic training data."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 10))
    # Make labels somewhat dependent on features
    scores = X[:, 0] + 0.5 * X[:, 1]
    y_1x2 = np.where(scores > 0.5, "H", np.where(scores < -0.5, "A", "D"))
    y_ou = (X[:, 2] + X[:, 3] > 0).astype(int)
    y_btts = (X[:, 4] > 0).astype(int)
    return X, y_1x2, y_ou, y_btts


def test_xgboost_fit_predict(synthetic_data):
    X, y_1x2, y_ou, y_btts = synthetic_data
    model = XGBoostPredictor(params={"n_estimators": 50, "max_depth": 3})
    model.fit(X, y_1x2, y_ou, y_btts)

    proba = model.predict_proba_1x2(X[:5])
    assert proba.shape == (5, 3)
    # Probabilities should sum to ~1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)


def test_xgboost_predict_labels(synthetic_data):
    X, y_1x2, _, _ = synthetic_data
    model = XGBoostPredictor(params={"n_estimators": 50, "max_depth": 3})
    model.fit(X, y_1x2)

    labels = model.predict_1x2(X[:10])
    assert all(l in ("H", "D", "A") for l in labels)


def test_xgboost_ou_prediction(synthetic_data):
    X, y_1x2, y_ou, _ = synthetic_data
    model = XGBoostPredictor(params={"n_estimators": 50, "max_depth": 3})
    model.fit(X, y_1x2, y_ou=y_ou)

    ou_proba = model.predict_proba_ou(X[:5])
    assert ou_proba.shape == (5,)
    assert all(0 <= p <= 1 for p in ou_proba)


def test_xgboost_btts_prediction(synthetic_data):
    X, y_1x2, _, y_btts = synthetic_data
    model = XGBoostPredictor(params={"n_estimators": 50, "max_depth": 3})
    model.fit(X, y_1x2, y_btts=y_btts)

    btts_proba = model.predict_proba_btts(X[:5])
    assert btts_proba.shape == (5,)
    assert all(0 <= p <= 1 for p in btts_proba)


def test_xgboost_ou_not_trained_error(synthetic_data):
    X, y_1x2, _, _ = synthetic_data
    model = XGBoostPredictor(params={"n_estimators": 50, "max_depth": 3})
    model.fit(X, y_1x2)  # No O/U
    with pytest.raises(ValueError, match="O/U model not trained"):
        model.predict_proba_ou(X[:5])
