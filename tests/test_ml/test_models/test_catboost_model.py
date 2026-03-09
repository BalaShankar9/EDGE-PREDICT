import numpy as np
import pytest
from sharpedge.ml.models.catboost_model import CatBoostPredictor


@pytest.fixture
def trained_model():
    np.random.seed(42)
    X = np.random.randn(500, 20).astype(np.float32)
    y_1x2 = np.array(["H", "D", "A"] * 166 + ["H", "D"])
    y_ou = (np.random.rand(500) > 0.5).astype(int)
    y_btts = (np.random.rand(500) > 0.5).astype(int)
    model = CatBoostPredictor()
    model.fit(X, y_1x2, y_ou=y_ou, y_btts=y_btts)
    return model, X


def test_predict_1x2_shape(trained_model):
    model, X = trained_model
    proba = model.predict_proba_1x2(X[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_ou(trained_model):
    model, X = trained_model
    proba = model.predict_proba_ou(X[:5])
    assert proba.shape == (5,)
    assert all(0 <= p <= 1 for p in proba)


def test_predict_btts(trained_model):
    model, X = trained_model
    proba = model.predict_proba_btts(X[:5])
    assert proba.shape == (5,)
    assert all(0 <= p <= 1 for p in proba)


def test_ou_returns_half_when_not_fitted():
    """If y_ou not provided, predict_proba_ou returns 0.5."""
    X = np.random.randn(100, 20).astype(np.float32)
    y_1x2 = np.array(["H", "D", "A"] * 33 + ["H"])
    model = CatBoostPredictor()
    model.fit(X, y_1x2)
    proba = model.predict_proba_ou(X[:5])
    assert np.allclose(proba, 0.5)


def test_btts_returns_half_when_not_fitted():
    """If y_btts not provided, predict_proba_btts returns 0.5."""
    X = np.random.randn(100, 20).astype(np.float32)
    y_1x2 = np.array(["H", "D", "A"] * 33 + ["H"])
    model = CatBoostPredictor()
    model.fit(X, y_1x2)
    proba = model.predict_proba_btts(X[:5])
    assert np.allclose(proba, 0.5)


def test_custom_params():
    """Custom params override defaults."""
    model = CatBoostPredictor(params={"iterations": 50, "depth": 3})
    assert model.params["iterations"] == 50
    assert model.params["depth"] == 3
    # Defaults should still be present
    assert model.params["learning_rate"] == 0.05
    assert model.params["random_seed"] == 42
