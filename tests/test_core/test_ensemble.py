"""Tests for the generalized N-outcome ensemble predictor."""
import numpy as np
import pytest

from sharpedge.core.ensemble import EnsemblePredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_model_2_outcome():
    """2 models, 2 outcomes (tennis-like: win/lose)."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    model_a = rng.dirichlet([3, 1], size=n)
    model_b = rng.dirichlet([1, 1], size=n)
    return {"model_a": model_a, "model_b": model_b}, y_true


@pytest.fixture
def two_model_3_outcome():
    """2 models, 3 outcomes (football: H/D/A)."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 3, size=n)
    model_a = rng.dirichlet([2, 1, 1], size=n)
    model_b = rng.dirichlet([1, 1, 1], size=n)
    return {"model_a": model_a, "model_b": model_b}, y_true


@pytest.fixture
def three_model_2_outcome():
    """3 models, 2 outcomes (tennis-like)."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    model_a = rng.dirichlet([3, 1], size=n)
    model_b = rng.dirichlet([1, 2], size=n)
    model_c = rng.dirichlet([1, 1], size=n)
    return {"model_a": model_a, "model_b": model_b, "model_c": model_c}, y_true


@pytest.fixture
def three_model_3_outcome():
    """3 models, 3 outcomes (football: H/D/A)."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 3, size=n)
    model_a = rng.dirichlet([2, 1, 1], size=n)
    model_b = rng.dirichlet([1, 2, 1], size=n)
    model_c = rng.dirichlet([1, 1, 1], size=n)
    return {"model_a": model_a, "model_b": model_b, "model_c": model_c}, y_true


# ---------------------------------------------------------------------------
# 2-model tests
# ---------------------------------------------------------------------------

class TestTwoModels:

    def test_2_outcome_weights_sum_to_one(self, two_model_2_outcome):
        preds, y_true = two_model_2_outcome
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0

    def test_2_outcome_predict_shape(self, two_model_2_outcome):
        preds, y_true = two_model_2_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert result.shape == (100, 2)

    def test_2_outcome_probabilities_sum_to_one(self, two_model_2_outcome):
        preds, y_true = two_model_2_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)

    def test_3_outcome_weights_sum_to_one(self, two_model_3_outcome):
        preds, y_true = two_model_3_outcome
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0

    def test_3_outcome_predict_shape(self, two_model_3_outcome):
        preds, y_true = two_model_3_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert result.shape == (100, 3)

    def test_3_outcome_probabilities_sum_to_one(self, two_model_3_outcome):
        preds, y_true = two_model_3_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# 3-model tests (SLSQP path)
# ---------------------------------------------------------------------------

class TestThreeModels:

    def test_2_outcome_slsqp_weights_sum_to_one(self, three_model_2_outcome):
        preds, y_true = three_model_2_outcome
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0

    def test_2_outcome_slsqp_predict_shape(self, three_model_2_outcome):
        preds, y_true = three_model_2_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert result.shape == (100, 2)

    def test_2_outcome_slsqp_probabilities_sum_to_one(self, three_model_2_outcome):
        preds, y_true = three_model_2_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)

    def test_3_outcome_slsqp_weights_sum_to_one(self, three_model_3_outcome):
        preds, y_true = three_model_3_outcome
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0

    def test_3_outcome_slsqp_predict_shape(self, three_model_3_outcome):
        preds, y_true = three_model_3_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert result.shape == (100, 3)

    def test_3_outcome_slsqp_probabilities_sum_to_one(self, three_model_3_outcome):
        preds, y_true = three_model_3_outcome
        ens = EnsemblePredictor()
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert np.allclose(result.sum(axis=1), 1.0, atol=0.01)

    def test_3_outcome_slsqp_weights_nonneg(self, three_model_3_outcome):
        preds, y_true = three_model_3_outcome
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        for w in weights.values():
            assert w >= -0.01  # allow small numerical noise


# ---------------------------------------------------------------------------
# Single model
# ---------------------------------------------------------------------------

class TestSingleModel:

    def test_single_model_weight_is_one(self):
        rng = np.random.default_rng(42)
        preds = {"only_model": rng.dirichlet([1, 1, 1], size=50)}
        y_true = rng.integers(0, 3, size=50)
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        assert weights["only_model"] == 1.0

    def test_single_model_2_outcome(self):
        rng = np.random.default_rng(42)
        preds = {"only_model": rng.dirichlet([1, 1], size=50)}
        y_true = rng.integers(0, 2, size=50)
        ens = EnsemblePredictor()
        weights = ens.fit_weights(preds, y_true)
        assert weights["only_model"] == 1.0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_predict_normalizes_rows(self):
        """Even if weights don't perfectly sum to 1, predict normalizes."""
        ens = EnsemblePredictor()
        ens.weights = {"a": 0.6, "b": 0.6}  # sum > 1 deliberately
        preds = {
            "a": np.array([[0.5, 0.3, 0.2]]),
            "b": np.array([[0.3, 0.4, 0.3]]),
        }
        result = ens.predict(preds)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_normalizes_2_outcome(self):
        ens = EnsemblePredictor()
        ens.weights = {"a": 0.7, "b": 0.7}
        preds = {
            "a": np.array([[0.6, 0.4]]),
            "b": np.array([[0.4, 0.6]]),
        }
        result = ens.predict(preds)
        assert result.shape == (1, 2)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Re-export from old location
# ---------------------------------------------------------------------------

class TestReExport:

    def test_import_from_old_location(self):
        from sharpedge.ml.models.ensemble import EnsemblePredictor as OldEnsemble
        from sharpedge.core.ensemble import EnsemblePredictor as NewEnsemble
        assert OldEnsemble is NewEnsemble

    def test_old_location_works(self):
        from sharpedge.ml.models.ensemble import EnsemblePredictor as Ens
        ens = Ens()
        rng = np.random.default_rng(42)
        preds = {"m": rng.dirichlet([1, 1, 1], size=20)}
        y_true = rng.integers(0, 3, size=20)
        ens.fit_weights(preds, y_true)
        result = ens.predict(preds)
        assert result.shape == (20, 3)
