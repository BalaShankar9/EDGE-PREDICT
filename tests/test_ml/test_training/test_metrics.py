# tests/test_ml/test_training/test_metrics.py
import numpy as np
import pytest
from sharpedge.ml.training.metrics import (
    ranked_probability_score,
    accuracy,
    log_loss_1x2,
    roi,
    closing_line_value,
    calibration_error,
)


class TestRPS:
    def test_perfect_prediction(self):
        """Perfect prediction = RPS of 0."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert ranked_probability_score(y_true, y_prob) == pytest.approx(0.0)

    def test_worst_prediction(self):
        """Completely wrong confident prediction > 0."""
        y_true = np.array([0])
        y_prob = np.array([[0.0, 0.0, 1.0]])
        rps = ranked_probability_score(y_true, y_prob)
        assert rps > 0.5

    def test_uncertain_prediction(self):
        """Uniform prediction should have moderate RPS."""
        y_true = np.array([0])
        y_prob = np.array([[1/3, 1/3, 1/3]])
        rps = ranked_probability_score(y_true, y_prob)
        assert 0 < rps < 0.5

    def test_rps_range(self):
        """RPS should always be between 0 and 1."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 3, size=100)
        y_prob = rng.dirichlet([1, 1, 1], size=100)
        rps = ranked_probability_score(y_true, y_prob)
        assert 0 <= rps <= 1


class TestAccuracy:
    def test_perfect(self):
        assert accuracy(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0

    def test_zero(self):
        assert accuracy(np.array([0, 0, 0]), np.array([1, 1, 1])) == 0.0

    def test_partial(self):
        assert accuracy(np.array([0, 1, 2, 0]), np.array([0, 1, 1, 1])) == 0.5


class TestROI:
    def test_all_winners(self):
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        r = roi(y_true, y_pred, odds)
        assert r == pytest.approx(1.0)

    def test_all_losers(self):
        y_true = np.array([0, 0])
        y_pred = np.array([2, 2])
        odds = np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        r = roi(y_true, y_pred, odds)
        assert r == pytest.approx(-1.0)


class TestCLV:
    def test_positive_clv(self):
        pick_odds = np.array([2.0, 2.0])
        closing_odds = np.array([1.8, 1.8])
        clv = closing_line_value(pick_odds, closing_odds)
        assert clv > 0

    def test_negative_clv(self):
        pick_odds = np.array([1.8, 1.8])
        closing_odds = np.array([2.0, 2.0])
        clv = closing_line_value(pick_odds, closing_odds)
        assert clv < 0


class TestCalibrationError:
    def test_perfect_calibration(self):
        rng = np.random.default_rng(42)
        n = 1000
        probs = rng.uniform(0.1, 0.9, n)
        outcomes = (rng.random(n) < probs).astype(int)
        ece = calibration_error(outcomes, probs, n_bins=10)
        assert ece < 0.1
