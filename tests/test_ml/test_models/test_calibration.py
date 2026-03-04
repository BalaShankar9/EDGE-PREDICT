import numpy as np
import pytest
from sharpedge.ml.models.calibration import ProbabilityCalibrator


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 3, size=n)
    # Slightly miscalibrated predictions
    y_prob = rng.dirichlet([2, 1, 1], size=n)
    return y_true, y_prob


def test_calibrator_platt(calibration_data):
    y_true, y_prob = calibration_data
    cal = ProbabilityCalibrator(method="platt")
    cal.fit(y_true, y_prob)
    calibrated = cal.calibrate(y_prob)
    assert calibrated.shape == y_prob.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=0.01)


def test_calibrator_isotonic(calibration_data):
    y_true, y_prob = calibration_data
    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(y_true, y_prob)
    calibrated = cal.calibrate(y_prob)
    assert calibrated.shape == y_prob.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=0.01)


def test_calibrator_preserves_ranking(calibration_data):
    """Calibration should preserve relative ordering."""
    y_true, y_prob = calibration_data
    cal = ProbabilityCalibrator(method="platt")
    cal.fit(y_true, y_prob)
    calibrated = cal.calibrate(y_prob)
    # Top predictions for class 0 should still be high after calibration
    top_before = np.argsort(y_prob[:, 0])[-10:]
    # These should still have relatively high calibrated probs
    avg_cal = calibrated[top_before, 0].mean()
    avg_all = calibrated[:, 0].mean()
    assert avg_cal >= avg_all  # Top should be above average


def test_calibrator_not_fitted_error():
    cal = ProbabilityCalibrator()
    with pytest.raises(ValueError, match="not fitted"):
        cal.calibrate(np.array([[0.5, 0.3, 0.2]]))
