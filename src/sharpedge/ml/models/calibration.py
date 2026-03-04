"""Probability calibration via Platt scaling (sigmoid) and isotonic regression.

Ensures that when the model says 80% confident, the event happens ~80% of the time.
"""
import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Calibrates probability predictions using Platt scaling or isotonic regression."""

    def __init__(self, method: str = "platt"):
        """
        Parameters
        ----------
        method : 'platt' or 'isotonic'
        """
        self.method = method
        self._calibrators: list = []  # One per class for multiclass

    def fit(self, y_true: NDArray[np.int_], y_prob: NDArray[np.float64]) -> "ProbabilityCalibrator":
        """Fit calibration on validation set.

        Parameters
        ----------
        y_true : (n_samples,) true class labels (0, 1, 2)
        y_prob : (n_samples, n_classes) predicted probabilities
        """
        n_classes = y_prob.shape[1]
        self._calibrators = []

        for c in range(n_classes):
            binary_true = (y_true == c).astype(int)
            probs = y_prob[:, c]

            if self.method == "platt":
                lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
                lr.fit(probs.reshape(-1, 1), binary_true)
                self._calibrators.append(lr)
            else:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(probs, binary_true)
                self._calibrators.append(ir)

        return self

    def calibrate(self, y_prob: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply calibration to raw probabilities.

        Returns
        -------
        (n_samples, n_classes) calibrated probabilities, normalised to sum to 1.
        """
        if not self._calibrators:
            raise ValueError("Calibrator not fitted. Call fit first.")

        n_classes = y_prob.shape[1]
        calibrated = np.zeros_like(y_prob)

        for c in range(n_classes):
            probs = y_prob[:, c]
            if self.method == "platt":
                calibrated[:, c] = self._calibrators[c].predict_proba(probs.reshape(-1, 1))[:, 1]
            else:
                calibrated[:, c] = self._calibrators[c].predict(probs)

        # Normalise
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return calibrated / row_sums
