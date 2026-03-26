"""Conformal prediction wrapper using MAPIE.

Provides statistically guaranteed prediction sets — e.g.,
'We are 90% confident the outcome is in {Home, Draw}'.

This is stronger than our current isotonic calibration because
conformal prediction has finite-sample coverage guarantees.
"""
import numpy as np
from numpy.typing import NDArray


class ConformalPredictor:
    """Wraps a classifier to produce conformal prediction sets."""

    def __init__(self, confidence_level: float = 0.90):
        """
        Parameters
        ----------
        confidence_level : target coverage (0.90 = 90% of prediction sets
                          will contain the true outcome)
        """
        self.confidence_level = confidence_level
        self._mapie = None
        self._fitted = False

    def fit(self, X: NDArray, y: NDArray, base_model=None) -> "ConformalPredictor":
        """Fit conformal predictor on calibration data.

        Parameters
        ----------
        X : feature matrix (calibration set)
        y : true labels (integer-encoded, calibration set)
        base_model : a fitted sklearn-compatible classifier with predict_proba
        """
        from mapie.classification import SplitConformalClassifier

        self._mapie = SplitConformalClassifier(
            estimator=base_model,
            confidence_level=self.confidence_level,
            conformity_score="lac",  # Least Ambiguous set-valued Classifier
            prefit=True,  # base_model is already fitted
        )
        self._mapie.conformalize(X, y)
        self._fitted = True
        return self

    def predict_sets(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Predict with conformal prediction sets.

        Returns
        -------
        y_pred : (n_samples,) — point predictions
        prediction_sets : (n_samples, n_classes) — boolean mask of included outcomes
                         True means that outcome is in the prediction set
        """
        if not self._fitted:
            raise ValueError("Not fitted. Call fit() first.")

        y_pred, prediction_sets = self._mapie.predict_set(X)
        # prediction_sets shape: (n_samples, n_classes, 1) — squeeze last dim
        if prediction_sets.ndim == 3:
            prediction_sets = prediction_sets[:, :, 0]

        return y_pred, prediction_sets

    def get_set_sizes(self, X: NDArray) -> NDArray:
        """Get the size of each prediction set (1=certain, 2=ambiguous, 3=clueless)."""
        _, sets = self.predict_sets(X)
        return sets.sum(axis=1)

    def get_confidence_score(self, X: NDArray) -> NDArray:
        """Conformal confidence: 1/set_size (1.0=certain, 0.33=all three possible)."""
        sizes = self.get_set_sizes(X)
        return 1.0 / np.maximum(sizes, 1)
