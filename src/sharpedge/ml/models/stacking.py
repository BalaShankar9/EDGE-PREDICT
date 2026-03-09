"""Stacking meta-learner for combining base model predictions.

Replaces naive weighted averaging with a learned Level-1 combiner.

Architecture:
  Level-0: Base models generate out-of-fold (OOF) predictions
  Level-1: LogisticRegression learns optimal combination weights + interactions

Why LogisticRegression as meta-learner:
  - Probability calibration guarantee (outputs are proper probabilities)
  - Resistant to overfitting with few features (15 inputs for 5 models × 3 classes)
  - Fast to train, no hyperparameter sensitivity
  - Proven in Kaggle stacking ensembles and academic football prediction papers
"""
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


class StackingMetaLearner:
    """Level-1 meta-learner that combines Level-0 model predictions."""

    def __init__(self, C: float = 1.0):
        """
        Parameters
        ----------
        C : regularization strength (lower = more regularization)
        """
        self.model = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=1000,
        )
        self.model_names: list[str] = []
        self._fitted = False

    def fit(
        self,
        oof_predictions: dict[str, NDArray[np.float64]],
        y_true: NDArray[np.int_],
    ) -> "StackingMetaLearner":
        """Fit meta-learner on out-of-fold predictions from base models.

        Parameters
        ----------
        oof_predictions : dict mapping model_name -> (n_samples, 3) array
            Out-of-fold probability predictions from each base model.
        y_true : (n_samples,) array of true labels (0=H, 1=D, 2=A)
        """
        self.model_names = sorted(oof_predictions.keys())
        X_meta = self._build_meta_features(oof_predictions)
        self.model.fit(X_meta, y_true)
        self._fitted = True
        return self

    def predict(
        self,
        model_predictions: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Combine base model predictions via the meta-learner.

        Parameters
        ----------
        model_predictions : dict mapping model_name -> (n_samples, 3) array

        Returns
        -------
        (n_samples, 3) array of [P(H), P(D), P(A)]
        """
        if not self._fitted:
            raise ValueError("Meta-learner not fitted. Call fit() first.")
        X_meta = self._build_meta_features(model_predictions)
        return self.model.predict_proba(X_meta)

    def _build_meta_features(
        self,
        model_predictions: dict[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Stack base model predictions into meta-feature matrix.

        For 3 models × 3 classes = 9 meta-features.
        """
        arrays = []
        for name in self.model_names:
            preds = model_predictions.get(name)
            if preds is None:
                raise ValueError(f"Missing predictions for model '{name}'")
            arrays.append(preds)
        return np.hstack(arrays)
