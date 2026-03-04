"""Weighted ensemble that combines XGBoost and Poisson predictions.

Weights are learned by minimising RPS on the validation set.
"""
import numpy as np
from numpy.typing import NDArray
from sharpedge.ml.training.metrics import ranked_probability_score


class EnsemblePredictor:
    """Combines multiple model predictions via learned weights."""

    def __init__(self):
        self.weights: dict[str, float] = {}

    def fit_weights(
        self,
        model_predictions: dict[str, NDArray[np.float64]],
        y_true: NDArray[np.int_],
        step: float = 0.05,
    ) -> dict[str, float]:
        """Learn optimal weights by grid search minimising RPS.

        Parameters
        ----------
        model_predictions : dict mapping model_name -> (n_samples, 3) probability array
        y_true : true outcomes (0=H, 1=D, 2=A)
        step : grid search step size

        Returns
        -------
        dict of model_name -> weight
        """
        model_names = list(model_predictions.keys())

        if len(model_names) == 1:
            self.weights = {model_names[0]: 1.0}
            return self.weights

        if len(model_names) == 2:
            best_rps = float("inf")
            best_w = 0.5
            for w in np.arange(0, 1 + step, step):
                combined = w * model_predictions[model_names[0]] + (1 - w) * model_predictions[model_names[1]]
                rps = ranked_probability_score(y_true, combined)
                if rps < best_rps:
                    best_rps = rps
                    best_w = w
            self.weights = {model_names[0]: best_w, model_names[1]: 1 - best_w}
            return self.weights

        # For 3+ models, use equal weights as starting point
        n = len(model_names)
        self.weights = {name: 1.0 / n for name in model_names}
        return self.weights

    def predict(self, model_predictions: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Weighted average of model predictions."""
        if not self.weights:
            raise ValueError("Weights not fitted. Call fit_weights first.")

        result = None
        for name, preds in model_predictions.items():
            w = self.weights.get(name, 0.0)
            if result is None:
                result = w * preds
            else:
                result = result + w * preds

        # Normalise rows to sum to 1
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return result / row_sums
