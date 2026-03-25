"""Weighted ensemble that combines multiple model predictions.

Weights are learned by minimising RPS on the validation set.
Supports 2+ models via constrained optimization.
Works for any number of outcomes (2 for tennis, 3 for football, N for other sports).
"""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from sharpedge.ml.training.metrics import ranked_probability_score


class EnsemblePredictor:
    """Combines multiple model predictions via learned weights.

    Supports predictions of shape (n_samples, n_outcomes) for any n_outcomes >= 2.
    """

    def __init__(self):
        self.weights: dict[str, float] = {}

    def fit_weights(
        self,
        model_predictions: dict[str, NDArray[np.float64]],
        y_true: NDArray[np.int_],
        step: float = 0.05,
    ) -> dict[str, float]:
        """Learn optimal weights by minimising RPS.

        Uses grid search for 2 models, constrained optimization for 3+.

        Parameters
        ----------
        model_predictions : dict mapping model_name -> (n_samples, n_outcomes) probability array
        y_true : true outcomes as integer class indices (0, 1, ..., n_outcomes-1)
        step : grid search step size (used for 2-model case)

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

        # 3+ models: constrained optimization (weights >= 0, sum = 1)
        pred_arrays = [model_predictions[name] for name in model_names]
        n = len(model_names)

        def objective(w):
            combined = sum(w[i] * pred_arrays[i] for i in range(n))
            return ranked_probability_score(y_true, combined)

        # Constraints: weights sum to 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        # Bounds: each weight in [0, 1]
        bounds = [(0.0, 1.0)] * n
        # Start from equal weights
        x0 = np.ones(n) / n

        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        self.weights = {name: float(result.x[i]) for i, name in enumerate(model_names)}
        return self.weights

    def predict(self, model_predictions: dict[str, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Weighted average of model predictions.

        Parameters
        ----------
        model_predictions : dict mapping model_name -> (n_samples, n_outcomes) probability array

        Returns
        -------
        NDArray of shape (n_samples, n_outcomes), rows normalized to sum to 1
        """
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
