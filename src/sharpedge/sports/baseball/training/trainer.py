"""Baseball model trainer.

Orchestrates model training and multi-market prediction for baseball.
"""

import logging

import numpy as np

from sharpedge.sports.baseball.models.run_model import RunModel

logger = logging.getLogger(__name__)


class BaseballTrainer:
    """Trains all baseball models and generates multi-market predictions."""

    def __init__(self):
        self.run_model: RunModel | None = None

    def train(self, X, y_margin) -> "BaseballTrainer":
        """Train the run model on features and margin data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_margin : array-like of shape (n_samples,)
            Run differential (positive = home win).
        """
        logger.info("Training baseball models on %d samples...", len(X))
        self.run_model = RunModel()
        self.run_model.fit(X=X, y_margin=y_margin)
        logger.info("Baseball models trained successfully")
        return self

    def predict(
        self,
        X,
        run_line: float = -1.5,
        total_line: float = 8.5,
    ) -> dict:
        """Generate predictions for all markets.

        Returns
        -------
        dict with keys: moneyline_probs, run_line_probs, total_probs
        """
        if self.run_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        ml_probs = self.run_model.predict_proba(X=X)
        total_probs = self.run_model.predict_total(X=X, total_line=total_line)

        return {
            "moneyline_probs": ml_probs,
            "run_line_probs": ml_probs,  # simplified: same model
            "total_probs": total_probs,
        }
