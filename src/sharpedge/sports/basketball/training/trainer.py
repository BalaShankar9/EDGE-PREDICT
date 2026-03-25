"""Basketball model trainer.

Orchestrates model training and multi-market prediction for basketball.
"""

import logging

import numpy as np

from sharpedge.sports.basketball.models.spread_model import SpreadModel

logger = logging.getLogger(__name__)


class BasketballTrainer:
    """Trains all basketball models and generates multi-market predictions."""

    def __init__(self):
        self.spread_model: SpreadModel | None = None

    def train(self, X, y_margin) -> "BasketballTrainer":
        """Train the spread model on features and margin data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_margin : array-like of shape (n_samples,)
            Margin of victory (positive = home win).
        """
        logger.info("Training basketball models on %d samples...", len(X))
        self.spread_model = SpreadModel()
        self.spread_model.fit(X, y_margin)
        logger.info("Basketball models trained successfully")
        return self

    def predict(
        self,
        X,
        spread_line: float = 0.0,
        total_line: float = 220.0,
    ) -> dict:
        """Generate predictions for all markets.

        Parameters
        ----------
        X : array-like
        spread_line : float
            Point spread line (negative = home favored).
        total_line : float
            Over/under total points line.

        Returns
        -------
        dict with keys: moneyline_probs, spread_probs, total_probs, predicted_margin
        """
        if self.spread_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        ml_probs = self.spread_model.predict_proba(X, spread_line=0.0)
        spread_probs = self.spread_model.predict_proba(X, spread_line=spread_line)
        total_probs = self.spread_model.predict_total(X, total_line=total_line)
        predicted_margin = self.spread_model.predict_margin(X)

        return {
            "moneyline_probs": ml_probs,
            "spread_probs": spread_probs,
            "total_probs": total_probs,
            "predicted_margin": predicted_margin,
        }
