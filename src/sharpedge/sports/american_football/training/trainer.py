"""NFL model trainer.

Orchestrates model training and multi-market prediction for american football.
"""

import logging

import numpy as np

from sharpedge.sports.american_football.models.spread_model import NFLSpreadModel

logger = logging.getLogger(__name__)


class NFLTrainer:
    """Trains all NFL models and generates multi-market predictions."""

    def __init__(self):
        self.spread_model: NFLSpreadModel | None = None

    def train(self, X, y_margin) -> "NFLTrainer":
        """Train the spread model on features and margin data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_margin : array-like of shape (n_samples,)
            Margin of victory (positive = home win).
        """
        logger.info("Training NFL models on %d samples...", len(X))
        self.spread_model = NFLSpreadModel()
        self.spread_model.fit(X, y_margin)
        logger.info("NFL models trained successfully")
        return self

    def predict(
        self,
        X,
        spread_line: float = 0.0,
        total_line: float = 45.0,
    ) -> dict:
        """Generate predictions for all markets.

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
