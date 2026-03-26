"""Ice hockey model trainer.

Orchestrates model training and multi-market prediction for ice hockey.
"""

import logging

import numpy as np

from sharpedge.sports.ice_hockey.models.goal_model import HockeyGoalModel

logger = logging.getLogger(__name__)


class HockeyTrainer:
    """Trains all hockey models and generates multi-market predictions."""

    def __init__(self):
        self.goal_model: HockeyGoalModel | None = None

    def train(self, X, y_margin) -> "HockeyTrainer":
        """Train the goal model on features and margin data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_margin : array-like of shape (n_samples,)
            Goal difference (positive = home win).
        """
        logger.info("Training hockey models on %d samples...", len(X))
        self.goal_model = HockeyGoalModel()
        self.goal_model.fit(X=X, y_margin=y_margin)
        logger.info("Hockey models trained successfully")
        return self

    def predict(
        self,
        X,
        puck_line: float = -1.5,
        total_line: float = 5.5,
    ) -> dict:
        """Generate predictions for all markets.

        Parameters
        ----------
        X : array-like
        puck_line : float
            Puck line (typically -1.5 for home team).
        total_line : float
            Over/under total goals line.

        Returns
        -------
        dict with keys: moneyline_probs, puck_line_probs, total_probs
        """
        if self.goal_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        ml_probs = self.goal_model.predict_proba(X=X)
        total_probs = self.goal_model.predict_total(X=X, total_line=total_line)

        return {
            "moneyline_probs": ml_probs,
            "puck_line_probs": ml_probs,  # simplified: same model
            "total_probs": total_probs,
        }
