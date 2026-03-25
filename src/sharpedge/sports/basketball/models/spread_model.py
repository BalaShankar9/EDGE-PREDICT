"""Gradient-boosted regression model for basketball margin prediction.

Predicts margin of victory, then converts to moneyline / spread / total
probabilities using a normal distribution assumption.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from sharpedge.core.base_model import BasePredictor

logger = logging.getLogger(__name__)

# Typical NBA residual standard deviations (empirically derived)
_MARGIN_STD = 12.0
_TOTAL_STD = 15.0
_DEFAULT_TOTAL = 220.0


class SpreadModel(BasePredictor):
    """Gradient-boosted regression for basketball margin prediction."""

    def __init__(self):
        self._model = None
        self._fitted = False
        self._residual_std = _MARGIN_STD

    @property
    def name(self) -> str:
        return "spread_model"

    def fit(self, X, y_margin, **kwargs) -> "SpreadModel":
        """Train on features + margin of victory.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_margin : array-like of shape (n_samples,)
            Margin of victory (positive = home win).
        """
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, y_margin)

        # Estimate residual std from training data
        preds = self._model.predict(X)
        residuals = y_margin - preds
        self._residual_std = max(float(np.std(residuals)), 5.0)

        self._fitted = True
        logger.info(
            "SpreadModel fitted: residual_std=%.2f", self._residual_std
        )
        return self

    def predict_proba(
        self, X, spread_line: float = 0.0, **kwargs
    ) -> NDArray[np.float64]:
        """Predict [P(home covers), P(away covers)] relative to spread line.

        When spread_line=0, this is equivalent to moneyline probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        spread_line : float
            The spread line to evaluate (negative = home favored).

        Returns
        -------
        NDArray of shape (n_samples, 2) with columns [P(home), P(away)].
        """
        from scipy.stats import norm

        predicted_margin = self._model.predict(X)
        std = self._residual_std

        # P(home covers) = P(margin > spread_line)
        p_home = norm.sf(spread_line, loc=predicted_margin, scale=std)
        p_away = 1.0 - p_home

        return np.column_stack([p_home, p_away])

    def predict_total(
        self, X, total_line: float = _DEFAULT_TOTAL, avg_total: float = _DEFAULT_TOTAL
    ) -> NDArray[np.float64]:
        """Predict [P(over), P(under)] for total points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        total_line : float
            The over/under line.
        avg_total : float
            Baseline average total for calibration.

        Returns
        -------
        NDArray of shape (n_samples, 2) with columns [P(over), P(under)].
        """
        from scipy.stats import norm

        predicted_margin = self._model.predict(X)
        # Predicted total = avg_total + small adjustment from margin
        predicted_total = avg_total + predicted_margin * 0.1
        std = _TOTAL_STD

        p_over = norm.sf(total_line, loc=predicted_total, scale=std)
        return np.column_stack([p_over, 1.0 - p_over])

    def predict_margin(self, X) -> NDArray[np.float64]:
        """Return raw predicted margin of victory."""
        return self._model.predict(X)
