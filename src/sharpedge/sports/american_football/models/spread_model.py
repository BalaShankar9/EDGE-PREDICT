"""Gradient-boosted regression model for NFL margin prediction.

Predicts margin of victory, then converts to moneyline / spread / total
probabilities using a normal distribution assumption.
NFL-specific: higher variance (std ~14 points) than basketball.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from sharpedge.core.base_model import BasePredictor

logger = logging.getLogger(__name__)

# Typical NFL residual standard deviations
_MARGIN_STD = 14.0
_TOTAL_STD = 10.0
_DEFAULT_TOTAL = 45.0


class NFLSpreadModel(BasePredictor):
    """Gradient-boosted regression for NFL margin prediction."""

    def __init__(self):
        self._model = None
        self._fitted = False
        self._residual_std = _MARGIN_STD

    @property
    def name(self) -> str:
        return "nfl_spread_model"

    def fit(self, X, y_margin, **kwargs) -> "NFLSpreadModel":
        """Train on features + margin of victory.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_margin : array-like of shape (n_samples,)
            Margin of victory (positive = home win).
        """
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, y_margin)

        preds = self._model.predict(X)
        residuals = y_margin - preds
        self._residual_std = max(float(np.std(residuals)), 7.0)

        self._fitted = True
        logger.info(
            "NFLSpreadModel fitted: residual_std=%.2f", self._residual_std
        )
        return self

    def predict_proba(
        self, X, spread_line: float = 0.0, **kwargs
    ) -> NDArray[np.float64]:
        """Predict [P(home covers), P(away covers)] relative to spread line.

        When spread_line=0, this is equivalent to moneyline probabilities.
        """
        from scipy.stats import norm

        predicted_margin = self._model.predict(X)
        std = self._residual_std

        p_home = norm.sf(spread_line, loc=predicted_margin, scale=std)
        p_away = 1.0 - p_home

        return np.column_stack([p_home, p_away])

    def predict_total(
        self, X, total_line: float = _DEFAULT_TOTAL, avg_total: float = _DEFAULT_TOTAL
    ) -> NDArray[np.float64]:
        """Predict [P(over), P(under)] for total points."""
        from scipy.stats import norm

        predicted_margin = self._model.predict(X)
        predicted_total = avg_total + predicted_margin * 0.05
        std = _TOTAL_STD

        p_over = norm.sf(total_line, loc=predicted_total, scale=std)
        return np.column_stack([p_over, 1.0 - p_over])

    def predict_margin(self, X) -> NDArray[np.float64]:
        """Return raw predicted margin of victory."""
        return self._model.predict(X)
