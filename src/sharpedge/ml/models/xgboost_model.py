"""XGBoost model for football predictions.

Wraps XGBoost with our specific requirements:
- Multi-class for 1X2 (Home/Draw/Away)
- Binary for Over/Under 2.5
- Binary for BTTS
- Outputs calibrated probabilities
"""
import logging
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


class XGBoostPredictor:
    """XGBoost predictor for 1X2, O/U, and BTTS markets."""

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model_1x2: xgb.XGBClassifier | None = None
        self.model_ou: xgb.XGBClassifier | None = None
        self.model_btts: xgb.XGBClassifier | None = None
        self._label_encoder = LabelEncoder()

    def fit(
        self,
        X: NDArray[np.float64],
        y_1x2: NDArray,  # "H", "D", "A"
        y_ou: NDArray[np.int_] | None = None,  # 1=over, 0=under
        y_btts: NDArray[np.int_] | None = None,  # 1=yes, 0=no
    ) -> "XGBoostPredictor":
        """Train all sub-models."""
        # 1X2 model (multi-class)
        y_encoded = self._label_encoder.fit_transform(y_1x2)
        self.model_1x2 = xgb.XGBClassifier(
            **self.params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )
        self.model_1x2.fit(X, y_encoded)
        logger.info(f"1X2 model trained on {len(X)} samples")

        # Over/Under model (binary)
        if y_ou is not None:
            self.model_ou = xgb.XGBClassifier(
                **self.params,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            self.model_ou.fit(X, y_ou)
            logger.info("O/U model trained")

        # BTTS model (binary)
        if y_btts is not None:
            self.model_btts = xgb.XGBClassifier(
                **self.params,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            self.model_btts.fit(X, y_btts)
            logger.info("BTTS model trained")

        return self

    def predict_proba_1x2(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict 1X2 probabilities. Returns (n, 3) array [P(H), P(D), P(A)]."""
        raw = self.model_1x2.predict_proba(X)
        # Ensure correct column order: H=0, D=1, A=2
        classes = self._label_encoder.classes_  # Alphabetically: ['A', 'D', 'H']
        h_idx = list(classes).index("H")
        d_idx = list(classes).index("D")
        a_idx = list(classes).index("A")
        return raw[:, [h_idx, d_idx, a_idx]]

    def predict_proba_ou(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict P(Over 2.5). Returns (n,) array."""
        if self.model_ou is None:
            raise ValueError("O/U model not trained")
        return self.model_ou.predict_proba(X)[:, 1]

    def predict_proba_btts(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict P(BTTS Yes). Returns (n,) array."""
        if self.model_btts is None:
            raise ValueError("BTTS model not trained")
        return self.model_btts.predict_proba(X)[:, 1]

    def predict_1x2(self, X: NDArray[np.float64]) -> NDArray:
        """Predict 1X2 result. Returns array of 'H'/'D'/'A'."""
        proba = self.predict_proba_1x2(X)
        indices = np.argmax(proba, axis=1)
        return np.array(["H", "D", "A"])[indices]
