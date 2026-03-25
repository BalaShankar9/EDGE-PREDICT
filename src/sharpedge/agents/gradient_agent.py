"""Gradient Agent -- ensemble of gradient-boosted tree models.

Philosophy: 'Let the data speak through diverse trees.'
Uses XGBoost + CatBoost + LightGBM, each with different tree construction
algorithms, providing genuine structural diversity.
"""
import logging

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext

logger = logging.getLogger(__name__)


class GradientAgent(BaseAgent):
    """Stacks XGBoost + CatBoost + LightGBM for pattern recognition."""

    def __init__(self):
        self._models: dict = {}  # name -> model instance
        self._weights: dict[str, float] = {}  # name -> weight
        self._fitted = False

    @property
    def name(self) -> str:
        return "gradient_agent"

    @property
    def agent_type(self) -> str:
        return "ml"

    @property
    def description(self) -> str:
        return (
            "XGBoost + CatBoost + LightGBM stacking. "
            "Finds non-linear patterns in 85+ features."
        )

    def fit(self, X_train, y_train_1x2, y_train_ou=None, y_train_btts=None, **kwargs):
        """Train all available gradient boosters.

        Parameters
        ----------
        X_train : feature matrix (n_samples, n_features)
        y_train_1x2 : "H"/"D"/"A" labels
        y_train_ou : optional over/under labels
        y_train_btts : optional BTTS labels
        """
        from sharpedge.ml.models.xgboost_model import XGBoostPredictor

        xgb = XGBoostPredictor()
        xgb.fit(X_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
        self._models["xgb"] = xgb
        self._weights["xgb"] = 1.0

        try:
            from sharpedge.ml.models.catboost_model import CatBoostPredictor

            cat = CatBoostPredictor()
            cat.fit(X_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
            self._models["catboost"] = cat
            self._weights["catboost"] = 1.0
        except ImportError:
            logger.info("CatBoost not available, skipping")

        try:
            from sharpedge.ml.models.lightgbm_model import LightGBMPredictor

            lgbm = LightGBMPredictor()
            lgbm.fit(X_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
            self._models["lgbm"] = lgbm
            self._weights["lgbm"] = 1.0
        except ImportError:
            logger.info("LightGBM not available, skipping")

        # Equal weights
        n = len(self._models)
        if n > 0:
            for k in self._weights:
                self._weights[k] = 1.0 / n
        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Generate a prediction using the gradient boosting ensemble.

        Returns None if not fitted or if no feature vector is available.
        """
        if not self._fitted or not self._models:
            return None
        if context.features is None:
            return None

        X = context.features.reshape(1, -1) if context.features.ndim == 1 else context.features

        # Collect predictions from all models
        model_probs: dict[str, np.ndarray] = {}
        for model_name, model in self._models.items():
            probs = model.predict_proba_1x2(X)[0]  # shape: (n_outcomes,)
            model_probs[model_name] = probs

        # Weighted average
        blended = np.zeros_like(list(model_probs.values())[0])
        for model_name, probs in model_probs.items():
            blended += self._weights[model_name] * probs
        blended = blended / blended.sum()

        pred_idx = int(np.argmax(blended))
        outcomes = context.outcomes

        # Uncertainty = mean std across models for each outcome
        all_probs = np.array(list(model_probs.values()))
        uncertainty = float(np.mean(np.std(all_probs, axis=0)))

        # Reasoning
        parts = [f"{name}: {p[pred_idx]:.2f}" for name, p in model_probs.items()]
        reasoning = (
            f"Models: {', '.join(parts)} -> {outcomes[pred_idx]} @ {blended[pred_idx]:.2f}"
        )

        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=outcomes,
            probabilities=blended,
            predicted_outcome=outcomes[pred_idx],
            confidence=float(blended[pred_idx]),
            uncertainty=uncertainty,
            reasoning=reasoning,
            features_used=[f"feature_{i}" for i in range(X.shape[1])],
            metadata={"model_weights": self._weights.copy()},
        )
