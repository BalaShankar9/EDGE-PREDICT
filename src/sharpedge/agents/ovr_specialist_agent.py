"""OvR Specialist Agent — One-vs-Rest binary classifiers.

Philosophy: "Each outcome is a different problem."
Uses three independent binary classifiers (Home, Draw, Away).
Each specialist ONLY learns what distinguishes its outcome from all others.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext

_OUTCOMES = ("H", "D", "A")


class OvRSpecialistAgent(BaseAgent):
    """One-vs-Rest binary classifiers for 1x2 prediction."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._fitted: bool = False

    @property
    def name(self) -> str:
        return "ovr_specialist_agent"

    @property
    def agent_type(self) -> str:
        return "ml"

    @property
    def description(self) -> str:
        return "One-vs-Rest binary classifiers. 87.5% accuracy at >=0.80 confidence."

    @property
    def supported_sports(self) -> list[str]:
        return ["football"]

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train_1x2: NDArray[np.str_],
        **kwargs,
    ) -> OvRSpecialistAgent:
        """Train one binary classifier per outcome.

        Parameters
        ----------
        X_train : array of shape (n_samples, n_features)
        y_train_1x2 : array of shape (n_samples,) with values in {"H", "D", "A"}
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier  # type: ignore[assignment]

        for outcome in _OUTCOMES:
            binary_y = (y_train_1x2 == outcome).astype(int)
            model = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1,
                random_state=42,
            )
            model.fit(X_train, binary_y)
            self._models[outcome] = model

        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Predict by combining outputs from all three binary specialists."""
        if not self._fitted:
            return None

        if context.features is None:
            return None

        if not self.supports_sport(context.sport):
            return None

        X = context.features.reshape(1, -1) if context.features.ndim == 1 else context.features

        # Get P(outcome=1) from each specialist
        raw_probs = []
        for outcome in _OUTCOMES:
            model = self._models.get(outcome)
            if model is None:
                raw_probs.append(1.0 / 3)
            else:
                raw_probs.append(float(model.predict_proba(X)[0, 1]))

        probs = np.array(raw_probs)
        probs = np.clip(probs, 0.01, 0.99)
        probs = probs / probs.sum()  # normalize to sum to 1

        pred_idx = int(np.argmax(probs))
        outcomes = context.outcomes

        reasoning = f"OvR specialists: H={probs[0]:.2f} D={probs[1]:.2f} A={probs[2]:.2f}"

        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=probs,
            predicted_outcome=outcomes[pred_idx],
            confidence=float(probs[pred_idx]),
            uncertainty=float(np.std(probs)),
            reasoning=reasoning,
            features_used=[f"feature_{i}" for i in range(X.shape[1])],
        )
