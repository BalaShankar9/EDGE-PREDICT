"""TabPFN Agent — Foundation Model for Tabular Prediction.

Philosophy: 'Pre-trained intelligence, zero training needed.'
Uses TabPFN, a transformer pre-trained on millions of synthetic datasets.
Makes predictions in a single forward pass — no gradient descent, no hyperparameters.
Especially powerful for small datasets (< 10K samples) like niche leagues.
"""
import logging

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext

logger = logging.getLogger(__name__)


class TabPFNAgent(BaseAgent):
    """Foundation model agent using TabPFN transformer."""

    def __init__(self, max_train_samples: int = 10000):
        self._model = None
        self._fitted = False
        self._max_samples = max_train_samples
        self._X_train = None
        self._y_train = None

    @property
    def name(self) -> str:
        return "tabpfn_agent"

    @property
    def agent_type(self) -> str:
        return "ml"

    @property
    def description(self) -> str:
        return "Pre-trained transformer (TabPFN). Zero hyperparameters, instant predictions."

    def fit(self, X_train, y_train_1x2, **kwargs) -> "TabPFNAgent":
        """Store training data for TabPFN (it uses in-context learning).

        TabPFN doesn't train in the traditional sense — it takes the training
        data as context and makes predictions in a single forward pass.
        """
        try:
            from tabpfn import TabPFNClassifier

            # Encode labels to integers
            label_map = {"H": 0, "D": 1, "A": 2}
            y_encoded = np.array([label_map.get(str(y), 1) for y in y_train_1x2])

            # TabPFN works best with <= 10K samples
            if len(X_train) > self._max_samples:
                idx = np.random.choice(len(X_train), self._max_samples, replace=False)
                X_train = X_train[idx]
                y_encoded = y_encoded[idx]

            # Replace NaN/inf with 0
            X_clean = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

            self._model = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
            self._model.fit(X_clean, y_encoded)
            self._fitted = True
            logger.info(f"TabPFN fitted on {len(X_clean)} samples")

        except ImportError:
            logger.warning("TabPFN not available")
        except Exception as e:
            logger.warning(f"TabPFN fit failed: {e}")

        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        if not self._fitted or self._model is None:
            return None
        if context.features is None:
            return None

        try:
            X = (
                context.features.reshape(1, -1)
                if context.features.ndim == 1
                else context.features
            )
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            probs = self._model.predict_proba(X_clean)[0]

            # Ensure correct shape (TabPFN may return different class count)
            if len(probs) < len(context.outcomes):
                probs = np.append(
                    probs,
                    [1.0 / len(context.outcomes)]
                    * (len(context.outcomes) - len(probs)),
                )
            probs = probs[: len(context.outcomes)]
            probs = np.clip(probs, 0.01, 0.99)
            probs = probs / probs.sum()

            pred_idx = int(np.argmax(probs))

            return AgentPrediction(
                agent_name=self.name,
                sport=context.sport,
                match_id=context.match_id,
                market=context.market,
                outcomes=context.outcomes,
                probabilities=probs,
                predicted_outcome=context.outcomes[pred_idx],
                confidence=float(probs[pred_idx]),
                uncertainty=float(1.0 - probs[pred_idx]),
                reasoning=f"TabPFN transformer: {context.outcomes[pred_idx]} @ {probs[pred_idx]:.2f}",
                features_used=[f"feature_{i}" for i in range(X.shape[1])],
                metadata={"model": "tabpfn", "ensemble_configs": 4},
            )
        except Exception as e:
            logger.warning(f"TabPFN predict failed: {e}")
            return None
