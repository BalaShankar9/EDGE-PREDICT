"""Meta Consensus Agent -- aggregates external prediction sites.

Philosophy: "Wisdom of the crowd, properly weighted."
Aggregates predictions from external prediction websites
(Forebet, PredictZ, WinDrawWin, etc.)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


class MetaConsensusAgent(BaseAgent):
    """Aggregates 6+ external prediction sites. Wisdom of the crowd."""

    def __init__(self) -> None:
        self._predictions: dict[str, list[dict[str, Any]]] = {}
        self._fitted = False

    @property
    def name(self) -> str:
        return "meta_consensus_agent"

    @property
    def agent_type(self) -> str:
        return "context"

    @property
    def description(self) -> str:
        return "Aggregates 6+ external prediction sites. Wisdom of the crowd."

    @property
    def supported_sports(self) -> list[str]:
        return ["football"]

    def fit(self, predictions_data: dict[str, list[dict[str, Any]]] | None = None, **kwargs) -> MetaConsensusAgent:
        """Load external prediction data.

        Parameters
        ----------
        predictions_data:
            Dict mapping match_id -> list of dicts like:
            ``[{"source": "forebet", "home": 0.55, "draw": 0.22, "away": 0.23}, ...]``
        """
        self._predictions = predictions_data or {}
        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        if not self._fitted:
            return None

        preds = self._predictions.get(context.match_id, [])
        if not preds:
            return None

        # Average across all sources
        n = len(preds)
        avg_probs = np.zeros(len(context.outcomes))
        for pred in preds:
            for i, outcome in enumerate(context.outcomes):
                avg_probs[i] += pred.get(outcome, 1 / len(context.outcomes))
        avg_probs /= n
        avg_probs = avg_probs / avg_probs.sum()

        # Agreement rate: what % of sources agree on the predicted outcome
        pred_idx = int(np.argmax(avg_probs))
        predicted = context.outcomes[pred_idx]
        agreements = sum(
            1
            for p in preds
            if max(p.get(o, 0) for o in context.outcomes) == p.get(predicted, 0)
        )
        agreement_rate = agreements / n

        reasoning = f"{n} sources, {agreement_rate:.0%} agree on {predicted}"

        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=avg_probs,
            predicted_outcome=predicted,
            confidence=float(avg_probs[pred_idx]),
            uncertainty=float(np.std([p.get(predicted, 0.33) for p in preds])),
            reasoning=reasoning,
            metadata={"n_sources": n, "agreement_rate": agreement_rate},
        )
