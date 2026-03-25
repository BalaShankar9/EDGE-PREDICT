"""Market Agent — odds-only devigged probability model.

Philosophy: "The market is usually right — find where it's wrong."
Uses ONLY odds-derived features. No match stats, no form, no ELO.
Pure market intelligence.
"""
from __future__ import annotations

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


class MarketAgent(BaseAgent):
    """Reverse-engineers sharp money from market prices."""

    @property
    def name(self) -> str:
        return "market_agent"

    @property
    def agent_type(self) -> str:
        return "market"

    @property
    def description(self) -> str:
        return "Odds-only model. Reverse-engineers sharp money from market prices."

    def fit(self, **kwargs) -> MarketAgent:  # noqa: D401
        """No training needed — this agent reads odds directly from context."""
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Convert odds to devigged probabilities."""
        # If no odds provided, skip
        if not context.odds:
            return None

        # Convert odds to implied probabilities
        raw_probs: dict[str, float] = {}
        for outcome, odd in context.odds.items():
            if odd > 0:
                raw_probs[outcome] = 1.0 / odd

        if not raw_probs:
            return None

        # Normalize (remove overround)
        total = sum(raw_probs.values())
        probs = {k: v / total for k, v in raw_probs.items()}

        # Map to outcome array matching context.outcomes
        prob_array = np.array(
            [probs.get(o, 1.0 / len(context.outcomes)) for o in context.outcomes]
        )
        prob_array = prob_array / prob_array.sum()

        pred_idx = int(np.argmax(prob_array))
        overround = (total - 1.0) * 100  # bookmaker margin percentage

        reasoning = (
            f"Market implied (devigged): "
            f"{dict(zip(context.outcomes, prob_array.round(3)))} | "
            f"Overround: {overround:.1f}%"
        )

        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=prob_array,
            predicted_outcome=context.outcomes[pred_idx],
            confidence=float(prob_array[pred_idx]),
            uncertainty=0.02,  # market is relatively certain
            reasoning=reasoning,
            features_used=["odds_home", "odds_draw", "odds_away", "overround"],
            metadata={"overround_pct": overround},
        )
