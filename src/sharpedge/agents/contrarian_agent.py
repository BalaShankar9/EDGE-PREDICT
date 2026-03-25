"""Contrarian Agent -- fades heavy public consensus.

Philosophy: "When everyone agrees, they're probably wrong."
This agent FADES heavy favorites. When odds heavily favor one side
(implied prob > threshold) but the edge is small, the contrarian
bets the OTHER way.
"""
from __future__ import annotations

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


class ContrarianAgent(BaseAgent):
    """Systematically fades heavy favorites. Finds value in unpopular picks."""

    def __init__(self, fade_threshold: float = 0.65):
        """Initialise contrarian agent.

        Parameters
        ----------
        fade_threshold:
            Implied probability above which we consider fading the favorite.
        """
        self._fade_threshold = fade_threshold

    @property
    def name(self) -> str:
        return "contrarian_agent"

    @property
    def agent_type(self) -> str:
        return "market"

    @property
    def description(self) -> str:
        return "Systematically fades heavy favorites. Finds value in unpopular picks."

    def fit(self, **kwargs) -> ContrarianAgent:  # noqa: D401
        """No training needed -- uses odds directly."""
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        if not context.odds:
            return None

        # Compute implied probabilities
        implied: dict[str, float] = {}
        for outcome, odd in context.odds.items():
            if odd > 0:
                implied[outcome] = 1.0 / odd
        total = sum(implied.values())
        if total == 0:
            return None
        implied = {k: v / total for k, v in implied.items()}

        # Find the heavy favorite
        fav_outcome = max(implied, key=implied.get)  # type: ignore[arg-type]
        fav_prob = implied[fav_outcome]

        # Only act if there's a heavy favorite to fade
        if fav_prob < self._fade_threshold:
            return None  # No strong enough signal

        # Fade: redistribute probability AWAY from the favorite
        # The contrarian gives the favorite LESS than market implies
        # and redistributes to underdogs
        fade_amount = (fav_prob - 0.5) * 0.3  # take 30% of the excess

        probs: dict[str, float] = {}
        for outcome in context.outcomes:
            if outcome == fav_outcome:
                probs[outcome] = implied.get(outcome, 1 / len(context.outcomes)) - fade_amount
            else:
                n_others = len(context.outcomes) - 1
                probs[outcome] = (
                    implied.get(outcome, 1 / len(context.outcomes)) + fade_amount / n_others
                )

        prob_array = np.array([max(0.05, probs.get(o, 0.1)) for o in context.outcomes])
        prob_array = prob_array / prob_array.sum()

        pred_idx = int(np.argmax(prob_array))

        reasoning = (
            f"Market heavy on {fav_outcome} ({fav_prob:.1%}). "
            f"Contrarian fades by {fade_amount:.1%}. "
            f"Pick: {context.outcomes[pred_idx]} @ {prob_array[pred_idx]:.2f}"
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
            uncertainty=0.12,
            reasoning=reasoning,
        )
