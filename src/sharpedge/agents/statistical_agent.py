"""Statistical Agent -- domain-specific probabilistic models.

Philosophy: 'Trust the math, not the hype.'
Uses Dixon-Coles and Bivariate Poisson for football.
These models encode deep domain knowledge about how football goals are generated.
"""
import logging

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext
from sharpedge.ml.models.bivariate_poisson import BivariatePoissonPredictor
from sharpedge.ml.models.poisson_model import PoissonPredictor

logger = logging.getLogger(__name__)


class StatisticalAgent(BaseAgent):
    """Combines Dixon-Coles and Bivariate Poisson for football predictions."""

    def __init__(self, dc_weight: float = 0.6, bvp_weight: float = 0.4):
        self._dc = PoissonPredictor()
        self._bvp = BivariatePoissonPredictor()
        self._dc_weight = dc_weight
        self._bvp_weight = bvp_weight
        self._fitted = False

    @property
    def name(self) -> str:
        return "statistical_agent"

    @property
    def agent_type(self) -> str:
        return "statistical"

    @property
    def description(self) -> str:
        return (
            "Domain-specific probabilistic models (Dixon-Coles + Bivariate Poisson). "
            "Trusts the math."
        )

    @property
    def supported_sports(self) -> list[str]:
        return ["football"]

    def fit(self, matches: list[dict], **kwargs) -> "StatisticalAgent":
        """Fit both models on historical match data.

        Parameters
        ----------
        matches : list of dicts with home_team_id, away_team_id, home_goals, away_goals
        """
        self._dc.fit(matches)
        self._bvp.fit(matches)
        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Generate a prediction for a single football match.

        Returns None if not fitted or if the sport is not football.
        """
        if not self._fitted:
            return None
        if context.sport != "football":
            return None

        home = context.home_team
        away = context.away_team

        # Get predictions from both models
        dc_probs = self._dc.predict_proba_1x2(home, away)
        bvp_probs = self._bvp.predict_proba_1x2(home, away)

        # Weighted blend
        blended = self._dc_weight * dc_probs + self._bvp_weight * bvp_probs
        blended = blended / blended.sum()  # normalize

        pred_idx = int(np.argmax(blended))
        outcomes = context.outcomes

        # Build reasoning
        reasoning = (
            f"DC: H={dc_probs[0]:.2f} D={dc_probs[1]:.2f} A={dc_probs[2]:.2f} | "
            f"BVP: H={bvp_probs[0]:.2f} D={bvp_probs[1]:.2f} A={bvp_probs[2]:.2f} | "
            f"Blend ({self._dc_weight:.0%}/{self._bvp_weight:.0%}): "
            f"{outcomes[pred_idx]} @ {blended[pred_idx]:.2f}"
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
            uncertainty=float(np.std([dc_probs[pred_idx], bvp_probs[pred_idx]])),
            reasoning=reasoning,
            features_used=[
                "dc_attack",
                "dc_defence",
                "bvp_correlation",
                "home_advantage",
            ],
        )
