"""League Specialist Agent -- per-league calibrated predictions.

Philosophy: "Know one thing better than anyone."
A configurable agent that can be instantiated per league with
league-specific adjustments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


class LeagueSpecialistAgent(BaseAgent):
    """Per-league specialist. One instance per league, deeply tuned."""

    def __init__(self, league: str, home_advantage: float = 0.46):
        self._league = league
        self._home_advantage = home_advantage  # league-specific home win rate
        self._league_avg_goals = 2.7  # default, updated on fit
        self._fitted = False

    @property
    def name(self) -> str:
        slug = self._league.lower().replace(" ", "_")
        return f"league_specialist_{slug}"

    @property
    def agent_type(self) -> str:
        return "niche"

    @property
    def description(self) -> str:
        return f"Deep specialist for {self._league}. Calibrated to league-specific patterns."

    def fit(self, matches_df: pd.DataFrame | None = None, **kwargs) -> LeagueSpecialistAgent:
        """Compute league-specific statistics from historical data."""
        if matches_df is None:
            self._fitted = True
            return self

        # Filter to this league's matches
        if "league" in matches_df.columns:
            league_df = matches_df[matches_df["league"] == self._league]
        elif "Div" in matches_df.columns:
            league_df = matches_df[matches_df["Div"] == self._league]
        else:
            league_df = matches_df

        if len(league_df) > 0:
            # Compute league-specific home advantage
            if "FTR" in league_df.columns:
                results = league_df["FTR"].value_counts(normalize=True)
                self._home_advantage = float(results.get("H", 0.46))

            # Compute average goals
            if "FTHG" in league_df.columns and "FTAG" in league_df.columns:
                self._league_avg_goals = float(
                    league_df["FTHG"].mean() + league_df["FTAG"].mean()
                )

        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        if not self._fitted:
            return None
        if context.league != self._league:
            return None  # Only predict for our league

        # Use league-calibrated base rates as prior
        h_prior = self._home_advantage
        d_prior = 0.27  # typical draw rate
        a_prior = 1.0 - h_prior - d_prior

        probs = np.array([h_prior, d_prior, a_prior])
        probs = np.clip(probs, 0.05, 0.90)
        probs = probs / probs.sum()

        pred_idx = int(np.argmax(probs))

        reasoning = (
            f"League prior ({self._league}): home={h_prior:.1%}, "
            f"avg goals={self._league_avg_goals:.1f}/match"
        )

        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=probs,
            predicted_outcome=context.outcomes[pred_idx],
            confidence=float(probs[pred_idx]),
            uncertainty=0.08,
            reasoning=reasoning,
            metadata={"league": self._league, "home_advantage": self._home_advantage},
        )
