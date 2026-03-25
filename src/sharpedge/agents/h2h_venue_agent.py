"""H2H / Venue Agent -- head-to-head specialist.

Philosophy: "History repeats in sports."
Specialises in head-to-head records and venue effects.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


class H2HVenueAgent(BaseAgent):
    """Head-to-head specialist. History repeats in sports."""

    def __init__(self, min_h2h_matches: int = 3):
        self._h2h_data: dict[tuple[str, str], dict[str, Any]] = {}
        self._min_matches = min_h2h_matches
        self._fitted = False

    @property
    def name(self) -> str:
        return "h2h_venue_agent"

    @property
    def agent_type(self) -> str:
        return "context"

    @property
    def description(self) -> str:
        return "Head-to-head specialist. History repeats in sports."

    def fit(self, matches_df: pd.DataFrame | None = None, **kwargs) -> H2HVenueAgent:
        """Build H2H profiles from historical matches.

        For each team pair, compute: win rates, avg goals, BTTS rate.
        """
        if matches_df is None:
            return self

        for _, row in matches_df.iterrows():
            home = row.get("home_team_id", row.get("HomeTeam", ""))
            away = row.get("away_team_id", row.get("AwayTeam", ""))
            key = (str(home), str(away))
            if key not in self._h2h_data:
                self._h2h_data[key] = {
                    "matches": 0,
                    "home_wins": 0,
                    "draws": 0,
                    "away_wins": 0,
                    "total_goals": 0,
                }

            self._h2h_data[key]["matches"] += 1
            self._h2h_data[key]["total_goals"] += int(row.get("FTHG", 0)) + int(
                row.get("FTAG", 0)
            )
            result = row.get("FTR", "D")
            if result == "H":
                self._h2h_data[key]["home_wins"] += 1
            elif result == "A":
                self._h2h_data[key]["away_wins"] += 1
            else:
                self._h2h_data[key]["draws"] += 1

        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        if not self._fitted:
            return None

        key = (context.home_team, context.away_team)
        h2h = self._h2h_data.get(key)

        if h2h is None or h2h["matches"] < self._min_matches:
            return None  # Not enough H2H history

        n = h2h["matches"]
        # Compute probabilities from H2H record with Bayesian smoothing (add 1 to each)
        h_rate = (h2h["home_wins"] + 1) / (n + 3)
        d_rate = (h2h["draws"] + 1) / (n + 3)
        a_rate = (h2h["away_wins"] + 1) / (n + 3)

        probs = np.array([h_rate, d_rate, a_rate])
        probs = probs / probs.sum()

        pred_idx = int(np.argmax(probs))

        reasoning = (
            f"H2H ({n} matches): {h2h['home_wins']}W {h2h['draws']}D {h2h['away_wins']}L | "
            f"Avg goals: {h2h['total_goals'] / n:.1f}"
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
            uncertainty=max(0.05, 1.0 / (n + 1)),  # less H2H data = more uncertain
            reasoning=reasoning,
            metadata={"h2h_matches": n},
        )
