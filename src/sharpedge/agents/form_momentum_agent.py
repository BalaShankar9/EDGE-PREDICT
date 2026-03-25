"""Form Momentum Agent — pure momentum tracker.

Philosophy: "Recent form is everything."
Uses ONLY the last 5-10 matches. Ignores long-term history.
Catches hot/cold streaks via exponentially weighted rolling stats.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


class FormMomentumAgent(BaseAgent):
    """Pure momentum tracker using last-10-match rolling statistics."""

    FORM_WINDOW = 10

    def __init__(self) -> None:
        self._form_data: dict[str, dict[str, float]] = {}
        self._fitted: bool = False

    @property
    def name(self) -> str:
        return "form_momentum_agent"

    @property
    def agent_type(self) -> str:
        return "context"

    @property
    def description(self) -> str:
        return "Pure momentum tracker. Uses only last 5-10 matches, exponentially weighted."

    @property
    def supported_sports(self) -> list[str]:
        return ["football"]

    def fit(self, matches_df: pd.DataFrame, **kwargs) -> FormMomentumAgent:
        """Build form profiles from historical match data.

        Parameters
        ----------
        matches_df : pd.DataFrame
            Must contain columns: home_team_id, away_team_id, FTHG, FTAG, FTR, match_date.
            FTR is 'H' / 'D' / 'A'.
        """
        required_cols = {"home_team_id", "away_team_id", "FTHG", "FTAG", "FTR", "match_date"}
        missing = required_cols - set(matches_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = matches_df.sort_values("match_date").reset_index(drop=True)

        # Collect per-team match results chronologically
        team_results: dict[str, list[dict]] = defaultdict(list)

        for _, row in df.iterrows():
            home = row["home_team_id"]
            away = row["away_team_id"]
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            result = row["FTR"]

            # Home team perspective
            if result == "H":
                home_pts, away_pts = 3, 0
            elif result == "D":
                home_pts, away_pts = 1, 1
            else:
                home_pts, away_pts = 0, 3

            team_results[home].append({
                "goals_for": hg,
                "goals_against": ag,
                "points": home_pts,
            })
            team_results[away].append({
                "goals_for": ag,
                "goals_against": hg,
                "points": away_pts,
            })

        # Compute form profiles from last N matches with exponential weighting
        for team, results in team_results.items():
            recent = results[-self.FORM_WINDOW:]
            n = len(recent)
            if n == 0:
                continue

            # Exponential weights: most recent match gets highest weight
            weights = np.array([np.exp(0.2 * i) for i in range(n)])
            weights = weights / weights.sum()

            goals_for = np.array([r["goals_for"] for r in recent], dtype=float)
            goals_against = np.array([r["goals_against"] for r in recent], dtype=float)
            points = np.array([r["points"] for r in recent], dtype=float)

            self._form_data[team] = {
                "points_pct": float(np.dot(weights, points) / 3.0),
                "goals_for_avg": float(np.dot(weights, goals_for)),
                "goals_against_avg": float(np.dot(weights, goals_against)),
                "matches": n,
            }

        self._fitted = True
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Predict based on recent form of both teams."""
        if not self._fitted:
            return None

        if not self.supports_sport(context.sport):
            return None

        home_form = self._form_data.get(context.home_team)
        away_form = self._form_data.get(context.away_team)

        if home_form is None or away_form is None:
            return None

        # Simple form-based prediction:
        # Higher points percentage -> higher win probability
        # Add home advantage factor (historical ~46% home win rate)
        home_strength = home_form["points_pct"] * 1.1  # home boost
        away_strength = away_form["points_pct"]
        draw_factor = 0.25  # ~25% of matches are draws

        raw_h = home_strength * (1 - draw_factor)
        raw_a = away_strength * (1 - draw_factor)
        raw_d = draw_factor

        total = raw_h + raw_d + raw_a
        probs = np.array([raw_h / total, raw_d / total, raw_a / total])
        probs = np.clip(probs, 0.05, 0.90)
        probs = probs / probs.sum()

        pred_idx = int(np.argmax(probs))

        reasoning = (
            f"Home form: {home_form['points_pct']:.1%} pts/match | "
            f"Away form: {away_form['points_pct']:.1%} pts/match | "
            f"Prediction: {context.outcomes[pred_idx]} @ {probs[pred_idx]:.2f}"
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
            uncertainty=0.10,  # form is noisy
            reasoning=reasoning,
            features_used=["home_points_pct", "away_points_pct", "home_advantage"],
        )
