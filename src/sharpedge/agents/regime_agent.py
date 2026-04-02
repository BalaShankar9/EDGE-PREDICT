"""Regime Detection Agent — detects structural shifts in league behaviour.

Philosophy: "The past is not the future. Detect when the rules change."
Compares a RECENT window of matches against a LONG-TERM baseline.
When a statistic deviates beyond a threshold, a regime shift is declared
and predictions are adjusted accordingly.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext

logger = logging.getLogger(__name__)

# Typical long-run football priors (used when a league has no baseline yet)
_DEFAULT_HOME_WIN_RATE = 0.46
_DEFAULT_DRAW_RATE = 0.27
_DEFAULT_AWAY_WIN_RATE = 0.27
_DEFAULT_AVG_GOALS = 2.70
_DEFAULT_OVER25_RATE = 0.52


class RegimeDetectionAgent(BaseAgent):
    """Detects regime shifts in league statistics and adjusts predictions.

    For each league seen during ``fit``, the agent records:
    - Long-term baseline statistics (last ``long_window`` matches).
    - Recent statistics (last ``short_window`` matches).

    At predict time it computes a Z-score for each stat. If any stat deviates
    beyond ``regime_threshold`` standard deviations, a regime shift is declared
    and the uniform 1x2 prior is adjusted before returning a prediction.

    If no regime shift is detected, ``predict`` returns ``None`` — the agent
    has nothing unusual to report.
    """

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 200,
        regime_threshold: float = 1.5,
    ) -> None:
        """Initialise RegimeDetectionAgent.

        Parameters
        ----------
        short_window:
            Number of recent matches per league used as the "current regime"
            window. Default 20.
        long_window:
            Number of historical matches per league used as the long-term
            baseline. Default 200.
        regime_threshold:
            Z-score threshold above which a stat is flagged as a regime shift.
            Default 1.5 (roughly top-7% tail of a normal distribution).
        """
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be < long_window ({long_window})"
            )
        if regime_threshold <= 0:
            raise ValueError(f"regime_threshold must be > 0, got {regime_threshold}")

        self._short_window = short_window
        self._long_window = long_window
        self._regime_threshold = regime_threshold

        # league_id -> regime summary dict (populated by fit)
        self._league_regimes: dict[str, dict[str, Any]] = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "regime_detection_agent"

    @property
    def agent_type(self) -> str:
        return "regime"

    @property
    def description(self) -> str:
        return (
            "Detects when a league is in a different statistical regime than normal "
            "(more goals, shifted home advantage, higher draw rate). Adjusts 1x2 "
            "prior accordingly. Silent when nothing unusual is happening."
        )

    @property
    def supported_sports(self) -> list[str]:
        return ["football"]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, matches_df: pd.DataFrame, **kwargs) -> RegimeDetectionAgent:
        """Compute per-league baselines and detect current regimes.

        Parameters
        ----------
        matches_df : pd.DataFrame
            Must contain columns: league, home_team_id, away_team_id,
            FTHG, FTAG, FTR, match_date.
            FTR values: 'H' (home win), 'D' (draw), 'A' (away win).
            match_date must be sortable (ISO string or datetime).
        """
        required_cols = {
            "league", "home_team_id", "away_team_id",
            "FTHG", "FTAG", "FTR", "match_date",
        }
        missing = required_cols - set(matches_df.columns)
        if missing:
            raise ValueError(f"matches_df is missing required columns: {missing}")

        df = (
            matches_df[list(required_cols)]
            .copy()
            .sort_values("match_date")
            .reset_index(drop=True)
        )

        # Collect per-league match records in chronological order
        league_records: dict[str, list[dict[str, float]]] = defaultdict(list)

        for _, row in df.iterrows():
            league = str(row["league"])
            fthg = float(row["FTHG"])
            ftag = float(row["FTAG"])
            ftr = str(row["FTR"])
            total_goals = fthg + ftag

            league_records[league].append({
                "total_goals": total_goals,
                "home_win": 1.0 if ftr == "H" else 0.0,
                "draw": 1.0 if ftr == "D" else 0.0,
                "away_win": 1.0 if ftr == "A" else 0.0,
                "over25": 1.0 if total_goals > 2.5 else 0.0,
            })

        self._league_regimes = {}

        for league, records in league_records.items():
            regime = self._compute_regime(league, records)
            if regime is not None:
                self._league_regimes[league] = regime
                logger.debug(
                    "[%s] League %s — shifts: %s",
                    self.name,
                    league,
                    regime.get("active_shifts", []),
                )

        self._fitted = True
        logger.info(
            "[%s] Fitted on %d leagues (%d with regime data)",
            self.name,
            len(league_records),
            len(self._league_regimes),
        )
        return self

    def _compute_regime(
        self, league: str, records: list[dict[str, float]]
    ) -> dict[str, Any] | None:
        """Compute baseline stats, recent stats, and Z-scores for one league."""
        n_total = len(records)
        if n_total < self._short_window:
            # Not enough data to do anything meaningful
            logger.debug(
                "[%s] League %s has only %d matches — need at least %d",
                self.name,
                league,
                n_total,
                self._short_window,
            )
            return None

        # Long-term baseline: last long_window matches (or all if fewer)
        baseline_records = records[-self._long_window:]
        # Recent window: last short_window matches (a sub-set of baseline)
        recent_records = records[-self._short_window:]

        def _stats(recs: list[dict[str, float]]) -> dict[str, float]:
            arr = {
                k: np.array([r[k] for r in recs], dtype=np.float64)
                for k in ("total_goals", "home_win", "draw", "away_win", "over25")
            }
            return {k: float(v.mean()) for k, v in arr.items()}

        def _stds(recs: list[dict[str, float]]) -> dict[str, float]:
            arr = {
                k: np.array([r[k] for r in recs], dtype=np.float64)
                for k in ("total_goals", "home_win", "draw", "away_win", "over25")
            }
            # Use population std; add small epsilon to avoid division-by-zero
            return {k: float(v.std() + 1e-6) for k, v in arr.items()}

        baseline_stats = _stats(baseline_records)
        baseline_stds = _stds(baseline_records)
        recent_stats = _stats(recent_records)

        # Z-scores: how many std deviations is recent vs baseline?
        z_scores: dict[str, float] = {}
        for stat in baseline_stats:
            z = (recent_stats[stat] - baseline_stats[stat]) / baseline_stds[stat]
            z_scores[stat] = float(z)

        # Detect which stats are in regime shift
        active_shifts: list[str] = [
            stat for stat, z in z_scores.items()
            if abs(z) >= self._regime_threshold
        ]

        return {
            "league": league,
            "n_baseline": len(baseline_records),
            "n_recent": len(recent_records),
            "baseline_stats": baseline_stats,
            "recent_stats": recent_stats,
            "z_scores": z_scores,
            "active_shifts": active_shifts,
            "has_shift": len(active_shifts) > 0,
        }

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Adjust 1x2 priors based on any detected regime shift in the league.

        Returns None if:
        - Agent not fitted.
        - Sport is not football.
        - League has no regime data (too few historical matches).
        - No regime shift is detected (nothing unusual to report).
        """
        if not self._fitted:
            return None

        if not self.supports_sport(context.sport):
            return None

        regime = self._league_regimes.get(context.league)

        if regime is None:
            logger.debug(
                "[%s] No regime data for league '%s' — skipping",
                self.name,
                context.league,
            )
            return None

        if not regime["has_shift"]:
            # Everything within normal range — nothing to report
            return None

        # ---- Start from league-calibrated baseline priors ----
        baseline = regime["baseline_stats"]
        recent = regime["recent_stats"]
        z_scores = regime["z_scores"]
        active_shifts = regime["active_shifts"]

        h_prob = baseline["home_win"]
        d_prob = baseline["draw"]
        a_prob = baseline["away_win"]

        # Normalise baseline priors in case they don't sum exactly to 1
        prior_sum = h_prob + d_prob + a_prob
        if prior_sum > 0:
            h_prob /= prior_sum
            d_prob /= prior_sum
            a_prob /= prior_sum
        else:
            h_prob = _DEFAULT_HOME_WIN_RATE
            d_prob = _DEFAULT_DRAW_RATE
            a_prob = _DEFAULT_AWAY_WIN_RATE

        shift_descriptions: list[str] = []

        # ---- Apply regime adjustments ----

        # Rule 1: Recent goals significantly higher → boost over predictions.
        # For 1x2 this manifests as slightly lower draw probability (draws tend
        # to be lower-scoring) and a slight home boost (home teams score more
        # in high-scoring environments).
        goals_z = z_scores.get("total_goals", 0.0)
        if abs(goals_z) >= self._regime_threshold:
            direction = "high" if goals_z > 0 else "low"
            goals_adjustment = np.clip(goals_z * 0.03, -0.08, 0.08)
            if direction == "high":
                # More goals → fewer draws, slightly better for home
                d_prob = max(0.05, d_prob - abs(goals_adjustment))
                h_prob = min(0.90, h_prob + abs(goals_adjustment) * 0.6)
                a_prob = min(0.90, a_prob + abs(goals_adjustment) * 0.4)
            else:
                # Fewer goals → more draws
                d_prob = min(0.50, d_prob + abs(goals_adjustment))
                h_prob = max(0.05, h_prob - abs(goals_adjustment) * 0.6)
                a_prob = max(0.05, a_prob - abs(goals_adjustment) * 0.4)
            shift_descriptions.append(
                f"goals regime: z={goals_z:+.2f} ({direction}, "
                f"recent avg={recent['total_goals']:.2f} vs "
                f"baseline={baseline['total_goals']:.2f})"
            )

        # Rule 2: Home advantage significantly shifted → adjust home/away balance.
        home_z = z_scores.get("home_win", 0.0)
        if abs(home_z) >= self._regime_threshold:
            direction = "stronger" if home_z > 0 else "weaker"
            home_adjustment = np.clip(home_z * 0.04, -0.10, 0.10)
            h_prob = np.clip(h_prob + home_adjustment, 0.05, 0.85)
            a_prob = np.clip(a_prob - home_adjustment * 0.7, 0.05, 0.85)
            shift_descriptions.append(
                f"home advantage: z={home_z:+.2f} ({direction}, "
                f"recent={recent['home_win']:.1%} vs "
                f"baseline={baseline['home_win']:.1%})"
            )

        # Rule 3: Draws are more common than normal → boost draw probability.
        draw_z = z_scores.get("draw", 0.0)
        if abs(draw_z) >= self._regime_threshold:
            direction = "more" if draw_z > 0 else "fewer"
            draw_adjustment = np.clip(draw_z * 0.04, -0.10, 0.10)
            d_prob = np.clip(d_prob + draw_adjustment, 0.05, 0.55)
            # Redistribute evenly from both home and away
            h_prob = np.clip(h_prob - draw_adjustment * 0.5, 0.05, 0.85)
            a_prob = np.clip(a_prob - draw_adjustment * 0.5, 0.05, 0.85)
            shift_descriptions.append(
                f"draw rate: z={draw_z:+.2f} ({direction} draws, "
                f"recent={recent['draw']:.1%} vs "
                f"baseline={baseline['draw']:.1%})"
            )

        # ---- Normalise adjusted probs ----
        probs = np.array([h_prob, d_prob, a_prob], dtype=np.float64)
        probs = np.clip(probs, 0.05, 0.90)
        probs = probs / probs.sum()

        # Align to context.outcomes (which might not always be home/draw/away order)
        outcome_map = {"home": 0, "draw": 1, "away": 2}
        prob_array = np.array(
            [probs[outcome_map[o]] if o in outcome_map else 1.0 / len(context.outcomes)
             for o in context.outcomes],
            dtype=np.float64,
        )
        prob_array = prob_array / prob_array.sum()

        pred_idx = int(np.argmax(prob_array))

        reasoning = (
            f"REGIME SHIFT in {context.league} — "
            + " | ".join(shift_descriptions)
            + f" | Adjusted: H={probs[0]:.3f} D={probs[1]:.3f} A={probs[2]:.3f}"
            + f" | Prediction: {context.outcomes[pred_idx]} @ {prob_array[pred_idx]:.3f}"
        )

        # Uncertainty is proportional to recency of data — smaller recent window means
        # we are less sure about the regime
        n_recent = regime["n_recent"]
        uncertainty = float(np.clip(0.15 - (n_recent / self._short_window) * 0.05, 0.05, 0.20))

        logger.info(
            "[%s] %s | league=%s | shifts=%s | pick=%s",
            self.name,
            context.match_id,
            context.league,
            active_shifts,
            context.outcomes[pred_idx],
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
            uncertainty=uncertainty,
            reasoning=reasoning,
            features_used=[
                "league_avg_goals",
                "league_home_win_rate",
                "league_draw_rate",
                "recent_avg_goals",
                "recent_home_win_rate",
                "recent_draw_rate",
            ],
            metadata={
                "league": context.league,
                "active_shifts": active_shifts,
                "z_scores": regime["z_scores"],
                "baseline_stats": regime["baseline_stats"],
                "recent_stats": regime["recent_stats"],
                "short_window": self._short_window,
                "long_window": self._long_window,
                "regime_threshold": self._regime_threshold,
            },
        )
