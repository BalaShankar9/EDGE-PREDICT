"""Agent Performance Tracker — persists predictions and tracks metrics.

Records every agent prediction to the database, resolves outcomes post-match,
and computes rolling performance snapshots for the Bayesian Arbiter.
"""
import logging
from datetime import date, datetime
from typing import Any

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction

logger = logging.getLogger(__name__)


class AgentTracker:
    """Tracks agent predictions and computes performance metrics."""

    def __init__(self):
        self._predictions: list[dict] = []  # in-memory buffer
        self._results: dict[str, list[dict]] = {}  # agent_name -> resolved predictions

    def record_prediction(self, prediction: AgentPrediction) -> None:
        """Record an agent prediction (pre-match)."""
        self._predictions.append({
            "agent_name": prediction.agent_name,
            "sport": prediction.sport,
            "match_id": prediction.match_id,
            "market": prediction.market,
            "predicted_outcome": prediction.predicted_outcome,
            "probabilities": {
                o: float(p) for o, p in zip(prediction.outcomes, prediction.probabilities)
            },
            "confidence": prediction.confidence,
            "reasoning": prediction.reasoning,
            "created_at": prediction.created_at.isoformat() if hasattr(prediction.created_at, 'isoformat') else str(prediction.created_at),
        })

    def record_batch(self, predictions: list[AgentPrediction]) -> None:
        """Record multiple predictions at once."""
        for pred in predictions:
            self.record_prediction(pred)

    def resolve(self, match_id: str, actual_outcome: str) -> dict[str, bool]:
        """Resolve all predictions for a match with the actual outcome.

        Returns dict: agent_name -> was_correct
        """
        results = {}
        remaining = []
        for pred in self._predictions:
            if pred["match_id"] == match_id:
                correct = pred["predicted_outcome"] == actual_outcome
                agent_name = pred["agent_name"]
                results[agent_name] = correct

                # Store resolved prediction
                if agent_name not in self._results:
                    self._results[agent_name] = []
                self._results[agent_name].append({
                    **pred,
                    "actual_outcome": actual_outcome,
                    "correct": correct,
                })
            else:
                remaining.append(pred)

        self._predictions = remaining
        return results

    def get_agent_stats(self, agent_name: str, last_n: int | None = None) -> dict[str, Any]:
        """Compute performance stats for an agent.

        Parameters
        ----------
        agent_name : agent to get stats for
        last_n : only consider last N predictions (None = all)

        Returns
        -------
        dict with: total_bets, wins, win_rate, avg_confidence,
                   brier_score, roi (if odds available)
        """
        history = self._results.get(agent_name, [])
        if last_n is not None:
            history = history[-last_n:]

        if not history:
            return {
                "total_bets": 0, "wins": 0, "win_rate": 0.0,
                "avg_confidence": 0.0, "brier_score": 1.0,
            }

        total = len(history)
        wins = sum(1 for h in history if h["correct"])
        win_rate = wins / total
        avg_confidence = np.mean([h["confidence"] for h in history])

        # Brier score: mean squared error of predicted probability vs outcome
        brier_scores = []
        for h in history:
            prob_of_actual = h["probabilities"].get(h["actual_outcome"], 0.0)
            brier_scores.append((1.0 - prob_of_actual) ** 2)
        brier_score = float(np.mean(brier_scores))

        return {
            "total_bets": total,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "avg_confidence": round(float(avg_confidence), 4),
            "brier_score": round(brier_score, 4),
        }

    def get_leaderboard(self, min_bets: int = 10) -> list[dict]:
        """Get all agents ranked by win rate.

        Only includes agents with at least min_bets resolved predictions.
        """
        leaderboard = []
        for agent_name in self._results:
            stats = self.get_agent_stats(agent_name)
            if stats["total_bets"] >= min_bets:
                leaderboard.append({"agent_name": agent_name, **stats})

        leaderboard.sort(key=lambda x: x["win_rate"], reverse=True)
        return leaderboard

    @property
    def pending_count(self) -> int:
        """Number of unresolved predictions."""
        return len(self._predictions)

    @property
    def resolved_count(self) -> int:
        """Total resolved predictions across all agents."""
        return sum(len(preds) for preds in self._results.values())
