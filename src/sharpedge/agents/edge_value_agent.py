"""Edge Value Agent — finds where the market is mispriced.

Philosophy: "Don't predict who wins. Predict where the market is wrong."
Uses ONLY odds + model probability to find mispriced markets.
High edge = high confidence. No edge = no prediction.
"""
from __future__ import annotations

import logging

import numpy as np

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext

logger = logging.getLogger(__name__)


class EdgeValueAgent(BaseAgent):
    """Finds mispriced markets by comparing model probability to implied probability.

    The agent scans every outcome in the 1x2 market. If the model's probability
    for any outcome exceeds the bookmaker-implied probability by more than
    ``min_edge``, the bet is flagged as value and a prediction is emitted.

    The confidence is proportional to edge size, adjusted by the overround
    (the "value score"). Kelly-optimal fraction is included in the reasoning
    so downstream systems can size positions appropriately.
    """

    def __init__(self, min_edge: float = 0.05, max_odds: float = 3.0) -> None:
        """Initialise the EdgeValueAgent.

        Parameters
        ----------
        min_edge:
            Minimum difference between model probability and implied probability
            required to emit a prediction. Default 0.05 (5 percentage points).
        max_odds:
            Maximum decimal odds to consider. Bets on very long shots are
            excluded even if the model edge is positive. Default 3.0.
        """
        if not 0.0 < min_edge < 1.0:
            raise ValueError(f"min_edge must be in (0, 1), got {min_edge}")
        if max_odds <= 1.0:
            raise ValueError(f"max_odds must be > 1.0, got {max_odds}")

        self._min_edge = min_edge
        self._max_odds = max_odds

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "edge_value_agent"

    @property
    def agent_type(self) -> str:
        return "market"

    @property
    def description(self) -> str:
        return (
            "Finds mispriced markets by comparing model probability to implied "
            "probability. No edge = no prediction."
        )

    @property
    def supported_sports(self) -> list[str]:
        return []  # all sports

    def fit(self, **kwargs) -> EdgeValueAgent:
        """No training needed — edge is computed live from odds and model probabilities."""
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Scan all outcomes for positive edge; predict the highest-edge outcome.

        Returns None if:
        - No odds are provided in the context.
        - No model probabilities are available (context.features is None and
          no ``model_probs`` key in context.metadata).
        - No outcome clears the min_edge threshold.
        - The best-edge outcome has odds above max_odds (too risky).

        The model probability source (in priority order):
        1. ``context.metadata["model_probs"]`` — dict keyed by outcome name.
        2. ``context.features`` — treated as a probability vector aligned to
           ``context.outcomes`` (will be normalised).
        """
        if not context.odds:
            logger.debug("[%s] No odds for match %s — skipping", self.name, context.match_id)
            return None

        # ---- 1. Build implied probabilities (raw, before removing overround) ----
        raw_implied: dict[str, float] = {}
        raw_odds: dict[str, float] = {}
        for outcome in context.outcomes:
            odd = context.odds.get(outcome, 0.0)
            if odd > 1.0:
                raw_implied[outcome] = 1.0 / odd
                raw_odds[outcome] = odd

        if len(raw_implied) < 2:
            logger.debug("[%s] Insufficient odds for match %s", self.name, context.match_id)
            return None

        overround: float = sum(raw_implied.values())  # typically 1.04–1.10
        # Devigged implied probabilities (bookmaker margin removed)
        implied: dict[str, float] = {k: v / overround for k, v in raw_implied.items()}

        # ---- 2. Resolve model probabilities ----
        model_probs: dict[str, float] | None = None

        if "model_probs" in context.metadata:
            mp = context.metadata["model_probs"]
            if isinstance(mp, dict):
                model_probs = {str(k): float(v) for k, v in mp.items()}

        if model_probs is None and context.features is not None:
            feat = np.asarray(context.features, dtype=np.float64)
            if feat.ndim == 1 and len(feat) == len(context.outcomes):
                feat = np.clip(feat, 0.0, None)
                total = feat.sum()
                if total > 0:
                    feat = feat / total
                    model_probs = dict(zip(context.outcomes, feat.tolist()))

        if model_probs is None:
            # Fall back: use devigged market probabilities as model (no edge possible,
            # but we still return None cleanly rather than crash).
            logger.debug(
                "[%s] No model probabilities for match %s — cannot compute edge",
                self.name,
                context.match_id,
            )
            return None

        # Normalise model_probs in case they don't sum to 1
        mp_total = sum(model_probs.get(o, 0.0) for o in context.outcomes)
        if mp_total <= 0:
            return None
        model_probs = {o: model_probs.get(o, 0.0) / mp_total for o in context.outcomes}

        # ---- 3. Compute edge per outcome ----
        # edge = model_prob - implied_prob  (positive = value bet)
        # value_score = edge * (1 / overround) — adjusts for book margin
        edges: dict[str, float] = {}
        value_scores: dict[str, float] = {}

        for outcome in context.outcomes:
            m_prob = model_probs.get(outcome, 0.0)
            i_prob = implied.get(outcome, 0.0)
            odd = raw_odds.get(outcome, 0.0)

            if i_prob <= 0 or odd <= 0:
                continue

            edge = m_prob - i_prob
            # Only consider outcomes within our odds range
            if odd > self._max_odds:
                continue

            edges[outcome] = edge
            value_scores[outcome] = edge * (1.0 / overround)

        # ---- 4. Filter to outcomes above min_edge ----
        valuable = {o: e for o, e in edges.items() if e >= self._min_edge}
        if not valuable:
            logger.debug(
                "[%s] No positive edge found for match %s (best edge: %.3f)",
                self.name,
                context.match_id,
                max(edges.values()) if edges else 0.0,
            )
            return None

        # ---- 5. Pick the outcome with the highest value score ----
        best_outcome = max(valuable, key=lambda o: value_scores.get(o, 0.0))
        best_edge = edges[best_outcome]
        best_value_score = value_scores[best_outcome]
        best_model_prob = model_probs[best_outcome]
        best_implied_prob = implied[best_outcome]
        best_odd = raw_odds.get(best_outcome, 0.0)

        # ---- 6. Build probability array ----
        # Weight the entire distribution toward the high-edge outcome proportionally.
        # Edge magnitude drives confidence — higher edge → tighter around best_outcome.
        prob_array = np.array(
            [model_probs.get(o, 0.0) for o in context.outcomes], dtype=np.float64
        )
        prob_array = np.clip(prob_array, 1e-6, None)
        prob_array = prob_array / prob_array.sum()

        pred_idx = list(context.outcomes).index(best_outcome)

        # ---- 7. Confidence: proportional to edge, capped at 0.95 ----
        # Confidence reflects edge magnitude, not raw probability.
        confidence = float(np.clip(best_model_prob + best_edge * 0.5, 0.01, 0.95))

        # ---- 8. Kelly fraction (full Kelly for informational purposes) ----
        # f* = (b*p - q) / b  where b = decimal_odds - 1, p = model_prob, q = 1 - p
        b = best_odd - 1.0
        p = best_model_prob
        q = 1.0 - p
        kelly_fraction = float(np.clip((b * p - q) / b, 0.0, 1.0)) if b > 0 else 0.0

        # ---- 9. Uncertainty: smaller with bigger edge ----
        uncertainty = float(np.clip(0.15 - best_edge * 0.5, 0.02, 0.15))

        # ---- 10. Build reasoning ----
        overround_pct = (overround - 1.0) * 100.0
        reasoning = (
            f"VALUE FOUND — {best_outcome}: "
            f"model={best_model_prob:.3f} vs implied={best_implied_prob:.3f} "
            f"| edge={best_edge:+.3f} | value_score={best_value_score:.4f} "
            f"| odds={best_odd:.2f} | overround={overround_pct:.1f}% "
            f"| Kelly={kelly_fraction:.3f}"
        )

        # Surface all edges in metadata for transparency
        all_edge_info = {
            o: {
                "model_prob": round(model_probs.get(o, 0.0), 4),
                "implied_prob": round(implied.get(o, 0.0), 4),
                "edge": round(edges.get(o, 0.0), 4),
                "value_score": round(value_scores.get(o, 0.0), 5),
            }
            for o in context.outcomes
        }

        logger.info(
            "[%s] %s | best outcome: %s | edge=%.3f | kelly=%.3f",
            self.name,
            context.match_id,
            best_outcome,
            best_edge,
            kelly_fraction,
        )

        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=prob_array,
            predicted_outcome=best_outcome,
            confidence=confidence,
            uncertainty=uncertainty,
            reasoning=reasoning,
            features_used=["odds_home", "odds_draw", "odds_away", "model_probs", "overround"],
            metadata={
                "best_edge": best_edge,
                "best_value_score": best_value_score,
                "kelly_fraction": kelly_fraction,
                "overround": overround,
                "overround_pct": overround_pct,
                "all_edges": all_edge_info,
                "min_edge_threshold": self._min_edge,
                "max_odds_threshold": self._max_odds,
            },
        )
