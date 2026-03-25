"""Bayesian Arbiter — the meta-agent that combines the Agent Swarm.

This is NOT a simple weighted average. It uses Bayesian Model Averaging:
1. Collect predictions from all agents
2. Weight each agent by its RECENT accuracy (exponentially decayed)
3. Agents that abstain (return None) are simply excluded
4. Final prediction = performance-weighted blend
5. Consensus score = what fraction of agents agree on the predicted outcome

The arbiter also provides:
- Agent agreement rate (high agreement = higher confidence)
- Prediction entropy (low entropy = strong signal)
- Per-agent weight transparency (for debugging and tracking)
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext

logger = logging.getLogger(__name__)


@dataclass
class ArbiterResult:
    """The arbiter's final combined prediction."""

    match_id: str
    sport: str
    market: str
    outcomes: tuple[str, ...]
    probabilities: NDArray[np.float64]  # final blended probabilities
    predicted_outcome: str  # majority/weighted prediction
    confidence: float  # peak probability
    consensus_score: float  # fraction of agents that agree (0-1)
    entropy: float  # prediction entropy (lower = more certain)
    n_agents_contributing: int  # how many agents made a prediction
    agent_predictions: list[AgentPrediction]  # raw predictions from each agent
    agent_weights: dict[str, float]  # name -> weight used
    reasoning: str


class BayesianArbiter:
    """Combines agent predictions using performance-weighted Bayesian Model Averaging."""

    def __init__(
        self, agents: list[BaseAgent] | None = None, decay: float = 0.95
    ):
        """
        Parameters
        ----------
        agents : list of BaseAgent instances
        decay : exponential decay factor for performance tracking
                0.95 means last 20 results dominate (half-life ~14 results)
        """
        self._agents: list[BaseAgent] = agents or []
        self._decay = decay
        # Performance tracking: agent_name -> list of correct booleans
        self._performance: dict[str, list[bool]] = {}
        # Cached weights (updated after each compute_weights call)
        self._weights: dict[str, float] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the swarm."""
        self._agents.append(agent)
        self._performance[agent.name] = []

    @property
    def agents(self) -> list[BaseAgent]:
        return self._agents

    def compute_weights(self) -> dict[str, float]:
        """Compute Bayesian weights from recent performance.

        Agents with no track record get equal weight (prior = 0.5 accuracy).
        Weights are proportional to exponentially-weighted accuracy.
        """
        weights: dict[str, float] = {}
        for agent in self._agents:
            history = self._performance.get(agent.name, [])
            if not history:
                # Prior: assume 50% accuracy
                weights[agent.name] = 0.5
            else:
                # Exponentially weighted accuracy
                n = len(history)
                decay_weights = np.array(
                    [self._decay ** (n - 1 - i) for i in range(n)]
                )
                decay_weights /= decay_weights.sum()
                accuracy = float(
                    np.dot(decay_weights, [1.0 if h else 0.0 for h in history])
                )
                # Floor at 5% to never fully zero out an agent
                weights[agent.name] = max(accuracy, 0.05)

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        self._weights = weights
        return weights

    def record_results(self, results: dict[str, bool]) -> None:
        """Record agent prediction outcomes.

        Parameters
        ----------
        results : dict mapping agent_name -> was_correct (bool)
        """
        for agent_name, correct in results.items():
            if agent_name not in self._performance:
                self._performance[agent_name] = []
            self._performance[agent_name].append(correct)

    def predict(self, context: MatchContext) -> ArbiterResult | None:
        """Combine all agent predictions for a single match.

        Steps:
        1. Collect predictions from all agents that support this sport
        2. Weight each by Bayesian performance weights
        3. Blend probabilities
        4. Compute consensus and entropy
        """
        if not self._agents:
            return None

        # Compute current weights
        weights = self.compute_weights()

        # Collect predictions
        agent_preds: list[AgentPrediction] = []
        for agent in self._agents:
            if not agent.supports_sport(context.sport):
                continue
            try:
                pred = agent.predict(context)
                if pred is not None:
                    agent_preds.append(pred)
            except Exception as e:
                logger.warning(
                    f"Agent {agent.name} failed on {context.match_id}: {e}"
                )

        if not agent_preds:
            return None

        # Weighted blend of probabilities
        n_outcomes = len(context.outcomes)
        blended = np.zeros(n_outcomes)
        total_weight = 0.0
        used_weights: dict[str, float] = {}

        for pred in agent_preds:
            w = weights.get(pred.agent_name, 1.0 / len(self._agents))
            blended += w * pred.probabilities
            total_weight += w
            used_weights[pred.agent_name] = w

        if total_weight > 0:
            blended /= total_weight
        blended = np.clip(blended, 0.01, None)
        blended /= blended.sum()

        pred_idx = int(np.argmax(blended))
        predicted_outcome = context.outcomes[pred_idx]

        # Consensus: fraction of agents that agree on predicted outcome
        agreements = sum(
            1 for p in agent_preds if p.predicted_outcome == predicted_outcome
        )
        consensus_score = agreements / len(agent_preds)

        # Entropy: -sum(p * log(p))
        entropy = float(
            -np.sum(blended * np.log(np.clip(blended, 1e-10, 1.0)))
        )

        # Build reasoning summary
        agent_summaries = []
        for pred in agent_preds:
            w = used_weights.get(pred.agent_name, 0)
            agent_summaries.append(
                f"{pred.agent_name}({w:.1%}): "
                f"{pred.predicted_outcome}@{pred.confidence:.2f}"
            )
        reasoning = (
            f"{len(agent_preds)} agents, "
            f"{consensus_score:.0%} consensus on {predicted_outcome}. "
            f"Entropy: {entropy:.3f}. " + " | ".join(agent_summaries)
        )

        return ArbiterResult(
            match_id=context.match_id,
            sport=context.sport,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=blended,
            predicted_outcome=predicted_outcome,
            confidence=float(blended[pred_idx]),
            consensus_score=consensus_score,
            entropy=entropy,
            n_agents_contributing=len(agent_preds),
            agent_predictions=agent_preds,
            agent_weights=used_weights,
            reasoning=reasoning,
        )

    def predict_batch(self, contexts: list[MatchContext]) -> list[ArbiterResult]:
        """Combine predictions for multiple matches."""
        results = []
        for ctx in contexts:
            result = self.predict(ctx)
            if result is not None:
                results.append(result)
        return results
