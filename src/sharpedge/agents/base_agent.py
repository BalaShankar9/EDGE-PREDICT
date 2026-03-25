"""Base agent interface for the SharpEdge Agent Swarm.

Every prediction agent must implement this interface. Agents compete
against each other — the Bayesian Arbiter tracks their performance
and dynamically reweights them.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class AgentPrediction:
    """A single prediction from one agent for one match.

    This is the universal output format — every agent produces these,
    regardless of sport or market type.
    """
    agent_name: str
    sport: str                              # "football", "tennis", "basketball"
    match_id: str                           # unique match identifier
    market: str                             # "1x2", "match_winner", "spread", etc.
    outcomes: tuple[str, ...]               # ("home", "draw", "away") or ("player1", "player2")
    probabilities: NDArray[np.float64]      # shape: (n_outcomes,) — must sum to ~1.0
    predicted_outcome: str                  # the outcome with highest probability
    confidence: float                       # peak probability (0.0 to 1.0)
    uncertainty: float                      # std from bootstrap/MC (0.0 = certain)
    reasoning: str                          # human-readable explanation
    features_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Validate prediction integrity."""
        if len(self.probabilities) != len(self.outcomes):
            raise ValueError(
                f"probabilities length ({len(self.probabilities)}) != "
                f"outcomes length ({len(self.outcomes)})"
            )
        prob_sum = float(np.sum(self.probabilities))
        if abs(prob_sum - 1.0) > 0.01:
            raise ValueError(f"probabilities must sum to ~1.0, got {prob_sum:.4f}")
        if self.predicted_outcome not in self.outcomes:
            raise ValueError(
                f"predicted_outcome '{self.predicted_outcome}' not in outcomes {self.outcomes}"
            )


@dataclass
class MatchContext:
    """All information about a match that agents receive.

    Sport-agnostic container — agents extract what they need.
    """
    match_id: str
    sport: str
    league: str
    match_date: str                        # ISO format
    home_team: str
    away_team: str
    features: NDArray[np.float64] | None = None  # feature vector (from pipeline)
    odds: dict[str, float] = field(default_factory=dict)  # {"home": 1.8, "draw": 3.5, "away": 4.2}
    market: str = "1x2"                    # which market to predict
    outcomes: tuple[str, ...] = ("home", "draw", "away")
    metadata: dict[str, Any] = field(default_factory=dict)  # sport-specific extra data


class BaseAgent(ABC):
    """Abstract base class for all prediction agents.

    Every agent in the swarm implements this interface. Agents are:
    - Autonomous: they make predictions independently
    - Transparent: they explain their reasoning
    - Tracked: the arbiter monitors their performance
    - Competing: poor performers get lower weight
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent name (e.g., 'statistical_agent', 'gradient_agent')."""

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Agent category: 'statistical', 'ml', 'market', 'context', 'regime', 'niche'."""

    @property
    def description(self) -> str:
        """Human-readable description of the agent's philosophy."""
        return ""

    @abstractmethod
    def predict(self, context: MatchContext) -> AgentPrediction | None:
        """Generate a prediction for a single match.

        Returns None if the agent cannot make a prediction for this match
        (e.g., insufficient data, wrong sport, low confidence).
        """

    def predict_batch(self, contexts: list[MatchContext]) -> list[AgentPrediction]:
        """Generate predictions for multiple matches.

        Default implementation calls predict() for each match.
        Override for vectorized implementations.
        """
        predictions = []
        for ctx in contexts:
            pred = self.predict(ctx)
            if pred is not None:
                predictions.append(pred)
        return predictions

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseAgent":
        """Train or update the agent's internal models.

        What this means varies by agent type:
        - Statistical: fit parameters (attack/defence strengths)
        - ML: train gradient boosters
        - Market: no training needed (uses odds directly)
        """

    @property
    def supported_sports(self) -> list[str]:
        """Sports this agent supports. Default: all."""
        return []  # empty = supports all

    def supports_sport(self, sport: str) -> bool:
        """Check if this agent supports a given sport."""
        supported = self.supported_sports
        return len(supported) == 0 or sport in supported
