"""SharpEdge Agent Swarm — competing prediction agents with Bayesian arbitration."""

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext
from sharpedge.agents.arbiter import BayesianArbiter, ArbiterResult
from sharpedge.agents.tracker import AgentTracker
from sharpedge.agents.market_agent import MarketAgent
from sharpedge.agents.statistical_agent import StatisticalAgent
from sharpedge.agents.form_momentum_agent import FormMomentumAgent
from sharpedge.agents.contrarian_agent import ContrarianAgent
from sharpedge.agents.h2h_venue_agent import H2HVenueAgent
from sharpedge.agents.gradient_agent import GradientAgent
from sharpedge.agents.edge_value_agent import EdgeValueAgent
from sharpedge.agents.regime_agent import RegimeDetectionAgent
from sharpedge.agents.evolution import AgentEvolutionEngine

ALL_AGENTS = [
    MarketAgent,
    StatisticalAgent,
    FormMomentumAgent,
    ContrarianAgent,
    H2HVenueAgent,
    GradientAgent,
    EdgeValueAgent,
    RegimeDetectionAgent,
]

__all__ = [
    "AgentPrediction", "BaseAgent", "MatchContext",
    "BayesianArbiter", "ArbiterResult",
    "AgentTracker",
    "MarketAgent", "StatisticalAgent", "FormMomentumAgent",
    "ContrarianAgent", "H2HVenueAgent", "GradientAgent",
    "EdgeValueAgent", "RegimeDetectionAgent",
    "AgentEvolutionEngine",
    "ALL_AGENTS",
]
