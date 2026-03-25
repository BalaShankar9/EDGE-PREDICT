"""Tests for the Bayesian Arbiter — the meta-agent combiner."""
import numpy as np
import pytest

from sharpedge.agents.arbiter import ArbiterResult, BayesianArbiter
from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


# ---------------------------------------------------------------------------
# Dummy agents for testing
# ---------------------------------------------------------------------------

class DummyAgentA(BaseAgent):
    """Always predicts home win with 60% confidence."""

    @property
    def name(self) -> str:
        return "dummy_a"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction:
        return AgentPrediction(
            agent_name="dummy_a",
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=np.array([0.6, 0.2, 0.2]),
            predicted_outcome="home",
            confidence=0.6,
            uncertainty=0.05,
            reasoning="test agent A",
        )


class DummyAgentB(BaseAgent):
    """Also predicts home win but with different probabilities."""

    @property
    def name(self) -> str:
        return "dummy_b"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction:
        return AgentPrediction(
            agent_name="dummy_b",
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=np.array([0.5, 0.3, 0.2]),
            predicted_outcome="home",
            confidence=0.5,
            uncertainty=0.08,
            reasoning="test agent B",
        )


class DummyAgentC(BaseAgent):
    """Predicts away win — disagrees with A and B."""

    @property
    def name(self) -> str:
        return "dummy_c"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction:
        return AgentPrediction(
            agent_name="dummy_c",
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=np.array([0.15, 0.15, 0.7]),
            predicted_outcome="away",
            confidence=0.7,
            uncertainty=0.03,
            reasoning="test agent C",
        )


class AbstainingAgent(BaseAgent):
    """Always returns None — abstains from prediction."""

    @property
    def name(self) -> str:
        return "abstainer"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        return None


class ExplodingAgent(BaseAgent):
    """Always raises an exception."""

    @property
    def name(self) -> str:
        return "exploder"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction:
        raise RuntimeError("Agent exploded!")


class CertainAgent(BaseAgent):
    """Predicts home with very high certainty (low entropy)."""

    @property
    def name(self) -> str:
        return "certain"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction:
        return AgentPrediction(
            agent_name="certain",
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=np.array([0.95, 0.03, 0.02]),
            predicted_outcome="home",
            confidence=0.95,
            uncertainty=0.01,
            reasoning="very certain",
        )


class UncertainAgent(BaseAgent):
    """Predicts with nearly uniform distribution (high entropy)."""

    @property
    def name(self) -> str:
        return "uncertain"

    @property
    def agent_type(self) -> str:
        return "test"

    def fit(self, **kw):
        return self

    def predict(self, context: MatchContext) -> AgentPrediction:
        return AgentPrediction(
            agent_name="uncertain",
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=np.array([0.34, 0.33, 0.33]),
            predicted_outcome="home",
            confidence=0.34,
            uncertainty=0.30,
            reasoning="very uncertain",
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def match_context() -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport="football",
        league="EPL",
        match_date="2026-03-25",
        home_team="Arsenal",
        away_team="Chelsea",
        odds={"home": 1.8, "draw": 3.5, "away": 4.2},
        market="1x2",
        outcomes=("home", "draw", "away"),
    )


@pytest.fixture
def match_context_2() -> MatchContext:
    return MatchContext(
        match_id="test_002",
        sport="football",
        league="EPL",
        match_date="2026-03-25",
        home_team="Liverpool",
        away_team="Spurs",
        odds={"home": 1.5, "draw": 4.0, "away": 6.0},
        market="1x2",
        outcomes=("home", "draw", "away"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBayesianArbiter:
    """Tests for the Bayesian Arbiter."""

    def test_no_agents_returns_none(self, match_context: MatchContext):
        """Arbiter with 0 agents returns None."""
        arbiter = BayesianArbiter(agents=[])
        result = arbiter.predict(match_context)
        assert result is None

    def test_single_agent(self, match_context: MatchContext):
        """Arbiter with 1 agent returns that agent's prediction (weighted)."""
        arbiter = BayesianArbiter(agents=[DummyAgentA()])
        result = arbiter.predict(match_context)

        assert result is not None
        assert result.predicted_outcome == "home"
        assert result.n_agents_contributing == 1
        assert result.consensus_score == 1.0
        # Probabilities should closely match the single agent
        np.testing.assert_allclose(
            result.probabilities, np.array([0.6, 0.2, 0.2]), atol=0.02
        )

    def test_three_agents_blend(self, match_context: MatchContext):
        """Arbiter with 3 agents blends probabilities correctly."""
        agents = [DummyAgentA(), DummyAgentB(), DummyAgentC()]
        arbiter = BayesianArbiter(agents=agents)
        result = arbiter.predict(match_context)

        assert result is not None
        assert result.n_agents_contributing == 3
        # Probabilities should sum to 1
        assert abs(result.probabilities.sum() - 1.0) < 0.01
        # All three agents contribute
        assert len(result.agent_weights) == 3

    def test_consensus_all_agree(self, match_context: MatchContext):
        """Consensus score = 1.0 when all agents agree."""
        agents = [DummyAgentA(), DummyAgentB()]  # both predict "home"
        arbiter = BayesianArbiter(agents=agents)
        result = arbiter.predict(match_context)

        assert result is not None
        assert result.consensus_score == 1.0

    def test_consensus_disagreement(self, match_context: MatchContext):
        """Consensus score < 1.0 when agents disagree."""
        agents = [DummyAgentA(), DummyAgentB(), DummyAgentC()]
        arbiter = BayesianArbiter(agents=agents)
        result = arbiter.predict(match_context)

        assert result is not None
        # A and B predict home, C predicts away; blended likely predicts home
        # So consensus = 2/3 (A and B agree on home)
        assert result.consensus_score < 1.0
        assert result.consensus_score == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_record_results_updates_history(self):
        """record_results updates performance history."""
        arbiter = BayesianArbiter(agents=[DummyAgentA()])
        arbiter.record_results({"dummy_a": True})
        arbiter.record_results({"dummy_a": False})
        arbiter.record_results({"dummy_a": True})

        assert arbiter._performance["dummy_a"] == [True, False, True]

    def test_compute_weights_better_agent_gets_higher_weight(self):
        """compute_weights gives higher weight to agents with better track records."""
        agents = [DummyAgentA(), DummyAgentB()]
        arbiter = BayesianArbiter(agents=agents)

        # Agent A: 8 correct out of 10
        for _ in range(8):
            arbiter.record_results({"dummy_a": True})
        for _ in range(2):
            arbiter.record_results({"dummy_a": False})

        # Agent B: 3 correct out of 10
        for _ in range(3):
            arbiter.record_results({"dummy_b": True})
        for _ in range(7):
            arbiter.record_results({"dummy_b": False})

        weights = arbiter.compute_weights()
        assert weights["dummy_a"] > weights["dummy_b"]

    def test_abstaining_agent_excluded(self, match_context: MatchContext):
        """Agent that returns None is excluded (doesn't crash)."""
        agents = [DummyAgentA(), AbstainingAgent()]
        arbiter = BayesianArbiter(agents=agents)
        result = arbiter.predict(match_context)

        assert result is not None
        assert result.n_agents_contributing == 1
        # Only dummy_a should be in weights
        assert "dummy_a" in result.agent_weights
        assert "abstainer" not in result.agent_weights

    def test_exploding_agent_caught(self, match_context: MatchContext):
        """Agent that raises exception is caught and excluded."""
        agents = [DummyAgentA(), ExplodingAgent()]
        arbiter = BayesianArbiter(agents=agents)
        result = arbiter.predict(match_context)

        assert result is not None
        assert result.n_agents_contributing == 1
        assert "exploder" not in result.agent_weights

    def test_entropy_lower_when_certain(self, match_context: MatchContext):
        """Entropy is lower when prediction is more certain."""
        certain_arbiter = BayesianArbiter(agents=[CertainAgent()])
        uncertain_arbiter = BayesianArbiter(agents=[UncertainAgent()])

        certain_result = certain_arbiter.predict(match_context)
        uncertain_result = uncertain_arbiter.predict(match_context)

        assert certain_result is not None
        assert uncertain_result is not None
        assert certain_result.entropy < uncertain_result.entropy

    def test_arbiter_result_fields(self, match_context: MatchContext):
        """ArbiterResult contains all expected fields."""
        arbiter = BayesianArbiter(agents=[DummyAgentA(), DummyAgentB()])
        result = arbiter.predict(match_context)

        assert result is not None
        assert result.match_id == "test_001"
        assert result.sport == "football"
        assert result.market == "1x2"
        assert result.outcomes == ("home", "draw", "away")
        assert isinstance(result.probabilities, np.ndarray)
        assert len(result.probabilities) == 3
        assert isinstance(result.predicted_outcome, str)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.consensus_score <= 1.0
        assert result.entropy >= 0.0
        assert result.n_agents_contributing == 2
        assert len(result.agent_predictions) == 2
        assert isinstance(result.agent_weights, dict)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_predict_batch(
        self, match_context: MatchContext, match_context_2: MatchContext
    ):
        """predict_batch works for multiple matches."""
        arbiter = BayesianArbiter(agents=[DummyAgentA(), DummyAgentB()])
        results = arbiter.predict_batch([match_context, match_context_2])

        assert len(results) == 2
        assert results[0].match_id == "test_001"
        assert results[1].match_id == "test_002"
        for r in results:
            assert isinstance(r, ArbiterResult)

    def test_register_agent(self):
        """register_agent adds to the swarm."""
        arbiter = BayesianArbiter()
        assert len(arbiter.agents) == 0

        arbiter.register_agent(DummyAgentA())
        assert len(arbiter.agents) == 1
        assert arbiter.agents[0].name == "dummy_a"

        arbiter.register_agent(DummyAgentB())
        assert len(arbiter.agents) == 2

    def test_all_agents_abstain_returns_none(self, match_context: MatchContext):
        """If all agents abstain, arbiter returns None."""
        arbiter = BayesianArbiter(agents=[AbstainingAgent()])
        result = arbiter.predict(match_context)
        assert result is None

    def test_all_agents_explode_returns_none(self, match_context: MatchContext):
        """If all agents raise exceptions, arbiter returns None."""
        arbiter = BayesianArbiter(agents=[ExplodingAgent()])
        result = arbiter.predict(match_context)
        assert result is None

    def test_weight_floor(self):
        """Agent with all-wrong history still gets minimum 5% weight (before normalization)."""
        agents = [DummyAgentA(), DummyAgentB()]
        arbiter = BayesianArbiter(agents=agents)

        # Agent A: always wrong
        for _ in range(20):
            arbiter.record_results({"dummy_a": False})
        # Agent B: always correct
        for _ in range(20):
            arbiter.record_results({"dummy_b": True})

        weights = arbiter.compute_weights()
        # Agent A should have the floor weight, not zero
        assert weights["dummy_a"] > 0
        # Agent B should dominate
        assert weights["dummy_b"] > weights["dummy_a"]
