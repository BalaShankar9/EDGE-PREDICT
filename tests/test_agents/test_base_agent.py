"""Tests for the base agent interface."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import AgentPrediction, BaseAgent, MatchContext


# ---------------------------------------------------------------------------
# AgentPrediction tests
# ---------------------------------------------------------------------------

class TestAgentPrediction:
    """Tests for AgentPrediction dataclass validation."""

    def test_valid_prediction(self):
        pred = AgentPrediction(
            agent_name="test_agent",
            sport="football",
            match_id="match_001",
            market="1x2",
            outcomes=("home", "draw", "away"),
            probabilities=np.array([0.5, 0.3, 0.2]),
            predicted_outcome="home",
            confidence=0.5,
            uncertainty=0.05,
            reasoning="Home team is stronger",
        )
        assert pred.agent_name == "test_agent"
        assert pred.predicted_outcome == "home"
        assert pred.confidence == 0.5
        assert len(pred.probabilities) == 3
        assert pred.features_used == []
        assert pred.metadata == {}

    def test_probabilities_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to ~1.0"):
            AgentPrediction(
                agent_name="test_agent",
                sport="football",
                match_id="match_001",
                market="1x2",
                outcomes=("home", "draw", "away"),
                probabilities=np.array([0.5, 0.5, 0.5]),
                predicted_outcome="home",
                confidence=0.5,
                uncertainty=0.05,
                reasoning="Bad prediction",
            )

    def test_probabilities_length_must_match_outcomes(self):
        with pytest.raises(ValueError, match="probabilities length"):
            AgentPrediction(
                agent_name="test_agent",
                sport="football",
                match_id="match_001",
                market="1x2",
                outcomes=("home", "draw", "away"),
                probabilities=np.array([0.6, 0.4]),
                predicted_outcome="home",
                confidence=0.6,
                uncertainty=0.05,
                reasoning="Wrong length",
            )

    def test_predicted_outcome_must_be_in_outcomes(self):
        with pytest.raises(ValueError, match="not in outcomes"):
            AgentPrediction(
                agent_name="test_agent",
                sport="football",
                match_id="match_001",
                market="1x2",
                outcomes=("home", "draw", "away"),
                probabilities=np.array([0.5, 0.3, 0.2]),
                predicted_outcome="over2.5",
                confidence=0.5,
                uncertainty=0.05,
                reasoning="Wrong outcome",
            )

    def test_probabilities_tolerance(self):
        """Probabilities summing to within 0.01 of 1.0 should be accepted."""
        pred = AgentPrediction(
            agent_name="test_agent",
            sport="football",
            match_id="match_001",
            market="1x2",
            outcomes=("home", "draw", "away"),
            probabilities=np.array([0.504, 0.3, 0.2]),
            predicted_outcome="home",
            confidence=0.504,
            uncertainty=0.05,
            reasoning="Slight rounding",
        )
        assert pred is not None


# ---------------------------------------------------------------------------
# MatchContext tests
# ---------------------------------------------------------------------------

class TestMatchContext:
    """Tests for MatchContext dataclass."""

    def test_creation_with_defaults(self):
        ctx = MatchContext(
            match_id="m1",
            sport="football",
            league="EPL",
            match_date="2026-03-25",
            home_team="Arsenal",
            away_team="Chelsea",
        )
        assert ctx.features is None
        assert ctx.odds == {}
        assert ctx.market == "1x2"
        assert ctx.outcomes == ("home", "draw", "away")
        assert ctx.metadata == {}

    def test_creation_with_full_data(self):
        features = np.array([1.0, 2.0, 3.0])
        ctx = MatchContext(
            match_id="m2",
            sport="tennis",
            league="ATP",
            match_date="2026-03-25",
            home_team="Djokovic",
            away_team="Sinner",
            features=features,
            odds={"player1": 1.5, "player2": 2.6},
            market="match_winner",
            outcomes=("player1", "player2"),
            metadata={"surface": "clay"},
        )
        assert ctx.sport == "tennis"
        assert ctx.market == "match_winner"
        assert len(ctx.outcomes) == 2
        assert ctx.metadata["surface"] == "clay"
        np.testing.assert_array_equal(ctx.features, features)


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------

class DummyAgent(BaseAgent):
    """Minimal concrete agent for testing."""

    @property
    def name(self) -> str:
        return "dummy_agent"

    @property
    def agent_type(self) -> str:
        return "statistical"

    @property
    def description(self) -> str:
        return "A dummy agent for testing"

    def predict(self, context: MatchContext) -> AgentPrediction | None:
        if not self.supports_sport(context.sport):
            return None
        probs = np.array([0.5, 0.3, 0.2])
        return AgentPrediction(
            agent_name=self.name,
            sport=context.sport,
            match_id=context.match_id,
            market=context.market,
            outcomes=context.outcomes,
            probabilities=probs,
            predicted_outcome=context.outcomes[0],
            confidence=0.5,
            uncertainty=0.05,
            reasoning="Dummy prediction",
            features_used=["feat_a", "feat_b"],
        )

    def fit(self, *args, **kwargs) -> "DummyAgent":
        return self


class FootballOnlyAgent(DummyAgent):
    """Agent that only supports football."""

    @property
    def name(self) -> str:
        return "football_only_agent"

    @property
    def supported_sports(self) -> list[str]:
        return ["football"]


class TestBaseAgent:
    """Tests for BaseAgent ABC and concrete implementations."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]

    def test_dummy_agent_properties(self):
        agent = DummyAgent()
        assert agent.name == "dummy_agent"
        assert agent.agent_type == "statistical"
        assert agent.description == "A dummy agent for testing"

    def test_predict_returns_prediction(self):
        agent = DummyAgent()
        ctx = MatchContext(
            match_id="m1",
            sport="football",
            league="EPL",
            match_date="2026-03-25",
            home_team="Arsenal",
            away_team="Chelsea",
        )
        pred = agent.predict(ctx)
        assert pred is not None
        assert isinstance(pred, AgentPrediction)
        assert pred.agent_name == "dummy_agent"
        assert pred.sport == "football"
        assert pred.match_id == "m1"

    def test_predict_batch_returns_list(self):
        agent = DummyAgent()
        contexts = [
            MatchContext(
                match_id=f"m{i}",
                sport="football",
                league="EPL",
                match_date="2026-03-25",
                home_team="Team A",
                away_team="Team B",
            )
            for i in range(3)
        ]
        preds = agent.predict_batch(contexts)
        assert isinstance(preds, list)
        assert len(preds) == 3
        assert all(isinstance(p, AgentPrediction) for p in preds)

    def test_fit_returns_self(self):
        agent = DummyAgent()
        result = agent.fit()
        assert result is agent

    def test_supports_sport_all_when_empty(self):
        agent = DummyAgent()
        assert agent.supported_sports == []
        assert agent.supports_sport("football") is True
        assert agent.supports_sport("tennis") is True
        assert agent.supports_sport("basketball") is True

    def test_supports_sport_filters_correctly(self):
        agent = FootballOnlyAgent()
        assert agent.supports_sport("football") is True
        assert agent.supports_sport("tennis") is False
        assert agent.supports_sport("basketball") is False

    def test_predict_returns_none_for_unsupported_sport(self):
        agent = FootballOnlyAgent()
        ctx = MatchContext(
            match_id="m1",
            sport="tennis",
            league="ATP",
            match_date="2026-03-25",
            home_team="Player A",
            away_team="Player B",
        )
        pred = agent.predict(ctx)
        assert pred is None

    def test_predict_batch_skips_none(self):
        agent = FootballOnlyAgent()
        contexts = [
            MatchContext(
                match_id="m1", sport="football", league="EPL",
                match_date="2026-03-25", home_team="A", away_team="B",
            ),
            MatchContext(
                match_id="m2", sport="tennis", league="ATP",
                match_date="2026-03-25", home_team="C", away_team="D",
            ),
        ]
        preds = agent.predict_batch(contexts)
        assert len(preds) == 1
        assert preds[0].match_id == "m1"

    def test_default_description_is_empty(self):
        """BaseAgent.description defaults to empty string."""
        # FootballOnlyAgent inherits from DummyAgent which overrides description,
        # so we need a minimal agent that doesn't override it.
        class MinimalAgent(BaseAgent):
            @property
            def name(self) -> str:
                return "minimal"

            @property
            def agent_type(self) -> str:
                return "statistical"

            def predict(self, context: MatchContext) -> AgentPrediction | None:
                return None

            def fit(self, *args, **kwargs) -> "MinimalAgent":
                return self

        agent = MinimalAgent()
        assert agent.description == ""
