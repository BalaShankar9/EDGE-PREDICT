"""Tests for the Contrarian Agent (fades heavy favorites)."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.contrarian_agent import ContrarianAgent


def _make_context(
    odds: dict[str, float] | None = None,
    sport: str = "football",
) -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport=sport,
        league="test_league",
        match_date="2026-03-25",
        home_team="TeamA",
        away_team="TeamB",
        odds=odds or {},
    )


class TestContrarianAgentProperties:
    def test_name(self):
        agent = ContrarianAgent()
        assert agent.name == "contrarian_agent"

    def test_agent_type(self):
        agent = ContrarianAgent()
        assert agent.agent_type == "market"

    def test_description_is_nonempty(self):
        agent = ContrarianAgent()
        assert len(agent.description) > 0

    def test_fit_returns_self(self):
        agent = ContrarianAgent()
        result = agent.fit()
        assert result is agent


class TestContrarianAgentPredict:
    def test_returns_none_when_no_odds(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={})
        assert agent.predict(ctx) is None

    def test_returns_none_when_odds_all_zero(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 0.0, "draw": 0.0, "away": 0.0})
        assert agent.predict(ctx) is None

    def test_returns_none_when_no_heavy_favorite(self):
        """Balanced odds should not trigger the contrarian."""
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 2.5, "draw": 3.3, "away": 2.8})
        assert agent.predict(ctx) is None

    def test_predict_with_heavy_favorite(self):
        """Heavy favorite (home=1.35) should trigger contrarian fade."""
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "contrarian_agent"
        assert pred.sport == "football"
        assert pred.match_id == "test_001"

    def test_probabilities_sum_to_one(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_fades_favorite_reduces_probability(self):
        """Contrarian should give the heavy favorite LOWER probability than market implies."""
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        # Implied probability for home is ~1/1.35 / total ~ 0.70+
        # After fading, home probability should be lower
        home_idx = list(pred.outcomes).index("home")
        assert pred.probabilities[home_idx] < 0.70

    def test_confidence_between_zero_and_one(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_reasoning_mentions_fade(self):
        agent = ContrarianAgent()
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        assert pred is not None
        assert "fade" in pred.reasoning.lower()

    def test_custom_fade_threshold(self):
        """Higher threshold means fewer triggers."""
        agent = ContrarianAgent(fade_threshold=0.80)
        ctx = _make_context(odds={"home": 1.35, "draw": 5.0, "away": 8.0})
        pred = agent.predict(ctx)
        # home implied prob ~0.70, below 0.80 threshold
        assert pred is None
