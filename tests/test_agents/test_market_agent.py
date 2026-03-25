"""Tests for the Market Agent (odds-only devigged probability model)."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.market_agent import MarketAgent


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


class TestMarketAgentProperties:
    def test_name(self):
        agent = MarketAgent()
        assert agent.name == "market_agent"

    def test_agent_type(self):
        agent = MarketAgent()
        assert agent.agent_type == "market"

    def test_description_is_nonempty(self):
        agent = MarketAgent()
        assert len(agent.description) > 0

    def test_fit_returns_self(self):
        agent = MarketAgent()
        result = agent.fit()
        assert result is agent


class TestMarketAgentPredict:
    def test_returns_none_when_no_odds(self):
        agent = MarketAgent()
        ctx = _make_context(odds={})
        assert agent.predict(ctx) is None

    def test_returns_none_when_odds_all_zero(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 0.0, "draw": 0.0, "away": 0.0})
        assert agent.predict(ctx) is None

    def test_predict_with_valid_odds(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "market_agent"
        assert pred.sport == "football"
        assert pred.match_id == "test_001"

    def test_probabilities_sum_to_one(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_home_favourite_predicted(self):
        """Lowest odds (home=1.8) should produce highest probability."""
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome == "home"

    def test_confidence_between_zero_and_one(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_is_nonnegative(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.uncertainty >= 0.0

    def test_reasoning_contains_overround(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert "Overround" in pred.reasoning

    def test_metadata_has_overround(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert "overround_pct" in pred.metadata
        assert pred.metadata["overround_pct"] > 0

    def test_features_used_populated(self):
        agent = MarketAgent()
        ctx = _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2})
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.features_used) > 0

    def test_predict_batch(self):
        agent = MarketAgent()
        contexts = [
            _make_context(odds={"home": 1.8, "draw": 3.5, "away": 4.2}),
            _make_context(odds={"home": 2.1, "draw": 3.3, "away": 3.5}),
        ]
        preds = agent.predict_batch(contexts)
        assert len(preds) == 2
        for p in preds:
            assert abs(float(np.sum(p.probabilities)) - 1.0) < 0.01
