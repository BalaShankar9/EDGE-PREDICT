"""Tests for the Meta Consensus Agent (external prediction aggregator)."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.meta_consensus_agent import MetaConsensusAgent


def _make_context(
    match_id: str = "test_001",
    sport: str = "football",
) -> MatchContext:
    return MatchContext(
        match_id=match_id,
        sport=sport,
        league="test_league",
        match_date="2026-03-25",
        home_team="TeamA",
        away_team="TeamB",
    )


MOCK_PREDICTIONS = {
    "test_001": [
        {"source": "forebet", "home": 0.55, "draw": 0.22, "away": 0.23},
        {"source": "predictz", "home": 0.50, "draw": 0.25, "away": 0.25},
        {"source": "windrawwin", "home": 0.60, "draw": 0.20, "away": 0.20},
    ]
}


class TestMetaConsensusAgentProperties:
    def test_name(self):
        agent = MetaConsensusAgent()
        assert agent.name == "meta_consensus_agent"

    def test_agent_type(self):
        agent = MetaConsensusAgent()
        assert agent.agent_type == "context"

    def test_description_is_nonempty(self):
        agent = MetaConsensusAgent()
        assert len(agent.description) > 0

    def test_supported_sports(self):
        agent = MetaConsensusAgent()
        assert "football" in agent.supported_sports

    def test_fit_returns_self(self):
        agent = MetaConsensusAgent()
        result = agent.fit(predictions_data=MOCK_PREDICTIONS)
        assert result is agent


class TestMetaConsensusAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = MetaConsensusAgent()
        ctx = _make_context()
        assert agent.predict(ctx) is None

    def test_returns_none_when_no_predictions_for_match(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context(match_id="unknown_match")
        assert agent.predict(ctx) is None

    def test_returns_none_when_fitted_with_empty_data(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data={})
        ctx = _make_context()
        assert agent.predict(ctx) is None

    def test_predict_with_mock_data(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "meta_consensus_agent"
        assert pred.sport == "football"

    def test_probabilities_sum_to_one(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_predicts_consensus_home(self):
        """All 3 sources favor home, so prediction should be home."""
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome == "home"

    def test_metadata_has_n_sources(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.metadata["n_sources"] == 3

    def test_metadata_has_agreement_rate(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.metadata["agreement_rate"] <= 1.0

    def test_confidence_between_zero_and_one(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_is_nonnegative(self):
        agent = MetaConsensusAgent()
        agent.fit(predictions_data=MOCK_PREDICTIONS)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.uncertainty >= 0.0
