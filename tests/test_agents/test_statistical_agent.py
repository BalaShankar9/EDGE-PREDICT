"""Tests for the Statistical Agent (Dixon-Coles + Bivariate Poisson)."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.statistical_agent import StatisticalAgent


def _make_matches(n_per_pair: int = 5) -> list[dict]:
    """Generate synthetic match data for 4 teams."""
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    matches = []
    rng = np.random.RandomState(42)
    for _ in range(n_per_pair):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                matches.append(
                    {
                        "home_team_id": home,
                        "away_team_id": away,
                        "home_goals": int(rng.poisson(1.5)),
                        "away_goals": int(rng.poisson(1.1)),
                    }
                )
    return matches


def _make_context(
    home: str = "TeamA",
    away: str = "TeamB",
    sport: str = "football",
) -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport=sport,
        league="test_league",
        match_date="2026-03-25",
        home_team=home,
        away_team=away,
    )


class TestStatisticalAgentProperties:
    def test_name(self):
        agent = StatisticalAgent()
        assert agent.name == "statistical_agent"

    def test_agent_type(self):
        agent = StatisticalAgent()
        assert agent.agent_type == "statistical"

    def test_supported_sports(self):
        agent = StatisticalAgent()
        assert agent.supported_sports == ["football"]

    def test_supports_football(self):
        agent = StatisticalAgent()
        assert agent.supports_sport("football") is True

    def test_does_not_support_tennis(self):
        agent = StatisticalAgent()
        assert agent.supports_sport("tennis") is False

    def test_description_is_nonempty(self):
        agent = StatisticalAgent()
        assert len(agent.description) > 0


class TestStatisticalAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = StatisticalAgent()
        ctx = _make_context()
        assert agent.predict(ctx) is None

    def test_returns_none_for_non_football(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context(sport="tennis")
        assert agent.predict(ctx) is None

    def test_fit_and_predict_returns_prediction(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "statistical_agent"
        assert pred.sport == "football"

    def test_probabilities_sum_to_one(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_confidence_between_zero_and_one(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_is_nonnegative(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.uncertainty >= 0.0

    def test_reasoning_is_nonempty(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.reasoning) > 0
        assert "DC:" in pred.reasoning
        assert "BVP:" in pred.reasoning

    def test_features_used_populated(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.features_used) > 0

    def test_custom_weights(self):
        agent = StatisticalAgent(dc_weight=0.8, bvp_weight=0.2)
        matches = _make_matches()
        agent.fit(matches)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert "80%/20%" in pred.reasoning

    def test_fit_returns_self(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        result = agent.fit(matches)
        assert result is agent

    def test_predict_batch(self):
        agent = StatisticalAgent()
        matches = _make_matches()
        agent.fit(matches)
        contexts = [
            _make_context(home="TeamA", away="TeamB"),
            _make_context(home="TeamC", away="TeamD"),
        ]
        preds = agent.predict_batch(contexts)
        assert len(preds) == 2
        for p in preds:
            assert abs(float(np.sum(p.probabilities)) - 1.0) < 0.01
