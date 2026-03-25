"""Tests for the OvR Specialist Agent (One-vs-Rest binary classifiers)."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.ovr_specialist_agent import OvRSpecialistAgent


def _make_training_data(n: int = 200, n_features: int = 10):
    """Generate synthetic training data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features)
    y = rng.choice(["H", "D", "A"], size=n)
    return X, y


def _make_context(
    features: np.ndarray | None = None,
    sport: str = "football",
) -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport=sport,
        league="test_league",
        match_date="2026-03-25",
        home_team="TeamA",
        away_team="TeamB",
        features=features,
    )


class TestOvRSpecialistAgentProperties:
    def test_name(self):
        agent = OvRSpecialistAgent()
        assert agent.name == "ovr_specialist_agent"

    def test_agent_type(self):
        agent = OvRSpecialistAgent()
        assert agent.agent_type == "ml"

    def test_supported_sports(self):
        agent = OvRSpecialistAgent()
        assert agent.supported_sports == ["football"]

    def test_supports_football(self):
        agent = OvRSpecialistAgent()
        assert agent.supports_sport("football") is True

    def test_does_not_support_tennis(self):
        agent = OvRSpecialistAgent()
        assert agent.supports_sport("tennis") is False

    def test_description_is_nonempty(self):
        agent = OvRSpecialistAgent()
        assert len(agent.description) > 0


class TestOvRSpecialistAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = OvRSpecialistAgent()
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        assert agent.predict(ctx) is None

    def test_returns_none_when_no_features(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        ctx = _make_context(features=None)
        assert agent.predict(ctx) is None

    def test_returns_none_for_non_football(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features, sport="tennis")
        assert agent.predict(ctx) is None

    def test_fit_and_predict_returns_prediction(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "ovr_specialist_agent"
        assert pred.sport == "football"

    def test_probabilities_sum_to_one(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_confidence_between_zero_and_one(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_is_nonnegative(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.uncertainty >= 0.0

    def test_reasoning_contains_ovr(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert "OvR" in pred.reasoning

    def test_features_used_populated(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.features_used) == 10

    def test_fit_returns_self(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        result = agent.fit(X, y)
        assert result is agent

    def test_predict_with_2d_features(self):
        """Features already shaped as (1, n_features) should work."""
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        features = np.random.randn(1, 10)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_predict_batch(self):
        agent = OvRSpecialistAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        contexts = [
            _make_context(features=np.random.randn(10)),
            _make_context(features=np.random.randn(10)),
        ]
        preds = agent.predict_batch(contexts)
        assert len(preds) == 2
        for p in preds:
            assert abs(float(np.sum(p.probabilities)) - 1.0) < 0.01
