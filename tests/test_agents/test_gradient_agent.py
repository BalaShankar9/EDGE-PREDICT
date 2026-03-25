"""Tests for the Gradient Agent (XGBoost + CatBoost + LightGBM)."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.gradient_agent import GradientAgent


def _make_training_data(n: int = 200, n_features: int = 10):
    """Generate synthetic training data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features).astype(np.float32)
    y = rng.choice(["H", "D", "A"], size=n)
    return X, y


def _make_context(features: np.ndarray | None = None) -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport="football",
        league="test_league",
        match_date="2026-03-25",
        home_team="TeamA",
        away_team="TeamB",
        features=features,
    )


class TestGradientAgentProperties:
    def test_name(self):
        agent = GradientAgent()
        assert agent.name == "gradient_agent"

    def test_agent_type(self):
        agent = GradientAgent()
        assert agent.agent_type == "ml"

    def test_description_is_nonempty(self):
        agent = GradientAgent()
        assert len(agent.description) > 0


class TestGradientAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = GradientAgent()
        features = np.random.randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        assert agent.predict(ctx) is None

    def test_returns_none_when_features_is_none(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)
        ctx = _make_context(features=None)
        assert agent.predict(ctx) is None

    def test_fit_and_predict_returns_prediction(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "gradient_agent"

    def test_probabilities_sum_to_one(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_confidence_between_zero_and_one(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_is_nonnegative(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.uncertainty >= 0.0

    def test_reasoning_is_nonempty(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.reasoning) > 0
        assert "Models:" in pred.reasoning

    def test_metadata_contains_model_weights(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert "model_weights" in pred.metadata
        weights = pred.metadata["model_weights"]
        assert isinstance(weights, dict)
        assert len(weights) > 0
        # Weights should sum to ~1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_features_used_matches_input_dim(self):
        agent = GradientAgent()
        X, y = _make_training_data(n_features=15)
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(15).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.features_used) == 15

    def test_fit_returns_self(self):
        agent = GradientAgent()
        X, y = _make_training_data()
        result = agent.fit(X, y)
        assert result is agent

    def test_predict_with_2d_features(self):
        """Features can be passed as 2D array (1, n_features)."""
        agent = GradientAgent()
        X, y = _make_training_data()
        agent.fit(X, y)

        features = np.random.RandomState(99).randn(1, 10).astype(np.float32)
        ctx = _make_context(features=features)
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01
