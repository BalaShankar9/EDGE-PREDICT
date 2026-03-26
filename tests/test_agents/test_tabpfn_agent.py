"""Tests for TabPFNAgent."""
import unittest.mock as mock

import numpy as np
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.tabpfn_agent import TabPFNAgent


@pytest.fixture
def agent():
    return TabPFNAgent()


@pytest.fixture
def match_context():
    return MatchContext(
        match_id="test_001",
        sport="football",
        league="EPL",
        match_date="2026-03-26",
        home_team="Arsenal",
        away_team="Chelsea",
        features=np.random.randn(10),
        odds={"home": 1.8, "draw": 3.5, "away": 4.2},
        market="1x2",
        outcomes=("home", "draw", "away"),
    )


class TestTabPFNAgentProperties:
    def test_name(self, agent):
        assert agent.name == "tabpfn_agent"

    def test_agent_type(self, agent):
        assert agent.agent_type == "ml"

    def test_description(self, agent):
        assert "TabPFN" in agent.description


class TestTabPFNAgentPredict:
    def test_predict_returns_none_when_not_fitted(self, agent, match_context):
        result = agent.predict(match_context)
        assert result is None

    def test_predict_returns_none_when_features_none(self, agent):
        ctx = MatchContext(
            match_id="test_001",
            sport="football",
            league="EPL",
            match_date="2026-03-26",
            home_team="Arsenal",
            away_team="Chelsea",
            features=None,
        )
        agent._fitted = True
        agent._model = mock.MagicMock()
        result = agent.predict(ctx)
        assert result is None

    def test_fit_graceful_when_tabpfn_unavailable(self, agent):
        """If TabPFN is not installed, fit should not raise."""
        with mock.patch.dict("sys.modules", {"tabpfn": None}):
            # Force ImportError by patching the import
            with mock.patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    (_ for _ in ()).throw(ImportError("No module named 'tabpfn'"))
                    if name == "tabpfn"
                    else mock.DEFAULT
                ),
            ):
                result = agent.fit(np.random.randn(50, 5), np.array(["H"] * 50))
                assert result is agent
                assert agent._fitted is False

    def test_fit_and_predict_with_mock_model(self, agent, match_context):
        """Test fit+predict using a mock TabPFN model."""
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])

        agent._model = mock_model
        agent._fitted = True

        pred = agent.predict(match_context)
        assert pred is not None
        assert pred.agent_name == "tabpfn_agent"
        assert pred.sport == "football"
        assert pred.match_id == "test_001"

    def test_probabilities_sum_to_one(self, agent, match_context):
        """Probabilities should sum to ~1.0."""
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.6, 0.25, 0.15]])

        agent._model = mock_model
        agent._fitted = True

        pred = agent.predict(match_context)
        assert pred is not None
        assert abs(pred.probabilities.sum() - 1.0) < 0.01

    def test_predicted_outcome_is_argmax(self, agent, match_context):
        """Predicted outcome should be the highest probability."""
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])

        agent._model = mock_model
        agent._fitted = True

        pred = agent.predict(match_context)
        assert pred is not None
        assert pred.predicted_outcome == "away"

    def test_metadata_contains_model_info(self, agent, match_context):
        mock_model = mock.MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.35, 0.25]])

        agent._model = mock_model
        agent._fitted = True

        pred = agent.predict(match_context)
        assert pred is not None
        assert pred.metadata["model"] == "tabpfn"
        assert pred.metadata["ensemble_configs"] == 4
