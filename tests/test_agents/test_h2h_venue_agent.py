"""Tests for the H2H/Venue Agent (head-to-head specialist)."""
import numpy as np
import pandas as pd
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.h2h_venue_agent import H2HVenueAgent


def _make_context(
    home_team: str = "TeamA",
    away_team: str = "TeamB",
) -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport="football",
        league="test_league",
        match_date="2026-03-25",
        home_team=home_team,
        away_team=away_team,
    )


def _make_h2h_df() -> pd.DataFrame:
    """5 matches between TeamA (home) and TeamB (away): 3H 1D 1A."""
    return pd.DataFrame(
        [
            {"HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 2, "FTAG": 1, "FTR": "H"},
            {"HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 1, "FTAG": 1, "FTR": "D"},
            {"HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 3, "FTAG": 0, "FTR": "H"},
            {"HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 0, "FTAG": 2, "FTR": "A"},
            {"HomeTeam": "TeamA", "AwayTeam": "TeamB", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        ]
    )


class TestH2HVenueAgentProperties:
    def test_name(self):
        agent = H2HVenueAgent()
        assert agent.name == "h2h_venue_agent"

    def test_agent_type(self):
        agent = H2HVenueAgent()
        assert agent.agent_type == "context"

    def test_description_is_nonempty(self):
        agent = H2HVenueAgent()
        assert len(agent.description) > 0

    def test_fit_returns_self(self):
        agent = H2HVenueAgent()
        result = agent.fit(matches_df=_make_h2h_df())
        assert result is agent


class TestH2HVenueAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = H2HVenueAgent()
        ctx = _make_context()
        assert agent.predict(ctx) is None

    def test_returns_none_when_no_h2h_data(self):
        agent = H2HVenueAgent()
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context(home_team="Unknown", away_team="Other")
        assert agent.predict(ctx) is None

    def test_returns_none_when_too_few_h2h_matches(self):
        """With min_h2h_matches=10, 5 matches is insufficient."""
        agent = H2HVenueAgent(min_h2h_matches=10)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        assert agent.predict(ctx) is None

    def test_predict_with_sufficient_h2h(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "h2h_venue_agent"
        assert pred.sport == "football"

    def test_probabilities_sum_to_one(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_predicts_home_when_dominant(self):
        """3 home wins out of 5 should favor home."""
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome == "home"

    def test_confidence_between_zero_and_one(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_decreases_with_more_matches(self):
        """Uncertainty = max(0.05, 1/(n+1)), so more matches -> less uncertainty."""
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        # 5 matches -> uncertainty = 1/6 ~ 0.167
        assert pred.uncertainty == pytest.approx(1 / 6, abs=0.01)

    def test_metadata_has_h2h_matches(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.metadata["h2h_matches"] == 5

    def test_reasoning_contains_record(self):
        agent = H2HVenueAgent(min_h2h_matches=3)
        agent.fit(matches_df=_make_h2h_df())
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert "H2H" in pred.reasoning
        assert "3W" in pred.reasoning
