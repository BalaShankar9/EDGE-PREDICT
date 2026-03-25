"""Tests for the League Specialist Agent (per-league calibrated predictions)."""
import numpy as np
import pandas as pd
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.league_specialist_agent import LeagueSpecialistAgent


def _make_context(
    league: str = "E0",
    sport: str = "football",
) -> MatchContext:
    return MatchContext(
        match_id="test_001",
        sport=sport,
        league=league,
        match_date="2026-03-25",
        home_team="TeamA",
        away_team="TeamB",
    )


def _make_league_df() -> pd.DataFrame:
    """10 matches in league E0: 5H 2D 3A, avg goals ~2.5."""
    rows = [
        {"Div": "E0", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 1, "FTR": "H"},
        {"Div": "E0", "HomeTeam": "C", "AwayTeam": "D", "FTHG": 0, "FTAG": 0, "FTR": "D"},
        {"Div": "E0", "HomeTeam": "E", "AwayTeam": "F", "FTHG": 3, "FTAG": 1, "FTR": "H"},
        {"Div": "E0", "HomeTeam": "G", "AwayTeam": "H", "FTHG": 0, "FTAG": 2, "FTR": "A"},
        {"Div": "E0", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Div": "E0", "HomeTeam": "B", "AwayTeam": "D", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        {"Div": "E0", "HomeTeam": "E", "AwayTeam": "G", "FTHG": 2, "FTAG": 0, "FTR": "H"},
        {"Div": "E0", "HomeTeam": "F", "AwayTeam": "H", "FTHG": 0, "FTAG": 1, "FTR": "A"},
        {"Div": "E0", "HomeTeam": "A", "AwayTeam": "E", "FTHG": 1, "FTAG": 0, "FTR": "H"},
        {"Div": "E0", "HomeTeam": "B", "AwayTeam": "F", "FTHG": 0, "FTAG": 3, "FTR": "A"},
    ]
    return pd.DataFrame(rows)


class TestLeagueSpecialistAgentProperties:
    def test_name_contains_league(self):
        agent = LeagueSpecialistAgent(league="E0")
        assert "e0" in agent.name

    def test_agent_type(self):
        agent = LeagueSpecialistAgent(league="E0")
        assert agent.agent_type == "niche"

    def test_description_contains_league(self):
        agent = LeagueSpecialistAgent(league="E0")
        assert "E0" in agent.description

    def test_fit_returns_self(self):
        agent = LeagueSpecialistAgent(league="E0")
        result = agent.fit()
        assert result is agent

    def test_fit_with_data_returns_self(self):
        agent = LeagueSpecialistAgent(league="E0")
        result = agent.fit(matches_df=_make_league_df())
        assert result is agent


class TestLeagueSpecialistAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = LeagueSpecialistAgent(league="E0")
        ctx = _make_context(league="E0")
        assert agent.predict(ctx) is None

    def test_returns_none_for_wrong_league(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit()
        ctx = _make_context(league="SP1")
        assert agent.predict(ctx) is None

    def test_predict_for_correct_league(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit()
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "league_specialist_e0"

    def test_probabilities_sum_to_one(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_fit_updates_home_advantage(self):
        """With 5H out of 10, home advantage should be ~0.50."""
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        assert agent._home_advantage == pytest.approx(0.50, abs=0.05)

    def test_fit_updates_avg_goals(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        # Total goals: 3+0+4+2+1+2+2+1+1+3 = 19, avg = 1.9
        assert agent._league_avg_goals == pytest.approx(1.9, abs=0.2)

    def test_confidence_between_zero_and_one(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_metadata_has_league(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.metadata["league"] == "E0"

    def test_reasoning_contains_league(self):
        agent = LeagueSpecialistAgent(league="E0")
        agent.fit(matches_df=_make_league_df())
        ctx = _make_context(league="E0")
        pred = agent.predict(ctx)
        assert pred is not None
        assert "E0" in pred.reasoning
