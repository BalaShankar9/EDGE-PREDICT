"""Tests for the Form Momentum Agent (rolling form tracker)."""
import numpy as np
import pandas as pd
import pytest

from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.form_momentum_agent import FormMomentumAgent


def _make_matches_df(n: int = 20) -> pd.DataFrame:
    """Generate a synthetic DataFrame of matches for 4 teams."""
    rng = np.random.RandomState(42)
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    rows = []
    base_date = pd.Timestamp("2026-01-01")
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 1) % len(teams)]
        hg = int(rng.poisson(1.5))
        ag = int(rng.poisson(1.1))
        if hg > ag:
            ftr = "H"
        elif hg == ag:
            ftr = "D"
        else:
            ftr = "A"
        rows.append({
            "home_team_id": home,
            "away_team_id": away,
            "FTHG": hg,
            "FTAG": ag,
            "FTR": ftr,
            "match_date": (base_date + pd.Timedelta(days=i)).isoformat(),
        })
    return pd.DataFrame(rows)


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


class TestFormMomentumAgentProperties:
    def test_name(self):
        agent = FormMomentumAgent()
        assert agent.name == "form_momentum_agent"

    def test_agent_type(self):
        agent = FormMomentumAgent()
        assert agent.agent_type == "context"

    def test_supported_sports(self):
        agent = FormMomentumAgent()
        assert agent.supported_sports == ["football"]

    def test_supports_football(self):
        agent = FormMomentumAgent()
        assert agent.supports_sport("football") is True

    def test_does_not_support_tennis(self):
        agent = FormMomentumAgent()
        assert agent.supports_sport("tennis") is False

    def test_description_is_nonempty(self):
        agent = FormMomentumAgent()
        assert len(agent.description) > 0


class TestFormMomentumAgentPredict:
    def test_returns_none_when_not_fitted(self):
        agent = FormMomentumAgent()
        ctx = _make_context()
        assert agent.predict(ctx) is None

    def test_returns_none_for_non_football(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context(sport="tennis")
        assert agent.predict(ctx) is None

    def test_returns_none_for_unknown_team(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context(home="UnknownFC", away="TeamA")
        assert agent.predict(ctx) is None

    def test_fit_and_predict_returns_prediction(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.agent_name == "form_momentum_agent"
        assert pred.sport == "football"

    def test_probabilities_sum_to_one(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert abs(float(np.sum(pred.probabilities)) - 1.0) < 0.01

    def test_probabilities_correct_length(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.probabilities) == len(pred.outcomes)

    def test_predicted_outcome_in_outcomes(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.predicted_outcome in pred.outcomes

    def test_confidence_between_zero_and_one(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert 0.0 <= pred.confidence <= 1.0

    def test_uncertainty_is_nonnegative(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert pred.uncertainty >= 0.0

    def test_reasoning_is_nonempty(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.reasoning) > 0
        assert "form" in pred.reasoning.lower() or "pts" in pred.reasoning.lower()

    def test_features_used_populated(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        ctx = _make_context()
        pred = agent.predict(ctx)
        assert pred is not None
        assert len(pred.features_used) > 0

    def test_fit_returns_self(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        result = agent.fit(df)
        assert result is agent

    def test_fit_raises_on_missing_columns(self):
        agent = FormMomentumAgent()
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            agent.fit(df)

    def test_predict_batch(self):
        agent = FormMomentumAgent()
        df = _make_matches_df()
        agent.fit(df)
        contexts = [
            _make_context(home="TeamA", away="TeamB"),
            _make_context(home="TeamC", away="TeamD"),
        ]
        preds = agent.predict_batch(contexts)
        assert len(preds) == 2
        for p in preds:
            assert abs(float(np.sum(p.probabilities)) - 1.0) < 0.01
