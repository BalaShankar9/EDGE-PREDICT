"""Tests for SuperiorPredictor — best-of-everything ensemble."""

import numpy as np
import pandas as pd
import pytest

from sharpedge.core.superior import SuperiorPrediction, SuperiorPredictor


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_synthetic_data(n: int = 200, n_features: int = 10):
    """Return (X, y, matches, train_df) for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = rng.choice(["H", "D", "A"], n)
    matches = [
        {
            "home_team_id": f"t{i % 10}",
            "away_team_id": f"t{(i + 5) % 10}",
            "home_goals": int(rng.poisson(1.3)),
            "away_goals": int(rng.poisson(1.1)),
        }
        for i in range(n)
    ]
    train_df = pd.DataFrame({"league": ["Premier League"] * n, "FTR": y})
    y_ou = rng.integers(0, 2, n)
    y_btts = rng.integers(0, 2, n)
    return X, y, matches, train_df, y_ou, y_btts


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestSuperiorPrediction:
    """Test the dataclass independently."""

    def test_basic_construction(self):
        pred = SuperiorPrediction(
            home_team="Arsenal",
            away_team="Chelsea",
            league="Premier League",
            match_date="2026-04-05",
            probabilities=np.array([0.5, 0.25, 0.25]),
            predicted_outcome="Home",
            confidence=0.5,
            model_probs={},
            model_agreement=0.8,
            prediction_entropy=0.9,
            mc_std=0.05,
        )
        assert pred.predicted_outcome == "Home"
        assert pred.confidence == 0.5
        assert pred.home_team == "Arsenal"
        assert pred.away_team == "Chelsea"

    def test_defaults(self):
        pred = SuperiorPrediction(
            home_team="A",
            away_team="B",
            league="L",
            match_date="",
            probabilities=np.array([0.4, 0.3, 0.3]),
            predicted_outcome="Home",
            confidence=0.4,
            model_probs={},
            model_agreement=0.6,
            prediction_entropy=1.0,
            mc_std=0.08,
        )
        assert pred.implied_probs is None
        assert pred.edge == 0.0
        assert pred.is_value_bet is False
        assert pred.kelly_stake == 0.0
        assert pred.confidence_factors == []
        assert pred.risk_factors == []


class TestSuperiorPredictor:
    """Integration tests for the full predictor."""

    @pytest.fixture(scope="class")
    def fitted_predictor(self):
        """Fit once, reuse across tests in this class."""
        X, y, matches, train_df, y_ou, y_btts = _make_synthetic_data()
        predictor = SuperiorPredictor()
        predictor.fit(X, y, train_df, matches, y_ou=y_ou, y_btts=y_btts)
        return predictor, X

    def test_fit_marks_fitted(self, fitted_predictor):
        predictor, _ = fitted_predictor
        assert predictor._fitted is True

    def test_fit_has_minimum_models(self, fitted_predictor):
        predictor, _ = fitted_predictor
        # At minimum: xgb, dc, bvp
        assert "xgb" in predictor._models
        assert "dc" in predictor._models
        assert "bvp" in predictor._models
        assert len(predictor._models) >= 3

    def test_predict_basic(self, fitted_predictor):
        predictor, X = fitted_predictor
        pred = predictor.predict(X[0], "t0", "t5", league="Premier League")
        assert isinstance(pred, SuperiorPrediction)
        assert abs(sum(pred.probabilities) - 1.0) < 0.01
        assert pred.predicted_outcome in ("Home", "Draw", "Away")
        assert 0 <= pred.confidence <= 1
        assert 0 <= pred.model_agreement <= 1
        assert len(pred.model_probs) >= 3  # at least XGB, DC, BVP

    def test_predict_probabilities_sum_to_one(self, fitted_predictor):
        predictor, X = fitted_predictor
        for i in range(min(5, len(X))):
            pred = predictor.predict(X[i], f"t{i % 10}", f"t{(i + 5) % 10}")
            assert abs(pred.probabilities.sum() - 1.0) < 0.01, (
                f"Probabilities sum to {pred.probabilities.sum()}"
            )

    def test_predict_entropy_nonnegative(self, fitted_predictor):
        predictor, X = fitted_predictor
        pred = predictor.predict(X[0], "t0", "t5")
        assert pred.prediction_entropy >= 0

    def test_predict_with_odds(self, fitted_predictor):
        predictor, X = fitted_predictor
        pred = predictor.predict(
            X[0],
            "t0",
            "t5",
            odds={"home": 1.8, "draw": 3.5, "away": 4.2},
        )
        assert pred.implied_probs is not None
        assert len(pred.implied_probs) == 3
        # Implied probs should sum close to 1
        assert abs(sum(pred.implied_probs.values()) - 1.0) < 0.01
        # Edge should be a real number (positive or negative)
        assert isinstance(pred.edge, float)

    def test_predict_with_extreme_odds(self, fitted_predictor):
        predictor, X = fitted_predictor
        pred = predictor.predict(
            X[0],
            "t0",
            "t5",
            odds={"home": 1.05, "draw": 20.0, "away": 30.0},
        )
        assert pred.implied_probs is not None

    def test_not_fitted_raises(self):
        predictor = SuperiorPredictor()
        with pytest.raises(ValueError, match="Not fitted"):
            predictor.predict(np.zeros(10), "A", "B")

    def test_league_draw_rates_learned(self, fitted_predictor):
        predictor, _ = fitted_predictor
        assert "Premier League" in predictor._league_draw_rates
        rate = predictor._league_draw_rates["Premier League"]
        assert 0 < rate < 1

    def test_reasoning_populated(self, fitted_predictor):
        predictor, X = fitted_predictor
        pred = predictor.predict(X[0], "t0", "t5")
        assert len(pred.reasoning) > 0
        assert "models" in pred.reasoning

    def test_value_bet_detection(self, fitted_predictor):
        """When a model is very confident and odds are generous, flag value."""
        predictor, X = fitted_predictor
        # Give very generous odds for the predicted outcome
        pred_no_odds = predictor.predict(X[0], "t0", "t5")
        # Create odds that are very generous for the predicted outcome
        if pred_no_odds.predicted_outcome == "Home":
            odds = {"home": 5.0, "draw": 3.0, "away": 2.0}
        elif pred_no_odds.predicted_outcome == "Draw":
            odds = {"home": 3.0, "draw": 8.0, "away": 2.0}
        else:
            odds = {"home": 2.0, "draw": 3.0, "away": 5.0}
        pred = predictor.predict(X[0], "t0", "t5", odds=odds)
        # We just verify the fields are populated — actual value depends on model confidence
        assert isinstance(pred.is_value_bet, bool)
        assert isinstance(pred.expected_value, float)
        assert isinstance(pred.kelly_stake, float)
