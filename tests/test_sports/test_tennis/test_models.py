"""Tests for tennis models: BradleyTerry, SetSimulator, and TennisTrainer."""

import numpy as np
import pytest

from sharpedge.sports.tennis.models.bradley_terry import BradleyTerryPredictor
from sharpedge.sports.tennis.models.set_simulator import SetSimulator
from sharpedge.sports.tennis.training.trainer import TennisTrainer

# ---------------------------------------------------------------------------
# Realistic test match data
# ---------------------------------------------------------------------------
TEST_MATCHES = [
    # Hard court: Djokovic dominates
    {"winner": "Djokovic", "loser": "Nadal", "surface": "Hard"},
    {"winner": "Djokovic", "loser": "Federer", "surface": "Hard"},
    {"winner": "Djokovic", "loser": "Nadal", "surface": "Hard"},
    {"winner": "Djokovic", "loser": "Medvedev", "surface": "Hard"},
    {"winner": "Djokovic", "loser": "Zverev", "surface": "Hard"},
    {"winner": "Medvedev", "loser": "Zverev", "surface": "Hard"},
    {"winner": "Nadal", "loser": "Medvedev", "surface": "Hard"},
    {"winner": "Federer", "loser": "Zverev", "surface": "Hard"},
    # Clay court: Nadal dominates
    {"winner": "Nadal", "loser": "Djokovic", "surface": "Clay"},
    {"winner": "Nadal", "loser": "Federer", "surface": "Clay"},
    {"winner": "Nadal", "loser": "Djokovic", "surface": "Clay"},
    {"winner": "Nadal", "loser": "Zverev", "surface": "Clay"},
    {"winner": "Nadal", "loser": "Medvedev", "surface": "Clay"},
    {"winner": "Djokovic", "loser": "Federer", "surface": "Clay"},
    {"winner": "Djokovic", "loser": "Zverev", "surface": "Clay"},
    # Grass court: Federer dominates
    {"winner": "Federer", "loser": "Nadal", "surface": "Grass"},
    {"winner": "Federer", "loser": "Djokovic", "surface": "Grass"},
    {"winner": "Federer", "loser": "Zverev", "surface": "Grass"},
    {"winner": "Federer", "loser": "Medvedev", "surface": "Grass"},
    {"winner": "Djokovic", "loser": "Nadal", "surface": "Grass"},
    {"winner": "Djokovic", "loser": "Zverev", "surface": "Grass"},
    # A few extra to bulk up
    {"winner": "Djokovic", "loser": "Federer", "surface": "Hard"},
    {"winner": "Nadal", "loser": "Zverev", "surface": "Clay"},
    {"winner": "Federer", "loser": "Medvedev", "surface": "Grass"},
]


# ====================================================================
# BradleyTerry tests
# ====================================================================
class TestBradleyTerry:
    def test_fit_and_predict_shape(self):
        model = BradleyTerryPredictor()
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Djokovic", "Nadal", "Hard")
        assert probs.shape == (2,)

    def test_probabilities_sum_to_one(self):
        model = BradleyTerryPredictor()
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Djokovic", "Nadal", "Hard")
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_stronger_player_higher_prob(self):
        """Djokovic should be favored over Nadal on Hard."""
        model = BradleyTerryPredictor()
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Djokovic", "Nadal", "Hard")
        assert probs[0] > probs[1], "Djokovic should be favored on Hard"

    def test_surface_specificity(self):
        """Nadal should be stronger than Djokovic on Clay."""
        model = BradleyTerryPredictor()
        model.fit(TEST_MATCHES)
        nadal_clay = model.get_strength("Nadal", "Clay")
        djokovic_clay = model.get_strength("Djokovic", "Clay")
        assert nadal_clay > djokovic_clay, "Nadal should be stronger on Clay"

    def test_federer_dominates_grass(self):
        model = BradleyTerryPredictor()
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Federer", "Djokovic", "Grass")
        assert probs[0] > probs[1], "Federer should be favored on Grass"

    def test_unknown_player_gets_default(self):
        """Unknown players should get the default 1500 strength."""
        model = BradleyTerryPredictor()
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Unknown1", "Unknown2", "Hard")
        assert abs(probs[0] - 0.5) < 1e-9, "Two unknowns should be 50/50"

    def test_empty_matches(self):
        model = BradleyTerryPredictor()
        result = model.fit([])
        assert result is model

    def test_name(self):
        assert BradleyTerryPredictor().name == "bradley_terry"


# ====================================================================
# SetSimulator tests
# ====================================================================
class TestSetSimulator:
    def test_fit_and_predict_shape(self):
        model = SetSimulator(n_sims=500)
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Djokovic", "Nadal")
        assert probs.shape == (2,)

    def test_probabilities_sum_to_one(self):
        model = SetSimulator(n_sims=500)
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Djokovic", "Nadal")
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_stronger_player_wins_more(self):
        """Player with more wins should be favored."""
        np.random.seed(42)
        model = SetSimulator(n_sims=2000)
        model.fit(TEST_MATCHES)
        # Djokovic has a high overall win rate
        probs = model.predict_proba("Djokovic", "Zverev")
        assert probs[0] > probs[1], "Djokovic should beat Zverev more often"

    def test_best_of_5_amplifies_favorite(self):
        """Best-of-5 should give the stronger player a higher win prob than best-of-3."""
        np.random.seed(42)
        model = SetSimulator(n_sims=3000)
        model.fit(TEST_MATCHES)
        probs_bo3 = model.predict_proba("Djokovic", "Zverev", best_of=3)
        np.random.seed(42)
        probs_bo5 = model.predict_proba("Djokovic", "Zverev", best_of=5)
        # With more sets, the favorite's edge compounds
        assert probs_bo5[0] > probs_bo3[0] - 0.05, (
            "Best-of-5 should not significantly hurt the favorite"
        )

    def test_equal_players_near_fifty_fifty(self):
        """Two unknown (equal) players should be close to 50/50."""
        np.random.seed(42)
        model = SetSimulator(n_sims=2000)
        model.fit(TEST_MATCHES)
        probs = model.predict_proba("Unknown1", "Unknown2")
        assert abs(probs[0] - 0.5) < 0.05, "Equal players should be near 50/50"

    def test_name(self):
        assert SetSimulator().name == "set_simulator"


# ====================================================================
# TennisTrainer tests
# ====================================================================
class TestTennisTrainer:
    def test_train_fits_both_models(self):
        trainer = TennisTrainer()
        trainer.train(TEST_MATCHES)
        assert trainer.bt_model is not None
        assert trainer.sim_model is not None
        assert trainer.bt_model._fitted
        assert trainer.sim_model._fitted

    def test_predict_returns_blended_results(self):
        np.random.seed(42)
        trainer = TennisTrainer()
        trainer.train(TEST_MATCHES)
        result = trainer.predict("Djokovic", "Nadal", surface="Hard")

        assert "bt_probs" in result
        assert "sim_probs" in result
        assert "blended_probs" in result
        assert "predicted_winner" in result
        assert "confidence" in result

        # Blended probs should sum to 1
        assert abs(result["blended_probs"].sum() - 1.0) < 1e-9

    def test_predicted_winner_has_higher_prob(self):
        np.random.seed(42)
        trainer = TennisTrainer()
        trainer.train(TEST_MATCHES)
        result = trainer.predict("Djokovic", "Nadal", surface="Hard")
        winner = result["predicted_winner"]
        blended = result["blended_probs"]

        if winner == "Djokovic":
            assert blended[0] >= blended[1]
        else:
            assert blended[1] >= blended[0]

    def test_confidence_range(self):
        np.random.seed(42)
        trainer = TennisTrainer()
        trainer.train(TEST_MATCHES)
        result = trainer.predict("Djokovic", "Nadal", surface="Hard")
        assert 0.5 <= result["confidence"] <= 1.0
