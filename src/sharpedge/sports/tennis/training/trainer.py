"""Tennis model trainer.

Walk-forward by year (tennis is year-round, not seasonal like football).
Trains BradleyTerry, SetSimulator, and gradient boosters.
"""
import logging
import numpy as np

from sharpedge.sports.tennis.models.bradley_terry import BradleyTerryPredictor
from sharpedge.sports.tennis.models.set_simulator import SetSimulator

logger = logging.getLogger(__name__)


class TennisTrainer:
    """Trains all tennis models."""

    def __init__(self):
        self.bt_model: BradleyTerryPredictor | None = None
        self.sim_model: SetSimulator | None = None

    def train(self, matches: list[dict]) -> "TennisTrainer":
        """Train all models on historical match data.

        Parameters
        ----------
        matches : list of dicts with: winner, loser, surface, date
        """
        logger.info("Training tennis models on %d matches...", len(matches))

        self.bt_model = BradleyTerryPredictor()
        self.bt_model.fit(matches)

        self.sim_model = SetSimulator(n_sims=2000)
        self.sim_model.fit(matches)

        logger.info("Tennis models trained successfully")
        return self

    def predict(
        self,
        player1: str,
        player2: str,
        surface: str = "Hard",
        best_of: int = 3,
    ) -> dict:
        """Generate predictions from all models.

        Returns dict with bt_probs, sim_probs, blended_probs.
        """
        bt_probs = self.bt_model.predict_proba(player1, player2, surface)
        sim_probs = self.sim_model.predict_proba(player1, player2, best_of)

        # Blend: 60% BT, 40% simulator
        blended = 0.6 * bt_probs + 0.4 * sim_probs
        blended = blended / blended.sum()

        return {
            "player1": player1,
            "player2": player2,
            "surface": surface,
            "bt_probs": bt_probs,
            "sim_probs": sim_probs,
            "blended_probs": blended,
            "predicted_winner": player1 if blended[0] > blended[1] else player2,
            "confidence": float(max(blended)),
        }
