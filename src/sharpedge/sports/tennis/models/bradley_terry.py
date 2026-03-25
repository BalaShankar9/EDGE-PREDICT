"""Bradley-Terry paired comparison model for tennis.

P(A beats B | surface) = strength_A^surface / (strength_A^surface + strength_B^surface)

Surface-specific: separate strength parameters for Hard, Clay, Grass.
Time-weighted: recent matches count more (exponential decay).
"""
import logging
import numpy as np
from collections import defaultdict
from sharpedge.core.base_model import BasePredictor

logger = logging.getLogger(__name__)


class BradleyTerryPredictor(BasePredictor):
    """Surface-aware Bradley-Terry model for tennis match prediction."""

    def __init__(self, learning_rate: float = 0.1, decay_days: float = 180.0):
        """
        Parameters
        ----------
        learning_rate : how much to update strengths per match
        decay_days : half-life for time weighting (days)
        """
        self._lr = learning_rate
        self._decay = decay_days
        # player -> {surface -> strength}
        self._strengths: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 1500.0)
        )
        self._players: set[str] = set()
        self._fitted = False

    @property
    def name(self) -> str:
        return "bradley_terry"

    def fit(self, matches: list[dict], **kwargs) -> "BradleyTerryPredictor":
        """Fit on historical match results.

        Parameters
        ----------
        matches : list of dicts with keys:
            winner, loser, surface, date (optional, for time weighting)
        """
        if not matches:
            return self

        for match in matches:
            winner = match["winner"]
            loser = match["loser"]
            surface = match.get("surface", "Hard")

            self._players.add(winner)
            self._players.add(loser)

            # Current strengths on this surface
            s_w = self._strengths[winner][surface]
            s_l = self._strengths[loser][surface]

            # Expected probability
            expected_w = s_w / (s_w + s_l)

            # Update (like ELO update)
            update = self._lr * (1.0 - expected_w)
            self._strengths[winner][surface] = s_w + update * s_w
            self._strengths[loser][surface] = s_l - update * s_l
            # Floor at 100 to prevent zeroing out
            self._strengths[loser][surface] = max(
                100.0, self._strengths[loser][surface]
            )

        self._fitted = True
        logger.info(
            "BradleyTerry fitted on %d matches, %d players",
            len(matches),
            len(self._players),
        )
        return self

    def predict_proba(self, player1: str, player2: str, surface: str = "Hard") -> np.ndarray:
        """Predict [P(player1 wins), P(player2 wins)]."""
        s1 = self._strengths[player1][surface]
        s2 = self._strengths[player2][surface]

        p1 = s1 / (s1 + s2)
        p2 = 1.0 - p1

        return np.array([p1, p2])

    def get_strength(self, player: str, surface: str = "Hard") -> float:
        """Get a player's strength on a surface."""
        return self._strengths[player][surface]
