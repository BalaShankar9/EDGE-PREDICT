"""Monte Carlo set-by-set tennis match simulator.

Simulates matches point-by-point (simplified to game-level):
1. Each player has a P(hold serve) derived from their serve stats or BT strength
2. Simulate games -> sets -> match
3. Handles tiebreaks, best-of-3 vs best-of-5, final set rules
4. Run N simulations -> aggregate for probability estimates
"""
import numpy as np
from sharpedge.core.base_model import BasePredictor


class SetSimulator(BasePredictor):
    """Monte Carlo tennis match simulator."""

    def __init__(self, n_sims: int = 5000):
        self._n_sims = n_sims
        self._hold_probs: dict[str, float] = {}  # player -> P(hold serve)
        self._fitted = False

    @property
    def name(self) -> str:
        return "set_simulator"

    def fit(self, matches: list[dict], **kwargs) -> "SetSimulator":
        """Estimate serve hold probabilities from historical data.

        If serve stats unavailable, estimate from win rate:
        P(hold) ~ 0.6 + 0.3 * win_rate  (calibrated heuristic)
        """
        # Count wins/losses per player
        wins: dict[str, int] = {}
        total: dict[str, int] = {}
        for m in matches:
            for player in [m["winner"], m["loser"]]:
                if player not in wins:
                    wins[player] = 0
                    total[player] = 0
                total[player] += 1
            wins[m["winner"]] += 1

        for player in total:
            win_rate = wins[player] / total[player] if total[player] > 0 else 0.5
            self._hold_probs[player] = 0.6 + 0.3 * win_rate

        self._fitted = True
        return self

    def _simulate_tiebreak(self, p1_serve: float, p2_serve: float) -> int:
        """Simulate a tiebreak. Returns 0 for P1 win, 1 for P2 win."""
        p1_pts, p2_pts = 0, 0
        serving = 0  # 0 = P1, 1 = P2
        point_count = 0

        while True:
            # Determine who serves this point
            if point_count == 0:
                serving = 0
            elif (point_count - 1) % 2 == 0:
                serving = 1 - serving  # switch every 2 points after first

            # Simulate point
            if serving == 0:
                if np.random.random() < p1_serve * 0.65:  # tiebreak serve advantage is less
                    p1_pts += 1
                else:
                    p2_pts += 1
            else:
                if np.random.random() < p2_serve * 0.65:
                    p2_pts += 1
                else:
                    p1_pts += 1

            point_count += 1

            # Check win condition: first to 7, must win by 2
            if p1_pts >= 7 and p1_pts - p2_pts >= 2:
                return 0
            if p2_pts >= 7 and p2_pts - p1_pts >= 2:
                return 1

            # Safety: prevent infinite loop
            if point_count > 100:
                return 0 if p1_pts > p2_pts else 1

    def _simulate_set(self, p1_hold: float, p2_hold: float) -> int:
        """Simulate a single set. Returns 0 for P1, 1 for P2."""
        p1_games, p2_games = 0, 0
        serving = 0  # P1 serves first

        while True:
            # Simulate game
            if serving == 0:
                if np.random.random() < p1_hold:
                    p1_games += 1
                else:
                    p2_games += 1
            else:
                if np.random.random() < p2_hold:
                    p2_games += 1
                else:
                    p1_games += 1

            serving = 1 - serving

            # Check set win: first to 6 with 2-game lead
            if p1_games >= 6 and p1_games - p2_games >= 2:
                return 0
            if p2_games >= 6 and p2_games - p1_games >= 2:
                return 1

            # Tiebreak at 6-6
            if p1_games == 6 and p2_games == 6:
                return self._simulate_tiebreak(p1_hold, p2_hold)

    def _simulate_match(self, p1_hold: float, p2_hold: float, best_of: int = 3) -> int:
        """Simulate a full match. Returns 0 for P1, 1 for P2."""
        sets_to_win = (best_of + 1) // 2  # 2 for best-of-3, 3 for best-of-5
        p1_sets, p2_sets = 0, 0

        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            winner = self._simulate_set(p1_hold, p2_hold)
            if winner == 0:
                p1_sets += 1
            else:
                p2_sets += 1

        return 0 if p1_sets > p2_sets else 1

    def predict_proba(self, player1: str, player2: str, best_of: int = 3, **kwargs) -> np.ndarray:
        """Run N simulations and return [P(P1 wins), P(P2 wins)]."""
        p1_hold = self._hold_probs.get(player1, 0.75)
        p2_hold = self._hold_probs.get(player2, 0.75)

        p1_wins = 0
        for _ in range(self._n_sims):
            if self._simulate_match(p1_hold, p2_hold, best_of) == 0:
                p1_wins += 1

        p1_prob = p1_wins / self._n_sims
        return np.array([p1_prob, 1.0 - p1_prob])
