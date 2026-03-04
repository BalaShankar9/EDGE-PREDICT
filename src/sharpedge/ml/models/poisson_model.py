"""Bivariate Poisson model for football goal predictions.

Models home and away goals as independent Poisson distributions,
with team attack/defence strengths estimated from historical data.
Derives: 1X2 probabilities, O/U probabilities, BTTS, correct score.
"""
import numpy as np
from scipy.stats import poisson
from numpy.typing import NDArray


class PoissonPredictor:
    """Bivariate Poisson goal model."""

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.home_advantage: float = 0.0
        self.attack_strength: dict[str, float] = {}
        self.defence_strength: dict[str, float] = {}
        self.league_avg_home: float = 1.5
        self.league_avg_away: float = 1.2

    def fit(self, matches: list[dict]) -> "PoissonPredictor":
        """Estimate team strengths from historical match data.

        Expects list of dicts with keys: home_team_id, away_team_id, home_goals, away_goals.
        """
        home_goals = [m["home_goals"] for m in matches]
        away_goals = [m["away_goals"] for m in matches]
        self.league_avg_home = np.mean(home_goals)
        self.league_avg_away = np.mean(away_goals)
        self.home_advantage = self.league_avg_home / max(self.league_avg_away, 0.01)

        team_home_scored = {}
        team_home_conceded = {}
        team_away_scored = {}
        team_away_conceded = {}

        for m in matches:
            h, a = m["home_team_id"], m["away_team_id"]
            hg, ag = m["home_goals"], m["away_goals"]

            team_home_scored.setdefault(h, []).append(hg)
            team_home_conceded.setdefault(h, []).append(ag)
            team_away_scored.setdefault(a, []).append(ag)
            team_away_conceded.setdefault(a, []).append(hg)

        all_teams = set(list(team_home_scored.keys()) + list(team_away_scored.keys()))
        for team in all_teams:
            h_scored = np.mean(team_home_scored.get(team, [self.league_avg_home]))
            a_scored = np.mean(team_away_scored.get(team, [self.league_avg_away]))
            h_conceded = np.mean(team_home_conceded.get(team, [self.league_avg_away]))
            a_conceded = np.mean(team_away_conceded.get(team, [self.league_avg_home]))

            self.attack_strength[team] = (h_scored / max(self.league_avg_home, 0.01) + a_scored / max(self.league_avg_away, 0.01)) / 2
            self.defence_strength[team] = (h_conceded / max(self.league_avg_away, 0.01) + a_conceded / max(self.league_avg_home, 0.01)) / 2

        return self

    def predict_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Predict expected goals for home and away teams."""
        home_att = self.attack_strength.get(home_team, 1.0)
        away_def = self.defence_strength.get(away_team, 1.0)
        away_att = self.attack_strength.get(away_team, 1.0)
        home_def = self.defence_strength.get(home_team, 1.0)

        home_xg = home_att * away_def * self.league_avg_home
        away_xg = away_att * home_def * self.league_avg_away

        return home_xg, away_xg

    def predict_scoreline_matrix(self, home_team: str, away_team: str) -> NDArray[np.float64]:
        """Compute full goal probability matrix P(home=i, away=j).
        Returns (max_goals+1, max_goals+1) matrix.
        """
        home_xg, away_xg = self.predict_goals(home_team, away_team)
        n = self.max_goals + 1
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
        return matrix

    def predict_proba_1x2(self, home_team: str, away_team: str) -> NDArray[np.float64]:
        """Predict 1X2 from scoreline matrix. Returns [P(H), P(D), P(A)]."""
        matrix = self.predict_scoreline_matrix(home_team, away_team)
        n = self.max_goals + 1
        p_home = sum(matrix[i, j] for i in range(n) for j in range(n) if i > j)
        p_draw = sum(matrix[i, i] for i in range(n))
        p_away = sum(matrix[i, j] for i in range(n) for j in range(n) if i < j)
        total = p_home + p_draw + p_away
        return np.array([p_home / total, p_draw / total, p_away / total])

    def predict_proba_ou(self, home_team: str, away_team: str, line: float = 2.5) -> float:
        """Predict P(Over given line)."""
        matrix = self.predict_scoreline_matrix(home_team, away_team)
        n = self.max_goals + 1
        p_over = sum(matrix[i, j] for i in range(n) for j in range(n) if i + j > line)
        return float(p_over)

    def predict_proba_btts(self, home_team: str, away_team: str) -> float:
        """Predict P(Both Teams To Score)."""
        matrix = self.predict_scoreline_matrix(home_team, away_team)
        n = self.max_goals + 1
        p_btts = sum(matrix[i, j] for i in range(1, n) for j in range(1, n))
        return float(p_btts)

    def predict_correct_score(self, home_team: str, away_team: str, top_n: int = 5) -> list[dict]:
        """Return top N most likely scorelines."""
        matrix = self.predict_scoreline_matrix(home_team, away_team)
        scores = []
        n = self.max_goals + 1
        for i in range(n):
            for j in range(n):
                scores.append({"home": i, "away": j, "prob": matrix[i, j]})
        scores.sort(key=lambda x: x["prob"], reverse=True)
        return scores[:top_n]
