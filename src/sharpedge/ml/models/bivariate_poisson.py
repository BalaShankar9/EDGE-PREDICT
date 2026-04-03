"""Bivariate Poisson goal model via penaltyblog.

Unlike the basic PoissonPredictor which treats goals as independent,
the bivariate model introduces a shared intensity parameter that
directly captures goal correlation between home and away teams.
"""
import logging
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class BivariatePoissonPredictor:
    """Bivariate Poisson model wrapping penaltyblog."""

    def __init__(self):
        self._model = None
        self._teams: list[str] = []
        self._fitted = False

    def fit(self, matches: list[dict]) -> "BivariatePoissonPredictor":
        """Fit bivariate Poisson model on historical match data.

        Parameters
        ----------
        matches : list of dicts with home_team_id, away_team_id, home_goals, away_goals
        """
        try:
            from penaltyblog.models import BivariatePoissonGoalModel

            # Build arrays
            home_teams = [m["home_team_id"] for m in matches if m["home_team_id"] is not None]
            away_teams = [m["away_team_id"] for m in matches if m["away_team_id"] is not None]

            # Filter out matches with None team IDs
            valid = [
                m for m in matches
                if m["home_team_id"] is not None and m["away_team_id"] is not None
            ]

            if len(valid) < 50:
                logger.warning(f"BVP: only {len(valid)} valid matches, need 50+")
                self._fitted = False
                return self

            home = [m["home_team_id"] for m in valid]
            away = [m["away_team_id"] for m in valid]
            hg = [int(m["home_goals"]) for m in valid]
            ag = [int(m["away_goals"]) for m in valid]

            self._teams = sorted(set(home + away))
            self._model = BivariatePoissonGoalModel(home, away, hg, ag)
            self._model.fit()
            self._fitted = True

            logger.info(
                f"BivariatePoissonPredictor fitted on {len(valid)} matches, "
                f"{len(self._teams)} teams"
            )
        except ImportError:
            logger.warning("penaltyblog not available — BVP disabled")
            self._fitted = False
        except Exception as e:
            logger.warning(f"BVP fit failed: {e}")
            self._fitted = False

        return self

    def predict_proba_1x2(
        self, home_team: str, away_team: str
    ) -> NDArray[np.float64]:
        """Predict [P(Home), P(Draw), P(Away)] for a match.

        Falls back to [1/3, 1/3, 1/3] for unknown teams.
        """
        if not self._fitted or self._model is None:
            return np.array([1 / 3, 1 / 3, 1 / 3])

        if home_team not in self._teams or away_team not in self._teams:
            return np.array([1 / 3, 1 / 3, 1 / 3])

        try:
            probs = self._model.predict(home_team, away_team)
            h = float(probs.home_win)
            d = float(probs.draw)
            a = float(probs.away_win)
            total = h + d + a
            if total > 0:
                return np.array([h / total, d / total, a / total])
        except Exception:
            pass

        return np.array([1 / 3, 1 / 3, 1 / 3])

    def predict_ou(
        self, home_team: str, away_team: str, line: float = 2.5
    ) -> float:
        """Predict P(over line) for a match."""
        if not self._fitted or self._model is None:
            return 0.5

        try:
            probs = self._model.predict(home_team, away_team)
            return float(probs.over(line))
        except Exception:
            return 0.5

    def predict_btts(self, home_team: str, away_team: str) -> float:
        """Predict P(both teams score)."""
        if not self._fitted or self._model is None:
            return 0.5

        try:
            probs = self._model.predict(home_team, away_team)
            return float(probs.btts_yes)
        except Exception:
            return 0.5
