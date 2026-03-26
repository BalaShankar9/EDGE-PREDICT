"""Poisson-based run expectancy model for baseball.

Baseball run scoring follows a distribution well-approximated by Poisson.
This model predicts expected runs per team, weighted heavily by starting
pitcher strength.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from sharpedge.core.base_model import BasePredictor

logger = logging.getLogger(__name__)

# Typical MLB averages
_AVG_RUNS_PER_TEAM = 4.5
_HOME_ADVANTAGE = 0.25  # home teams score slightly more
_PITCHER_WEIGHT = 0.6  # pitcher contributes 60% of run prevention


class RunModel(BasePredictor):
    """Poisson-based run expectancy model for baseball."""

    def __init__(self):
        self._team_offense: dict[str, float] = {}
        self._team_pitching: dict[str, float] = {}
        self._league_avg: float = _AVG_RUNS_PER_TEAM
        self._fitted = False
        self._gbm = None

    @property
    def name(self) -> str:
        return "baseball_run_model"

    def fit(self, X=None, y_margin=None, matches: list[dict] | None = None,
            **kwargs) -> "RunModel":
        """Train on historical scores or feature matrix."""
        if X is not None and y_margin is not None:
            return self._fit_gbm(X, y_margin)

        if matches is not None:
            return self._fit_poisson(matches)

        raise ValueError("Must provide either (X, y_margin) or matches")

    def _fit_poisson(self, matches: list[dict]) -> "RunModel":
        """Compute team offense/pitching strengths from historical scores."""
        team_rf: dict[str, list] = {}
        team_ra: dict[str, list] = {}

        for m in matches:
            ht = m["home_team"]
            at = m["away_team"]
            hr = m["home_score"]
            ar = m["away_score"]

            team_rf.setdefault(ht, []).append(hr)
            team_ra.setdefault(ht, []).append(ar)
            team_rf.setdefault(at, []).append(ar)
            team_ra.setdefault(at, []).append(hr)

        if not team_rf:
            self._fitted = True
            return self

        all_runs = []
        for rf in team_rf.values():
            all_runs.extend(rf)
        self._league_avg = np.mean(all_runs) if all_runs else _AVG_RUNS_PER_TEAM

        for team in team_rf:
            self._team_offense[team] = np.mean(team_rf[team]) / max(self._league_avg, 0.1)
            self._team_pitching[team] = np.mean(team_ra[team]) / max(self._league_avg, 0.1)

        self._fitted = True
        logger.info(
            "RunModel fitted (Poisson): %d teams, league_avg=%.2f",
            len(self._team_offense), self._league_avg,
        )
        return self

    def _fit_gbm(self, X, y_margin) -> "RunModel":
        """Train gradient booster on features + margin."""
        import lightgbm as lgb

        self._gbm = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbose=-1,
        )
        self._gbm.fit(X, y_margin)
        self._fitted = True
        logger.info("RunModel fitted (GBM) on %d samples", len(X))
        return self

    def predict_proba(self, X=None, home_team: str | None = None,
                      away_team: str | None = None,
                      **kwargs) -> NDArray[np.float64]:
        """Predict [P(home), P(away)] for moneyline."""
        if X is not None and self._gbm is not None:
            return self._predict_gbm(X)

        if home_team is not None and away_team is not None:
            return self._predict_poisson(home_team, away_team)

        raise ValueError("Must provide either X or (home_team, away_team)")

    def _predict_poisson(self, home_team: str, away_team: str) -> NDArray[np.float64]:
        """Predict using Poisson run model."""
        off_h = self._team_offense.get(home_team, 1.0)
        pit_h = self._team_pitching.get(home_team, 1.0)
        off_a = self._team_offense.get(away_team, 1.0)
        pit_a = self._team_pitching.get(away_team, 1.0)

        # Expected runs: offense * opponent pitching weakness + home advantage
        lambda_home = self._league_avg * off_h * pit_a + _HOME_ADVANTAGE
        lambda_away = self._league_avg * off_a * pit_h

        lambda_home = max(lambda_home, 1.0)
        lambda_away = max(lambda_away, 1.0)

        max_runs = 15
        p_home_win = 0.0
        p_away_win = 0.0

        for h in range(max_runs + 1):
            for a in range(max_runs + 1):
                p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                if h > a:
                    p_home_win += p
                elif a > h:
                    p_away_win += p
                # Ties go to extra innings (slight home advantage)
                else:
                    p_home_win += p * 0.52
                    p_away_win += p * 0.48

        total = p_home_win + p_away_win
        if total > 0:
            p_home_win /= total
            p_away_win /= total

        return np.array([[p_home_win, p_away_win]])

    def _predict_gbm(self, X) -> NDArray[np.float64]:
        """Predict using gradient booster margin model."""
        from scipy.stats import norm

        predicted_margin = self._gbm.predict(X)
        std = 3.5  # MLB run differential std

        p_home = norm.sf(0, loc=predicted_margin, scale=std)
        p_away = 1.0 - p_home

        return np.column_stack([p_home, p_away])

    def predict_total(self, X=None, total_line: float = 8.5,
                      **kwargs) -> NDArray[np.float64]:
        """Predict [P(over), P(under)] for total runs."""
        from scipy.stats import norm

        if self._gbm is not None and X is not None:
            predicted_margin = self._gbm.predict(X)
            predicted_total = 9.0 + predicted_margin * 0.1
        else:
            predicted_total = np.array([9.0])

        std = 2.5  # MLB total runs std
        p_over = norm.sf(total_line, loc=predicted_total, scale=std)
        return np.column_stack([p_over, 1.0 - p_over])
