"""Poisson-based goal model for hockey, with overtime adjustment.

Hockey goals follow a Poisson distribution well. This model predicts
regulation-time goal distributions and then models overtime as a slight
home-favored coin flip.
"""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from sharpedge.core.base_model import BasePredictor

logger = logging.getLogger(__name__)

# Typical NHL averages
_AVG_GOALS_PER_TEAM = 3.0
_HOME_ADVANTAGE = 0.15  # home teams score ~0.15 more goals
_OT_HOME_EDGE = 0.52  # slight home advantage in overtime


class HockeyGoalModel(BasePredictor):
    """Poisson-based goal model for hockey, with overtime adjustment."""

    def __init__(self):
        self._team_attack: dict[str, float] = {}
        self._team_defence: dict[str, float] = {}
        self._league_avg: float = _AVG_GOALS_PER_TEAM
        self._fitted = False
        self._gbm = None

    @property
    def name(self) -> str:
        return "hockey_goal_model"

    def fit(self, X=None, y_margin=None, matches: list[dict] | None = None,
            **kwargs) -> "HockeyGoalModel":
        """Train on historical scores or feature matrix.

        Supports two modes:
        1. matches: list of dicts with home_team, away_team, home_score, away_score
        2. X, y_margin: feature matrix and margin for gradient boosting
        """
        if X is not None and y_margin is not None:
            return self._fit_gbm(X, y_margin)

        if matches is not None:
            return self._fit_poisson(matches)

        raise ValueError("Must provide either (X, y_margin) or matches")

    def _fit_poisson(self, matches: list[dict]) -> "HockeyGoalModel":
        """Compute team attack/defence strengths from historical scores."""
        team_gf: dict[str, list] = {}
        team_ga: dict[str, list] = {}

        for m in matches:
            ht = m["home_team"]
            at = m["away_team"]
            hg = m["home_score"]
            ag = m["away_score"]

            team_gf.setdefault(ht, []).append(hg)
            team_ga.setdefault(ht, []).append(ag)
            team_gf.setdefault(at, []).append(ag)
            team_ga.setdefault(at, []).append(hg)

        if not team_gf:
            self._fitted = True
            return self

        all_goals = []
        for gf in team_gf.values():
            all_goals.extend(gf)
        self._league_avg = np.mean(all_goals) if all_goals else _AVG_GOALS_PER_TEAM

        for team in team_gf:
            self._team_attack[team] = np.mean(team_gf[team]) / max(self._league_avg, 0.1)
            self._team_defence[team] = np.mean(team_ga[team]) / max(self._league_avg, 0.1)

        self._fitted = True
        logger.info(
            "HockeyGoalModel fitted (Poisson): %d teams, league_avg=%.2f",
            len(self._team_attack), self._league_avg,
        )
        return self

    def _fit_gbm(self, X, y_margin) -> "HockeyGoalModel":
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
        logger.info("HockeyGoalModel fitted (GBM) on %d samples", len(X))
        return self

    def predict_proba(self, X=None, home_team: str | None = None,
                      away_team: str | None = None,
                      **kwargs) -> NDArray[np.float64]:
        """Predict [P(home), P(away)] — no draws in hockey (OT decides).

        Supports two modes:
        1. X: feature matrix (uses GBM)
        2. home_team, away_team: team names (uses Poisson)
        """
        if X is not None and self._gbm is not None:
            return self._predict_gbm(X)

        if home_team is not None and away_team is not None:
            return self._predict_poisson(home_team, away_team)

        raise ValueError("Must provide either X or (home_team, away_team)")

    def _predict_poisson(self, home_team: str, away_team: str) -> NDArray[np.float64]:
        """Predict using Poisson model with overtime adjustment."""
        att_h = self._team_attack.get(home_team, 1.0)
        def_h = self._team_defence.get(home_team, 1.0)
        att_a = self._team_attack.get(away_team, 1.0)
        def_a = self._team_defence.get(away_team, 1.0)

        # Expected goals
        lambda_home = self._league_avg * att_h * def_a + _HOME_ADVANTAGE
        lambda_away = self._league_avg * att_a * def_h

        lambda_home = max(lambda_home, 0.5)
        lambda_away = max(lambda_away, 0.5)

        max_goals = 10
        p_home_win = 0.0
        p_away_win = 0.0
        p_draw = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lambda_home) * poisson.pmf(a, lambda_away)
                if h > a:
                    p_home_win += p
                elif a > h:
                    p_away_win += p
                else:
                    p_draw += p

        # Overtime: distribute draw probability with slight home edge
        p_home_final = p_home_win + p_draw * _OT_HOME_EDGE
        p_away_final = p_away_win + p_draw * (1.0 - _OT_HOME_EDGE)

        total = p_home_final + p_away_final
        if total > 0:
            p_home_final /= total
            p_away_final /= total

        return np.array([[p_home_final, p_away_final]])

    def _predict_gbm(self, X) -> NDArray[np.float64]:
        """Predict using gradient booster margin model."""
        from scipy.stats import norm

        predicted_margin = self._gbm.predict(X)
        std = 2.5  # NHL margin std is ~2.5 goals

        p_home = norm.sf(0, loc=predicted_margin, scale=std)
        p_away = 1.0 - p_home

        return np.column_stack([p_home, p_away])

    def predict_total(self, X=None, total_line: float = 5.5,
                      **kwargs) -> NDArray[np.float64]:
        """Predict [P(over), P(under)] for total goals."""
        from scipy.stats import norm

        if self._gbm is not None and X is not None:
            predicted_margin = self._gbm.predict(X)
            predicted_total = 5.5 + predicted_margin * 0.1
        else:
            predicted_total = np.array([5.5])

        std = 1.8  # NHL total goals std
        p_over = norm.sf(total_line, loc=predicted_total, scale=std)
        return np.column_stack([p_over, 1.0 - p_over])
