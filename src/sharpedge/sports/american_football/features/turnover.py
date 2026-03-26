"""NFL turnover features (4 total).

Features capturing turnover tendencies and luck regression.

Features:
  nfl_to_margin                - Turnover margin (rolling)
  nfl_fumble_rate              - Fumble rate (proxy from score variance)
  nfl_int_rate                 - Interception rate proxy
  nfl_turnover_luck_regression - Regression toward mean of turnover luck
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nfl_to_margin",
    "nfl_fumble_rate",
    "nfl_int_rate",
    "nfl_turnover_luck_regression",
]

_ROLLING_N = 5


def _score_variance(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Compute score variance as proxy for turnover-related volatility."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) < 2:
        return np.nan

    scores = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            scores.append(g.get("home_score", np.nan))
        else:
            scores.append(g.get("away_score", np.nan))

    valid = [s for s in scores if pd.notna(s)]
    return np.std(valid) if len(valid) >= 2 else np.nan


def _point_diff(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Average point differential for a team."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    diffs = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            diffs.append(g.get("home_score", 0) - g.get("away_score", 0))
        else:
            diffs.append(g.get("away_score", 0) - g.get("home_score", 0))

    return np.mean(diffs) if diffs else np.nan


class TurnoverFeatures(FeatureGroup):
    name = "nfl_turnover"
    feature_count = 4

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["game_date"] = pd.to_datetime(df.get("game_date", df.get("date")))
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            match_date = row["game_date"]
            prior = df[df["game_date"] < match_date]

            if len(prior) == 0:
                continue

            pd_h = _point_diff(prior, home)
            pd_a = _point_diff(prior, away)

            # Turnover margin proxy (from point differential)
            if pd.notna(pd_h) and pd.notna(pd_a):
                result.loc[idx, "nfl_to_margin"] = (pd_h - pd_a) / 14.0

            # Fumble rate proxy
            var_h = _score_variance(prior, home)
            if pd.notna(var_h):
                result.loc[idx, "nfl_fumble_rate"] = var_h / 21.0

            # INT rate proxy
            var_a = _score_variance(prior, away)
            if pd.notna(var_a):
                result.loc[idx, "nfl_int_rate"] = var_a / 21.0

            # Turnover luck regression: extreme TO margins regress toward 0
            if pd.notna(pd_h) and pd.notna(pd_a):
                raw_margin = (pd_h - pd_a) / 14.0
                result.loc[idx, "nfl_turnover_luck_regression"] = raw_margin * 0.6

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
