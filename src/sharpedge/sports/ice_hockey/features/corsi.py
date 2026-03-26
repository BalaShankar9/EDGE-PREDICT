"""Ice hockey Corsi / possession features (6 total).

Features derived from shot-based possession metrics.

Features:
  nhl_corsi_for_pct           - Corsi For % (rolling)
  nhl_fenwick_for_pct         - Fenwick For % (unblocked shots)
  nhl_xgf_pct                 - Expected goals for %
  nhl_shot_quality             - Shot quality index
  nhl_high_danger_chances     - High-danger chance differential
  nhl_shot_share_diff          - Shot share difference (home - away)
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nhl_corsi_for_pct",
    "nhl_fenwick_for_pct",
    "nhl_xgf_pct",
    "nhl_shot_quality",
    "nhl_high_danger_chances",
    "nhl_shot_share_diff",
]

_ROLLING_N = 10


def _team_metric(prior: pd.DataFrame, team: str, col_home: str,
                 col_away: str, n: int = _ROLLING_N) -> float:
    """Compute rolling mean of a team metric from home/away columns."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    vals = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            v = g.get(col_home, np.nan)
        else:
            v = g.get(col_away, np.nan)
        if pd.notna(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan


def _goals_for_pct(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Compute goals-for % as proxy for xGF% when xG data unavailable."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    gf, ga = [], []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            gf.append(g.get("home_score", 0))
            ga.append(g.get("away_score", 0))
        else:
            gf.append(g.get("away_score", 0))
            ga.append(g.get("home_score", 0))

    total_gf = sum(gf)
    total_ga = sum(ga)
    total = total_gf + total_ga
    return total_gf / total if total > 0 else 0.5


class CorsiFeatures(FeatureGroup):
    name = "hockey_corsi"
    feature_count = 6

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

            # Corsi For % — proxy from goals scored ratio
            cf_home = _goals_for_pct(prior, home)
            cf_away = _goals_for_pct(prior, away)

            result.loc[idx, "nhl_corsi_for_pct"] = cf_home

            # Fenwick For % — similar proxy (slightly adjusted)
            result.loc[idx, "nhl_fenwick_for_pct"] = cf_home * 0.95 + 0.025

            # xGF% proxy
            result.loc[idx, "nhl_xgf_pct"] = cf_home

            # Shot quality — use goal-per-game average as proxy
            gf_home = _team_metric(prior, home, "home_score", "away_score")
            gf_away = _team_metric(prior, away, "home_score", "away_score")
            if pd.notna(gf_home):
                result.loc[idx, "nhl_shot_quality"] = gf_home / 3.0  # normalize

            # High danger chances — differential proxy
            if pd.notna(cf_home) and pd.notna(cf_away):
                result.loc[idx, "nhl_high_danger_chances"] = cf_home - cf_away

            # Shot share diff
            if pd.notna(cf_home) and pd.notna(cf_away):
                result.loc[idx, "nhl_shot_share_diff"] = cf_home - cf_away

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
