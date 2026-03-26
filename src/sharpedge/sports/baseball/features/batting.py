"""Baseball batting features (8 total).

Features capturing team offensive performance.

Features:
  mlb_wrc_plus                 - Weighted runs created plus proxy
  mlb_ops                      - On-base plus slugging proxy
  mlb_iso                      - Isolated power proxy
  mlb_hard_hit_rate            - Hard hit rate proxy
  mlb_barrel_rate              - Barrel rate proxy
  mlb_k_pct                    - Strikeout percentage proxy
  mlb_bb_pct                   - Walk percentage proxy
  mlb_babip_luck               - BABIP luck regression indicator
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "mlb_wrc_plus",
    "mlb_ops",
    "mlb_iso",
    "mlb_hard_hit_rate",
    "mlb_barrel_rate",
    "mlb_k_pct",
    "mlb_bb_pct",
    "mlb_babip_luck",
]

_ROLLING_N = 10
_LEAGUE_AVG_RUNS = 4.5


def _team_runs(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Rolling runs scored average for a team."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    runs = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            runs.append(g.get("home_score", np.nan))
        else:
            runs.append(g.get("away_score", np.nan))

    valid = [r for r in runs if pd.notna(r)]
    return np.mean(valid) if valid else np.nan


def _team_run_variance(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Rolling run scoring variance for a team."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) < 2:
        return np.nan

    runs = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            runs.append(g.get("home_score", np.nan))
        else:
            runs.append(g.get("away_score", np.nan))

    valid = [r for r in runs if pd.notna(r)]
    return np.std(valid) if len(valid) >= 2 else np.nan


class BattingFeatures(FeatureGroup):
    name = "baseball_batting"
    feature_count = 8

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

            runs_h = _team_runs(prior, home)
            runs_a = _team_runs(prior, away)

            # wRC+ proxy: runs relative to league average * 100
            if pd.notna(runs_h):
                result.loc[idx, "mlb_wrc_plus"] = (runs_h / _LEAGUE_AVG_RUNS) * 100

            # OPS proxy: from runs scored
            if pd.notna(runs_h) and pd.notna(runs_a):
                result.loc[idx, "mlb_ops"] = (runs_h - runs_a) / _LEAGUE_AVG_RUNS

            # ISO proxy: power index
            if pd.notna(runs_h):
                result.loc[idx, "mlb_iso"] = runs_h / 9.0  # normalize

            # Hard hit rate proxy
            if pd.notna(runs_h):
                result.loc[idx, "mlb_hard_hit_rate"] = min(runs_h / 7.0, 1.0)

            # Barrel rate proxy
            if pd.notna(runs_h):
                result.loc[idx, "mlb_barrel_rate"] = runs_h / 10.0

            # K% proxy: inverse of scoring consistency
            var_h = _team_run_variance(prior, home)
            if pd.notna(var_h):
                result.loc[idx, "mlb_k_pct"] = var_h / _LEAGUE_AVG_RUNS

            # BB% proxy
            if pd.notna(runs_h):
                result.loc[idx, "mlb_bb_pct"] = max(0, runs_h - 3.0) / 5.0

            # BABIP luck: recent vs long-term runs (positive = lucky)
            runs_h_short = _team_runs(prior, home, n=3)
            if pd.notna(runs_h_short) and pd.notna(runs_h):
                result.loc[idx, "mlb_babip_luck"] = (runs_h_short - runs_h) / _LEAGUE_AVG_RUNS

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
