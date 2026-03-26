"""Baseball bullpen features (6 total).

Features capturing bullpen quality and availability.

Features:
  mlb_bullpen_era              - Bullpen ERA proxy (rolling)
  mlb_workload_3d              - Bullpen workload in last 3 days
  mlb_closer_available         - Closer availability flag
  mlb_high_leverage_era        - High-leverage bullpen ERA proxy
  mlb_bullpen_depth            - Bullpen depth (inverse of workload)
  mlb_rest_quality             - Bullpen rest quality composite
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "mlb_bullpen_era",
    "mlb_workload_3d",
    "mlb_closer_available",
    "mlb_high_leverage_era",
    "mlb_bullpen_depth",
    "mlb_rest_quality",
]

_ROLLING_N = 10


def _team_late_runs_allowed(prior: pd.DataFrame, team: str,
                             n: int = _ROLLING_N) -> float:
    """Proxy for bullpen ERA: runs allowed in recent games."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    ra = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            ra.append(g.get("away_score", np.nan))
        else:
            ra.append(g.get("home_score", np.nan))

    valid = [v for v in ra if pd.notna(v)]
    return np.mean(valid) if valid else np.nan


def _games_in_window(prior: pd.DataFrame, team: str, match_date, days: int) -> int:
    """Count team games in the last N days."""
    cutoff = match_date - pd.Timedelta(days=days)
    recent = prior[
        (prior["game_date"] >= cutoff) &
        ((prior["home_team"] == team) | (prior["away_team"] == team))
    ]
    return len(recent)


class BullpenFeatures(FeatureGroup):
    name = "baseball_bullpen"
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

            ra_h = _team_late_runs_allowed(prior, home)
            ra_a = _team_late_runs_allowed(prior, away)

            # Bullpen ERA proxy
            if pd.notna(ra_h) and pd.notna(ra_a):
                result.loc[idx, "mlb_bullpen_era"] = ra_h - ra_a

            # Workload in last 3 days
            w3_h = _games_in_window(prior, home, match_date, days=3)
            w3_a = _games_in_window(prior, away, match_date, days=3)
            result.loc[idx, "mlb_workload_3d"] = float(w3_h - w3_a)

            # Closer availability (available if not overworked)
            result.loc[idx, "mlb_closer_available"] = 1.0 if w3_h < 3 else 0.0

            # High leverage ERA proxy (slightly adjusted)
            if pd.notna(ra_h):
                result.loc[idx, "mlb_high_leverage_era"] = ra_h * 0.9

            # Bullpen depth (inverse of workload burden)
            result.loc[idx, "mlb_bullpen_depth"] = max(0, 3.0 - w3_h)

            # Rest quality composite
            w5_h = _games_in_window(prior, home, match_date, days=5)
            result.loc[idx, "mlb_rest_quality"] = max(0, 5.0 - w5_h) / 5.0

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
