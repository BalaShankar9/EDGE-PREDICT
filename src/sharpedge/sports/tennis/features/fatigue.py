"""Tennis fatigue features (6 total).

Features tracking player workload and rest periods.

Features:
  tnf_matches_7d_p1      — P1 matches played in last 7 days
  tnf_matches_7d_p2      — P2 matches in last 7 days
  tnf_matches_28d_p1     — P1 matches in last 28 days
  tnf_matches_28d_p2     — P2 matches in last 28 days
  tnf_days_since_last_p1 — Days since P1's last match
  tnf_days_since_last_p2 — Days since P2's last match
"""

import pandas as pd
import numpy as np
from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "tnf_matches_7d_p1",
    "tnf_matches_7d_p2",
    "tnf_matches_28d_p1",
    "tnf_matches_28d_p2",
    "tnf_days_since_last_p1",
    "tnf_days_since_last_p2",
]


def _player_matches_in_window(prior: pd.DataFrame, player: str, match_date, days: int) -> int:
    """Count matches a player had in the last N days."""
    cutoff = match_date - pd.Timedelta(days=days)
    window = prior[(prior["date"] >= cutoff) & (prior["date"] < match_date)]
    return int(((window["player1"] == player) | (window["player2"] == player)).sum())


def _days_since_last(prior: pd.DataFrame, player: str, match_date) -> float:
    """Days since the player's most recent match."""
    player_prior = prior[
        (prior["player1"] == player) | (prior["player2"] == player)
    ]
    if len(player_prior) == 0:
        return np.nan
    last_date = player_prior["date"].max()
    return (match_date - last_date).days


class FatigueFeatures(FeatureGroup):
    name = "tennis_fatigue"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            p1 = row["player1"]
            p2 = row["player2"]
            match_date = row["date"]

            prior = df[df["date"] < match_date]

            result.loc[idx, "tnf_matches_7d_p1"] = _player_matches_in_window(prior, p1, match_date, 7)
            result.loc[idx, "tnf_matches_7d_p2"] = _player_matches_in_window(prior, p2, match_date, 7)
            result.loc[idx, "tnf_matches_28d_p1"] = _player_matches_in_window(prior, p1, match_date, 28)
            result.loc[idx, "tnf_matches_28d_p2"] = _player_matches_in_window(prior, p2, match_date, 28)
            result.loc[idx, "tnf_days_since_last_p1"] = _days_since_last(prior, p1, match_date)
            result.loc[idx, "tnf_days_since_last_p2"] = _days_since_last(prior, p2, match_date)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
