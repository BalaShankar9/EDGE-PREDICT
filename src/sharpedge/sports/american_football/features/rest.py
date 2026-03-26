"""NFL rest features (4 total).

Features capturing schedule and rest advantages.

Features:
  nfl_bye_week_advantage       - 1 if team is coming off bye week
  nfl_days_rest                - Days since last game
  nfl_short_week_flag          - 1 if playing on short rest (< 6 days)
  nfl_travel_distance          - Travel distance proxy (timezone diff)
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nfl_bye_week_advantage",
    "nfl_days_rest",
    "nfl_short_week_flag",
    "nfl_travel_distance",
]

# Simplified timezone mapping
_EASTERN = {"BUF", "MIA", "NE", "NYJ", "NYG", "BAL", "CIN", "CLE", "PIT",
            "JAX", "ATL", "CAR", "TB", "WAS", "PHI"}
_CENTRAL = {"CHI", "DET", "GB", "MIN", "DAL", "HOU", "IND", "TEN",
            "KC", "NO"}
_MOUNTAIN = {"DEN", "ARI"}
_PACIFIC = {"LAR", "LAC", "SF", "SEA", "LV"}


def _timezone(team: str) -> int:
    if team in _EASTERN:
        return 0
    if team in _CENTRAL:
        return 1
    if team in _MOUNTAIN:
        return 2
    if team in _PACIFIC:
        return 3
    return 1


def _days_since_last(prior: pd.DataFrame, team: str, match_date) -> float:
    """Days since team's most recent game."""
    team_games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ]
    if len(team_games) == 0:
        return np.nan
    last_date = team_games["game_date"].max()
    delta = (match_date - last_date).days
    return float(delta)


class NFLRestFeatures(FeatureGroup):
    name = "nfl_rest"
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
                # Travel distance can always be computed
                tz_diff = abs(_timezone(home) - _timezone(away))
                result.loc[idx, "nfl_travel_distance"] = float(tz_diff)
                continue

            rest_h = _days_since_last(prior, home, match_date)
            rest_a = _days_since_last(prior, away, match_date)

            # Bye week advantage: rest >= 12 days suggests a bye
            bye_h = 1.0 if (pd.notna(rest_h) and rest_h >= 12) else 0.0
            bye_a = 1.0 if (pd.notna(rest_a) and rest_a >= 12) else 0.0
            result.loc[idx, "nfl_bye_week_advantage"] = bye_h - bye_a

            # Days rest differential
            if pd.notna(rest_h) and pd.notna(rest_a):
                result.loc[idx, "nfl_days_rest"] = rest_h - rest_a
            elif pd.notna(rest_h):
                result.loc[idx, "nfl_days_rest"] = rest_h
            elif pd.notna(rest_a):
                result.loc[idx, "nfl_days_rest"] = -rest_a

            # Short week flag
            short_h = 1.0 if (pd.notna(rest_h) and rest_h < 6) else 0.0
            short_a = 1.0 if (pd.notna(rest_a) and rest_a < 6) else 0.0
            result.loc[idx, "nfl_short_week_flag"] = short_h - short_a

            # Travel distance
            tz_diff = abs(_timezone(home) - _timezone(away))
            result.loc[idx, "nfl_travel_distance"] = float(tz_diff)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
