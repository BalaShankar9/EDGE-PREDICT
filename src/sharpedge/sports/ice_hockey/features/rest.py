"""Ice hockey rest and fatigue features (6 total).

Features capturing schedule advantages and fatigue.

Features:
  nhl_rest_days_home           - Days since last game (home)
  nhl_rest_days_away           - Days since last game (away)
  nhl_back_to_back_home        - 1 if playing second game in 2 days (home)
  nhl_back_to_back_away        - 1 if playing second game in 2 days (away)
  nhl_rest_advantage           - Rest days differential (home - away)
  nhl_fatigue_index            - Composite fatigue score
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nhl_rest_days_home",
    "nhl_rest_days_away",
    "nhl_back_to_back_home",
    "nhl_back_to_back_away",
    "nhl_rest_advantage",
    "nhl_fatigue_index",
]


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


def _games_in_window(prior: pd.DataFrame, team: str, match_date, days: int = 7) -> int:
    """Count team games in the last N days."""
    cutoff = match_date - pd.Timedelta(days=days)
    recent = prior[
        (prior["game_date"] >= cutoff) &
        ((prior["home_team"] == team) | (prior["away_team"] == team))
    ]
    return len(recent)


class HockeyRestFeatures(FeatureGroup):
    name = "hockey_rest"
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

            rest_h = _days_since_last(prior, home, match_date)
            rest_a = _days_since_last(prior, away, match_date)

            result.loc[idx, "nhl_rest_days_home"] = rest_h
            result.loc[idx, "nhl_rest_days_away"] = rest_a

            if pd.notna(rest_h):
                result.loc[idx, "nhl_back_to_back_home"] = 1.0 if rest_h <= 1 else 0.0
            if pd.notna(rest_a):
                result.loc[idx, "nhl_back_to_back_away"] = 1.0 if rest_a <= 1 else 0.0

            if pd.notna(rest_h) and pd.notna(rest_a):
                result.loc[idx, "nhl_rest_advantage"] = rest_h - rest_a
            else:
                result.loc[idx, "nhl_rest_advantage"] = np.nan

            # Fatigue index: games in last 7 days, weighted by back-to-back
            games_7d_h = _games_in_window(prior, home, match_date, days=7)
            games_7d_a = _games_in_window(prior, away, match_date, days=7)
            b2b_h = 1.0 if (pd.notna(rest_h) and rest_h <= 1) else 0.0
            b2b_a = 1.0 if (pd.notna(rest_a) and rest_a <= 1) else 0.0
            fatigue_h = games_7d_h + b2b_h * 0.5
            fatigue_a = games_7d_a + b2b_a * 0.5
            result.loc[idx, "nhl_fatigue_index"] = fatigue_h - fatigue_a

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
