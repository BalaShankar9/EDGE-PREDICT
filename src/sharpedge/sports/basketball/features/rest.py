"""Basketball rest features (6 total).

Features capturing schedule advantages and fatigue.

Features:
  bkr_rest_days_home          - days since last game (home)
  bkr_rest_days_away          - days since last game (away)
  bkr_back_to_back_home       - 1 if playing second game in 2 days (home)
  bkr_back_to_back_away       - 1 if playing second game in 2 days (away)
  bkr_rest_advantage          - rest_home - rest_away
  bkr_travel_flag             - 1 if cross-timezone travel detected
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "bkr_rest_days_home",
    "bkr_rest_days_away",
    "bkr_back_to_back_home",
    "bkr_back_to_back_away",
    "bkr_rest_advantage",
    "bkr_travel_flag",
]

# Teams in different time zones (simplified: East vs West)
_EASTERN_TEAMS = {
    "BOS", "BKN", "NYK", "PHI", "TOR",
    "CHI", "CLE", "DET", "IND", "MIL",
    "ATL", "CHA", "MIA", "ORL", "WAS",
}
_WESTERN_TEAMS = {
    "DAL", "HOU", "MEM", "NOP", "SAS",
    "DEN", "MIN", "OKC", "POR", "UTA",
    "GSW", "LAC", "LAL", "PHX", "SAC",
}


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


def _is_travel(home: str, away: str) -> float:
    """Check if away team is travelling across time zones."""
    home_east = home in _EASTERN_TEAMS
    away_east = away in _EASTERN_TEAMS
    home_west = home in _WESTERN_TEAMS
    away_west = away in _WESTERN_TEAMS

    # Cross-conference travel
    if (home_east and away_west) or (home_west and away_east):
        return 1.0
    return 0.0


class RestFeatures(FeatureGroup):
    name = "basketball_rest"
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
                result.loc[idx, "bkr_travel_flag"] = _is_travel(home, away)
                continue

            rest_h = _days_since_last(prior, home, match_date)
            rest_a = _days_since_last(prior, away, match_date)

            result.loc[idx, "bkr_rest_days_home"] = rest_h
            result.loc[idx, "bkr_rest_days_away"] = rest_a

            if pd.notna(rest_h):
                result.loc[idx, "bkr_back_to_back_home"] = 1.0 if rest_h <= 1 else 0.0
            if pd.notna(rest_a):
                result.loc[idx, "bkr_back_to_back_away"] = 1.0 if rest_a <= 1 else 0.0

            if pd.notna(rest_h) and pd.notna(rest_a):
                result.loc[idx, "bkr_rest_advantage"] = rest_h - rest_a
            else:
                result.loc[idx, "bkr_rest_advantage"] = np.nan

            result.loc[idx, "bkr_travel_flag"] = _is_travel(home, away)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
