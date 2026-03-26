"""NFL situational features (8 total).

Features capturing game context and situational advantages.

Features:
  nfl_home_dome                - 1 if home team plays in a dome
  nfl_grass_vs_turf            - 1 if surface is grass (vs artificial turf)
  nfl_altitude                 - Stadium altitude factor (Denver = 1.0)
  nfl_timezone_travel          - Timezone difference (away team)
  nfl_divisional               - 1 if divisional game
  nfl_revenge                  - 1 if rematch within season
  nfl_primetime                - 1 if primetime game (SNF/MNF/TNF)
  nfl_playoff_experience       - Playoff appearances differential
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nfl_home_dome",
    "nfl_grass_vs_turf",
    "nfl_altitude",
    "nfl_timezone_travel",
    "nfl_divisional",
    "nfl_revenge",
    "nfl_primetime",
    "nfl_playoff_experience",
]

# Teams that play in domes
_DOME_TEAMS = {
    "IND", "NO", "MIN", "DET", "ATL", "LV", "ARI", "DAL", "LAR", "LAC",
}

# Teams by timezone (simplified)
_EASTERN = {"BUF", "MIA", "NE", "NYJ", "NYG", "BAL", "CIN", "CLE", "PIT",
            "JAX", "ATL", "CAR", "TB", "WAS", "PHI"}
_CENTRAL = {"CHI", "DET", "GB", "MIN", "DAL", "HOU", "IND", "TEN",
            "KC", "NO"}
_MOUNTAIN = {"DEN", "ARI"}
_PACIFIC = {"LAR", "LAC", "SF", "SEA", "LV"}

# Divisions
_AFC_EAST = {"BUF", "MIA", "NE", "NYJ"}
_AFC_NORTH = {"BAL", "CIN", "CLE", "PIT"}
_AFC_SOUTH = {"HOU", "IND", "JAX", "TEN"}
_AFC_WEST = {"DEN", "KC", "LV", "LAC"}
_NFC_EAST = {"DAL", "NYG", "PHI", "WAS"}
_NFC_NORTH = {"CHI", "DET", "GB", "MIN"}
_NFC_SOUTH = {"ATL", "CAR", "NO", "TB"}
_NFC_WEST = {"ARI", "LAR", "SF", "SEA"}

_DIVISIONS = [
    _AFC_EAST, _AFC_NORTH, _AFC_SOUTH, _AFC_WEST,
    _NFC_EAST, _NFC_NORTH, _NFC_SOUTH, _NFC_WEST,
]


def _timezone(team: str) -> int:
    if team in _EASTERN:
        return 0
    if team in _CENTRAL:
        return 1
    if team in _MOUNTAIN:
        return 2
    if team in _PACIFIC:
        return 3
    return 1  # default central


def _same_division(home: str, away: str) -> bool:
    for div in _DIVISIONS:
        if home in div and away in div:
            return True
    return False


class SituationalFeatures(FeatureGroup):
    name = "nfl_situational"
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

            # Dome
            result.loc[idx, "nfl_home_dome"] = 1.0 if home in _DOME_TEAMS else 0.0

            # Surface — default to turf for dome teams
            result.loc[idx, "nfl_grass_vs_turf"] = 0.0 if home in _DOME_TEAMS else 1.0

            # Altitude (Denver advantage)
            result.loc[idx, "nfl_altitude"] = 1.0 if home == "DEN" else 0.0

            # Timezone travel
            tz_diff = abs(_timezone(home) - _timezone(away))
            result.loc[idx, "nfl_timezone_travel"] = float(tz_diff)

            # Divisional
            result.loc[idx, "nfl_divisional"] = 1.0 if _same_division(home, away) else 0.0

            # Revenge — did these teams play earlier?
            if len(prior) > 0:
                prev_meetings = prior[
                    ((prior["home_team"] == home) & (prior["away_team"] == away)) |
                    ((prior["home_team"] == away) & (prior["away_team"] == home))
                ]
                result.loc[idx, "nfl_revenge"] = 1.0 if len(prev_meetings) > 0 else 0.0
            else:
                result.loc[idx, "nfl_revenge"] = 0.0

            # Primetime proxy (from column or default 0)
            result.loc[idx, "nfl_primetime"] = float(row.get("primetime", 0))

            # Playoff experience proxy (from win rate)
            if len(prior) > 0:
                h_wins = prior[
                    ((prior["home_team"] == home) & (prior["home_win"] == 1)) |
                    ((prior["away_team"] == home) & (prior["home_win"] == 0))
                ]
                a_wins = prior[
                    ((prior["home_team"] == away) & (prior["home_win"] == 1)) |
                    ((prior["away_team"] == away) & (prior["home_win"] == 0))
                ]
                h_games = prior[
                    (prior["home_team"] == home) | (prior["away_team"] == home)
                ]
                a_games = prior[
                    (prior["home_team"] == away) | (prior["away_team"] == away)
                ]
                wr_h = len(h_wins) / max(len(h_games), 1)
                wr_a = len(a_wins) / max(len(a_games), 1)
                result.loc[idx, "nfl_playoff_experience"] = wr_h - wr_a
            else:
                result.loc[idx, "nfl_playoff_experience"] = 0.0

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
