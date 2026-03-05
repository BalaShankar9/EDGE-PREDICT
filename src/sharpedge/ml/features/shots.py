"""Shot-based features (8 total).

Features derived from shot statistics (available from Football-Data UK).
Shot efficiency and dominance are strong predictors of future results.

Features:
  shot_home_conversion_5    - Home team shot conversion rate, last 5 home games
  shot_home_accuracy_5      - Home team shots on target %, last 5 home games
  shot_away_conversion_5    - Away team shot conversion rate, last 5 away games
  shot_away_accuracy_5      - Away team shots on target %, last 5 away games
  shot_home_dominance_5     - Home shot share (home_shots / total_shots), last 5 home
  shot_away_dominance_5     - Away shot share (away_shots / total_shots), last 5 away
  shot_home_sot_ratio_5     - Home SOT per goal conceded (defensive quality)
  shot_away_sot_ratio_5     - Away SOT per goal conceded (defensive quality)
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

WINDOW = 5

FEATURE_NAMES = [
    "shot_home_conversion_5",
    "shot_home_accuracy_5",
    "shot_away_conversion_5",
    "shot_away_accuracy_5",
    "shot_home_dominance_5",
    "shot_away_dominance_5",
    "shot_home_sot_ratio_5",
    "shot_away_sot_ratio_5",
]


class ShotFeatures(FeatureGroup):
    name = "shots"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        has_shots = "HS" in df.columns and "AS" in df.columns

        if not has_shots:
            return result

        for idx in range(len(df)):
            row = df.iloc[idx]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
            prior = df.iloc[:idx]

            # Home team's recent HOME matches with shot data
            home_home = prior[prior["home_team_id"] == home_id].tail(WINDOW)
            home_home = home_home.dropna(subset=["HS"])

            if len(home_home) >= 3:
                total_shots = home_home["HS"].sum()
                total_sot = home_home["HST"].sum() if "HST" in home_home.columns else 0
                total_goals = home_home["FTHG"].sum()
                opp_shots = home_home["AS"].sum()

                if total_shots > 0:
                    result.iloc[idx, result.columns.get_loc("shot_home_conversion_5")] = total_goals / total_shots
                    if total_sot > 0:
                        result.iloc[idx, result.columns.get_loc("shot_home_accuracy_5")] = total_sot / total_shots
                    total_all = total_shots + opp_shots
                    if total_all > 0:
                        result.iloc[idx, result.columns.get_loc("shot_home_dominance_5")] = total_shots / total_all

                # Defensive: opponent SOT per goal conceded
                goals_conceded = home_home["FTAG"].sum()
                opp_sot = home_home["AST"].sum() if "AST" in home_home.columns else 0
                if goals_conceded > 0 and opp_sot > 0:
                    result.iloc[idx, result.columns.get_loc("shot_home_sot_ratio_5")] = opp_sot / goals_conceded

            # Away team's recent AWAY matches with shot data
            away_away = prior[prior["away_team_id"] == away_id].tail(WINDOW)
            away_away = away_away.dropna(subset=["AS"])

            if len(away_away) >= 3:
                total_shots = away_away["AS"].sum()
                total_sot = away_away["AST"].sum() if "AST" in away_away.columns else 0
                total_goals = away_away["FTAG"].sum()
                opp_shots = away_away["HS"].sum()

                if total_shots > 0:
                    result.iloc[idx, result.columns.get_loc("shot_away_conversion_5")] = total_goals / total_shots
                    if total_sot > 0:
                        result.iloc[idx, result.columns.get_loc("shot_away_accuracy_5")] = total_sot / total_shots
                    total_all = total_shots + opp_shots
                    if total_all > 0:
                        result.iloc[idx, result.columns.get_loc("shot_away_dominance_5")] = total_shots / total_all

                goals_conceded = away_away["FTHG"].sum()
                opp_sot = away_away["HST"].sum() if "HST" in away_away.columns else 0
                if goals_conceded > 0 and opp_sot > 0:
                    result.iloc[idx, result.columns.get_loc("shot_away_sot_ratio_5")] = opp_sot / goals_conceded

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
