"""Goal pattern features (8 total).

Features capturing scoring/conceding patterns beyond simple averages.
These features capture team tendencies (high-scoring, defensive, volatile)
that are predictive of future results.

Features:
  goal_home_clean_sheet_rate_5  - Home team clean sheet %, last 5 home games
  goal_away_clean_sheet_rate_5  - Away team clean sheet %, last 5 away games
  goal_home_btts_rate_5         - BTTS %, last 5 home games
  goal_away_btts_rate_5         - BTTS %, last 5 away games
  goal_home_over25_rate_5       - Over 2.5 %, last 5 home games
  goal_away_over25_rate_5       - Over 2.5 %, last 5 away games
  goal_home_first_half_rate_5   - Home team 1H goal %, last 5 home games
  goal_away_first_half_rate_5   - Away team 1H goal %, last 5 away games
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

WINDOW = 5

FEATURE_NAMES = [
    "goal_home_clean_sheet_rate_5",
    "goal_away_clean_sheet_rate_5",
    "goal_home_btts_rate_5",
    "goal_away_btts_rate_5",
    "goal_home_over25_rate_5",
    "goal_away_over25_rate_5",
    "goal_home_first_half_rate_5",
    "goal_away_first_half_rate_5",
]


class GoalPatternFeatures(FeatureGroup):
    name = "goals"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        has_ht = "HTHG" in df.columns and "HTAG" in df.columns

        for idx in range(len(df)):
            row = df.iloc[idx]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
            prior = df.iloc[:idx]

            # Home team's recent HOME matches
            home_home = prior[prior["home_team_id"] == home_id].tail(WINDOW)

            if len(home_home) >= 3:
                # Clean sheet rate (conceded 0 goals)
                clean_sheets = (home_home["FTAG"] == 0).sum()
                result.iloc[idx, result.columns.get_loc("goal_home_clean_sheet_rate_5")] = clean_sheets / len(home_home)

                # BTTS rate
                btts = ((home_home["FTHG"] > 0) & (home_home["FTAG"] > 0)).sum()
                result.iloc[idx, result.columns.get_loc("goal_home_btts_rate_5")] = btts / len(home_home)

                # Over 2.5 rate
                over25 = ((home_home["FTHG"] + home_home["FTAG"]) > 2.5).sum()
                result.iloc[idx, result.columns.get_loc("goal_home_over25_rate_5")] = over25 / len(home_home)

                # First half goal rate
                if has_ht:
                    hh_ht = home_home.dropna(subset=["HTHG"])
                    if len(hh_ht) >= 2:
                        fh_goals = (hh_ht["HTHG"] > 0).sum()
                        result.iloc[idx, result.columns.get_loc("goal_home_first_half_rate_5")] = fh_goals / len(hh_ht)

            # Away team's recent AWAY matches
            away_away = prior[prior["away_team_id"] == away_id].tail(WINDOW)

            if len(away_away) >= 3:
                clean_sheets = (away_away["FTHG"] == 0).sum()
                result.iloc[idx, result.columns.get_loc("goal_away_clean_sheet_rate_5")] = clean_sheets / len(away_away)

                btts = ((away_away["FTHG"] > 0) & (away_away["FTAG"] > 0)).sum()
                result.iloc[idx, result.columns.get_loc("goal_away_btts_rate_5")] = btts / len(away_away)

                over25 = ((away_away["FTHG"] + away_away["FTAG"]) > 2.5).sum()
                result.iloc[idx, result.columns.get_loc("goal_away_over25_rate_5")] = over25 / len(away_away)

                if has_ht:
                    aa_ht = away_away.dropna(subset=["HTAG"])
                    if len(aa_ht) >= 2:
                        fh_goals = (aa_ht["HTAG"] > 0).sum()
                        result.iloc[idx, result.columns.get_loc("goal_away_first_half_rate_5")] = fh_goals / len(aa_ht)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
