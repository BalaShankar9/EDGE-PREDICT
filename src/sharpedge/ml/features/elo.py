"""ELO rating features (6 total).

Features derived from ClubELO ratings. For each match, looks up the most
recent ELO rating before the match date for each team.

Features:
  elo_home          - Home team ELO rating
  elo_away          - Away team ELO rating
  elo_diff          - Home ELO minus Away ELO
  elo_home_momentum - Home team ELO change (current - 5 periods ago)
  elo_away_momentum - Away team ELO change (current - 5 periods ago)
  elo_home_pctile   - Home team ELO percentile in dataset
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "elo_home_momentum",
    "elo_away_momentum",
    "elo_home_pctile",
]


class EloFeatures(FeatureGroup):
    name = "elo"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        elo_df = context.get("elo_df")

        result = pd.DataFrame(index=matches.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        if elo_df is None or elo_df.empty:
            return result

        # Ensure date columns are datetime
        elo = elo_df.copy()
        elo["date"] = pd.to_datetime(elo["date"])

        for idx, row in matches.iterrows():
            match_date = pd.to_datetime(row["match_date"])
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            # Get most recent ELO before match date
            home_elo = elo[(elo["team_id"] == home_id) & (elo["date"] < match_date)]
            away_elo = elo[(elo["team_id"] == away_id) & (elo["date"] < match_date)]

            if not home_elo.empty:
                home_elo_sorted = home_elo.sort_values("date")
                current_home = home_elo_sorted.iloc[-1]["elo"]
                result.loc[idx, "elo_home"] = current_home

                # Momentum: change from 5 periods ago
                if len(home_elo_sorted) >= 6:
                    old_home = home_elo_sorted.iloc[-6]["elo"]
                    result.loc[idx, "elo_home_momentum"] = current_home - old_home

            if not away_elo.empty:
                away_elo_sorted = away_elo.sort_values("date")
                current_away = away_elo_sorted.iloc[-1]["elo"]
                result.loc[idx, "elo_away"] = current_away

                if len(away_elo_sorted) >= 6:
                    old_away = away_elo_sorted.iloc[-6]["elo"]
                    result.loc[idx, "elo_away_momentum"] = current_away - old_away

            # Diff
            if pd.notna(result.loc[idx, "elo_home"]) and pd.notna(result.loc[idx, "elo_away"]):
                result.loc[idx, "elo_diff"] = result.loc[idx, "elo_home"] - result.loc[idx, "elo_away"]

            # Percentile: home team ELO relative to all teams at that time
            if pd.notna(result.loc[idx, "elo_home"]):
                recent_all = elo[elo["date"] < match_date]
                if not recent_all.empty:
                    latest_per_team = recent_all.sort_values("date").groupby("team_id").last()
                    all_ratings = latest_per_team["elo"].values
                    pctile = np.mean(all_ratings <= result.loc[idx, "elo_home"])
                    result.loc[idx, "elo_home_pctile"] = pctile

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
