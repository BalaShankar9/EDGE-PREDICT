# src/sharpedge/ml/features/xg_perf.py
"""xG (expected goals) performance features (8 total).

Computes features from Understat xG data merged with match results.
xG measures the quality of chances created — comparing actual goals to xG
reveals whether a team is over/underperforming.

Features:
  xg_home_xg_5         - Home team avg xG, last 5 home matches
  xg_home_xga_5        - Home team avg xG against, last 5 home matches
  xg_away_xg_5         - Away team avg xG, last 5 away matches
  xg_away_xga_5        - Away team avg xG against, last 5 away matches
  xg_home_overperform  - Home: (goals - xG) over last 10 matches
  xg_away_overperform  - Away: (goals - xG) over last 10 matches
  xg_home_variance     - xG variance for home team (consistency)
  xg_home_shot_quality - Home: xG per shot
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "xg_home_xg_5",
    "xg_home_xga_5",
    "xg_away_xg_5",
    "xg_away_xga_5",
    "xg_home_overperform",
    "xg_away_overperform",
    "xg_home_variance",
    "xg_home_shot_quality",
]

WINDOW = 5


class XGPerformanceFeatures(FeatureGroup):
    name = "xg_perf"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        xg_df = context.get("xg_df")

        result = pd.DataFrame(index=matches.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        if xg_df is None or xg_df.empty:
            return result

        # xg_df expected columns: match_date, home_team_id, away_team_id, home_xg, away_xg
        # Also may have: home_shots, away_shots
        xg = xg_df.copy()
        xg["match_date"] = pd.to_datetime(xg["match_date"])

        df = matches.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])

        for idx, row in df.iterrows():
            match_date = row["match_date"]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            prior_xg = xg[xg["match_date"] < match_date]

            # Home team xG from their home matches
            home_home_xg = prior_xg[prior_xg["home_team_id"] == home_id].sort_values("match_date").tail(WINDOW)
            if len(home_home_xg) >= 3:
                result.loc[idx, "xg_home_xg_5"] = home_home_xg["home_xg"].mean()
                result.loc[idx, "xg_home_xga_5"] = home_home_xg["away_xg"].mean()
                result.loc[idx, "xg_home_variance"] = home_home_xg["home_xg"].var()
                if "home_shots" in home_home_xg.columns:
                    total_shots = home_home_xg["home_shots"].sum()
                    total_xg = home_home_xg["home_xg"].sum()
                    if total_shots > 0:
                        result.loc[idx, "xg_home_shot_quality"] = total_xg / total_shots

            # Away team xG from their away matches
            away_away_xg = prior_xg[prior_xg["away_team_id"] == away_id].sort_values("match_date").tail(WINDOW)
            if len(away_away_xg) >= 3:
                result.loc[idx, "xg_away_xg_5"] = away_away_xg["away_xg"].mean()
                result.loc[idx, "xg_away_xga_5"] = away_away_xg["home_xg"].mean()

            # Overperformance: (actual goals - xG) over last 10 all matches
            home_all = prior_xg[
                (prior_xg["home_team_id"] == home_id) | (prior_xg["away_team_id"] == home_id)
            ].sort_values("match_date").tail(10)
            if len(home_all) >= 3:
                goals = []
                xgs = []
                for _, m in home_all.iterrows():
                    if m["home_team_id"] == home_id:
                        if "FTHG" in m.index:
                            goals.append(m["FTHG"])
                        elif "home_goals" in m.index:
                            goals.append(m["home_goals"])
                        else:
                            goals.append(0)
                        xgs.append(m["home_xg"])
                    else:
                        if "FTAG" in m.index:
                            goals.append(m["FTAG"])
                        elif "away_goals" in m.index:
                            goals.append(m["away_goals"])
                        else:
                            goals.append(0)
                        xgs.append(m["away_xg"])
                result.loc[idx, "xg_home_overperform"] = np.mean(np.array(goals) - np.array(xgs))

            away_all = prior_xg[
                (prior_xg["home_team_id"] == away_id) | (prior_xg["away_team_id"] == away_id)
            ].sort_values("match_date").tail(10)
            if len(away_all) >= 3:
                goals = []
                xgs = []
                for _, m in away_all.iterrows():
                    if m["home_team_id"] == away_id:
                        if "FTHG" in m.index:
                            goals.append(m["FTHG"])
                        elif "home_goals" in m.index:
                            goals.append(m["home_goals"])
                        else:
                            goals.append(0)
                        xgs.append(m["home_xg"])
                    else:
                        if "FTAG" in m.index:
                            goals.append(m["FTAG"])
                        elif "away_goals" in m.index:
                            goals.append(m["away_goals"])
                        else:
                            goals.append(0)
                        xgs.append(m["away_xg"])
                result.loc[idx, "xg_away_overperform"] = np.mean(np.array(goals) - np.array(xgs))

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
