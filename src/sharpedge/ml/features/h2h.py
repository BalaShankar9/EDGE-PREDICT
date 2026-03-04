"""Head-to-head features (6 total).

Features from historical meetings between two teams.

Features:
  h2h_home_win_rate_5  - Home win rate in last 5 H2H meetings
  h2h_avg_goals_5      - Average total goals in last 5 H2H
  h2h_home_goals_avg   - Avg goals by home team in H2H
  h2h_away_goals_avg   - Avg goals by away team in H2H
  h2h_btts_rate        - Both Teams To Score rate in H2H
  h2h_home_advantage   - Home advantage factor (home wins / total H2H)
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "h2h_home_win_rate_5",
    "h2h_avg_goals_5",
    "h2h_home_goals_avg",
    "h2h_away_goals_avg",
    "h2h_btts_rate",
    "h2h_home_advantage",
]

class H2HFeatures(FeatureGroup):
    name = "h2h"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
            match_date = row["match_date"]

            # Get prior H2H meetings (either direction)
            prior = df[df["match_date"] < match_date]
            h2h = prior[
                ((prior["home_team_id"] == home_id) & (prior["away_team_id"] == away_id)) |
                ((prior["home_team_id"] == away_id) & (prior["away_team_id"] == home_id))
            ].tail(5)

            if len(h2h) < 2:
                continue

            # Count home wins for the current home team
            home_wins = 0
            total_goals_h = []
            total_goals_a = []
            btts_count = 0
            for _, m in h2h.iterrows():
                if m["home_team_id"] == home_id:
                    hg, ag = m["FTHG"], m["FTAG"]
                    if hg > ag:
                        home_wins += 1
                else:
                    hg, ag = m["FTAG"], m["FTHG"]  # Swap perspective
                    if hg > ag:
                        home_wins += 1
                total_goals_h.append(hg)
                total_goals_a.append(ag)
                if m["FTHG"] > 0 and m["FTAG"] > 0:
                    btts_count += 1

            n = len(h2h)
            result.loc[idx, "h2h_home_win_rate_5"] = home_wins / n
            result.loc[idx, "h2h_avg_goals_5"] = (np.sum(total_goals_h) + np.sum(total_goals_a)) / n
            result.loc[idx, "h2h_home_goals_avg"] = np.mean(total_goals_h)
            result.loc[idx, "h2h_away_goals_avg"] = np.mean(total_goals_a)
            result.loc[idx, "h2h_btts_rate"] = btts_count / n
            result.loc[idx, "h2h_home_advantage"] = home_wins / n

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
