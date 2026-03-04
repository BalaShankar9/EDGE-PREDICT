# src/sharpedge/ml/features/form.py
"""Rolling form features (12 total).

Computes rolling 5-match averages for key stats, separately for home and away.
These features capture current team form/momentum.

Features produced:
  form_home_goals_scored_5    - Home team avg goals scored, last 5 home games
  form_home_goals_conceded_5  - Home team avg goals conceded, last 5 home games
  form_away_goals_scored_5    - Away team avg goals scored, last 5 away games
  form_away_goals_conceded_5  - Away team avg goals conceded, last 5 away games
  form_home_points_5          - Home team avg points per game, last 5 home games
  form_away_points_5          - Away team avg points per game, last 5 away games
  form_home_win_streak        - Current home win streak
  form_away_win_streak        - Current away win streak (away games only)
  form_home_unbeaten_5        - Home team unbeaten run length (home)
  form_away_unbeaten_5        - Away team unbeaten run length (away)
  form_home_momentum          - Home: last 3 avg pts - last 10 avg pts (positive = improving)
  form_away_momentum          - Away: last 3 avg pts - last 10 avg pts
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

WINDOW = 5
FEATURE_NAMES = [
    "form_home_goals_scored_5",
    "form_home_goals_conceded_5",
    "form_away_goals_scored_5",
    "form_away_goals_conceded_5",
    "form_home_points_5",
    "form_away_points_5",
    "form_home_win_streak",
    "form_away_win_streak",
    "form_home_unbeaten_5",
    "form_away_unbeaten_5",
    "form_home_momentum",
    "form_away_momentum",
]


class FormFeatures(FeatureGroup):
    name = "form"
    feature_count = 12

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        """Compute rolling form features.

        Requires matches sorted by date with columns:
        match_date, home_team_id, away_team_id, FTHG, FTAG, FTR
        """
        df = matches.copy()
        df = df.sort_values("match_date").reset_index(drop=True)

        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx in range(len(df)):
            row = df.iloc[idx]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            # Get prior matches for each team (before this match date)
            prior = df.iloc[:idx]

            # Home team's recent HOME matches
            home_home = prior[prior["home_team_id"] == home_id].tail(WINDOW)
            # Away team's recent AWAY matches
            away_away = prior[prior["away_team_id"] == away_id].tail(WINDOW)

            # Home team all recent matches (for momentum)
            home_all = prior[
                (prior["home_team_id"] == home_id) | (prior["away_team_id"] == home_id)
            ].tail(10)
            away_all = prior[
                (prior["home_team_id"] == away_id) | (prior["away_team_id"] == away_id)
            ].tail(10)

            # Home form features
            if len(home_home) >= 3:
                result.loc[idx, "form_home_goals_scored_5"] = home_home["FTHG"].mean()
                result.loc[idx, "form_home_goals_conceded_5"] = home_home["FTAG"].mean()
                pts = home_home["FTR"].map({"H": 3, "D": 1, "A": 0}).fillna(0)
                result.loc[idx, "form_home_points_5"] = pts.mean()
                # Win streak
                streak = 0
                for r in reversed(pts.values):
                    if r == 3:
                        streak += 1
                    else:
                        break
                result.loc[idx, "form_home_win_streak"] = streak
                # Unbeaten run
                unbeaten = 0
                for r in reversed(pts.values):
                    if r >= 1:
                        unbeaten += 1
                    else:
                        break
                result.loc[idx, "form_home_unbeaten_5"] = unbeaten

            # Away form features
            if len(away_away) >= 3:
                result.loc[idx, "form_away_goals_scored_5"] = away_away["FTAG"].mean()
                result.loc[idx, "form_away_goals_conceded_5"] = away_away["FTHG"].mean()
                pts = away_away["FTR"].map({"H": 0, "D": 1, "A": 3}).fillna(0)
                result.loc[idx, "form_away_points_5"] = pts.mean()
                streak = 0
                for r in reversed(pts.values):
                    if r == 3:
                        streak += 1
                    else:
                        break
                result.loc[idx, "form_away_win_streak"] = streak
                unbeaten = 0
                for r in reversed(pts.values):
                    if r >= 1:
                        unbeaten += 1
                    else:
                        break
                result.loc[idx, "form_away_unbeaten_5"] = unbeaten

            # Momentum: compare last 3 vs last 10
            if len(home_all) >= 3:
                home_pts = []
                for _, m in home_all.iterrows():
                    if m["home_team_id"] == home_id:
                        home_pts.append({"H": 3, "D": 1, "A": 0}.get(m["FTR"], 0))
                    else:
                        home_pts.append({"H": 0, "D": 1, "A": 3}.get(m["FTR"], 0))
                if len(home_pts) >= 3:
                    last3 = np.mean(home_pts[-3:])
                    all_avg = np.mean(home_pts)
                    result.loc[idx, "form_home_momentum"] = last3 - all_avg

            if len(away_all) >= 3:
                away_pts = []
                for _, m in away_all.iterrows():
                    if m["home_team_id"] == away_id:
                        away_pts.append({"H": 3, "D": 1, "A": 0}.get(m["FTR"], 0))
                    else:
                        away_pts.append({"H": 0, "D": 1, "A": 3}.get(m["FTR"], 0))
                if len(away_pts) >= 3:
                    last3 = np.mean(away_pts[-3:])
                    all_avg = np.mean(away_pts)
                    result.loc[idx, "form_away_momentum"] = last3 - all_avg

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
