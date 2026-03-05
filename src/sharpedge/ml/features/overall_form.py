"""Overall (cross-venue) form features (8 total).

Captures team strength across ALL matches regardless of venue.
Complements the venue-specific form features by providing a broader
picture of team quality and consistency.

Features:
  oform_home_ppg_10        - Home team points per game, last 10 ALL matches
  oform_away_ppg_10        - Away team points per game, last 10 ALL matches
  oform_home_gd_10         - Home team goal difference per game, last 10 ALL matches
  oform_away_gd_10         - Away team goal difference per game, last 10 ALL matches
  oform_ppg_diff           - Home overall PPG - Away overall PPG
  oform_gd_diff            - Home overall GD/game - Away overall GD/game
  oform_home_consistency   - Home team std dev of points (low = consistent)
  oform_away_consistency   - Away team std dev of points (low = consistent)
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

WINDOW = 10

FEATURE_NAMES = [
    "oform_home_ppg_10",
    "oform_away_ppg_10",
    "oform_home_gd_10",
    "oform_away_gd_10",
    "oform_ppg_diff",
    "oform_gd_diff",
    "oform_home_consistency",
    "oform_away_consistency",
]


def _team_match_stats(matches_subset: pd.DataFrame, team_id: str) -> tuple[list, list]:
    """Extract points and goal differences for a team from match rows."""
    points = []
    gds = []
    for _, m in matches_subset.iterrows():
        if m["home_team_id"] == team_id:
            pts = {"H": 3, "D": 1, "A": 0}.get(m["FTR"], 0)
            gd = m["FTHG"] - m["FTAG"]
        else:
            pts = {"H": 0, "D": 1, "A": 3}.get(m["FTR"], 0)
            gd = m["FTAG"] - m["FTHG"]
        points.append(pts)
        gds.append(gd)
    return points, gds


class OverallFormFeatures(FeatureGroup):
    name = "overall_form"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])
        df = df.sort_values("match_date").reset_index(drop=True)

        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx in range(len(df)):
            row = df.iloc[idx]
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
            prior = df.iloc[:idx]

            # Home team: last 10 ALL matches
            home_all = prior[
                (prior["home_team_id"] == home_id) | (prior["away_team_id"] == home_id)
            ].tail(WINDOW)

            # Away team: last 10 ALL matches
            away_all = prior[
                (prior["home_team_id"] == away_id) | (prior["away_team_id"] == away_id)
            ].tail(WINDOW)

            if len(home_all) >= 5:
                h_pts, h_gds = _team_match_stats(home_all, home_id)
                h_ppg = np.mean(h_pts)
                h_gd = np.mean(h_gds)
                result.iloc[idx, result.columns.get_loc("oform_home_ppg_10")] = h_ppg
                result.iloc[idx, result.columns.get_loc("oform_home_gd_10")] = h_gd
                result.iloc[idx, result.columns.get_loc("oform_home_consistency")] = np.std(h_pts)

            if len(away_all) >= 5:
                a_pts, a_gds = _team_match_stats(away_all, away_id)
                a_ppg = np.mean(a_pts)
                a_gd = np.mean(a_gds)
                result.iloc[idx, result.columns.get_loc("oform_away_ppg_10")] = a_ppg
                result.iloc[idx, result.columns.get_loc("oform_away_gd_10")] = a_gd
                result.iloc[idx, result.columns.get_loc("oform_away_consistency")] = np.std(a_pts)

            # Diffs (only if both available)
            h_ppg_val = result.iloc[idx, result.columns.get_loc("oform_home_ppg_10")]
            a_ppg_val = result.iloc[idx, result.columns.get_loc("oform_away_ppg_10")]
            if not np.isnan(h_ppg_val) and not np.isnan(a_ppg_val):
                result.iloc[idx, result.columns.get_loc("oform_ppg_diff")] = h_ppg_val - a_ppg_val

            h_gd_val = result.iloc[idx, result.columns.get_loc("oform_home_gd_10")]
            a_gd_val = result.iloc[idx, result.columns.get_loc("oform_away_gd_10")]
            if not np.isnan(h_gd_val) and not np.isnan(a_gd_val):
                result.iloc[idx, result.columns.get_loc("oform_gd_diff")] = h_gd_val - a_gd_val

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
