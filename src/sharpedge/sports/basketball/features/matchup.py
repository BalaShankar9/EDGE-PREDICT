"""Basketball matchup features (6 total).

Features capturing head-to-head and stylistic matchup dynamics.

Features:
  bkm_pace_mismatch                - difference in preferred pace
  bkm_three_point_rate_diff        - 3PT attempt rate difference
  bkm_paint_scoring_diff           - paint points difference
  bkm_defensive_rating_vs_style    - how well defense matches opponent's offense style
  bkm_historical_margin            - average margin in last 5 H2H
  bkm_divisional_flag              - 1 if division rivals
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "bkm_pace_mismatch",
    "bkm_three_point_rate_diff",
    "bkm_paint_scoring_diff",
    "bkm_defensive_rating_vs_style",
    "bkm_historical_margin",
    "bkm_divisional_flag",
]

# NBA divisions (simplified)
_DIVISIONS = {
    "Atlantic": {"BOS", "BKN", "NYK", "PHI", "TOR"},
    "Central": {"CHI", "CLE", "DET", "IND", "MIL"},
    "Southeast": {"ATL", "CHA", "MIA", "ORL", "WAS"},
    "Northwest": {"DEN", "MIN", "OKC", "POR", "UTA"},
    "Pacific": {"GSW", "LAC", "LAL", "PHX", "SAC"},
    "Southwest": {"DAL", "HOU", "MEM", "NOP", "SAS"},
}

_TEAM_TO_DIV: dict[str, str] = {}
for div, teams in _DIVISIONS.items():
    for t in teams:
        _TEAM_TO_DIV[t] = div

_ROLLING_N = 10


def _team_pace_proxy(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Team's average total score per game as pace proxy."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)
    if len(games) == 0:
        return np.nan
    totals = []
    for _, g in games.iterrows():
        t = g.get("home_score", np.nan) + g.get("away_score", np.nan)
        if pd.notna(t):
            totals.append(t)
    return np.mean(totals) if totals else np.nan


def _team_stat(prior: pd.DataFrame, team: str, col_home: str, col_away: str,
               n: int = _ROLLING_N) -> float:
    """Generic rolling average of a team stat column."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)
    if len(games) == 0:
        return np.nan
    vals = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            v = g.get(col_home, np.nan)
        else:
            v = g.get(col_away, np.nan)
        if pd.notna(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan


def _h2h_margin(prior: pd.DataFrame, home: str, away: str, n: int = 5) -> float:
    """Average margin in last n H2H meetings (from home perspective)."""
    h2h = prior[
        ((prior["home_team"] == home) & (prior["away_team"] == away))
        | ((prior["home_team"] == away) & (prior["away_team"] == home))
    ].tail(n)
    if len(h2h) == 0:
        return np.nan
    margins = []
    for _, g in h2h.iterrows():
        hs = g.get("home_score", np.nan)
        aws = g.get("away_score", np.nan)
        if pd.notna(hs) and pd.notna(aws):
            if g["home_team"] == home:
                margins.append(hs - aws)
            else:
                margins.append(aws - hs)
    return np.mean(margins) if margins else np.nan


class MatchupFeatures(FeatureGroup):
    name = "basketball_matchup"
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
                # Divisional flag can be computed without prior data
                result.loc[idx, "bkm_divisional_flag"] = (
                    1.0 if _TEAM_TO_DIV.get(home) == _TEAM_TO_DIV.get(away)
                    and _TEAM_TO_DIV.get(home) is not None
                    else 0.0
                )
                continue

            # Pace mismatch
            pace_h = _team_pace_proxy(prior, home)
            pace_a = _team_pace_proxy(prior, away)
            if pd.notna(pace_h) and pd.notna(pace_a):
                result.loc[idx, "bkm_pace_mismatch"] = abs(pace_h - pace_a)

            # 3PT rate diff
            three_h = _team_stat(prior, home, "three_rate_home", "three_rate_away")
            three_a = _team_stat(prior, away, "three_rate_home", "three_rate_away")
            if pd.notna(three_h) and pd.notna(three_a):
                result.loc[idx, "bkm_three_point_rate_diff"] = three_h - three_a

            # Paint scoring diff
            paint_h = _team_stat(prior, home, "paint_pts_home", "paint_pts_away")
            paint_a = _team_stat(prior, away, "paint_pts_home", "paint_pts_away")
            if pd.notna(paint_h) and pd.notna(paint_a):
                result.loc[idx, "bkm_paint_scoring_diff"] = paint_h - paint_a

            # Defensive rating vs opponent style
            # Use off rating for opponent and def rating for team
            def_h = _team_stat(prior, home, "def_rating_home", "def_rating_away")
            off_a = _team_stat(prior, away, "off_rating_home", "off_rating_away")
            if pd.notna(def_h) and pd.notna(off_a):
                result.loc[idx, "bkm_defensive_rating_vs_style"] = def_h - off_a

            # H2H historical margin
            result.loc[idx, "bkm_historical_margin"] = _h2h_margin(prior, home, away)

            # Divisional flag
            result.loc[idx, "bkm_divisional_flag"] = (
                1.0 if _TEAM_TO_DIV.get(home) == _TEAM_TO_DIV.get(away)
                and _TEAM_TO_DIV.get(home) is not None
                else 0.0
            )

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
