"""Basketball pace features (6 total).

Features derived from team pace / tempo statistics.

Features:
  bkp_pace_home               - possessions per game (home, rolling)
  bkp_pace_away               - possessions per game (away, rolling)
  bkp_pace_matchup_adj        - adjusted pace for this specific matchup
  bkp_fast_break_rate_home    - transition scoring rate (home)
  bkp_fast_break_rate_away    - transition scoring rate (away)
  bkp_pace_rank_diff          - pace ranking difference (home - away)
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "bkp_pace_home",
    "bkp_pace_away",
    "bkp_pace_matchup_adj",
    "bkp_fast_break_rate_home",
    "bkp_fast_break_rate_away",
    "bkp_pace_rank_diff",
]

# Default NBA pace ~100 possessions per game
_DEFAULT_PACE = 100.0
_ROLLING_N = 10


def _team_rolling_pace(
    prior: pd.DataFrame, team: str, n: int = _ROLLING_N
) -> float:
    """Estimate possessions per game from scores (rough proxy)."""
    team_games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(team_games) == 0:
        return np.nan

    paces = []
    for _, g in team_games.iterrows():
        # Pace proxy: (home_score + away_score) / 2
        total = g.get("home_score", np.nan) + g.get("away_score", np.nan)
        if pd.notna(total):
            paces.append(total / 2.0)
    return np.mean(paces) if paces else np.nan


def _team_fast_break_rate(
    prior: pd.DataFrame, team: str, n: int = _ROLLING_N
) -> float:
    """Estimate fast break scoring rate from fast_break_pts column."""
    team_games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(team_games) == 0 or "fast_break_pts_home" not in team_games.columns:
        return np.nan

    rates = []
    for _, g in team_games.iterrows():
        if g["home_team"] == team:
            fb = g.get("fast_break_pts_home", np.nan)
            total = g.get("home_score", np.nan)
        else:
            fb = g.get("fast_break_pts_away", np.nan)
            total = g.get("away_score", np.nan)
        if pd.notna(fb) and pd.notna(total) and total > 0:
            rates.append(fb / total)
    return np.mean(rates) if rates else np.nan


class PaceFeatures(FeatureGroup):
    name = "basketball_pace"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["game_date"] = pd.to_datetime(df.get("game_date", df.get("date")))
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        # Build per-team pace rankings from all prior data
        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            match_date = row["game_date"]
            prior = df[df["game_date"] < match_date]

            if len(prior) == 0:
                continue

            pace_h = _team_rolling_pace(prior, home)
            pace_a = _team_rolling_pace(prior, away)

            result.loc[idx, "bkp_pace_home"] = pace_h
            result.loc[idx, "bkp_pace_away"] = pace_a

            if pd.notna(pace_h) and pd.notna(pace_a):
                result.loc[idx, "bkp_pace_matchup_adj"] = (pace_h + pace_a) / 2.0
            else:
                result.loc[idx, "bkp_pace_matchup_adj"] = np.nan

            result.loc[idx, "bkp_fast_break_rate_home"] = _team_fast_break_rate(
                prior, home
            )
            result.loc[idx, "bkp_fast_break_rate_away"] = _team_fast_break_rate(
                prior, away
            )

            # Pace rank diff: compute ranks over all teams in prior data
            teams = set(prior["home_team"]) | set(prior["away_team"])
            team_paces = {}
            for t in teams:
                tp = _team_rolling_pace(prior, t)
                if pd.notna(tp):
                    team_paces[t] = tp

            if home in team_paces and away in team_paces:
                sorted_teams = sorted(team_paces, key=team_paces.get, reverse=True)
                rank_h = sorted_teams.index(home) + 1
                rank_a = sorted_teams.index(away) + 1
                result.loc[idx, "bkp_pace_rank_diff"] = rank_h - rank_a
            else:
                result.loc[idx, "bkp_pace_rank_diff"] = np.nan

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
