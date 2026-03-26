"""Ice hockey goaltending features (8 total).

Features capturing goaltender performance and workload.

Features:
  nhl_save_pct_home            - Save percentage (home goalie, rolling)
  nhl_save_pct_away            - Save percentage (away goalie, rolling)
  nhl_gsax_home                - Goals saved above expected (home)
  nhl_gsax_away                - Goals saved above expected (away)
  nhl_high_danger_save_pct     - High-danger save % differential
  nhl_goalie_form_10           - Goalie form index (last 10 games)
  nhl_backup_flag              - 1 if backup goalie is likely starting
  nhl_workload_7d              - Games started in last 7 days
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nhl_save_pct_home",
    "nhl_save_pct_away",
    "nhl_gsax_home",
    "nhl_gsax_away",
    "nhl_high_danger_save_pct",
    "nhl_goalie_form_10",
    "nhl_backup_flag",
    "nhl_workload_7d",
]

_ROLLING_N = 10
_AVG_GOALS_AGAINST = 3.0  # league avg for GSAX baseline


def _goals_against_avg(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Rolling goals against average for a team."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    ga = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            ga.append(g.get("away_score", np.nan))
        else:
            ga.append(g.get("home_score", np.nan))

    valid = [v for v in ga if pd.notna(v)]
    return np.mean(valid) if valid else np.nan


def _games_in_window(prior: pd.DataFrame, team: str, match_date, days: int = 7) -> int:
    """Count team games in the last N days."""
    cutoff = match_date - pd.Timedelta(days=days)
    recent = prior[
        (prior["game_date"] >= cutoff) &
        ((prior["home_team"] == team) | (prior["away_team"] == team))
    ]
    return len(recent)


class GoaltendingFeatures(FeatureGroup):
    name = "hockey_goaltending"
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

            if len(prior) == 0:
                continue

            gaa_h = _goals_against_avg(prior, home)
            gaa_a = _goals_against_avg(prior, away)

            # Save % proxy: 1 - (GA / expected shots per game ~32)
            if pd.notna(gaa_h):
                result.loc[idx, "nhl_save_pct_home"] = 1.0 - gaa_h / 32.0
            if pd.notna(gaa_a):
                result.loc[idx, "nhl_save_pct_away"] = 1.0 - gaa_a / 32.0

            # GSAX: goals saved above expected (positive = better)
            if pd.notna(gaa_h):
                result.loc[idx, "nhl_gsax_home"] = _AVG_GOALS_AGAINST - gaa_h
            if pd.notna(gaa_a):
                result.loc[idx, "nhl_gsax_away"] = _AVG_GOALS_AGAINST - gaa_a

            # High-danger save % differential
            if pd.notna(gaa_h) and pd.notna(gaa_a):
                sv_h = 1.0 - gaa_h / 32.0
                sv_a = 1.0 - gaa_a / 32.0
                result.loc[idx, "nhl_high_danger_save_pct"] = sv_h - sv_a

            # Goalie form — recent GAA trend (lower is better, so invert)
            if pd.notna(gaa_h):
                result.loc[idx, "nhl_goalie_form_10"] = _AVG_GOALS_AGAINST - gaa_h

            # Backup flag — if team played 3+ games in last 5 days, likely backup
            workload = _games_in_window(prior, home, match_date, days=5)
            result.loc[idx, "nhl_backup_flag"] = 1.0 if workload >= 3 else 0.0

            # Workload in last 7 days
            result.loc[idx, "nhl_workload_7d"] = float(
                _games_in_window(prior, home, match_date, days=7)
            )

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
