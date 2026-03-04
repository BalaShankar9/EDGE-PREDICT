"""Contextual features (6 total).

Match context that doesn't come from performance stats.

Features:
  ctx_home_rest_days   - Days since home team's last match
  ctx_away_rest_days   - Days since away team's last match
  ctx_home_advantage   - League-specific home win rate (historical)
  ctx_match_importance - 1.0 for title/relegation, 0.5 mid-table, 0.0 nothing
  ctx_is_derby         - Local derby flag
  ctx_temperature      - Temperature at match time (placeholder)
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "ctx_home_rest_days",
    "ctx_away_rest_days",
    "ctx_home_advantage",
    "ctx_match_importance",
    "ctx_is_derby",
    "ctx_temperature",
]

# Known derby pairs (canonical team IDs)
DERBIES = {
    frozenset({"arsenal", "tottenham"}),
    frozenset({"liverpool", "everton"}),
    frozenset({"man_united", "man_city"}),
    frozenset({"real_madrid", "barcelona"}),
    frozenset({"ac_milan", "inter"}),
    frozenset({"roma", "lazio"}),
    frozenset({"bayern_munich", "dortmund"}),
    frozenset({"psg", "marseille"}),
    frozenset({"chelsea", "arsenal"}),
    frozenset({"man_united", "liverpool"}),
}


class ContextFeatures(FeatureGroup):
    name = "context"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["match_date"] = pd.to_datetime(df["match_date"])

        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        # Compute league-level home advantage
        league_home_advantage = {}
        if "league" in df.columns and "FTR" in df.columns:
            for league in df["league"].unique():
                league_matches = df[df["league"] == league]
                total = len(league_matches)
                home_wins = (league_matches["FTR"] == "H").sum()
                if total > 0:
                    league_home_advantage[league] = home_wins / total

        for idx, row in df.iterrows():
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]
            match_date = row["match_date"]

            prior = df[df["match_date"] < match_date]

            # Rest days
            home_last = prior[
                (prior["home_team_id"] == home_id) | (prior["away_team_id"] == home_id)
            ]
            if not home_last.empty:
                last_date = home_last["match_date"].max()
                result.loc[idx, "ctx_home_rest_days"] = (match_date - last_date).days

            away_last = prior[
                (prior["home_team_id"] == away_id) | (prior["away_team_id"] == away_id)
            ]
            if not away_last.empty:
                last_date = away_last["match_date"].max()
                result.loc[idx, "ctx_away_rest_days"] = (match_date - last_date).days

            # League home advantage
            league = row.get("league")
            if league and league in league_home_advantage:
                result.loc[idx, "ctx_home_advantage"] = league_home_advantage[league]

            # Match importance: default 0.5 (would need table position data for real calc)
            result.loc[idx, "ctx_match_importance"] = 0.5

            # Derby check
            pair = frozenset({home_id, away_id})
            result.loc[idx, "ctx_is_derby"] = 1.0 if pair in DERBIES else 0.0

            # Temperature: placeholder
            result.loc[idx, "ctx_temperature"] = 15.0  # Default average

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
