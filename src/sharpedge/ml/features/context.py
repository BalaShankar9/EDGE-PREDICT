"""Contextual features (6 total).

Match context that doesn't come from performance stats.

Features:
  ctx_home_rest_days   - Days since home team's last match
  ctx_away_rest_days   - Days since away team's last match
  ctx_home_advantage   - League-specific home win rate (historical)
  ctx_rest_advantage   - Rest days difference (positive = home more rested)
  ctx_is_derby         - Local derby flag
  ctx_month            - Month of year (seasonality signal)
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "ctx_home_rest_days",
    "ctx_away_rest_days",
    "ctx_home_advantage",
    "ctx_rest_advantage",
    "ctx_is_derby",
    "ctx_month",
]

# Known derby pairs (canonical team IDs)
DERBIES = {
    frozenset({"Arsenal", "Tottenham"}),
    frozenset({"Liverpool", "Everton"}),
    frozenset({"Man United", "Man City"}),
    frozenset({"Real Madrid", "Barcelona"}),
    frozenset({"AC Milan", "Inter"}),
    frozenset({"Roma", "Lazio"}),
    frozenset({"Bayern Munich", "Dortmund"}),
    frozenset({"Paris SG", "Marseille"}),
    frozenset({"Chelsea", "Arsenal"}),
    frozenset({"Man United", "Liverpool"}),
    frozenset({"Chelsea", "Tottenham"}),
    frozenset({"Juventus", "Inter"}),
    frozenset({"Ath Madrid", "Real Madrid"}),
    frozenset({"Leverkusen", "Dortmund"}),
    frozenset({"Lyon", "Marseille"}),
    frozenset({"Napoli", "Juventus"}),
    frozenset({"West Ham", "Tottenham"}),
    frozenset({"Newcastle", "Sunderland"}),
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
            home_rest = np.nan
            home_last = prior[
                (prior["home_team_id"] == home_id) | (prior["away_team_id"] == home_id)
            ]
            if not home_last.empty:
                last_date = home_last["match_date"].max()
                home_rest = (match_date - last_date).days
                result.loc[idx, "ctx_home_rest_days"] = home_rest

            away_rest = np.nan
            away_last = prior[
                (prior["home_team_id"] == away_id) | (prior["away_team_id"] == away_id)
            ]
            if not away_last.empty:
                last_date = away_last["match_date"].max()
                away_rest = (match_date - last_date).days
                result.loc[idx, "ctx_away_rest_days"] = away_rest

            # Rest advantage (positive = home team more rested)
            if not np.isnan(home_rest) and not np.isnan(away_rest):
                result.loc[idx, "ctx_rest_advantage"] = home_rest - away_rest

            # League home advantage
            league = row.get("league")
            if league and league in league_home_advantage:
                result.loc[idx, "ctx_home_advantage"] = league_home_advantage[league]

            # Derby check (use team names from home_team_name column if available)
            home_name = row.get("home_team_name", str(home_id))
            away_name = row.get("away_team_name", str(away_id))
            pair = frozenset({home_name, away_name})
            result.loc[idx, "ctx_is_derby"] = 1.0 if pair in DERBIES else 0.0

            # Month (seasonality — early/late season effects)
            result.loc[idx, "ctx_month"] = match_date.month

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
