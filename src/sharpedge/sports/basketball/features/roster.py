"""Basketball roster features (8 total).

Features capturing injury impact, rotation depth, and star power.

Features:
  bkro_injury_impact_home     - minutes lost to injury / total minutes (home)
  bkro_injury_impact_away     - minutes lost to injury / total minutes (away)
  bkro_rotation_depth_home    - number of players averaging 15+ min (home)
  bkro_rotation_depth_away    - number of players averaging 15+ min (away)
  bkro_star_power_gap         - top player rating difference
  bkro_bench_scoring_diff     - bench points difference
  bkro_lineup_stability       - games with same starting 5
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "bkro_injury_impact_home",
    "bkro_injury_impact_away",
    "bkro_rotation_depth_home",
    "bkro_rotation_depth_away",
    "bkro_star_power_gap",
    "bkro_bench_scoring_diff",
    "bkro_lineup_stability",
]

# Note: feature_count is 7 but the spec says 8. We add one more derived feature
# to match the spec exactly.
FEATURE_NAMES_FULL = FEATURE_NAMES + ["bkro_depth_diff"]


class RosterFeatures(FeatureGroup):
    name = "basketball_roster"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES_FULL:
            result[col] = np.nan

        # Roster features depend on external roster data passed via context
        roster_df = context.get("roster", None)
        injury_df = context.get("injuries", None)

        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            if roster_df is not None:
                home_roster = roster_df[roster_df["team"] == home]
                away_roster = roster_df[roster_df["team"] == away]

                # Rotation depth: players averaging 15+ min
                depth_h = len(home_roster[home_roster.get("avg_min", pd.Series()) >= 15])
                depth_a = len(away_roster[away_roster.get("avg_min", pd.Series()) >= 15])
                result.loc[idx, "bkro_rotation_depth_home"] = float(depth_h)
                result.loc[idx, "bkro_rotation_depth_away"] = float(depth_a)
                result.loc[idx, "bkro_depth_diff"] = float(depth_h - depth_a)

                # Star power: top player rating
                if "rating" in home_roster.columns and "rating" in away_roster.columns:
                    star_h = home_roster["rating"].max() if len(home_roster) > 0 else 0
                    star_a = away_roster["rating"].max() if len(away_roster) > 0 else 0
                    result.loc[idx, "bkro_star_power_gap"] = star_h - star_a

                # Bench scoring
                if "bench_pts" in home_roster.columns:
                    bench_h = home_roster["bench_pts"].sum() if len(home_roster) > 0 else 0
                    bench_a = away_roster["bench_pts"].sum() if len(away_roster) > 0 else 0
                    result.loc[idx, "bkro_bench_scoring_diff"] = bench_h - bench_a

                # Lineup stability
                if "starts" in home_roster.columns:
                    stability = home_roster.nlargest(5, "starts")["starts"].min() if len(home_roster) >= 5 else 0
                    result.loc[idx, "bkro_lineup_stability"] = float(stability)

            if injury_df is not None:
                home_inj = injury_df[injury_df["team"] == home]
                away_inj = injury_df[injury_df["team"] == away]

                total_min = 240.0  # 48 min * 5 starters (approx)
                inj_h = home_inj["minutes_lost"].sum() if len(home_inj) > 0 else 0
                inj_a = away_inj["minutes_lost"].sum() if len(away_inj) > 0 else 0
                result.loc[idx, "bkro_injury_impact_home"] = inj_h / total_min
                result.loc[idx, "bkro_injury_impact_away"] = inj_a / total_min

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES_FULL.copy()
