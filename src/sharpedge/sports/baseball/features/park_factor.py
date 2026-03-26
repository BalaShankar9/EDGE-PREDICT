"""Baseball park factor features (4 total).

Features capturing ballpark effects on scoring.

Features:
  mlb_park_run_factor          - Park run factor (relative to neutral)
  mlb_park_hr_factor           - Park home run factor
  mlb_park_factor_handedness   - Park factor by batter handedness
  mlb_altitude                 - Stadium altitude factor (Coors = 1.0)
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "mlb_park_run_factor",
    "mlb_park_hr_factor",
    "mlb_park_factor_handedness",
    "mlb_altitude",
]

# Known hitter-friendly parks
_HITTER_PARKS = {"COL", "CIN", "TEX", "PHI", "MIL", "BAL", "BOS"}
_PITCHER_PARKS = {"SF", "SD", "MIA", "OAK", "SEA", "NYM", "TB"}
_HIGH_ALTITUDE = {"COL"}  # Coors Field

_ROLLING_N = 20


class ParkFactorFeatures(FeatureGroup):
    name = "baseball_park_factor"
    feature_count = 4

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["game_date"] = pd.to_datetime(df.get("game_date", df.get("date")))
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            home = row["home_team"]
            match_date = row["game_date"]
            prior = df[df["game_date"] < match_date]

            # Park run factor from historical data at this venue
            if len(prior) > 0:
                home_games = prior[prior["home_team"] == home].tail(_ROLLING_N)
                if len(home_games) > 0:
                    avg_total = (
                        home_games["home_score"].mean() + home_games["away_score"].mean()
                    )
                    # Normalize: league average ~9 runs total
                    result.loc[idx, "mlb_park_run_factor"] = avg_total / 9.0
                else:
                    # Use known park type as fallback
                    if home in _HITTER_PARKS:
                        result.loc[idx, "mlb_park_run_factor"] = 1.10
                    elif home in _PITCHER_PARKS:
                        result.loc[idx, "mlb_park_run_factor"] = 0.90
                    else:
                        result.loc[idx, "mlb_park_run_factor"] = 1.00
            else:
                if home in _HITTER_PARKS:
                    result.loc[idx, "mlb_park_run_factor"] = 1.10
                elif home in _PITCHER_PARKS:
                    result.loc[idx, "mlb_park_run_factor"] = 0.90
                else:
                    result.loc[idx, "mlb_park_run_factor"] = 1.00

            # HR factor proxy (correlated with run factor)
            pf = result.loc[idx, "mlb_park_run_factor"]
            if pd.notna(pf):
                result.loc[idx, "mlb_park_hr_factor"] = pf * 1.05

            # Handedness factor (default neutral without detailed data)
            result.loc[idx, "mlb_park_factor_handedness"] = 0.0

            # Altitude
            result.loc[idx, "mlb_altitude"] = 1.0 if home in _HIGH_ALTITUDE else 0.0

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
