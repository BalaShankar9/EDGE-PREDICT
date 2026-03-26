"""NFL market features (8 total).

Features derived from bookmaker odds and lines.

Features:
  nfl_mkt_implied_home         - Devigged implied probability (home)
  nfl_mkt_implied_away         - Devigged implied probability (away)
  nfl_mkt_spread_line          - Bookmaker spread line
  nfl_mkt_total_line           - Over/under line
  nfl_mkt_overround            - Market overround
  nfl_mkt_sharp_vs_soft_home   - Pinnacle vs soft book difference
  nfl_mkt_spread_movement      - Spread change from opening
  nfl_mkt_total_movement       - Total change from opening
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nfl_mkt_implied_home",
    "nfl_mkt_implied_away",
    "nfl_mkt_spread_line",
    "nfl_mkt_total_line",
    "nfl_mkt_overround",
    "nfl_mkt_sharp_vs_soft_home",
    "nfl_mkt_spread_movement",
    "nfl_mkt_total_movement",
]


def _devig(odds_home: float, odds_away: float) -> tuple[float, float]:
    """Remove overround from decimal odds to get fair probabilities."""
    if odds_home <= 1 or odds_away <= 1:
        return np.nan, np.nan
    raw_h = 1.0 / odds_home
    raw_a = 1.0 / odds_away
    total = raw_h + raw_a
    if total <= 0:
        return np.nan, np.nan
    return raw_h / total, raw_a / total


class NFLMarketFeatures(FeatureGroup):
    name = "nfl_market"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            odds_h = row.get("odds_home", np.nan)
            odds_a = row.get("odds_away", np.nan)

            if pd.isna(odds_h):
                odds_h = row.get("b365_home", np.nan)
            if pd.isna(odds_a):
                odds_a = row.get("b365_away", np.nan)

            if pd.notna(odds_h) and pd.notna(odds_a):
                p_h, p_a = _devig(odds_h, odds_a)
                result.loc[idx, "nfl_mkt_implied_home"] = p_h
                result.loc[idx, "nfl_mkt_implied_away"] = p_a

                if odds_h > 1 and odds_a > 1:
                    result.loc[idx, "nfl_mkt_overround"] = (
                        (1 / odds_h + 1 / odds_a - 1) * 100
                    )

            spread = row.get("spread_line", np.nan)
            if pd.notna(spread):
                result.loc[idx, "nfl_mkt_spread_line"] = spread

            total = row.get("total_line", np.nan)
            if pd.notna(total):
                result.loc[idx, "nfl_mkt_total_line"] = total

            pin_h = row.get("pinnacle_home", np.nan)
            soft_h = row.get("b365_home", np.nan)
            if pd.notna(pin_h) and pd.notna(soft_h) and pin_h > 1 and soft_h > 1:
                result.loc[idx, "nfl_mkt_sharp_vs_soft_home"] = (
                    1 / pin_h - 1 / soft_h
                )

            spread_open = row.get("spread_open", np.nan)
            spread_close = row.get("spread_line", np.nan)
            if pd.notna(spread_open) and pd.notna(spread_close):
                result.loc[idx, "nfl_mkt_spread_movement"] = spread_close - spread_open

            total_open = row.get("total_open", np.nan)
            total_close = row.get("total_line", np.nan)
            if pd.notna(total_open) and pd.notna(total_close):
                result.loc[idx, "nfl_mkt_total_movement"] = total_close - total_open

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
