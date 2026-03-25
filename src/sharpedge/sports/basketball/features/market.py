"""Basketball market features (8 total).

Features derived from bookmaker odds and lines.

Features:
  bkmkt_implied_home          - devigged implied probability (home)
  bkmkt_implied_away          - devigged implied probability (away)
  bkmkt_spread_line           - bookmaker spread line
  bkmkt_total_line            - over/under line
  bkmkt_overround             - market overround
  bkmkt_sharp_vs_soft_home    - Pinnacle vs soft book difference
  bkmkt_spread_movement       - spread change from opening
  bkmkt_total_movement        - total change from opening
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "bkmkt_implied_home",
    "bkmkt_implied_away",
    "bkmkt_spread_line",
    "bkmkt_total_line",
    "bkmkt_overround",
    "bkmkt_sharp_vs_soft_home",
    "bkmkt_spread_movement",
    "bkmkt_total_movement",
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


class BasketballMarketFeatures(FeatureGroup):
    name = "basketball_market"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            # ML odds devigging
            odds_h = row.get("odds_home", np.nan)
            odds_a = row.get("odds_away", np.nan)

            # Try alternative column names
            if pd.isna(odds_h):
                odds_h = row.get("b365_home", np.nan)
            if pd.isna(odds_a):
                odds_a = row.get("b365_away", np.nan)

            if pd.notna(odds_h) and pd.notna(odds_a):
                p_h, p_a = _devig(odds_h, odds_a)
                result.loc[idx, "bkmkt_implied_home"] = p_h
                result.loc[idx, "bkmkt_implied_away"] = p_a

                # Overround
                if odds_h > 1 and odds_a > 1:
                    result.loc[idx, "bkmkt_overround"] = (
                        (1 / odds_h + 1 / odds_a - 1) * 100
                    )

            # Spread line
            spread = row.get("spread_line", np.nan)
            if pd.notna(spread):
                result.loc[idx, "bkmkt_spread_line"] = spread

            # Total line
            total = row.get("total_line", np.nan)
            if pd.notna(total):
                result.loc[idx, "bkmkt_total_line"] = total

            # Sharp vs soft
            pin_h = row.get("pinnacle_home", np.nan)
            soft_h = row.get("b365_home", np.nan)
            if pd.notna(pin_h) and pd.notna(soft_h) and pin_h > 1 and soft_h > 1:
                result.loc[idx, "bkmkt_sharp_vs_soft_home"] = (
                    1 / pin_h - 1 / soft_h
                )

            # Spread movement
            spread_open = row.get("spread_open", np.nan)
            spread_close = row.get("spread_line", np.nan)
            if pd.notna(spread_open) and pd.notna(spread_close):
                result.loc[idx, "bkmkt_spread_movement"] = spread_close - spread_open

            # Total movement
            total_open = row.get("total_open", np.nan)
            total_close = row.get("total_line", np.nan)
            if pd.notna(total_open) and pd.notna(total_close):
                result.loc[idx, "bkmkt_total_movement"] = total_close - total_open

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
