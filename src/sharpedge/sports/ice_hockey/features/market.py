"""Ice hockey market features (6 total).

Features derived from bookmaker odds and lines.

Features:
  nhl_mkt_implied_home         - Devigged implied probability (home)
  nhl_mkt_implied_away         - Devigged implied probability (away)
  nhl_mkt_puck_line            - Puck line value
  nhl_mkt_overround            - Market overround
  nhl_mkt_sharp_vs_soft        - Pinnacle vs soft book difference
  nhl_mkt_total_line           - Over/under line
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nhl_mkt_implied_home",
    "nhl_mkt_implied_away",
    "nhl_mkt_puck_line",
    "nhl_mkt_overround",
    "nhl_mkt_sharp_vs_soft",
    "nhl_mkt_total_line",
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


class HockeyMarketFeatures(FeatureGroup):
    name = "hockey_market"
    feature_count = 6

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
                result.loc[idx, "nhl_mkt_implied_home"] = p_h
                result.loc[idx, "nhl_mkt_implied_away"] = p_a

                if odds_h > 1 and odds_a > 1:
                    result.loc[idx, "nhl_mkt_overround"] = (
                        (1 / odds_h + 1 / odds_a - 1) * 100
                    )

            puck_line = row.get("puck_line", np.nan)
            if pd.notna(puck_line):
                result.loc[idx, "nhl_mkt_puck_line"] = puck_line

            pin_h = row.get("pinnacle_home", np.nan)
            soft_h = row.get("b365_home", np.nan)
            if pd.notna(pin_h) and pd.notna(soft_h) and pin_h > 1 and soft_h > 1:
                result.loc[idx, "nhl_mkt_sharp_vs_soft"] = (
                    1 / pin_h - 1 / soft_h
                )

            total = row.get("total_line", np.nan)
            if pd.notna(total):
                result.loc[idx, "nhl_mkt_total_line"] = total

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
