"""Market / odds features (8 total).

Features derived from bookmaker odds in the match data.

Features:
  mkt_best_odds_home     - Best available home odds
  mkt_best_odds_draw     - Best available draw odds
  mkt_best_odds_away     - Best available away odds
  mkt_avg_odds_home      - Average home odds (market consensus)
  mkt_pinnacle_implied_h - Pinnacle (sharpest) implied home prob
  mkt_implied_draw       - Market-implied draw probability (devigged)
  mkt_market_disagree    - Max - Min home odds (bookmaker disagreement)
  mkt_overround          - Total implied probability (measures margin)
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "mkt_best_odds_home",
    "mkt_best_odds_draw",
    "mkt_best_odds_away",
    "mkt_avg_odds_home",
    "mkt_pinnacle_implied_h",
    "mkt_implied_draw",
    "mkt_market_disagree",
    "mkt_overround",
]

# Odds column sets from Football-Data.co.uk
ODDS_COLS_HOME = ["B365H", "PSH", "WHH"]
ODDS_COLS_DRAW = ["B365D", "PSD", "WHD"]
ODDS_COLS_AWAY = ["B365A", "PSA", "WHA"]


class MarketFeatures(FeatureGroup):
    name = "market"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        result = pd.DataFrame(index=matches.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in matches.iterrows():
            home_odds = [row.get(c) for c in ODDS_COLS_HOME if c in row.index and pd.notna(row.get(c))]
            draw_odds = [row.get(c) for c in ODDS_COLS_DRAW if c in row.index and pd.notna(row.get(c))]
            away_odds = [row.get(c) for c in ODDS_COLS_AWAY if c in row.index and pd.notna(row.get(c))]

            if home_odds:
                result.loc[idx, "mkt_best_odds_home"] = max(home_odds)
                result.loc[idx, "mkt_avg_odds_home"] = np.mean(home_odds)
                result.loc[idx, "mkt_market_disagree"] = max(home_odds) - min(home_odds)
                # Pinnacle = PSH if available, else best
                psh = row.get("PSH")
                if pd.notna(psh) and psh > 0:
                    result.loc[idx, "mkt_pinnacle_implied_h"] = 1.0 / psh
                elif home_odds:
                    result.loc[idx, "mkt_pinnacle_implied_h"] = 1.0 / np.mean(home_odds)

            if draw_odds:
                result.loc[idx, "mkt_best_odds_draw"] = max(draw_odds)

            if away_odds:
                result.loc[idx, "mkt_best_odds_away"] = max(away_odds)

            # Devigged implied draw probability (removes bookmaker margin)
            avg_h = np.mean(home_odds) if home_odds else None
            avg_d = np.mean(draw_odds) if draw_odds else None
            avg_a = np.mean(away_odds) if away_odds else None
            if avg_h and avg_d and avg_a and avg_h > 0 and avg_d > 0 and avg_a > 0:
                overround = 1/avg_h + 1/avg_d + 1/avg_a
                result.loc[idx, "mkt_overround"] = overround * 100
                # Devigged: normalize implied probs to sum to 1
                result.loc[idx, "mkt_implied_draw"] = (1/avg_d) / overround

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
