"""Market / odds features (20 total).

Features derived from bookmaker odds in the match data.
Uses Shin (1993) method for devigging — mathematically superior to
basic normalization because it models the favourite-longshot bias.

V2 additions (6 new features):
  mkt_sharp_vs_soft_home  - Pinnacle vs B365 implied prob difference (smart money signal)
  mkt_sharp_vs_soft_away  - Same for away
  mkt_max_avg_ratio_home  - Max/Avg home odds ratio (line movement proxy)
  mkt_max_avg_ratio_away  - Max/Avg away odds ratio
  mkt_pinnacle_edge_home  - How much Pinnacle disagrees with market average
  mkt_ou_overround        - Over/Under market overround (tightness = more info)

V5 additions (3 new features):
  mkt_goto_home  - goto_conversion devigged home probability (Kaggle gold technique)
  mkt_goto_draw  - goto_conversion devigged draw probability
  mkt_goto_away  - goto_conversion devigged away probability
"""
import logging

import pandas as pd
import numpy as np
from penaltyblog.implied import calculate_implied
from sharpedge.ml.features.base import FeatureGroup

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    # Original 11
    "mkt_best_odds_home",
    "mkt_best_odds_draw",
    "mkt_best_odds_away",
    "mkt_avg_odds_home",
    "mkt_pinnacle_implied_h",
    "mkt_implied_draw",
    "mkt_market_disagree",
    "mkt_overround",
    "mkt_shin_home",
    "mkt_shin_draw",
    "mkt_shin_away",
    # V2: Smart money / line movement signals (6 new)
    "mkt_sharp_vs_soft_home",
    "mkt_sharp_vs_soft_away",
    "mkt_max_avg_ratio_home",
    "mkt_max_avg_ratio_away",
    "mkt_pinnacle_edge_home",
    "mkt_ou_overround",
    # V5: goto_conversion devigged probabilities (3 new)
    "mkt_goto_home",
    "mkt_goto_draw",
    "mkt_goto_away",
]

# Odds column sets from Football-Data.co.uk
ODDS_COLS_HOME = ["B365H", "PSH", "WHH"]
ODDS_COLS_DRAW = ["B365D", "PSD", "WHD"]
ODDS_COLS_AWAY = ["B365A", "PSA", "WHA"]


class MarketFeatures(FeatureGroup):
    name = "market"
    feature_count = 20

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
                result.loc[idx, "mkt_implied_draw"] = (1/avg_d) / overround

                # Shin implied probabilities (models favourite-longshot bias)
                try:
                    shin = calculate_implied(
                        [avg_h, avg_d, avg_a], method="shin"
                    )
                    probs = shin.probabilities
                    result.loc[idx, "mkt_shin_home"] = probs[0]
                    result.loc[idx, "mkt_shin_draw"] = probs[1]
                    result.loc[idx, "mkt_shin_away"] = probs[2]
                except Exception:
                    pass

                # goto_conversion devigged probabilities (Kaggle gold technique)
                try:
                    from goto_conversion import Goto
                    goto = Goto()
                    goto_probs = goto.convert([avg_h, avg_d, avg_a])
                    result.loc[idx, "mkt_goto_home"] = goto_probs[0]
                    result.loc[idx, "mkt_goto_draw"] = goto_probs[1]
                    result.loc[idx, "mkt_goto_away"] = goto_probs[2]
                except Exception:
                    pass

            # === V2: Smart money / line movement features ===

            # Sharp vs Soft: Pinnacle implied prob minus B365 implied prob
            # Positive = Pinnacle thinks home is MORE likely than B365 (sharp money on home)
            psh_val = row.get("PSH")
            b365h = row.get("B365H")
            psa_val = row.get("PSA")
            b365a = row.get("B365A")

            if pd.notna(psh_val) and pd.notna(b365h) and psh_val > 0 and b365h > 0:
                sharp_home = 1.0 / psh_val
                soft_home = 1.0 / b365h
                result.loc[idx, "mkt_sharp_vs_soft_home"] = sharp_home - soft_home

            if pd.notna(psa_val) and pd.notna(b365a) and psa_val > 0 and b365a > 0:
                sharp_away = 1.0 / psa_val
                soft_away = 1.0 / b365a
                result.loc[idx, "mkt_sharp_vs_soft_away"] = sharp_away - soft_away

            # Max/Avg odds ratio: >1 means someone is offering much better odds
            # (line hasn't moved at that book yet = potential value)
            max_h = row.get("MaxH")
            avg_h_col = row.get("AvgH")
            max_a = row.get("MaxA")
            avg_a_col = row.get("AvgA")

            if pd.notna(max_h) and pd.notna(avg_h_col) and avg_h_col > 0:
                result.loc[idx, "mkt_max_avg_ratio_home"] = max_h / avg_h_col
            if pd.notna(max_a) and pd.notna(avg_a_col) and avg_a_col > 0:
                result.loc[idx, "mkt_max_avg_ratio_away"] = max_a / avg_a_col

            # Pinnacle edge: how much the sharpest book disagrees with the avg
            if pd.notna(psh_val) and avg_h and psh_val > 0:
                pin_implied = 1.0 / psh_val
                avg_implied = 1.0 / avg_h
                result.loc[idx, "mkt_pinnacle_edge_home"] = pin_implied - avg_implied

            # O/U market overround: tighter market = more information priced in
            ou_over = row.get("Avg>2.5")
            ou_under = row.get("Avg<2.5")
            if pd.notna(ou_over) and pd.notna(ou_under) and ou_over > 0 and ou_under > 0:
                ou_or = (1.0 / ou_over + 1.0 / ou_under) * 100
                result.loc[idx, "mkt_ou_overround"] = ou_or

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
