"""Tennis market / odds features (6 total).

Features derived from bookmaker odds for tennis matches.

Features:
  tnm_best_odds_p1       — Best available odds for P1
  tnm_best_odds_p2       — Best available odds for P2
  tnm_implied_p1         — Devigged implied probability for P1
  tnm_implied_p2         — Devigged implied probability for P2
  tnm_overround          — Total market overround
  tnm_sharp_vs_soft_p1   — Pinnacle vs B365 implied prob difference for P1
"""

import pandas as pd
import numpy as np
from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "tnm_best_odds_p1",
    "tnm_best_odds_p2",
    "tnm_implied_p1",
    "tnm_implied_p2",
    "tnm_overround",
    "tnm_sharp_vs_soft_p1",
]

# Tennis odds columns (winner/loser perspective from tennis-data.co.uk)
ODDS_COLS_WINNER = ["b365_winner", "ps_winner"]
ODDS_COLS_LOSER = ["b365_loser", "ps_loser"]


class TennisMarketFeatures(FeatureGroup):
    name = "tennis_market"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            p1 = row["player1"]
            winner = row["winner"]

            # Map winner/loser odds to P1/P2
            if winner == p1:
                p1_odds_cols = ODDS_COLS_WINNER
                p2_odds_cols = ODDS_COLS_LOSER
            else:
                p1_odds_cols = ODDS_COLS_LOSER
                p2_odds_cols = ODDS_COLS_WINNER

            p1_odds = [row.get(c, np.nan) for c in p1_odds_cols
                       if c in row.index and pd.notna(row.get(c)) and row.get(c) > 0]
            p2_odds = [row.get(c, np.nan) for c in p2_odds_cols
                       if c in row.index and pd.notna(row.get(c)) and row.get(c) > 0]

            if p1_odds:
                result.loc[idx, "tnm_best_odds_p1"] = max(p1_odds)
            if p2_odds:
                result.loc[idx, "tnm_best_odds_p2"] = max(p2_odds)

            # Devigged implied probabilities
            if p1_odds and p2_odds:
                avg_p1 = np.mean(p1_odds)
                avg_p2 = np.mean(p2_odds)
                if avg_p1 > 0 and avg_p2 > 0:
                    raw_p1 = 1.0 / avg_p1
                    raw_p2 = 1.0 / avg_p2
                    overround = raw_p1 + raw_p2
                    result.loc[idx, "tnm_overround"] = overround * 100
                    result.loc[idx, "tnm_implied_p1"] = raw_p1 / overround
                    result.loc[idx, "tnm_implied_p2"] = raw_p2 / overround

            # Sharp vs Soft: Pinnacle vs B365 implied prob difference for P1
            if winner == p1:
                ps_col, b365_col = "ps_winner", "b365_winner"
            else:
                ps_col, b365_col = "ps_loser", "b365_loser"

            ps_val = row.get(ps_col, np.nan)
            b365_val = row.get(b365_col, np.nan)
            if pd.notna(ps_val) and pd.notna(b365_val) and ps_val > 0 and b365_val > 0:
                result.loc[idx, "tnm_sharp_vs_soft_p1"] = (1.0 / ps_val) - (1.0 / b365_val)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
