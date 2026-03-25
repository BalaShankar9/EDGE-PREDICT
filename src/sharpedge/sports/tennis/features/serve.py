"""Tennis serve features (8 total).

Features derived from serve statistics. Gracefully handles missing columns
since tennis-data.co.uk CSVs may not include detailed serve stats.

Features:
  tns_ace_rate_p1        — P1 aces per match (rolling 10)
  tns_ace_rate_p2        — P2 aces per match
  tns_df_rate_p1         — P1 double faults per match
  tns_df_rate_p2         — P2 double faults per match
  tns_1st_serve_pct_p1   — P1 first serve percentage
  tns_1st_serve_pct_p2   — P2 first serve percentage
  tns_bp_save_pct_p1     — P1 break point save percentage
  tns_bp_convert_pct_p1  — P1 break point conversion percentage
"""

import pandas as pd
import numpy as np
from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "tns_ace_rate_p1",
    "tns_ace_rate_p2",
    "tns_df_rate_p1",
    "tns_df_rate_p2",
    "tns_1st_serve_pct_p1",
    "tns_1st_serve_pct_p2",
    "tns_bp_save_pct_p1",
    "tns_bp_convert_pct_p1",
]

# Column mappings for serve stats (winner/loser perspective)
# These are common column names in ATP match datasets
SERVE_COLS = {
    "w_ace": "aces",
    "l_ace": "aces",
    "w_df": "double_faults",
    "l_df": "double_faults",
    "w_1stIn": "first_serve_in",
    "w_svpt": "serve_points",
    "l_1stIn": "first_serve_in",
    "l_svpt": "serve_points",
    "w_bpSaved": "bp_saved",
    "w_bpFaced": "bp_faced",
    "l_bpSaved": "bp_saved",
    "l_bpFaced": "bp_faced",
}


def _rolling_stat(prior_matches: pd.DataFrame, player: str, stat_col_w: str,
                  stat_col_l: str, n: int = 10) -> float:
    """Compute rolling average of a stat for a player over last n matches."""
    player_matches = prior_matches[
        (prior_matches["player1"] == player) | (prior_matches["player2"] == player)
    ].tail(n)

    if len(player_matches) == 0:
        return np.nan

    values = []
    for _, m in player_matches.iterrows():
        if m["winner"] == player:
            val = m.get(stat_col_w, np.nan)
        else:
            val = m.get(stat_col_l, np.nan)
        if pd.notna(val):
            values.append(val)

    return np.mean(values) if values else np.nan


def _first_serve_pct(prior_matches: pd.DataFrame, player: str, n: int = 10) -> float:
    """Compute rolling first serve percentage."""
    player_matches = prior_matches[
        (prior_matches["player1"] == player) | (prior_matches["player2"] == player)
    ].tail(n)

    if len(player_matches) == 0:
        return np.nan

    pcts = []
    for _, m in player_matches.iterrows():
        if m["winner"] == player:
            sin = m.get("w_1stIn", np.nan)
            svpt = m.get("w_svpt", np.nan)
        else:
            sin = m.get("l_1stIn", np.nan)
            svpt = m.get("l_svpt", np.nan)
        if pd.notna(sin) and pd.notna(svpt) and svpt > 0:
            pcts.append(sin / svpt)

    return np.mean(pcts) if pcts else np.nan


def _bp_save_pct(prior_matches: pd.DataFrame, player: str, n: int = 10) -> float:
    """Break point save percentage."""
    player_matches = prior_matches[
        (prior_matches["player1"] == player) | (prior_matches["player2"] == player)
    ].tail(n)

    total_saved = 0
    total_faced = 0
    for _, m in player_matches.iterrows():
        if m["winner"] == player:
            saved = m.get("w_bpSaved", np.nan)
            faced = m.get("w_bpFaced", np.nan)
        else:
            saved = m.get("l_bpSaved", np.nan)
            faced = m.get("l_bpFaced", np.nan)
        if pd.notna(saved) and pd.notna(faced):
            total_saved += saved
            total_faced += faced

    return total_saved / total_faced if total_faced > 0 else np.nan


def _bp_convert_pct(prior_matches: pd.DataFrame, player: str, n: int = 10) -> float:
    """Break point conversion percentage (how well the player breaks opponent)."""
    player_matches = prior_matches[
        (prior_matches["player1"] == player) | (prior_matches["player2"] == player)
    ].tail(n)

    total_converted = 0
    total_opp_faced = 0
    for _, m in player_matches.iterrows():
        # When player is the winner, opponent's bp stats are loser's
        if m["winner"] == player:
            opp_faced = m.get("l_bpFaced", np.nan)
            opp_saved = m.get("l_bpSaved", np.nan)
        else:
            opp_faced = m.get("w_bpFaced", np.nan)
            opp_saved = m.get("w_bpSaved", np.nan)
        if pd.notna(opp_faced) and pd.notna(opp_saved):
            total_converted += (opp_faced - opp_saved)
            total_opp_faced += opp_faced

    return total_converted / total_opp_faced if total_opp_faced > 0 else np.nan


class ServeFeatures(FeatureGroup):
    name = "tennis_serve"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        # Check if serve stat columns exist at all
        has_serve_data = any(c in df.columns for c in ["w_ace", "l_ace", "w_df", "l_df"])
        if not has_serve_data:
            return result  # Return all NaN gracefully

        for idx, row in df.iterrows():
            p1 = row["player1"]
            p2 = row["player2"]
            match_date = row["date"]

            prior = df[df["date"] < match_date]
            if len(prior) == 0:
                continue

            result.loc[idx, "tns_ace_rate_p1"] = _rolling_stat(prior, p1, "w_ace", "l_ace")
            result.loc[idx, "tns_ace_rate_p2"] = _rolling_stat(prior, p2, "w_ace", "l_ace")
            result.loc[idx, "tns_df_rate_p1"] = _rolling_stat(prior, p1, "w_df", "l_df")
            result.loc[idx, "tns_df_rate_p2"] = _rolling_stat(prior, p2, "w_df", "l_df")
            result.loc[idx, "tns_1st_serve_pct_p1"] = _first_serve_pct(prior, p1)
            result.loc[idx, "tns_1st_serve_pct_p2"] = _first_serve_pct(prior, p2)
            result.loc[idx, "tns_bp_save_pct_p1"] = _bp_save_pct(prior, p1)
            result.loc[idx, "tns_bp_convert_pct_p1"] = _bp_convert_pct(prior, p1)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
