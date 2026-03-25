"""Tennis head-to-head features (6 total).

Features from historical meetings between two players.

Features:
  tnh_h2h_win_rate_p1    — P1 overall win rate vs P2
  tnh_h2h_count          — Total H2H matches
  tnh_h2h_recent_3_p1    — P1 win rate in last 3 H2H meetings
  tnh_h2h_surface_p1     — P1 win rate vs P2 on current surface
  tnh_h2h_sets_won_pct   — P1 sets won percentage in H2H
  tnh_h2h_rank_gap       — Avg ranking gap in H2H meetings
"""

import pandas as pd
import numpy as np
from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "tnh_h2h_win_rate_p1",
    "tnh_h2h_count",
    "tnh_h2h_recent_3_p1",
    "tnh_h2h_surface_p1",
    "tnh_h2h_sets_won_pct",
    "tnh_h2h_rank_gap",
]


def _parse_sets_from_score(score: str, winner: str, p1: str) -> tuple[int, int]:
    """Parse sets won by P1 and P2 from a score string like '6-3 6-4'."""
    if not isinstance(score, str) or not score.strip():
        return 0, 0
    p1_sets = 0
    p2_sets = 0
    for set_score in score.strip().split():
        parts = set_score.split("-")
        if len(parts) != 2:
            continue
        try:
            s1, s2 = int(parts[0].strip("()")), int(parts[1].strip("()"))
        except (ValueError, IndexError):
            continue
        if winner == p1:
            if s1 > s2:
                p1_sets += 1
            else:
                p2_sets += 1
        else:
            if s1 > s2:
                p2_sets += 1
            else:
                p1_sets += 1
    return p1_sets, p2_sets


class H2HFeatures(FeatureGroup):
    name = "tennis_h2h"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["date"] = pd.to_datetime(df["date"])
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            p1 = row["player1"]
            p2 = row["player2"]
            match_date = row["date"]
            surface = row.get("surface", None)

            prior = df[df["date"] < match_date]
            h2h = prior[
                ((prior["player1"] == p1) & (prior["player2"] == p2))
                | ((prior["player1"] == p2) & (prior["player2"] == p1))
            ]

            if len(h2h) == 0:
                continue

            p1_wins = (h2h["winner"] == p1).sum()
            n = len(h2h)
            result.loc[idx, "tnh_h2h_win_rate_p1"] = p1_wins / n
            result.loc[idx, "tnh_h2h_count"] = n

            # Last 3 H2H
            recent_3 = h2h.tail(3)
            result.loc[idx, "tnh_h2h_recent_3_p1"] = (recent_3["winner"] == p1).sum() / len(recent_3)

            # Surface-specific H2H
            if surface:
                h2h_surf = h2h[h2h["surface"] == surface]
                if len(h2h_surf) > 0:
                    result.loc[idx, "tnh_h2h_surface_p1"] = (h2h_surf["winner"] == p1).sum() / len(h2h_surf)

            # Sets won percentage
            if "score" in h2h.columns:
                total_p1_sets = 0
                total_p2_sets = 0
                for _, m in h2h.iterrows():
                    s1, s2 = _parse_sets_from_score(m.get("score", ""), m["winner"], p1)
                    total_p1_sets += s1
                    total_p2_sets += s2
                total_sets = total_p1_sets + total_p2_sets
                if total_sets > 0:
                    result.loc[idx, "tnh_h2h_sets_won_pct"] = total_p1_sets / total_sets

            # Avg ranking gap
            rank_gaps = []
            for _, m in h2h.iterrows():
                wr = m.get("winner_rank", np.nan)
                lr = m.get("loser_rank", np.nan)
                if pd.notna(wr) and pd.notna(lr):
                    rank_gaps.append(abs(wr - lr))
            if rank_gaps:
                result.loc[idx, "tnh_h2h_rank_gap"] = np.mean(rank_gaps)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
