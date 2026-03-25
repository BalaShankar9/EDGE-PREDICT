"""Tennis ranking features (8 total).

Features derived from ATP/WTA rankings at the time of each match.

Features:
  tnk_rank_diff          — Winner rank minus loser rank (negative = favourite)
  tnk_rank_ratio         — Lower rank / higher rank
  tnk_rank_p1            — Player 1 ranking
  tnk_rank_p2            — Player 2 ranking
  tnk_points_diff        — Ranking points difference
  tnk_rank_momentum_p1   — Rank change over last 4 weeks for P1
  tnk_rank_momentum_p2   — Rank change over last 4 weeks for P2
  tnk_top10_flag         — 1 if either player is top 10
"""

import pandas as pd
import numpy as np
from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "tnk_rank_diff",
    "tnk_rank_ratio",
    "tnk_rank_p1",
    "tnk_rank_p2",
    "tnk_points_diff",
    "tnk_rank_momentum_p1",
    "tnk_rank_momentum_p2",
    "tnk_top10_flag",
]


class RankingFeatures(FeatureGroup):
    name = "tennis_ranking"
    feature_count = 8

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

            # Determine P1/P2 ranks from winner/loser columns
            if row["winner"] == p1:
                p1_rank = row.get("winner_rank", np.nan)
                p2_rank = row.get("loser_rank", np.nan)
                p1_pts = row.get("winner_points", np.nan)
                p2_pts = row.get("loser_points", np.nan)
            else:
                p1_rank = row.get("loser_rank", np.nan)
                p2_rank = row.get("winner_rank", np.nan)
                p1_pts = row.get("loser_points", np.nan)
                p2_pts = row.get("winner_points", np.nan)

            result.loc[idx, "tnk_rank_p1"] = p1_rank
            result.loc[idx, "tnk_rank_p2"] = p2_rank

            if pd.notna(p1_rank) and pd.notna(p2_rank):
                result.loc[idx, "tnk_rank_diff"] = p1_rank - p2_rank
                higher = max(p1_rank, p2_rank)
                lower = min(p1_rank, p2_rank)
                result.loc[idx, "tnk_rank_ratio"] = lower / higher if higher > 0 else np.nan
                result.loc[idx, "tnk_top10_flag"] = 1.0 if min(p1_rank, p2_rank) <= 10 else 0.0

            if pd.notna(p1_pts) and pd.notna(p2_pts):
                result.loc[idx, "tnk_points_diff"] = p1_pts - p2_pts

            # Rank momentum: look at matches in last 28 days for each player
            prior = df[df["date"] < match_date]
            cutoff = match_date - pd.Timedelta(days=28)
            recent = prior[prior["date"] >= cutoff]

            for player, col_name in [(p1, "tnk_rank_momentum_p1"), (p2, "tnk_rank_momentum_p2")]:
                player_matches = recent[
                    (recent["winner"] == player) | (recent["player1"] == player) | (recent["player2"] == player)
                ]
                if len(player_matches) >= 2:
                    # Get earliest and latest rank for this player in the window
                    ranks = []
                    for _, m in player_matches.iterrows():
                        if m["winner"] == player:
                            r = m.get("winner_rank", np.nan)
                        else:
                            r = m.get("loser_rank", np.nan)
                        if pd.notna(r):
                            ranks.append((m["date"], r))
                    if len(ranks) >= 2:
                        ranks.sort(key=lambda x: x[0])
                        # Positive momentum = rank number decreasing (improving)
                        result.loc[idx, col_name] = ranks[0][1] - ranks[-1][1]

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
