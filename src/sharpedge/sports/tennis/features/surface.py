"""Tennis surface features (6 total).

Features based on player performance on specific court surfaces.

Features:
  tns_surface_win_rate_p1   — P1 win rate on this surface (last 20 matches)
  tns_surface_win_rate_p2   — P2 win rate on this surface
  tns_surface_matches_p1    — P1 matches played on this surface (experience)
  tns_surface_matches_p2    — P2 matches played on surface
  tns_clay_specialist_diff  — Difference in clay specialist score
  tns_surface_h2h           — P1 win rate vs P2 on this surface specifically
"""

import pandas as pd
import numpy as np
from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "tns_surface_win_rate_p1",
    "tns_surface_win_rate_p2",
    "tns_surface_matches_p1",
    "tns_surface_matches_p2",
    "tns_clay_specialist_diff",
    "tns_surface_h2h",
]


def _player_surface_stats(prior_surface: pd.DataFrame, player: str, n: int = 20):
    """Get player's win rate on a surface from prior matches (last n)."""
    player_matches = prior_surface[
        (prior_surface["player1"] == player) | (prior_surface["player2"] == player)
    ].tail(n)
    if len(player_matches) == 0:
        return np.nan, 0
    wins = (player_matches["winner"] == player).sum()
    return wins / len(player_matches), len(player_matches)


def _clay_specialist_score(prior: pd.DataFrame, player: str) -> float:
    """Score > 0.6 win rate on clay = specialist."""
    clay = prior[prior["surface"] == "Clay"]
    player_clay = clay[(clay["player1"] == player) | (clay["player2"] == player)]
    if len(player_clay) < 5:
        return 0.0
    wins = (player_clay["winner"] == player).sum()
    rate = wins / len(player_clay)
    return rate if rate > 0.6 else 0.0


class SurfaceFeatures(FeatureGroup):
    name = "tennis_surface"
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

            if surface and len(prior) > 0:
                prior_surface = prior[prior["surface"] == surface]

                wr1, mc1 = _player_surface_stats(prior_surface, p1)
                wr2, mc2 = _player_surface_stats(prior_surface, p2)

                result.loc[idx, "tns_surface_win_rate_p1"] = wr1
                result.loc[idx, "tns_surface_win_rate_p2"] = wr2
                result.loc[idx, "tns_surface_matches_p1"] = mc1
                result.loc[idx, "tns_surface_matches_p2"] = mc2

                # Clay specialist diff
                cs1 = _clay_specialist_score(prior, p1)
                cs2 = _clay_specialist_score(prior, p2)
                result.loc[idx, "tns_clay_specialist_diff"] = cs1 - cs2

                # Surface-specific H2H
                h2h_surface = prior_surface[
                    ((prior_surface["player1"] == p1) & (prior_surface["player2"] == p2))
                    | ((prior_surface["player1"] == p2) & (prior_surface["player2"] == p1))
                ]
                if len(h2h_surface) > 0:
                    p1_wins = (h2h_surface["winner"] == p1).sum()
                    result.loc[idx, "tns_surface_h2h"] = p1_wins / len(h2h_surface)

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
