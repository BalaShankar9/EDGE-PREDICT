"""Ice hockey special teams features (6 total).

Features capturing power play and penalty kill efficiency.

Features:
  nhl_pp_pct_home              - Power play % (home, rolling)
  nhl_pp_pct_away              - Power play % (away, rolling)
  nhl_pk_pct_home              - Penalty kill % (home, rolling)
  nhl_pk_pct_away              - Penalty kill % (away, rolling)
  nhl_pp_opportunities_diff    - PP opportunities differential
  nhl_shorthanded_goals_diff   - Shorthanded goals differential
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nhl_pp_pct_home",
    "nhl_pp_pct_away",
    "nhl_pk_pct_home",
    "nhl_pk_pct_away",
    "nhl_pp_opportunities_diff",
    "nhl_shorthanded_goals_diff",
]

_ROLLING_N = 10


def _team_special_teams_proxy(prior: pd.DataFrame, team: str,
                               n: int = _ROLLING_N) -> tuple[float, float]:
    """Compute PP% and PK% proxies from score data.

    Uses goals scored/allowed ratio as a proxy when detailed
    special teams data is not available.
    """
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan, np.nan

    gf, ga = [], []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            gf.append(g.get("home_score", 0))
            ga.append(g.get("away_score", 0))
        else:
            gf.append(g.get("away_score", 0))
            ga.append(g.get("home_score", 0))

    avg_gf = np.mean(gf) if gf else 0
    avg_ga = np.mean(ga) if ga else 0

    # PP% proxy: ~20% league avg, scaled by offensive output
    pp_pct = min(0.20 * (avg_gf / 3.0), 0.40) if avg_gf > 0 else 0.20

    # PK% proxy: ~80% league avg, scaled inversely by goals against
    pk_pct = min(0.80 * (3.0 / max(avg_ga, 1.0)), 0.95) if avg_ga > 0 else 0.80

    return pp_pct, pk_pct


class SpecialTeamsFeatures(FeatureGroup):
    name = "hockey_special_teams"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        df = matches.copy()
        df["game_date"] = pd.to_datetime(df.get("game_date", df.get("date")))
        result = pd.DataFrame(index=df.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            match_date = row["game_date"]
            prior = df[df["game_date"] < match_date]

            if len(prior) == 0:
                continue

            pp_h, pk_h = _team_special_teams_proxy(prior, home)
            pp_a, pk_a = _team_special_teams_proxy(prior, away)

            result.loc[idx, "nhl_pp_pct_home"] = pp_h
            result.loc[idx, "nhl_pp_pct_away"] = pp_a
            result.loc[idx, "nhl_pk_pct_home"] = pk_h
            result.loc[idx, "nhl_pk_pct_away"] = pk_a

            # PP opportunities differential proxy
            if pd.notna(pp_h) and pd.notna(pp_a):
                result.loc[idx, "nhl_pp_opportunities_diff"] = pp_h - pp_a

            # Shorthanded goals differential proxy
            if pd.notna(pk_h) and pd.notna(pk_a):
                result.loc[idx, "nhl_shorthanded_goals_diff"] = pk_h - pk_a

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
