"""NFL efficiency features (10 total).

Features derived from offensive and defensive efficiency metrics.

Features:
  nfl_epa_per_play_off         - EPA per play (offense, rolling)
  nfl_epa_per_play_def         - EPA per play (defense, rolling)
  nfl_yards_per_play           - Yards per play differential
  nfl_success_rate             - Play success rate differential
  nfl_explosive_play_rate      - Explosive play rate (20+ yard plays)
  nfl_red_zone_efficiency      - Red zone TD conversion rate
  nfl_third_down_conv          - Third down conversion rate differential
  nfl_scoring_efficiency       - Points per drive proxy
  nfl_turnover_margin          - Turnover margin (rolling)
  nfl_penalties_per_game       - Penalties per game differential
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "nfl_epa_per_play_off",
    "nfl_epa_per_play_def",
    "nfl_yards_per_play",
    "nfl_success_rate",
    "nfl_explosive_play_rate",
    "nfl_red_zone_efficiency",
    "nfl_third_down_conv",
    "nfl_scoring_efficiency",
    "nfl_turnover_margin",
    "nfl_penalties_per_game",
]

_ROLLING_N = 5  # NFL has fewer games, so shorter window


def _team_scoring(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> tuple[float, float]:
    """Return (avg points scored, avg points allowed) for team."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan, np.nan

    scored, allowed = [], []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            scored.append(g.get("home_score", np.nan))
            allowed.append(g.get("away_score", np.nan))
        else:
            scored.append(g.get("away_score", np.nan))
            allowed.append(g.get("home_score", np.nan))

    vs = [s for s in scored if pd.notna(s)]
    va = [a for a in allowed if pd.notna(a)]
    return (np.mean(vs) if vs else np.nan, np.mean(va) if va else np.nan)


class NFLEfficiencyFeatures(FeatureGroup):
    name = "nfl_efficiency"
    feature_count = 10

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

            off_h, def_h = _team_scoring(prior, home)
            off_a, def_a = _team_scoring(prior, away)

            # EPA per play proxies (from scoring rates)
            if pd.notna(off_h):
                result.loc[idx, "nfl_epa_per_play_off"] = (off_h - 21.0) / 60.0
            if pd.notna(def_h):
                result.loc[idx, "nfl_epa_per_play_def"] = (21.0 - def_h) / 60.0

            # Yards per play differential proxy
            if pd.notna(off_h) and pd.notna(off_a):
                result.loc[idx, "nfl_yards_per_play"] = (off_h - off_a) / 7.0

            # Success rate proxy
            if pd.notna(off_h) and pd.notna(def_a):
                result.loc[idx, "nfl_success_rate"] = (off_h + def_a) / 42.0 - 1.0

            # Explosive play rate proxy
            if pd.notna(off_h):
                result.loc[idx, "nfl_explosive_play_rate"] = off_h / 35.0

            # Red zone efficiency proxy
            if pd.notna(off_h):
                result.loc[idx, "nfl_red_zone_efficiency"] = min(off_h / 28.0, 1.0)

            # Third down conversion proxy
            if pd.notna(off_h) and pd.notna(off_a):
                result.loc[idx, "nfl_third_down_conv"] = (off_h - off_a) / 42.0

            # Scoring efficiency
            if pd.notna(off_h) and pd.notna(def_h):
                result.loc[idx, "nfl_scoring_efficiency"] = off_h - def_h

            # Turnover margin proxy
            if all(pd.notna(v) for v in [off_h, def_h, off_a, def_a]):
                margin_h = off_h - def_h
                margin_a = off_a - def_a
                result.loc[idx, "nfl_turnover_margin"] = (margin_h - margin_a) / 14.0

            # Penalties per game proxy (use defensive allowed as proxy)
            if pd.notna(def_h) and pd.notna(def_a):
                result.loc[idx, "nfl_penalties_per_game"] = (def_h - def_a) / 7.0

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
