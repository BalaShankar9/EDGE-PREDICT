"""Basketball efficiency features (10 total).

Features derived from offensive / defensive ratings and shooting efficiency.

Features:
  bke_off_rating_home         - points per 100 possessions (offense, home)
  bke_off_rating_away         - points per 100 possessions (offense, away)
  bke_def_rating_home         - points allowed per 100 possessions (home)
  bke_def_rating_away         - points allowed per 100 possessions (away)
  bke_net_rating_diff         - net rating difference (home - away)
  bke_efg_pct_home            - effective FG% (home)
  bke_efg_pct_away            - effective FG% (away)
  bke_ts_pct_diff             - true shooting % difference
  bke_tov_pct_diff            - turnover % difference
  bke_four_factors_composite  - weighted composite of 4 factors
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "bke_off_rating_home",
    "bke_off_rating_away",
    "bke_def_rating_home",
    "bke_def_rating_away",
    "bke_net_rating_diff",
    "bke_efg_pct_home",
    "bke_efg_pct_away",
    "bke_ts_pct_diff",
    "bke_tov_pct_diff",
    "bke_four_factors_composite",
]

_ROLLING_N = 10


def _team_off_rating(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Offensive rating proxy: points scored per game (rolling)."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    pts = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            pts.append(g.get("home_score", np.nan))
        else:
            pts.append(g.get("away_score", np.nan))
    valid = [p for p in pts if pd.notna(p)]
    return np.mean(valid) if valid else np.nan


def _team_def_rating(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Defensive rating proxy: points allowed per game (rolling)."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    pts = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            pts.append(g.get("away_score", np.nan))
        else:
            pts.append(g.get("home_score", np.nan))
    valid = [p for p in pts if pd.notna(p)]
    return np.mean(valid) if valid else np.nan


def _team_efg(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Effective FG% from efg_pct_home/away columns."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    vals = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            v = g.get("efg_pct_home", np.nan)
        else:
            v = g.get("efg_pct_away", np.nan)
        if pd.notna(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan


def _team_ts(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """True shooting % from ts_pct_home/away columns."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    vals = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            v = g.get("ts_pct_home", np.nan)
        else:
            v = g.get("ts_pct_away", np.nan)
        if pd.notna(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan


def _team_tov(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Turnover % from tov_pct_home/away columns."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    vals = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            v = g.get("tov_pct_home", np.nan)
        else:
            v = g.get("tov_pct_away", np.nan)
        if pd.notna(v):
            vals.append(v)
    return np.mean(vals) if vals else np.nan


class EfficiencyFeatures(FeatureGroup):
    name = "basketball_efficiency"
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

            off_h = _team_off_rating(prior, home)
            off_a = _team_off_rating(prior, away)
            def_h = _team_def_rating(prior, home)
            def_a = _team_def_rating(prior, away)

            result.loc[idx, "bke_off_rating_home"] = off_h
            result.loc[idx, "bke_off_rating_away"] = off_a
            result.loc[idx, "bke_def_rating_home"] = def_h
            result.loc[idx, "bke_def_rating_away"] = def_a

            # Net rating diff
            if all(pd.notna(v) for v in [off_h, def_h, off_a, def_a]):
                net_h = off_h - def_h
                net_a = off_a - def_a
                result.loc[idx, "bke_net_rating_diff"] = net_h - net_a
            else:
                result.loc[idx, "bke_net_rating_diff"] = np.nan

            efg_h = _team_efg(prior, home)
            efg_a = _team_efg(prior, away)
            result.loc[idx, "bke_efg_pct_home"] = efg_h
            result.loc[idx, "bke_efg_pct_away"] = efg_a

            ts_h = _team_ts(prior, home)
            ts_a = _team_ts(prior, away)
            if pd.notna(ts_h) and pd.notna(ts_a):
                result.loc[idx, "bke_ts_pct_diff"] = ts_h - ts_a
            else:
                result.loc[idx, "bke_ts_pct_diff"] = np.nan

            tov_h = _team_tov(prior, home)
            tov_a = _team_tov(prior, away)
            if pd.notna(tov_h) and pd.notna(tov_a):
                result.loc[idx, "bke_tov_pct_diff"] = tov_h - tov_a
            else:
                result.loc[idx, "bke_tov_pct_diff"] = np.nan

            # Four factors composite: eFG% (40%), TOV% (25%), OREB% (20%), FT rate (15%)
            # We approximate with available data
            if pd.notna(efg_h) and pd.notna(efg_a):
                efg_adv = efg_h - efg_a
            else:
                efg_adv = 0.0
            if pd.notna(tov_h) and pd.notna(tov_a):
                tov_adv = tov_a - tov_h  # lower TOV is better
            else:
                tov_adv = 0.0
            if all(pd.notna(v) for v in [off_h, def_h, off_a, def_a]):
                off_adv = (off_h - def_h) - (off_a - def_a)
            else:
                off_adv = 0.0
            result.loc[idx, "bke_four_factors_composite"] = (
                0.4 * efg_adv + 0.25 * tov_adv + 0.35 * off_adv
            )

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
