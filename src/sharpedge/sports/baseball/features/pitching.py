"""Baseball pitching features (12 total).

Features capturing starting pitcher and pitching staff quality.

Features:
  mlb_era_home                 - ERA (home starter, rolling)
  mlb_era_away                 - ERA (away starter, rolling)
  mlb_fip                      - Fielding-independent pitching proxy
  mlb_xfip                     - Expected FIP proxy
  mlb_whip                     - WHIP differential
  mlb_k_per_9                  - Strikeouts per 9 innings differential
  mlb_bb_per_9                 - Walks per 9 innings differential
  mlb_pitch_quality            - Overall pitch quality composite
  mlb_spin_rate_trend          - Spin rate trend proxy
  mlb_days_rest_pitcher        - Pitcher days rest
  mlb_pitch_count_last         - Recent pitch count burden
  mlb_platoon_advantage        - Platoon advantage flag
"""

import numpy as np
import pandas as pd

from sharpedge.core.base_features import FeatureGroup

FEATURE_NAMES = [
    "mlb_era_home",
    "mlb_era_away",
    "mlb_fip",
    "mlb_xfip",
    "mlb_whip",
    "mlb_k_per_9",
    "mlb_bb_per_9",
    "mlb_pitch_quality",
    "mlb_spin_rate_trend",
    "mlb_days_rest_pitcher",
    "mlb_pitch_count_last",
    "mlb_platoon_advantage",
]

_ROLLING_N = 10
_LEAGUE_AVG_RUNS = 4.5


def _team_runs_against(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Rolling runs allowed average for a team (ERA proxy)."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    ra = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            ra.append(g.get("away_score", np.nan))
        else:
            ra.append(g.get("home_score", np.nan))

    valid = [v for v in ra if pd.notna(v)]
    return np.mean(valid) if valid else np.nan


def _team_runs_for(prior: pd.DataFrame, team: str, n: int = _ROLLING_N) -> float:
    """Rolling runs scored average for a team."""
    games = prior[
        (prior["home_team"] == team) | (prior["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return np.nan

    rf = []
    for _, g in games.iterrows():
        if g["home_team"] == team:
            rf.append(g.get("home_score", np.nan))
        else:
            rf.append(g.get("away_score", np.nan))

    valid = [v for v in rf if pd.notna(v)]
    return np.mean(valid) if valid else np.nan


class PitchingFeatures(FeatureGroup):
    name = "baseball_pitching"
    feature_count = 12

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

            ra_h = _team_runs_against(prior, home)
            ra_a = _team_runs_against(prior, away)
            rf_h = _team_runs_for(prior, home)
            rf_a = _team_runs_for(prior, away)

            # ERA proxy (runs against per game)
            result.loc[idx, "mlb_era_home"] = ra_h
            result.loc[idx, "mlb_era_away"] = ra_a

            # FIP proxy: adjusted ERA (slightly regressed toward mean)
            if pd.notna(ra_h) and pd.notna(ra_a):
                fip_h = ra_h * 0.8 + _LEAGUE_AVG_RUNS * 0.2
                fip_a = ra_a * 0.8 + _LEAGUE_AVG_RUNS * 0.2
                result.loc[idx, "mlb_fip"] = fip_h - fip_a

            # xFIP proxy: further regressed
            if pd.notna(ra_h) and pd.notna(ra_a):
                xfip_h = ra_h * 0.6 + _LEAGUE_AVG_RUNS * 0.4
                xfip_a = ra_a * 0.6 + _LEAGUE_AVG_RUNS * 0.4
                result.loc[idx, "mlb_xfip"] = xfip_h - xfip_a

            # WHIP proxy: from runs allowed
            if pd.notna(ra_h) and pd.notna(ra_a):
                result.loc[idx, "mlb_whip"] = (ra_h - ra_a) / _LEAGUE_AVG_RUNS

            # K/9 proxy
            if pd.notna(ra_h) and pd.notna(ra_a):
                result.loc[idx, "mlb_k_per_9"] = (_LEAGUE_AVG_RUNS - ra_h) - (_LEAGUE_AVG_RUNS - ra_a)

            # BB/9 proxy
            if pd.notna(ra_h) and pd.notna(ra_a):
                result.loc[idx, "mlb_bb_per_9"] = (ra_h - ra_a) / 9.0

            # Pitch quality composite
            if pd.notna(ra_h):
                result.loc[idx, "mlb_pitch_quality"] = max(0, _LEAGUE_AVG_RUNS - ra_h)

            # Spin rate trend proxy (use recent vs overall difference)
            ra_h_short = _team_runs_against(prior, home, n=3)
            if pd.notna(ra_h_short) and pd.notna(ra_h):
                result.loc[idx, "mlb_spin_rate_trend"] = ra_h - ra_h_short

            # Pitcher days rest proxy
            home_games = prior[
                (prior["home_team"] == home) | (prior["away_team"] == home)
            ]
            if len(home_games) > 0:
                last_date = home_games["game_date"].max()
                result.loc[idx, "mlb_days_rest_pitcher"] = float(
                    (match_date - last_date).days
                )

            # Pitch count proxy (games in last 5 days)
            if len(prior) > 0:
                cutoff = match_date - pd.Timedelta(days=5)
                recent = prior[
                    (prior["game_date"] >= cutoff) &
                    ((prior["home_team"] == home) | (prior["away_team"] == home))
                ]
                result.loc[idx, "mlb_pitch_count_last"] = float(len(recent))

            # Platoon advantage (default 0 without handedness data)
            result.loc[idx, "mlb_platoon_advantage"] = 0.0

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
