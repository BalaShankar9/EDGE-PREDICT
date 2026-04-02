# src/sharpedge/ml/features/elite.py
"""Elite feature groups using enriched data sources (32 total features).

Six new feature groups that transform data from the specialist collectors
into XGBoost-ready numeric features.

Feature groups and counts:
  RefereeFeatures       — 6 features
  ManagerFeatures       — 5 features
  WageFeatures          — 4 features
  FatigueFeatures       — 5 features
  LineupFeatures        — 4 features
  AdvancedStatsFeatures — 8 features
"""
import logging

import numpy as np
import pandas as pd

from sharpedge.ml.features.base import FeatureGroup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_lookup(source_df: pd.DataFrame, key_col: str, key_val, value_col: str):
    """Return value_col for the first row where key_col == key_val, or NaN."""
    if source_df is None or source_df.empty:
        return np.nan
    mask = source_df[key_col] == key_val
    if not mask.any():
        return np.nan
    return source_df.loc[mask.idxmax(), value_col]


def _lookup_team(df: pd.DataFrame, team_col: str, team_val, value_col: str):
    """Wrapper around _safe_lookup for team-keyed DataFrames."""
    return _safe_lookup(df, team_col, team_val, value_col)


def _nan_frame(index, feature_names: list[str]) -> pd.DataFrame:
    """Return an all-NaN DataFrame with the given index and columns."""
    return pd.DataFrame(np.nan, index=index, columns=feature_names)


# ---------------------------------------------------------------------------
# 1. RefereeFeatures
# ---------------------------------------------------------------------------

REFEREE_FEATURE_NAMES = [
    "referee_cards_per_game",
    "referee_penalties_per_game",
    "referee_home_bias",
    "referee_goals_per_game",
    "referee_strictness",
    "referee_known",
]

_MIN_REF_GAMES = 5  # minimum games before a referee is considered "known"


class RefereeFeatures(FeatureGroup):
    """Six features describing the assigned referee's historical tendencies.

    Context key: ``referee_df``
    Required columns: referee_name, yellow_cards_per_game, red_cards_per_game,
                      penalties_per_game, home_win_pct, avg_goals_per_game,
                      games (optional — used for referee_known threshold).

    Match referee from ``df["Referee"]`` column.
    """

    name = "referee"
    feature_count = 6

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        referee_df: pd.DataFrame | None = context.get("referee_df")

        result = _nan_frame(matches.index, REFEREE_FEATURE_NAMES)

        if referee_df is None or referee_df.empty:
            return result

        ref_df = referee_df.copy()

        # Pre-compute league-average home win pct from the referee data itself
        league_home_win_pct = (
            ref_df["home_win_pct"].mean()
            if "home_win_pct" in ref_df.columns
            else np.nan
        )

        # Compute strictness min/max for 0-1 normalisation (cards + pens per game)
        if {"yellow_cards_per_game", "red_cards_per_game", "penalties_per_game"}.issubset(ref_df.columns):
            ref_df["_raw_strictness"] = (
                ref_df["yellow_cards_per_game"]
                + ref_df["red_cards_per_game"]
                + ref_df["penalties_per_game"]
            )
            s_min = ref_df["_raw_strictness"].min()
            s_max = ref_df["_raw_strictness"].max()
            s_range = s_max - s_min if s_max != s_min else 1.0
            ref_df["_strictness_norm"] = (ref_df["_raw_strictness"] - s_min) / s_range
        else:
            ref_df["_strictness_norm"] = np.nan

        # Determine if "Referee" column exists
        if "Referee" not in matches.columns:
            return result

        for idx, row in matches.iterrows():
            ref_name = row.get("Referee", None)
            if pd.isna(ref_name) or ref_name == "":
                continue

            ref_row = ref_df[ref_df["referee_name"] == ref_name]
            if ref_row.empty:
                continue

            r = ref_row.iloc[0]

            # referee_cards_per_game = yellow + red
            if "yellow_cards_per_game" in ref_df.columns and "red_cards_per_game" in ref_df.columns:
                result.loc[idx, "referee_cards_per_game"] = (
                    r["yellow_cards_per_game"] + r["red_cards_per_game"]
                )

            if "penalties_per_game" in ref_df.columns:
                result.loc[idx, "referee_penalties_per_game"] = r["penalties_per_game"]

            if "home_win_pct" in ref_df.columns and pd.notna(league_home_win_pct):
                result.loc[idx, "referee_home_bias"] = r["home_win_pct"] - league_home_win_pct

            if "avg_goals_per_game" in ref_df.columns:
                result.loc[idx, "referee_goals_per_game"] = r["avg_goals_per_game"]

            result.loc[idx, "referee_strictness"] = r.get("_strictness_norm", np.nan)

            # referee_known: 1 if enough games recorded
            if "games" in ref_df.columns:
                result.loc[idx, "referee_known"] = int(r["games"] >= _MIN_REF_GAMES)
            else:
                # Assume known if the referee appears in the data at all
                result.loc[idx, "referee_known"] = 1.0

        return result

    def get_feature_names(self) -> list[str]:
        return REFEREE_FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# 2. ManagerFeatures
# ---------------------------------------------------------------------------

MANAGER_FEATURE_NAMES = [
    "home_manager_win_pct",
    "away_manager_win_pct",
    "home_is_new_manager",
    "away_is_new_manager",
    "new_manager_bounce",
]


class ManagerFeatures(FeatureGroup):
    """Five features capturing managerial tenure and new-manager effects.

    Context key: ``manager_df``
    Required columns: team, career_win_pct, is_new_manager,
                      new_manager_bounce_expected.
    """

    name = "manager"
    feature_count = 5

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        manager_df: pd.DataFrame | None = context.get("manager_df")

        result = _nan_frame(matches.index, MANAGER_FEATURE_NAMES)

        if manager_df is None or manager_df.empty:
            return result

        for idx, row in matches.iterrows():
            home_id = row.get("home_team_id")
            away_id = row.get("away_team_id")

            home_win = _lookup_team(manager_df, "team", home_id, "career_win_pct")
            away_win = _lookup_team(manager_df, "team", away_id, "career_win_pct")
            home_new = _lookup_team(manager_df, "team", home_id, "is_new_manager")
            away_new = _lookup_team(manager_df, "team", away_id, "is_new_manager")
            home_bounce = _lookup_team(manager_df, "team", home_id, "new_manager_bounce_expected")
            away_bounce = _lookup_team(manager_df, "team", away_id, "new_manager_bounce_expected")

            result.loc[idx, "home_manager_win_pct"] = home_win
            result.loc[idx, "away_manager_win_pct"] = away_win
            result.loc[idx, "home_is_new_manager"] = float(home_new) if pd.notna(home_new) else np.nan
            result.loc[idx, "away_is_new_manager"] = float(away_new) if pd.notna(away_new) else np.nan

            # new_manager_bounce: 1 if either team is in the bounce window
            h_bounce = float(home_bounce) if pd.notna(home_bounce) else 0.0
            a_bounce = float(away_bounce) if pd.notna(away_bounce) else 0.0
            if pd.notna(home_bounce) or pd.notna(away_bounce):
                result.loc[idx, "new_manager_bounce"] = float(bool(h_bounce or a_bounce))

        return result

    def get_feature_names(self) -> list[str]:
        return MANAGER_FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# 3. WageFeatures
# ---------------------------------------------------------------------------

WAGE_FEATURE_NAMES = [
    "home_wage_rank",
    "away_wage_rank",
    "wage_ratio",
    "wage_rank_diff",
]


class WageFeatures(FeatureGroup):
    """Four features quantifying the wage disparity between the two teams.

    Context key: ``wage_df``
    Required columns: team, total_wage_bill_weekly, wage_bill_rank.
    """

    name = "wage"
    feature_count = 4

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        wage_df: pd.DataFrame | None = context.get("wage_df")

        result = _nan_frame(matches.index, WAGE_FEATURE_NAMES)

        if wage_df is None or wage_df.empty:
            return result

        for idx, row in matches.iterrows():
            home_id = row.get("home_team_id")
            away_id = row.get("away_team_id")

            home_rank = _lookup_team(wage_df, "team", home_id, "wage_bill_rank")
            away_rank = _lookup_team(wage_df, "team", away_id, "wage_bill_rank")
            home_wage = _lookup_team(wage_df, "team", home_id, "total_wage_bill_weekly")
            away_wage = _lookup_team(wage_df, "team", away_id, "total_wage_bill_weekly")

            result.loc[idx, "home_wage_rank"] = home_rank
            result.loc[idx, "away_wage_rank"] = away_rank

            if pd.notna(home_wage) and pd.notna(away_wage) and away_wage != 0:
                result.loc[idx, "wage_ratio"] = home_wage / away_wage

            if pd.notna(home_rank) and pd.notna(away_rank):
                # positive = home team has lower (better) rank number => spends more
                result.loc[idx, "wage_rank_diff"] = away_rank - home_rank

        return result

    def get_feature_names(self) -> list[str]:
        return WAGE_FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# 4. FatigueFeatures
# ---------------------------------------------------------------------------

FATIGUE_FEATURE_NAMES = [
    "home_european_fatigue",
    "away_european_fatigue",
    "home_days_rest",
    "away_days_rest",
    "rest_advantage",
]


class FatigueFeatures(FeatureGroup):
    """Five features capturing match congestion and European fixture fatigue.

    Context key: ``fatigue_df``
    Required columns: team, fatigue_score, days_between_matches.
    """

    name = "fatigue"
    feature_count = 5

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        fatigue_df: pd.DataFrame | None = context.get("fatigue_df")

        result = _nan_frame(matches.index, FATIGUE_FEATURE_NAMES)

        if fatigue_df is None or fatigue_df.empty:
            return result

        for idx, row in matches.iterrows():
            home_id = row.get("home_team_id")
            away_id = row.get("away_team_id")

            home_fatigue = _lookup_team(fatigue_df, "team", home_id, "fatigue_score")
            away_fatigue = _lookup_team(fatigue_df, "team", away_id, "fatigue_score")
            home_rest = _lookup_team(fatigue_df, "team", home_id, "days_between_matches")
            away_rest = _lookup_team(fatigue_df, "team", away_id, "days_between_matches")

            result.loc[idx, "home_european_fatigue"] = home_fatigue
            result.loc[idx, "away_european_fatigue"] = away_fatigue
            result.loc[idx, "home_days_rest"] = home_rest
            result.loc[idx, "away_days_rest"] = away_rest

            if pd.notna(home_rest) and pd.notna(away_rest):
                result.loc[idx, "rest_advantage"] = home_rest - away_rest

        return result

    def get_feature_names(self) -> list[str]:
        return FATIGUE_FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# 5. LineupFeatures
# ---------------------------------------------------------------------------

LINEUP_FEATURE_NAMES = [
    "home_key_absences",
    "away_key_absences",
    "lineup_confirmed",
    "absence_advantage",
]


class LineupFeatures(FeatureGroup):
    """Four features derived from confirmed or predicted lineups.

    Context key: ``lineup_df``
    Required columns: home_team, away_team, key_absences_home,
                      key_absences_away, lineup_status.

    Matching is done against ``home_team_id`` / ``away_team_id`` in matches.
    lineup_status == "confirmed" -> lineup_confirmed = 1, else 0.
    """

    name = "lineup"
    feature_count = 4

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        lineup_df: pd.DataFrame | None = context.get("lineup_df")

        result = _nan_frame(matches.index, LINEUP_FEATURE_NAMES)

        if lineup_df is None or lineup_df.empty:
            return result

        for idx, row in matches.iterrows():
            home_id = row.get("home_team_id")
            away_id = row.get("away_team_id")

            # Match on both home_team AND away_team columns simultaneously
            mask = (lineup_df["home_team"] == home_id) & (lineup_df["away_team"] == away_id)
            if not mask.any():
                continue

            lr = lineup_df.loc[mask].iloc[0]

            home_abs = lr.get("key_absences_home", np.nan)
            away_abs = lr.get("key_absences_away", np.nan)
            status = lr.get("lineup_status", None)

            result.loc[idx, "home_key_absences"] = home_abs
            result.loc[idx, "away_key_absences"] = away_abs
            result.loc[idx, "lineup_confirmed"] = (
                1.0 if str(status).lower() == "confirmed" else 0.0
            )

            if pd.notna(home_abs) and pd.notna(away_abs):
                # positive = home has fewer absences (advantage)
                result.loc[idx, "absence_advantage"] = away_abs - home_abs

        return result

    def get_feature_names(self) -> list[str]:
        return LINEUP_FEATURE_NAMES.copy()


# ---------------------------------------------------------------------------
# 6. AdvancedStatsFeatures
# ---------------------------------------------------------------------------

ADVANCED_FEATURE_NAMES = [
    "home_xg_per90",
    "away_xg_per90",
    "home_pressing_intensity",
    "away_pressing_intensity",
    "home_progressive_passes",
    "away_progressive_passes",
    "home_pass_completion",
    "away_pass_completion",
]


class AdvancedStatsFeatures(FeatureGroup):
    """Eight features from advanced statistical data (xG, pressing, passing).

    Context key: ``advanced_df``
    Required columns: team, xg_90, pressing_intensity,
                      progressive_passes_90, pass_completion_pct.
    """

    name = "advanced_stats"
    feature_count = 8

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        advanced_df: pd.DataFrame | None = context.get("advanced_df")

        result = _nan_frame(matches.index, ADVANCED_FEATURE_NAMES)

        if advanced_df is None or advanced_df.empty:
            return result

        for idx, row in matches.iterrows():
            home_id = row.get("home_team_id")
            away_id = row.get("away_team_id")

            result.loc[idx, "home_xg_per90"] = _lookup_team(advanced_df, "team", home_id, "xg_90")
            result.loc[idx, "away_xg_per90"] = _lookup_team(advanced_df, "team", away_id, "xg_90")
            result.loc[idx, "home_pressing_intensity"] = _lookup_team(
                advanced_df, "team", home_id, "pressing_intensity"
            )
            result.loc[idx, "away_pressing_intensity"] = _lookup_team(
                advanced_df, "team", away_id, "pressing_intensity"
            )
            result.loc[idx, "home_progressive_passes"] = _lookup_team(
                advanced_df, "team", home_id, "progressive_passes_90"
            )
            result.loc[idx, "away_progressive_passes"] = _lookup_team(
                advanced_df, "team", away_id, "progressive_passes_90"
            )
            result.loc[idx, "home_pass_completion"] = _lookup_team(
                advanced_df, "team", home_id, "pass_completion_pct"
            )
            result.loc[idx, "away_pass_completion"] = _lookup_team(
                advanced_df, "team", away_id, "pass_completion_pct"
            )

        return result

    def get_feature_names(self) -> list[str]:
        return ADVANCED_FEATURE_NAMES.copy()
