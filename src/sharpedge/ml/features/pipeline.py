"""Feature assembly pipeline.

Orchestrates all FeatureGroups to produce the full 78-feature matrix
from raw match data + supplementary DataFrames.

Note: XGPerformanceFeatures, MetaPredictionFeatures, GoalPatternFeatures, and
OverallFormFeatures are excluded. The first two produce 100% NaN (no data available).
Goal/overall form features were tested but hurt accuracy due to noise.
"""
import logging
import pandas as pd
import numpy as np
from sharpedge.ml.features.form import FormFeatures
from sharpedge.ml.features.elo import EloFeatures
from sharpedge.ml.features.h2h import H2HFeatures
from sharpedge.ml.features.market import MarketFeatures
from sharpedge.ml.features.context import ContextFeatures
from sharpedge.ml.features.shots import ShotFeatures
from sharpedge.ml.features.elite import (
    RefereeFeatures,
    ManagerFeatures,
    WageFeatures,
    FatigueFeatures,
    LineupFeatures,
    AdvancedStatsFeatures,
)

logger = logging.getLogger(__name__)

ALL_GROUPS = [
    FormFeatures,           # 12 features
    EloFeatures,            # 6 features
    H2HFeatures,            # 6 features
    MarketFeatures,         # 8 features
    ContextFeatures,        # 6 features
    ShotFeatures,           # 8 features
    RefereeFeatures,        # 6 features
    ManagerFeatures,        # 5 features
    WageFeatures,           # 4 features
    FatigueFeatures,        # 5 features
    LineupFeatures,         # 4 features
    AdvancedStatsFeatures,  # 8 features
]


class FeaturePipeline:
    """Assembles all feature groups into a single feature matrix."""

    def __init__(self, groups: list | None = None):
        self.groups = [cls() for cls in (groups or ALL_GROUPS)]

    def build(
        self,
        matches: pd.DataFrame,
        elo_df: pd.DataFrame | None = None,
        xg_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
        referee_df: pd.DataFrame | None = None,
        manager_df: pd.DataFrame | None = None,
        wage_df: pd.DataFrame | None = None,
        fatigue_df: pd.DataFrame | None = None,
        lineup_df: pd.DataFrame | None = None,
        advanced_df: pd.DataFrame | None = None,
        injuries_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build complete feature matrix.

        Parameters
        ----------
        matches : DataFrame with match results + odds columns
        elo_df : ELO ratings from ClubELO
        xg_df : xG data from Understat
        predictions_df : Competitor predictions from Forebet etc.
        referee_df : Referee historical stats (referee_name, yellow_cards_per_game, …)
        manager_df : Manager tenure / win percentage data (team, career_win_pct, …)
        wage_df : Wage bill data (team, total_wage_bill_weekly, wage_bill_rank)
        fatigue_df : Fatigue / fixture congestion data (team, fatigue_score, …)
        lineup_df : Lineup confirmation data (home_team, away_team, key_absences_*, …)
        advanced_df : Advanced stats (team, xg_90, pressing_intensity, …)
        injuries_df : Injury data — reserved for future use

        Returns
        -------
        DataFrame with all feature columns, same index as matches.
        """
        context = {
            "elo_df": elo_df,
            "xg_df": xg_df,
            "predictions_df": predictions_df,
            "referee_df": referee_df,
            "manager_df": manager_df,
            "wage_df": wage_df,
            "fatigue_df": fatigue_df,
            "lineup_df": lineup_df,
            "advanced_df": advanced_df,
            "injuries_df": injuries_df,
        }

        feature_frames = []
        for group in self.groups:
            try:
                features = group.compute(matches, **context)
                feature_frames.append(features)
                logger.info(f"  {group.name}: {features.shape[1]} features computed")
            except Exception as e:
                logger.warning(f"  {group.name}: FAILED — {e}. Filling with NaN.")
                nan_df = pd.DataFrame(
                    np.nan,
                    index=matches.index,
                    columns=group.get_feature_names(),
                )
                feature_frames.append(nan_df)

        result = pd.concat(feature_frames, axis=1)
        logger.info(f"Feature matrix: {result.shape[0]} matches x {result.shape[1]} features")
        return result

    def get_all_feature_names(self) -> list[str]:
        """Return ordered list of all feature names across all groups."""
        names = []
        for g in self.groups:
            names.extend(g.get_feature_names())
        return names

    @property
    def total_features(self) -> int:
        """Total number of features across all groups."""
        return sum(g.feature_count for g in self.groups)
