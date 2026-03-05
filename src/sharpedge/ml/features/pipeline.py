"""Feature assembly pipeline.

Orchestrates all FeatureGroups to produce the full 46-feature matrix
from raw match data + supplementary DataFrames.

Note: XGPerformanceFeatures and MetaPredictionFeatures are excluded
because they produce 100% NaN (no xG or competitor prediction data available).
Re-enable them once Understat xG and competitor prediction data are ingested.
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

logger = logging.getLogger(__name__)

ALL_GROUPS = [
    FormFeatures,       # 12 features
    EloFeatures,        # 6 features
    H2HFeatures,        # 6 features
    MarketFeatures,     # 8 features
    ContextFeatures,    # 6 features
    ShotFeatures,       # 8 features
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
    ) -> pd.DataFrame:
        """Build complete feature matrix.

        Parameters
        ----------
        matches : DataFrame with match results + odds columns
        elo_df : ELO ratings from ClubELO
        xg_df : xG data from Understat
        predictions_df : Competitor predictions from Forebet etc.

        Returns
        -------
        DataFrame with all feature columns, same index as matches.
        """
        context = {
            "elo_df": elo_df,
            "xg_df": xg_df,
            "predictions_df": predictions_df,
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
