"""Tennis feature assembly pipeline.

Orchestrates all 6 tennis FeatureGroups to produce the full ~40-feature matrix.
"""

import logging
import pandas as pd
import numpy as np
from sharpedge.sports.tennis.features.ranking import RankingFeatures
from sharpedge.sports.tennis.features.surface import SurfaceFeatures
from sharpedge.sports.tennis.features.fatigue import FatigueFeatures
from sharpedge.sports.tennis.features.h2h import H2HFeatures
from sharpedge.sports.tennis.features.serve import ServeFeatures
from sharpedge.sports.tennis.features.market import TennisMarketFeatures

logger = logging.getLogger(__name__)

ALL_GROUPS = [
    RankingFeatures,       # 8 features
    SurfaceFeatures,       # 6 features
    FatigueFeatures,       # 6 features
    H2HFeatures,           # 6 features
    ServeFeatures,         # 8 features
    TennisMarketFeatures,  # 6 features
]


class TennisFeaturePipeline:
    """Assembles all tennis feature groups into a single feature matrix."""

    def __init__(self, groups: list | None = None):
        self.groups = [cls() for cls in (groups or ALL_GROUPS)]

    def build(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        """Build complete tennis feature matrix.

        Parameters
        ----------
        matches : DataFrame with tennis match results + odds columns.
            Expected columns: date, player1, player2, winner, loser, surface,
            winner_rank, loser_rank, winner_points, loser_points, score,
            b365_winner, b365_loser, ps_winner, ps_loser.
        context : dict
            Additional DataFrames if needed.

        Returns
        -------
        DataFrame with all feature columns, same index as matches.
        """
        feature_frames = []
        for group in self.groups:
            try:
                features = group.compute(matches, **context)
                feature_frames.append(features)
                logger.info(f"  {group.name}: {features.shape[1]} features computed")
            except Exception as e:
                logger.warning(f"  {group.name}: FAILED - {e}. Filling with NaN.")
                nan_df = pd.DataFrame(
                    np.nan,
                    index=matches.index,
                    columns=group.get_feature_names(),
                )
                feature_frames.append(nan_df)

        result = pd.concat(feature_frames, axis=1)
        logger.info(f"Tennis feature matrix: {result.shape[0]} matches x {result.shape[1]} features")
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
