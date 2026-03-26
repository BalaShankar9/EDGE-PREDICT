"""Ice hockey feature assembly pipeline.

Orchestrates all 5 hockey FeatureGroups to produce the full ~32-feature matrix.
"""

import logging

import numpy as np
import pandas as pd

from sharpedge.sports.ice_hockey.features.corsi import CorsiFeatures
from sharpedge.sports.ice_hockey.features.goaltending import GoaltendingFeatures
from sharpedge.sports.ice_hockey.features.special_teams import SpecialTeamsFeatures
from sharpedge.sports.ice_hockey.features.rest import HockeyRestFeatures
from sharpedge.sports.ice_hockey.features.market import HockeyMarketFeatures

logger = logging.getLogger(__name__)

ALL_GROUPS = [
    CorsiFeatures,           # 6 features
    GoaltendingFeatures,     # 8 features
    SpecialTeamsFeatures,    # 6 features
    HockeyRestFeatures,      # 6 features
    HockeyMarketFeatures,    # 6 features
]


class HockeyFeaturePipeline:
    """Assembles all hockey feature groups into a single feature matrix."""

    def __init__(self, groups: list | None = None):
        self.groups = [cls() for cls in (groups or ALL_GROUPS)]

    def build(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        """Build complete hockey feature matrix.

        Parameters
        ----------
        matches : DataFrame with hockey game results + odds columns.
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
                logger.info("  %s: %d features computed", group.name, features.shape[1])
            except Exception as e:
                logger.warning("  %s: FAILED - %s. Filling with NaN.", group.name, e)
                nan_df = pd.DataFrame(
                    np.nan,
                    index=matches.index,
                    columns=group.get_feature_names(),
                )
                feature_frames.append(nan_df)

        result = pd.concat(feature_frames, axis=1)
        logger.info(
            "Hockey feature matrix: %d matches x %d features",
            result.shape[0],
            result.shape[1],
        )
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
