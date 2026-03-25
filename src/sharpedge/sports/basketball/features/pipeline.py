"""Basketball feature assembly pipeline.

Orchestrates all 6 basketball FeatureGroups to produce the full ~44-feature matrix.
"""

import logging

import numpy as np
import pandas as pd

from sharpedge.sports.basketball.features.pace import PaceFeatures
from sharpedge.sports.basketball.features.efficiency import EfficiencyFeatures
from sharpedge.sports.basketball.features.rest import RestFeatures
from sharpedge.sports.basketball.features.roster import RosterFeatures
from sharpedge.sports.basketball.features.matchup import MatchupFeatures
from sharpedge.sports.basketball.features.market import BasketballMarketFeatures

logger = logging.getLogger(__name__)

ALL_GROUPS = [
    PaceFeatures,              # 6 features
    EfficiencyFeatures,        # 10 features
    RestFeatures,              # 6 features
    RosterFeatures,            # 8 features
    MatchupFeatures,           # 6 features
    BasketballMarketFeatures,  # 8 features
]


class BasketballFeaturePipeline:
    """Assembles all basketball feature groups into a single feature matrix."""

    def __init__(self, groups: list | None = None):
        self.groups = [cls() for cls in (groups or ALL_GROUPS)]

    def build(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        """Build complete basketball feature matrix.

        Parameters
        ----------
        matches : DataFrame with basketball game results + odds columns.
        context : dict
            Additional DataFrames if needed (roster, injuries, etc.).

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
            "Basketball feature matrix: %d matches x %d features",
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
