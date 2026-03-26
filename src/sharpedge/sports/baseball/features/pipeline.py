"""Baseball feature assembly pipeline.

Orchestrates all 5 baseball FeatureGroups to produce the full ~38-feature matrix.
"""

import logging

import numpy as np
import pandas as pd

from sharpedge.sports.baseball.features.pitching import PitchingFeatures
from sharpedge.sports.baseball.features.batting import BattingFeatures
from sharpedge.sports.baseball.features.park_factor import ParkFactorFeatures
from sharpedge.sports.baseball.features.bullpen import BullpenFeatures
from sharpedge.sports.baseball.features.market import BaseballMarketFeatures

logger = logging.getLogger(__name__)

ALL_GROUPS = [
    PitchingFeatures,        # 12 features
    BattingFeatures,         # 8 features
    ParkFactorFeatures,      # 4 features
    BullpenFeatures,         # 6 features
    BaseballMarketFeatures,  # 8 features
]


class BaseballFeaturePipeline:
    """Assembles all baseball feature groups into a single feature matrix."""

    def __init__(self, groups: list | None = None):
        self.groups = [cls() for cls in (groups or ALL_GROUPS)]

    def build(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        """Build complete baseball feature matrix."""
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
            "Baseball feature matrix: %d matches x %d features",
            result.shape[0],
            result.shape[1],
        )
        return result

    def get_all_feature_names(self) -> list[str]:
        names = []
        for g in self.groups:
            names.extend(g.get_feature_names())
        return names

    @property
    def total_features(self) -> int:
        return sum(g.feature_count for g in self.groups)
