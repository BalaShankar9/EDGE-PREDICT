"""Base class for feature groups.

Each FeatureGroup computes a set of related features from raw match data.
All feature groups follow the same interface: compute(matches_df) -> features_df.
"""
from abc import ABC, abstractmethod
import pandas as pd


class FeatureGroup(ABC):
    """Abstract base for feature computation groups."""

    name: str = ""  # e.g. "form", "elo", "xg_perf"
    feature_count: int = 0  # Expected number of features produced

    @abstractmethod
    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        """Compute features for each match.

        Parameters
        ----------
        matches : DataFrame
            Must contain at minimum: match_date, home_team_id, away_team_id,
            and any source-specific columns this group needs.
        context : dict
            Additional DataFrames needed (e.g., elo_df, xg_df, odds_df).

        Returns
        -------
        DataFrame with same index as matches, containing only the new feature columns.
        Column names should be prefixed with the group name (e.g. "form_home_xg_5").
        """
        ...

    def get_feature_names(self) -> list[str]:
        """Return list of feature column names this group produces."""
        ...
