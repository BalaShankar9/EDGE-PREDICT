"""Walk-forward (expanding window) cross-validation for time-series.

NEVER use random cross-validation for football predictions — it leaks
future information. Walk-forward respects temporal ordering.

Splits:
  Fold 1: Train [S1, S2, S3] → Validate [S4]
  Fold 2: Train [S1, S2, S3, S4] → Validate [S5]
"""
import pandas as pd
import numpy as np
from typing import Iterator


class WalkForwardCV:
    """Walk-forward cross-validation with expanding training window."""

    def __init__(self, n_splits: int = 2, min_train_seasons: int = 3):
        """
        Parameters
        ----------
        n_splits : max number of validation folds
        min_train_seasons : minimum number of seasons in training set
        """
        self.n_splits = n_splits
        self.min_train_seasons = min_train_seasons

    def split(
        self,
        matches_df: pd.DataFrame,
        season_col: str = "season",
    ) -> Iterator[tuple[pd.Index, pd.Index]]:
        """Yield (train_indices, val_indices) respecting temporal order.

        Parameters
        ----------
        matches_df : DataFrame with a season column
        season_col : column name containing season labels

        Yields
        ------
        (train_indices, val_indices) tuples
        """
        seasons = sorted(matches_df[season_col].unique())

        if len(seasons) <= self.min_train_seasons:
            raise ValueError(
                f"Need >{self.min_train_seasons} seasons, got {len(seasons)}"
            )

        folds_produced = 0
        for i in range(self.min_train_seasons, len(seasons)):
            if folds_produced >= self.n_splits:
                break

            train_seasons = seasons[:i]
            val_season = seasons[i]

            train_mask = matches_df[season_col].isin(train_seasons)
            val_mask = matches_df[season_col] == val_season

            train_idx = matches_df[train_mask].index
            val_idx = matches_df[val_mask].index

            if len(train_idx) > 0 and len(val_idx) > 0:
                folds_produced += 1
                yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits
