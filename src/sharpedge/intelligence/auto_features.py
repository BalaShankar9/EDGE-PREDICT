"""Auto-Feature Generator using tsfresh.

Generates hundreds of time-series features from team match history:
entropy, autocorrelation, wavelet coefficients, trend statistics, etc.
These are features that human engineers would never think to create.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AutoFeatureGenerator:
    """Generates time-series features using tsfresh."""

    def __init__(self, window: int = 10, n_jobs: int = 1):
        """
        Parameters
        ----------
        window : number of recent matches per team to use
        n_jobs : parallel workers for tsfresh
        """
        self.window = window
        self.n_jobs = n_jobs

    def generate(
        self,
        matches_df: pd.DataFrame,
        team_col: str = "home_team_id",
        value_col: str = "FTHG",
    ) -> pd.DataFrame:
        """Generate auto-features for each team from match history.

        Parameters
        ----------
        matches_df : historical matches with date sorting
        team_col : column identifying the team
        value_col : column to extract features from (goals, shots, etc.)

        Returns
        -------
        DataFrame with tsfresh features per team
        """
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import MinimalFCParameters
            from tsfresh.utilities.dataframe_functions import impute

            # Build time-series per team (last N matches)
            teams = matches_df[team_col].unique()
            ts_data = []

            for team in teams:
                team_matches = matches_df[matches_df[team_col] == team].tail(
                    self.window
                )
                for i, (_, row) in enumerate(team_matches.iterrows()):
                    val = row.get(value_col, 0)
                    if pd.isna(val):
                        val = 0
                    ts_data.append(
                        {
                            "id": str(team),
                            "time": i,
                            "value": float(val),
                        }
                    )

            if not ts_data:
                return pd.DataFrame()

            ts_df = pd.DataFrame(ts_data)

            # Extract minimal features (fast — ~30 features per team instead of 700+)
            # Use MinimalFCParameters for speed in backtest; switch to full for production
            features = extract_features(
                ts_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=MinimalFCParameters(),
                n_jobs=self.n_jobs,
                disable_progressbar=True,
            )

            features = impute(features)

            # Prefix column names
            features.columns = [f"tsf_{value_col}_{c}" for c in features.columns]

            logger.info(
                f"tsfresh generated {features.shape[1]} features for {len(teams)} teams"
            )
            return features

        except ImportError:
            logger.warning("tsfresh not available")
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"tsfresh feature generation failed: {e}")
            return pd.DataFrame()

    def generate_multi(
        self,
        matches_df: pd.DataFrame,
        value_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate features for multiple value columns and merge.

        Default columns: FTHG (goals for), FTAG (goals against),
        HS (shots), HST (shots on target)
        """
        if value_cols is None:
            value_cols = ["FTHG", "FTAG"]
            # Only include columns that exist
            value_cols = [c for c in value_cols if c in matches_df.columns]

        all_features = []
        for col in value_cols:
            feat = self.generate(matches_df, value_col=col)
            if not feat.empty:
                all_features.append(feat)

        if not all_features:
            return pd.DataFrame()

        # Merge all feature sets
        result = all_features[0]
        for feat in all_features[1:]:
            result = result.join(feat, how="outer")

        return result.fillna(0)
