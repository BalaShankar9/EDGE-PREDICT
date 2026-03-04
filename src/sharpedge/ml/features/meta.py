"""Meta-prediction features (4 total).

Features derived from competitor prediction sites (Forebet, PredictZ, etc.).

Features:
  meta_forebet_prob_home  - Forebet predicted probability for home win
  meta_consensus_result   - Mode prediction across sites (0=H, 1=D, 2=A)
  meta_prediction_agree   - Agreement rate: fraction of sites agreeing
  meta_avg_pred_total     - Average predicted total goals
"""
import pandas as pd
import numpy as np
from sharpedge.ml.features.base import FeatureGroup

FEATURE_NAMES = [
    "meta_forebet_prob_home",
    "meta_consensus_result",
    "meta_prediction_agree",
    "meta_avg_pred_total",
]


class MetaPredictionFeatures(FeatureGroup):
    name = "meta"
    feature_count = 4

    def compute(self, matches: pd.DataFrame, **context) -> pd.DataFrame:
        predictions_df = context.get("predictions_df")

        result = pd.DataFrame(index=matches.index)
        for col in FEATURE_NAMES:
            result[col] = np.nan

        if predictions_df is None or predictions_df.empty:
            return result

        pred = predictions_df.copy()
        # Expected columns: home_team_id, away_team_id, match_date, source,
        # prob_home, prob_draw, prob_away, predicted_result, predicted_total

        if "match_date" in pred.columns:
            pred["match_date"] = pd.to_datetime(pred["match_date"])

        for idx, row in matches.iterrows():
            home_id = row["home_team_id"]
            away_id = row["away_team_id"]

            # Match predictions for this fixture
            match_preds = pred[
                (pred["home_team_id"] == home_id) & (pred["away_team_id"] == away_id)
            ]

            if match_preds.empty:
                continue

            # Forebet probability
            forebet = match_preds[match_preds["source"] == "forebet"]
            if not forebet.empty and "prob_home" in forebet.columns:
                result.loc[idx, "meta_forebet_prob_home"] = forebet.iloc[0]["prob_home"]

            # Consensus
            if "predicted_result" in match_preds.columns:
                results = match_preds["predicted_result"].dropna()
                if not results.empty:
                    mode_result = results.mode()
                    if not mode_result.empty:
                        mapping = {"H": 0, "D": 1, "A": 2}
                        result.loc[idx, "meta_consensus_result"] = mapping.get(mode_result.iloc[0], np.nan)
                        agreement = (results == mode_result.iloc[0]).mean()
                        result.loc[idx, "meta_prediction_agree"] = agreement

            # Average predicted total goals
            if "predicted_total" in match_preds.columns:
                totals = match_preds["predicted_total"].dropna()
                if not totals.empty:
                    result.loc[idx, "meta_avg_pred_total"] = totals.mean()

        return result

    def get_feature_names(self) -> list[str]:
        return FEATURE_NAMES.copy()
