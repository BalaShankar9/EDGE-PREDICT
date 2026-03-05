"""Training orchestrator — full pipeline from data to trained model.

Coordinates feature building, model training, ensemble weighting,
calibration, and evaluation using walk-forward validation.
"""
import logging
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.xgboost_model import XGBoostPredictor
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.ml.models.ensemble import EnsemblePredictor
from sharpedge.ml.models.calibration import ProbabilityCalibrator
from sharpedge.ml.training.walk_forward import WalkForwardCV
from sharpedge.ml.training.metrics import (
    ranked_probability_score,
    accuracy,
    log_loss_1x2,
    roi,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Holds training results and metrics."""

    fold_metrics: list[dict] = field(default_factory=list)
    aggregate_metrics: dict = field(default_factory=dict)
    ensemble_weights: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)


class ModelTrainer:
    """Orchestrates the full training pipeline."""

    def __init__(
        self,
        xgb_params: dict | None = None,
        n_splits: int = 2,
        min_train_seasons: int = 3,
        calibration_method: str = "platt",
    ):
        self.xgb_params = xgb_params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.01,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "random_state": 42,
        }
        self.n_splits = n_splits
        self.min_train_seasons = min_train_seasons
        self.calibration_method = calibration_method

        # Trained components (after training)
        self.feature_pipeline: FeaturePipeline | None = None
        self.xgb_model: XGBoostPredictor | None = None
        self.poisson_model: PoissonPredictor | None = None
        self.ensemble: EnsemblePredictor | None = None
        self.calibrator: ProbabilityCalibrator | None = None

    def train(
        self,
        matches_df: pd.DataFrame,
        elo_df: pd.DataFrame | None = None,
        xg_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
    ) -> TrainingResult:
        """Full training pipeline.

        1. Build feature matrix via FeaturePipeline
        2. Create target variables (y_1x2, y_ou, y_btts)
        3. Walk-forward split
        4. For each fold: train both models, ensemble, calibrate, evaluate
        5. Retrain final models on all data
        6. Return results
        """
        result = TrainingResult()

        # Build features
        self.feature_pipeline = FeaturePipeline()
        feature_matrix = self.feature_pipeline.build(
            matches_df, elo_df=elo_df, xg_df=xg_df, predictions_df=predictions_df
        )
        result.feature_names = self.feature_pipeline.get_all_feature_names()

        # Create target variables
        y_1x2 = matches_df["FTR"].values  # "H", "D", "A" strings
        y_1x2_encoded = np.array(
            [{"H": 0, "D": 1, "A": 2}.get(r, 1) for r in y_1x2]
        )

        total_goals = matches_df["FTHG"].values + matches_df["FTAG"].values
        y_ou = (total_goals > 2.5).astype(int)
        y_btts = (
            (matches_df["FTHG"].values > 0) & (matches_df["FTAG"].values > 0)
        ).astype(int)

        # Fill NaN features with column median
        X = feature_matrix.values.astype(float)
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

        # Walk-forward validation
        cv = WalkForwardCV(
            n_splits=self.n_splits, min_train_seasons=self.min_train_seasons
        )

        all_rps = []
        all_acc = []

        for fold_num, (train_idx, val_idx) in enumerate(cv.split(matches_df)):
            logger.info(
                f"Fold {fold_num + 1}: train={len(train_idx)}, val={len(val_idx)}"
            )

            # Convert pd.Index to numpy arrays for reliable indexing
            train_idx_arr = np.array(train_idx)
            val_idx_arr = np.array(val_idx)

            X_train, X_val = X[train_idx_arr], X[val_idx_arr]
            y_train_1x2 = y_1x2[train_idx_arr]
            y_val_1x2 = y_1x2_encoded[val_idx_arr]
            y_train_ou = y_ou[train_idx_arr]
            y_train_btts = y_btts[train_idx_arr]

            # Train XGBoost (expects string labels for 1x2)
            xgb = XGBoostPredictor(params=self.xgb_params)
            xgb.fit(X_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
            xgb_proba = xgb.predict_proba_1x2(X_val)

            # Train Poisson
            poisson_model = PoissonPredictor()
            train_matches = [
                {
                    "home_team_id": matches_df.iloc[i]["home_team_id"],
                    "away_team_id": matches_df.iloc[i]["away_team_id"],
                    "home_goals": int(matches_df.iloc[i]["FTHG"]),
                    "away_goals": int(matches_df.iloc[i]["FTAG"]),
                }
                for i in train_idx_arr
            ]
            poisson_model.fit(train_matches)

            # Get Poisson predictions for validation set
            poisson_proba = np.zeros((len(val_idx_arr), 3))
            for k, vi in enumerate(val_idx_arr):
                home = matches_df.iloc[vi]["home_team_id"]
                away = matches_df.iloc[vi]["away_team_id"]
                poisson_proba[k] = poisson_model.predict_proba_1x2(home, away)

            # Ensemble
            ensemble = EnsemblePredictor()
            model_preds = {"xgboost": xgb_proba, "poisson": poisson_proba}
            ensemble.fit_weights(model_preds, y_val_1x2)
            combined = ensemble.predict(model_preds)

            # Calibrate
            calibrator = ProbabilityCalibrator(method=self.calibration_method)
            calibrator.fit(y_val_1x2, combined)
            calibrated = calibrator.calibrate(combined)

            # Evaluate
            fold_rps = ranked_probability_score(y_val_1x2, calibrated)
            fold_pred = np.argmax(calibrated, axis=1)
            fold_acc = accuracy(y_val_1x2, fold_pred)

            fold_metrics = {
                "fold": fold_num + 1,
                "train_size": len(train_idx_arr),
                "val_size": len(val_idx_arr),
                "rps": float(fold_rps),
                "accuracy": float(fold_acc),
                "ensemble_weights": ensemble.weights,
            }
            result.fold_metrics.append(fold_metrics)
            all_rps.append(fold_rps)
            all_acc.append(fold_acc)

            logger.info(
                f"  Fold {fold_num + 1}: RPS={fold_rps:.4f}, Acc={fold_acc:.3f}"
            )

        # Aggregate metrics
        result.aggregate_metrics = {
            "mean_rps": float(np.mean(all_rps)),
            "mean_accuracy": float(np.mean(all_acc)),
        }
        result.ensemble_weights = (
            result.fold_metrics[-1]["ensemble_weights"]
            if result.fold_metrics
            else {}
        )

        # Final retrain on all data
        logger.info("Retraining final models on all data...")
        self.xgb_model = XGBoostPredictor(params=self.xgb_params)
        self.xgb_model.fit(X, y_1x2, y_ou=y_ou, y_btts=y_btts)

        self.poisson_model = PoissonPredictor()
        all_matches = [
            {
                "home_team_id": matches_df.iloc[i]["home_team_id"],
                "away_team_id": matches_df.iloc[i]["away_team_id"],
                "home_goals": int(matches_df.iloc[i]["FTHG"]),
                "away_goals": int(matches_df.iloc[i]["FTAG"]),
            }
            for i in range(len(matches_df))
        ]
        self.poisson_model.fit(all_matches)

        self.ensemble = EnsemblePredictor()
        self.ensemble.weights = result.ensemble_weights

        # Fit calibrator on full-data predictions
        xgb_proba_all = self.xgb_model.predict_proba_1x2(X)
        poisson_proba_all = np.zeros((len(matches_df), 3))
        for k in range(len(matches_df)):
            home = matches_df.iloc[k]["home_team_id"]
            away = matches_df.iloc[k]["away_team_id"]
            poisson_proba_all[k] = self.poisson_model.predict_proba_1x2(home, away)
        combined_all = self.ensemble.predict(
            {"xgboost": xgb_proba_all, "poisson": poisson_proba_all}
        )
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)
        self.calibrator.fit(y_1x2_encoded, combined_all)

        logger.info(
            f"Training complete. Mean RPS: {result.aggregate_metrics['mean_rps']:.4f}"
        )
        return result

    def save(self, path: str | Path) -> None:
        """Save trained model pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "xgb_model": self.xgb_model,
                    "poisson_model": self.poisson_model,
                    "ensemble": self.ensemble,
                    "calibrator": self.calibrator,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "ModelTrainer":
        """Load trained model pipeline from disk."""
        path = Path(path)
        trainer = cls()
        with open(path / "model.pkl", "rb") as f:
            data = pickle.load(f)
        trainer.xgb_model = data["xgb_model"]
        trainer.poisson_model = data["poisson_model"]
        trainer.ensemble = data["ensemble"]
        trainer.calibrator = data["calibrator"]
        return trainer
