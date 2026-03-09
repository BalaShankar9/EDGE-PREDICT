"""Training orchestrator — full pipeline from data to trained model.

Coordinates feature building, model training, ensemble weighting,
calibration, and evaluation using walk-forward validation.

Architecture:
  Level-0: XGBoost, Dixon-Coles, Bivariate Poisson, CatBoost, LightGBM generate OOF predictions
  Level-1: Stacking meta-learner (LogisticRegression) combines them
  Calibration: Isotonic regression on OOF predictions (leak-free)
  OvR specialists: LightGBM binary classifiers for high-confidence picks
"""
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.xgboost_model import XGBoostPredictor
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.ml.models.bivariate_poisson import BivariatePoissonPredictor
from sharpedge.ml.models.catboost_model import CatBoostPredictor
from sharpedge.ml.models.lightgbm_model import LightGBMPredictor
from sharpedge.ml.models.ensemble import EnsemblePredictor
from sharpedge.ml.models.stacking import StackingMetaLearner
from sharpedge.ml.models.calibration import ProbabilityCalibrator
from sharpedge.ml.models.ovr_model import OvRPredictor
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
        n_splits: int = 3,
        min_train_seasons: int = 3,
        calibration_method: str = "isotonic",
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
        self.bvp_model: BivariatePoissonPredictor | None = None
        self.ensemble: EnsemblePredictor | None = None
        self.stacking: StackingMetaLearner | None = None
        self.calibrator: ProbabilityCalibrator | None = None
        self.catboost_model: CatBoostPredictor | None = None
        self.lgbm_model: LightGBMPredictor | None = None
        self.ovr_model: OvRPredictor | None = None
        self.ou_calibrator: IsotonicRegression | None = None
        self.btts_calibrator: IsotonicRegression | None = None

        # Draw probability floor — prevents model from ignoring draws
        # Draws are ~26% of Big 5 outcomes but model predicts <1%
        self.draw_floor: float = 0.18

    def _build_match_dicts(
        self, matches_df: pd.DataFrame, indices: np.ndarray
    ) -> list[dict]:
        """Extract match dicts for Poisson model training."""
        return [
            {
                "home_team_id": matches_df.iloc[i]["home_team_id"],
                "away_team_id": matches_df.iloc[i]["away_team_id"],
                "home_goals": int(matches_df.iloc[i]["FTHG"]),
                "away_goals": int(matches_df.iloc[i]["FTAG"]),
            }
            for i in indices
        ]

    def _get_poisson_proba(
        self,
        model: PoissonPredictor,
        matches_df: pd.DataFrame,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Get Dixon-Coles 1x2 probabilities for given indices."""
        proba = np.zeros((len(indices), 3))
        for k, vi in enumerate(indices):
            home = matches_df.iloc[vi]["home_team_id"]
            away = matches_df.iloc[vi]["away_team_id"]
            proba[k] = model.predict_proba_1x2(home, away)
        return proba

    def _get_bvp_proba(
        self,
        model: BivariatePoissonPredictor,
        matches_df: pd.DataFrame,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Get Bivariate Poisson 1x2 probabilities for given indices."""
        proba = np.zeros((len(indices), 3))
        for k, vi in enumerate(indices):
            home = matches_df.iloc[vi]["home_team_id"]
            away = matches_df.iloc[vi]["away_team_id"]
            proba[k] = model.predict_proba_1x2(home, away)
        return proba

    def train(
        self,
        matches_df: pd.DataFrame,
        elo_df: pd.DataFrame | None = None,
        xg_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
    ) -> TrainingResult:
        """Full training pipeline with stacking ensemble.

        1. Build feature matrix via FeaturePipeline
        2. Create target variables (y_1x2, y_ou, y_btts)
        3. Walk-forward split
        4. For each fold: train Level-0 models, collect OOF predictions
        5. Fit Level-1 stacking meta-learner on OOF predictions
        6. Retrain final models on all data
        7. Return results
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

        # Keep NaN — XGBoost handles missing values natively
        X = feature_matrix.values.astype(np.float32)

        # Walk-forward validation
        cv = WalkForwardCV(
            n_splits=self.n_splits, min_train_seasons=self.min_train_seasons
        )

        all_rps = []
        all_acc = []
        all_fold_weights = []

        # Collect out-of-fold predictions for stacking + leak-free calibration
        n_samples = len(matches_df)
        oof_xgb_preds = np.full((n_samples, 3), np.nan)
        oof_dc_preds = np.full((n_samples, 3), np.nan)
        oof_bvp_preds = np.full((n_samples, 3), np.nan)
        oof_catboost_preds = np.full((n_samples, 3), np.nan)
        oof_lgbm_preds = np.full((n_samples, 3), np.nan)
        oof_ensemble_preds = np.full((n_samples, 3), np.nan)
        oof_ovr_preds = np.full((n_samples, 3), np.nan)
        oof_labels = np.full(n_samples, -1, dtype=int)

        # OOF O/U and BTTS predictions for calibration
        oof_ou_preds = np.full(n_samples, np.nan)
        oof_btts_preds = np.full(n_samples, np.nan)
        oof_ou_labels = y_ou.copy()
        oof_btts_labels = y_btts.copy()

        for fold_num, (train_idx, val_idx) in enumerate(cv.split(matches_df)):
            logger.info(
                f"Fold {fold_num + 1}: train={len(train_idx)}, val={len(val_idx)}"
            )

            train_idx_arr = np.array(train_idx)
            val_idx_arr = np.array(val_idx)

            # --- Level-0 Model 1: Dixon-Coles ---
            poisson_model = PoissonPredictor()
            train_matches = self._build_match_dicts(matches_df, train_idx_arr)
            poisson_model.fit(train_matches)

            # Augment feature matrix with Dixon-Coles team strengths
            X_aug_train = self._augment_with_dc(
                X[train_idx_arr], matches_df, train_idx_arr, poisson_model
            )
            X_aug_val = self._augment_with_dc(
                X[val_idx_arr], matches_df, val_idx_arr, poisson_model
            )

            y_train_1x2 = y_1x2[train_idx_arr]
            y_train_encoded = y_1x2_encoded[train_idx_arr]
            y_val_1x2 = y_1x2_encoded[val_idx_arr]
            y_train_ou = y_ou[train_idx_arr]
            y_train_btts = y_btts[train_idx_arr]

            # --- Level-0 Model 2: XGBoost ---
            xgb = XGBoostPredictor(params=self.xgb_params)
            xgb.fit(X_aug_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
            xgb_proba = xgb.predict_proba_1x2(X_aug_val)

            # Dixon-Coles validation predictions
            dc_proba = self._get_poisson_proba(poisson_model, matches_df, val_idx_arr)

            # --- Level-0 Model 3: Bivariate Poisson ---
            bvp_model = BivariatePoissonPredictor()
            bvp_model.fit(train_matches)
            bvp_proba = self._get_bvp_proba(bvp_model, matches_df, val_idx_arr)

            # --- Level-0 Model 4: CatBoost ---
            catboost = CatBoostPredictor()
            catboost.fit(X_aug_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
            catboost_proba = catboost.predict_proba_1x2(X_aug_val)
            oof_catboost_preds[val_idx_arr] = catboost_proba

            # --- Level-0 Model 5: LightGBM ---
            lgbm_model = LightGBMPredictor()
            lgbm_model.fit(X_aug_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
            lgbm_proba = lgbm_model.predict_proba_1x2(X_aug_val)
            oof_lgbm_preds[val_idx_arr] = lgbm_proba

            # Store OOF predictions for stacking
            oof_xgb_preds[val_idx_arr] = xgb_proba
            oof_dc_preds[val_idx_arr] = dc_proba
            oof_bvp_preds[val_idx_arr] = bvp_proba
            oof_labels[val_idx_arr] = y_val_1x2

            # Weighted ensemble (kept as fallback + for weight tracking)
            ensemble = EnsemblePredictor()
            model_preds = {
                "xgboost": xgb_proba,
                "poisson": dc_proba,
                "bivariate_poisson": bvp_proba,
                "catboost": catboost_proba,
                "lightgbm": lgbm_proba,
            }
            ensemble.fit_weights(model_preds, y_val_1x2)
            combined = ensemble.predict(model_preds)
            all_fold_weights.append(ensemble.weights)
            oof_ensemble_preds[val_idx_arr] = combined

            # OvR: train on training data, collect OOF predictions
            ovr = OvRPredictor()
            ovr.fit(X_aug_train, y_train_encoded)
            oof_ovr_preds[val_idx_arr] = ovr.predict_proba_raw(X_aug_val)

            # Collect OOF O/U and BTTS predictions (XGBoost + DC weighted)
            xgb_ou_val = xgb.predict_proba_ou(X_aug_val)
            xgb_btts_val = xgb.predict_proba_btts(X_aug_val)
            dc_ou_val = np.zeros(len(val_idx_arr))
            dc_btts_val = np.zeros(len(val_idx_arr))
            for k, vi in enumerate(val_idx_arr):
                home = matches_df.iloc[vi]["home_team_id"]
                away = matches_df.iloc[vi]["away_team_id"]
                dc_ou_val[k] = poisson_model.predict_proba_ou(home, away, 2.5)
                dc_btts_val[k] = poisson_model.predict_proba_btts(home, away)
            # Weight: XGBoost 60% + DC 40% (BVP excluded — 45.9% O/U accuracy)
            oof_ou_preds[val_idx_arr] = 0.6 * xgb_ou_val + 0.4 * dc_ou_val
            oof_btts_preds[val_idx_arr] = 0.6 * xgb_btts_val + 0.4 * dc_btts_val

            # Evaluate
            fold_rps = ranked_probability_score(y_val_1x2, combined)
            fold_pred = np.argmax(combined, axis=1)
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
                f"  Fold {fold_num + 1}: RPS={fold_rps:.4f}, Acc={fold_acc:.3f}, "
                f"weights={ensemble.weights}"
            )

        # --- Fit stacking meta-learner on OOF predictions ---
        oof_mask = oof_labels >= 0
        logger.info("Fitting stacking meta-learner on OOF predictions...")
        self.stacking = StackingMetaLearner()
        self.stacking.fit(
            {
                "xgboost": oof_xgb_preds[oof_mask],
                "poisson": oof_dc_preds[oof_mask],
                "bivariate_poisson": oof_bvp_preds[oof_mask],
                "catboost": oof_catboost_preds[oof_mask],
                "lightgbm": oof_lgbm_preds[oof_mask],
            },
            oof_labels[oof_mask],
        )
        # Evaluate stacking vs simple ensemble
        stacked_preds = self.stacking.predict(
            {
                "xgboost": oof_xgb_preds[oof_mask],
                "poisson": oof_dc_preds[oof_mask],
                "bivariate_poisson": oof_bvp_preds[oof_mask],
                "catboost": oof_catboost_preds[oof_mask],
                "lightgbm": oof_lgbm_preds[oof_mask],
            },
        )
        stacked_rps = ranked_probability_score(oof_labels[oof_mask], stacked_preds)
        ensemble_rps = ranked_probability_score(
            oof_labels[oof_mask], oof_ensemble_preds[oof_mask]
        )
        logger.info(
            f"Stacking RPS: {stacked_rps:.4f} vs Weighted Ensemble RPS: {ensemble_rps:.4f}"
        )

        # Average ensemble weights across all folds
        avg_weights: dict[str, float] = {}
        if all_fold_weights:
            all_keys = all_fold_weights[0].keys()
            for key in all_keys:
                avg_weights[key] = float(
                    np.mean([fw[key] for fw in all_fold_weights])
                )
            total = sum(avg_weights.values())
            if total > 0:
                avg_weights = {k: v / total for k, v in avg_weights.items()}

        result.aggregate_metrics = {
            "mean_rps": float(np.mean(all_rps)),
            "mean_accuracy": float(np.mean(all_acc)),
            "stacking_rps": float(stacked_rps),
        }
        result.ensemble_weights = avg_weights

        # --- Final retrain on all data ---
        logger.info("Retraining final models on all data...")

        # Dixon-Coles
        all_matches = self._build_match_dicts(
            matches_df, np.arange(len(matches_df))
        )
        self.poisson_model = PoissonPredictor()
        self.poisson_model.fit(all_matches)

        # Bivariate Poisson
        self.bvp_model = BivariatePoissonPredictor()
        self.bvp_model.fit(all_matches)

        # XGBoost (with DC feature augmentation)
        all_idx = np.arange(len(matches_df))
        X_aug = self._augment_with_dc(X, matches_df, all_idx, self.poisson_model)
        self.xgb_model = XGBoostPredictor(params=self.xgb_params)
        self.xgb_model.fit(X_aug, y_1x2, y_ou=y_ou, y_btts=y_btts)

        # CatBoost
        self.catboost_model = CatBoostPredictor()
        self.catboost_model.fit(X_aug, y_1x2, y_ou=y_ou, y_btts=y_btts)

        # LightGBM
        self.lgbm_model = LightGBMPredictor()
        self.lgbm_model.fit(X_aug, y_1x2, y_ou=y_ou, y_btts=y_btts)

        # Weighted ensemble (fallback)
        self.ensemble = EnsemblePredictor()
        self.ensemble.weights = avg_weights

        # Fit calibrator on OOF stacked predictions (leak-free)
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)
        self.calibrator.fit(oof_labels[oof_mask], stacked_preds)

        # Train OvR on all data, calibrate on OOF predictions (leak-free)
        logger.info("Training OvR specialist classifiers...")
        self.ovr_model = OvRPredictor()
        self.ovr_model.fit(X_aug, y_1x2_encoded)
        oof_ovr_valid = oof_ovr_preds[oof_mask]
        oof_labels_valid = oof_labels[oof_mask]
        self.ovr_model.calibrators = []
        for c in range(3):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(oof_ovr_valid[:, c], (oof_labels_valid == c).astype(float))
            self.ovr_model.calibrators.append(ir)

        # --- Fit O/U and BTTS calibrators on OOF predictions (leak-free) ---
        ou_mask = ~np.isnan(oof_ou_preds)
        if ou_mask.sum() > 50:
            logger.info("Fitting O/U isotonic calibrator on OOF predictions...")
            self.ou_calibrator = IsotonicRegression(out_of_bounds="clip")
            self.ou_calibrator.fit(oof_ou_preds[ou_mask], oof_ou_labels[ou_mask])
            # Evaluate calibration improvement
            raw_ou_acc = np.mean(
                (oof_ou_preds[ou_mask] > 0.5) == oof_ou_labels[ou_mask]
            )
            cal_preds = self.ou_calibrator.predict(oof_ou_preds[ou_mask])
            cal_ou_acc = np.mean((cal_preds > 0.5) == oof_ou_labels[ou_mask])
            logger.info(f"  O/U accuracy: raw={raw_ou_acc:.3f} → calibrated={cal_ou_acc:.3f}")

        btts_mask = ~np.isnan(oof_btts_preds)
        if btts_mask.sum() > 50:
            logger.info("Fitting BTTS isotonic calibrator on OOF predictions...")
            self.btts_calibrator = IsotonicRegression(out_of_bounds="clip")
            self.btts_calibrator.fit(oof_btts_preds[btts_mask], oof_btts_labels[btts_mask])
            raw_btts_acc = np.mean(
                (oof_btts_preds[btts_mask] > 0.5) == oof_btts_labels[btts_mask]
            )
            cal_btts_preds = self.btts_calibrator.predict(oof_btts_preds[btts_mask])
            cal_btts_acc = np.mean((cal_btts_preds > 0.5) == oof_btts_labels[btts_mask])
            logger.info(f"  BTTS accuracy: raw={raw_btts_acc:.3f} → calibrated={cal_btts_acc:.3f}")

        logger.info(
            f"Training complete. Mean RPS: {result.aggregate_metrics['mean_rps']:.4f}, "
            f"Stacking RPS: {stacked_rps:.4f}"
        )
        return result

    @staticmethod
    def _augment_with_dc(
        X: np.ndarray,
        matches_df: pd.DataFrame,
        idx: np.ndarray,
        poisson: PoissonPredictor,
    ) -> np.ndarray:
        """Add Dixon-Coles team strengths as extra XGBoost features.

        Adds 4 columns: home_attack, home_defence, away_attack, away_defence.
        """
        n = len(idx)
        dc_features = np.full((n, 4), np.nan, dtype=np.float32)
        for k, i in enumerate(idx):
            home = matches_df.iloc[i]["home_team_id"]
            away = matches_df.iloc[i]["away_team_id"]
            dc_features[k, 0] = poisson.attack_strength.get(home, np.nan)
            dc_features[k, 1] = poisson.defence_strength.get(home, np.nan)
            dc_features[k, 2] = poisson.attack_strength.get(away, np.nan)
            dc_features[k, 3] = poisson.defence_strength.get(away, np.nan)
        return np.hstack([X, dc_features])

    def save(self, path: str | Path) -> None:
        """Save trained model pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(
                {
                    "xgb_model": self.xgb_model,
                    "poisson_model": self.poisson_model,
                    "bvp_model": self.bvp_model,
                    "catboost_model": self.catboost_model,
                    "lgbm_model": self.lgbm_model,
                    "ensemble": self.ensemble,
                    "stacking": self.stacking,
                    "calibrator": self.calibrator,
                    "ovr_model": self.ovr_model,
                    "ou_calibrator": self.ou_calibrator,
                    "btts_calibrator": self.btts_calibrator,
                    "draw_floor": self.draw_floor,
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
        trainer.bvp_model = data.get("bvp_model")
        trainer.catboost_model = data.get("catboost_model")
        trainer.lgbm_model = data.get("lgbm_model")
        trainer.ensemble = data["ensemble"]
        trainer.stacking = data.get("stacking")
        trainer.calibrator = data["calibrator"]
        trainer.ovr_model = data.get("ovr_model")
        trainer.ou_calibrator = data.get("ou_calibrator")
        trainer.btts_calibrator = data.get("btts_calibrator")
        trainer.draw_floor = data.get("draw_floor", 0.18)
        return trainer
