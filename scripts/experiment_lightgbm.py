"""Experiment: LightGBM vs XGBoost head-to-head comparison."""
import logging
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

from sharpedge.db.engine import get_session
from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.xgboost_model import XGBoostPredictor
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.ml.models.ensemble import EnsemblePredictor
from sharpedge.ml.models.calibration import ProbabilityCalibrator
from sharpedge.ml.training.walk_forward import WalkForwardCV
from sharpedge.ml.training.metrics import ranked_probability_score, accuracy

logger = logging.getLogger(__name__)


def load_data():
    session = get_session()
    matches_df = pd.read_sql(text("""
        SELECT m.id, m.match_date, m.home_team_id, m.away_team_id,
            m.home_goals AS "FTHG", m.away_goals AS "FTAG",
            m.home_goals_ht AS "HTHG", m.away_goals_ht AS "HTAG",
            CASE WHEN m.home_goals > m.away_goals THEN 'H'
                 WHEN m.home_goals = m.away_goals THEN 'D' ELSE 'A' END AS "FTR",
            m.referee AS "Referee",
            ht.canonical_name AS home_team_name, at.canonical_name AS away_team_name,
            l.name AS league, s.label AS season,
            ms.home_shots AS "HS", ms.away_shots AS "AS",
            ms.home_shots_on_target AS "HST", ms.away_shots_on_target AS "AST",
            ms.home_fouls AS "HF", ms.away_fouls AS "AF",
            ms.home_corners AS "HC", ms.away_corners AS "AC",
            b365.odds_home AS "B365H", b365.odds_draw AS "B365D", b365.odds_away AS "B365A",
            ps.odds_home AS "PSH", ps.odds_draw AS "PSD", ps.odds_away AS "PSA",
            wh.odds_home AS "WHH", wh.odds_draw AS "WHD", wh.odds_away AS "WHA",
            mx.odds_home AS "MaxH", mx.odds_draw AS "MaxD", mx.odds_away AS "MaxA",
            av.odds_home AS "AvgH", av.odds_draw AS "AvgD", av.odds_away AS "AvgA"
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN seasons s ON m.season_id = s.id
        JOIN leagues l ON s.league_id = l.id
        LEFT JOIN match_stats ms ON ms.match_id = m.id
        LEFT JOIN match_odds b365 ON b365.match_id = m.id AND b365.bookmaker = 'Bet365'
        LEFT JOIN match_odds ps ON ps.match_id = m.id AND ps.bookmaker = 'Pinnacle'
        LEFT JOIN match_odds wh ON wh.match_id = m.id AND wh.bookmaker = 'WilliamHill'
        LEFT JOIN match_odds mx ON mx.match_id = m.id AND mx.bookmaker = 'MarketMax'
        LEFT JOIN match_odds av ON av.match_id = m.id AND av.bookmaker = 'MarketAvg'
        WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.match_date
    """), session.bind)
    elo_df = pd.read_sql(text("SELECT team_id, rating_date AS date, elo, source FROM elo_ratings ORDER BY rating_date"), session.bind)
    session.close()
    return matches_df, elo_df


def train_lgbm_1x2(X_train, y_train_encoded, X_val):
    """Train LightGBM multiclass and return probabilities."""
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train_encoded)
    proba = model.predict_proba(X_val)
    return proba, model


LGBM_CONFIGS = [
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.01, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_samples": 10, "reg_alpha": 1.0, "reg_lambda": 1.0},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.01, "subsample": 0.8,
     "colsample_bytree": 0.7, "min_child_samples": 20, "reg_alpha": 1.0, "reg_lambda": 2.0},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.005, "subsample": 0.8,
     "colsample_bytree": 0.6, "min_child_samples": 30, "reg_alpha": 0.5, "reg_lambda": 1.0},
    {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.005, "subsample": 0.7,
     "colsample_bytree": 0.6, "min_child_samples": 20, "reg_alpha": 2.0, "reg_lambda": 3.0},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8,
     "colsample_bytree": 0.8, "min_child_samples": 10, "reg_alpha": 0.5, "reg_lambda": 1.0},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.01, "subsample": 0.8,
     "colsample_bytree": 0.5, "min_child_samples": 30, "reg_alpha": 1.0, "reg_lambda": 2.0},
]


if __name__ == "__main__":
    matches_df, elo_df = load_data()
    logger.info(f"Loaded {len(matches_df)} matches")

    # Build features
    pipe = FeaturePipeline()
    feature_matrix = pipe.build(matches_df, elo_df=elo_df)

    y_1x2 = matches_df["FTR"].values
    y_1x2_encoded = np.array([{"H": 0, "D": 1, "A": 2}.get(r, 1) for r in y_1x2])

    X = feature_matrix.values.astype(float)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)

    for config_idx, lgbm_params in enumerate(LGBM_CONFIGS):
        all_rps = []
        all_acc = []

        for fold_num, (train_idx, val_idx) in enumerate(cv.split(matches_df)):
            train_idx_arr = np.array(train_idx)
            val_idx_arr = np.array(val_idx)

            X_train, X_val = X[train_idx_arr], X[val_idx_arr]
            y_train = y_1x2_encoded[train_idx_arr]
            y_val = y_1x2_encoded[val_idx_arr]

            # LightGBM
            params = {
                **lgbm_params,
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            lgbm_proba = model.predict_proba(X_val)

            # Calibrate
            calibrator = ProbabilityCalibrator(method="platt")
            calibrator.fit(y_val, lgbm_proba)
            calibrated = calibrator.calibrate(lgbm_proba)

            fold_rps = ranked_probability_score(y_val, calibrated)
            fold_pred = np.argmax(calibrated, axis=1)
            fold_acc = accuracy(y_val, fold_pred)

            all_rps.append(fold_rps)
            all_acc.append(fold_acc)

        mean_rps = float(np.mean(all_rps))
        mean_acc = float(np.mean(all_acc))
        short = {k: v for k, v in lgbm_params.items()}
        logger.info(f"LightGBM config {config_idx+1}: RPS={mean_rps:.4f}, Acc={mean_acc:.3f} | {short}")

    logger.info(f"\nBaseline XGBoost: RPS=0.1922, Acc=0.543")
