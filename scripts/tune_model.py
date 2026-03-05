"""Hyperparameter tuning — computes features ONCE, then grid-searches XGBoost params.

Now includes Poisson stacking features (5 additional features per match).
"""
import logging
import sys
import itertools

import numpy as np
import pandas as pd
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tuning.log"),
    ],
)

from sharpedge.db.engine import get_session
from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.xgboost_model import XGBoostPredictor
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.ml.models.ensemble import EnsemblePredictor
from sharpedge.ml.models.calibration import ProbabilityCalibrator
from sharpedge.ml.training.walk_forward import WalkForwardCV
from sharpedge.ml.training.metrics import ranked_probability_score, accuracy
from sharpedge.ml.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def load_training_data():
    session = get_session()
    matches_query = text("""
        SELECT
            m.id, m.match_date, m.home_team_id, m.away_team_id,
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
    """)
    matches_df = pd.read_sql(matches_query, session.bind)
    elo_df = pd.read_sql(text("SELECT team_id, rating_date AS date, elo, source FROM elo_ratings ORDER BY rating_date"), session.bind)
    xg_df = pd.read_sql(text("SELECT x.match_id, x.source, x.home_xg, x.away_xg, m.match_date, m.home_team_id, m.away_team_id FROM match_xg x JOIN matches m ON x.match_id = m.id"), session.bind)
    predictions_df = pd.read_sql(text("SELECT match_id, source, predicted_result, prob_home, prob_draw, prob_away FROM competitor_predictions"), session.bind)
    session.close()
    return matches_df, elo_df, xg_df, predictions_df


PARAM_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.005, 0.01, 0.03],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "min_child_weight": [10, 20, 30],
    "reg_alpha": [0.5, 1.0, 2.0],
    "reg_lambda": [1.0, 2.0, 3.0],
}


def generate_configs(grid: dict, max_configs: int = 30) -> list[dict]:
    keys = list(grid.keys())
    all_combos = list(itertools.product(*grid.values()))
    np.random.seed(42)
    if len(all_combos) > max_configs:
        indices = np.random.choice(len(all_combos), max_configs, replace=False)
        selected = [all_combos[i] for i in indices]
    else:
        selected = all_combos
    return [{**dict(zip(keys, combo)), "random_state": 42} for combo in selected]


def compute_poisson_features(poisson_model, matches_df, indices):
    """Compute 5 Poisson stacking features: P(H), P(D), P(A), xG_home, xG_away."""
    features = np.zeros((len(indices), 5))
    for k, idx in enumerate(indices):
        home = matches_df.iloc[idx]["home_team_id"]
        away = matches_df.iloc[idx]["away_team_id"]
        proba = poisson_model.predict_proba_1x2(home, away)
        xg_home, xg_away = poisson_model.predict_goals(home, away)
        features[k] = [proba[0], proba[1], proba[2], xg_home, xg_away]
    return features


def evaluate_config(config, X_base, y_1x2, y_1x2_encoded, y_ou, y_btts,
                    matches_df, fold_data):
    """Evaluate a single XGBoost config using pre-computed features and Poisson stacking."""
    all_rps = []
    all_acc = []

    for fold_num, (train_idx_arr, val_idx_arr, poisson_train, poisson_val) in enumerate(fold_data):
        # Stack Poisson features onto base features
        X_train = np.hstack([X_base[train_idx_arr], poisson_train])
        X_val = np.hstack([X_base[val_idx_arr], poisson_val])

        y_train_1x2 = y_1x2[train_idx_arr]
        y_val_1x2 = y_1x2_encoded[val_idx_arr]
        y_train_ou = y_ou[train_idx_arr]
        y_train_btts = y_btts[train_idx_arr]

        # Train XGBoost with stacked features
        xgb = XGBoostPredictor(params=config)
        xgb.fit(X_train, y_train_1x2, y_ou=y_train_ou, y_btts=y_train_btts)
        xgb_proba = xgb.predict_proba_1x2(X_val)

        # Poisson 1x2 for ensemble
        poisson_proba = poisson_val[:, :3]

        # Ensemble
        ensemble = EnsemblePredictor()
        model_preds = {"xgboost": xgb_proba, "poisson": poisson_proba}
        ensemble.fit_weights(model_preds, y_val_1x2)
        combined = ensemble.predict(model_preds)

        # Calibrate
        calibrator = ProbabilityCalibrator(method="platt")
        calibrator.fit(y_val_1x2, combined)
        calibrated = calibrator.calibrate(combined)

        # Evaluate
        fold_rps = ranked_probability_score(y_val_1x2, calibrated)
        fold_pred = np.argmax(calibrated, axis=1)
        fold_acc = accuracy(y_val_1x2, fold_pred)

        all_rps.append(fold_rps)
        all_acc.append(fold_acc)

    return float(np.mean(all_rps)), float(np.mean(all_acc))


if __name__ == "__main__":
    logger.info("Loading training data...")
    matches_df, elo_df, xg_df, predictions_df = load_training_data()
    logger.info(f"Loaded {len(matches_df)} matches")

    # ---- STEP 1: Compute base features ONCE ----
    logger.info("Computing features (one-time)...")
    pipe = FeaturePipeline()
    feature_matrix = pipe.build(
        matches_df,
        elo_df=elo_df if len(elo_df) > 0 else None,
        xg_df=xg_df if len(xg_df) > 0 else None,
        predictions_df=predictions_df if len(predictions_df) > 0 else None,
    )

    # Prepare arrays
    y_1x2 = matches_df["FTR"].values
    y_1x2_encoded = np.array([{"H": 0, "D": 1, "A": 2}.get(r, 1) for r in y_1x2])
    total_goals = matches_df["FTHG"].values + matches_df["FTAG"].values
    y_ou = (total_goals > 2.5).astype(int)
    y_btts = ((matches_df["FTHG"].values > 0) & (matches_df["FTAG"].values > 0)).astype(int)

    X_base = feature_matrix.values.astype(float)
    col_medians = np.nanmedian(X_base, axis=0)
    for j in range(X_base.shape[1]):
        mask = np.isnan(X_base[:, j])
        X_base[mask, j] = col_medians[j] if not np.isnan(col_medians[j]) else 0.0

    logger.info(f"Base feature matrix: {X_base.shape}")

    # ---- STEP 2: Pre-compute Poisson stacking features for each fold ----
    logger.info("Pre-computing Poisson stacking features for each fold...")
    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)
    fold_data = []

    for fold_num, (train_idx, val_idx) in enumerate(cv.split(matches_df)):
        train_idx_arr = np.array(train_idx)
        val_idx_arr = np.array(val_idx)

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

        poisson_train = compute_poisson_features(poisson_model, matches_df, train_idx_arr)
        poisson_val = compute_poisson_features(poisson_model, matches_df, val_idx_arr)
        fold_data.append((train_idx_arr, val_idx_arr, poisson_train, poisson_val))

    logger.info("Poisson stacking features cached for all folds.")

    # ---- STEP 3: Grid search over XGBoost params ----
    configs = generate_configs(PARAM_GRID, max_configs=30)
    logger.info(f"Testing {len(configs)} hyperparameter configurations...")

    results = []
    for i, config in enumerate(configs):
        short_config = {k: v for k, v in config.items() if k != "random_state"}
        logger.info(f"Config {i+1}/{len(configs)}: {short_config}")

        try:
            rps, acc = evaluate_config(
                config, X_base, y_1x2, y_1x2_encoded, y_ou, y_btts,
                matches_df, fold_data,
            )
            results.append({"config": config, "rps": rps, "accuracy": acc})
            logger.info(f"  -> RPS={rps:.4f}, Acc={acc:.3f}")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")

    # Sort by RPS (lower is better)
    results.sort(key=lambda x: x["rps"])

    logger.info(f"\n{'='*60}")
    logger.info("TOP 5 CONFIGURATIONS (by RPS)")
    logger.info(f"{'='*60}")
    for rank, r in enumerate(results[:5], 1):
        short = {k: v for k, v in r["config"].items() if k != "random_state"}
        logger.info(f"  #{rank}: RPS={r['rps']:.4f}, Acc={r['accuracy']:.3f}")
        logger.info(f"     {short}")

    if results:
        best = results[0]
        logger.info(f"\nBEST CONFIG: RPS={best['rps']:.4f}, Acc={best['accuracy']:.3f}")
        logger.info(f"  Params: {best['config']}")

        # Retrain full model with best config and save
        logger.info("\nRetraining final model with best config...")
        trainer = ModelTrainer(
            xgb_params=best["config"],
            n_splits=2,
            min_train_seasons=3,
        )
        final_result = trainer.train(
            matches_df,
            elo_df=elo_df if len(elo_df) > 0 else None,
            xg_df=xg_df if len(xg_df) > 0 else None,
            predictions_df=predictions_df if len(predictions_df) > 0 else None,
        )
        trainer.save("models/latest.pkl")
        logger.info(f"Model saved! Final RPS={final_result.aggregate_metrics['mean_rps']:.4f}, "
                     f"Acc={final_result.aggregate_metrics['mean_accuracy']:.3f}")
        logger.info(f"Ensemble weights: {final_result.ensemble_weights}")
