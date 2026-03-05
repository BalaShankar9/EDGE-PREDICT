"""Experiment: compare calibration methods and fold counts on the proven 46-feature set."""
import logging
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])

from sharpedge.db.engine import get_session
from sharpedge.ml.training.trainer import ModelTrainer

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


if __name__ == "__main__":
    matches_df, elo_df = load_data()
    logger.info(f"Loaded {len(matches_df)} matches")

    # Best known XGBoost params
    best_params = {
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

    experiments = [
        ("platt_2fold", "platt", 2, 3),
        ("isotonic_2fold", "isotonic", 2, 3),
        ("platt_3fold", "platt", 3, 2),
        ("isotonic_3fold", "isotonic", 3, 2),
    ]

    for name, cal_method, n_splits, min_seasons in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"Experiment: {name}")
        logger.info(f"{'='*50}")

        trainer = ModelTrainer(
            xgb_params=best_params,
            n_splits=n_splits,
            min_train_seasons=min_seasons,
            calibration_method=cal_method,
        )
        result = trainer.train(matches_df, elo_df=elo_df)
        logger.info(f"RESULT [{name}]: RPS={result.aggregate_metrics['mean_rps']:.4f}, "
                     f"Acc={result.aggregate_metrics['mean_accuracy']:.3f}")
        for fold in result.fold_metrics:
            logger.info(f"  Fold {fold['fold']}: RPS={fold['rps']:.4f}, Acc={fold['accuracy']:.3f}")
