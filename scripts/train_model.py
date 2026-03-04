"""Train the model on real data from the database."""
import logging
import sys

import pandas as pd
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log"),
    ],
)

from sharpedge.db.engine import get_session
from sharpedge.ml.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def load_training_data():
    """Load matches + supporting data from the database."""
    session = get_session()

    matches_query = text("""
        SELECT
            m.id, m.match_date, m.home_team_id, m.away_team_id,
            m.home_goals AS "FTHG", m.away_goals AS "FTAG",
            CASE
                WHEN m.home_goals > m.away_goals THEN 'H'
                WHEN m.home_goals = m.away_goals THEN 'D'
                ELSE 'A'
            END AS "FTR",
            ht.canonical_name AS home_team_name,
            at.canonical_name AS away_team_name,
            l.name AS league,
            s.label AS season
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN seasons s ON m.season_id = s.id
        JOIN leagues l ON s.league_id = l.id
        WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.match_date
    """)
    matches_df = pd.read_sql(matches_query, session.bind)

    elo_query = text("""
        SELECT team_id, rating_date, elo, source
        FROM elo_ratings
        ORDER BY rating_date
    """)
    elo_df = pd.read_sql(elo_query, session.bind)

    xg_query = text("""
        SELECT match_id, source, home_xg, away_xg
        FROM match_xg
    """)
    xg_df = pd.read_sql(xg_query, session.bind)

    pred_query = text("""
        SELECT match_id, source, predicted_result, prob_home, prob_draw, prob_away
        FROM competitor_predictions
    """)
    predictions_df = pd.read_sql(pred_query, session.bind)

    session.close()
    return matches_df, elo_df, xg_df, predictions_df


if __name__ == "__main__":
    logger.info("Loading training data from database...")
    matches_df, elo_df, xg_df, predictions_df = load_training_data()

    logger.info(f"Loaded {len(matches_df)} matches, {len(elo_df)} ELO records, "
                f"{len(xg_df)} xG records, {len(predictions_df)} competitor predictions")

    if len(matches_df) < 100:
        logger.error("Not enough matches to train. Run the backfill first.")
        sys.exit(1)

    logger.info("Starting model training...")
    trainer = ModelTrainer(n_splits=2, min_train_seasons=3)
    result = trainer.train(
        matches_df,
        elo_df=elo_df if len(elo_df) > 0 else None,
        xg_df=xg_df if len(xg_df) > 0 else None,
        predictions_df=predictions_df if len(predictions_df) > 0 else None,
    )

    logger.info("Training complete!")
    logger.info(f"  Mean RPS: {result.aggregate_metrics.get('mean_rps', 'N/A')}")
    logger.info(f"  Mean Accuracy: {result.aggregate_metrics.get('mean_accuracy', 'N/A')}")
    logger.info(f"  Ensemble weights: {result.ensemble_weights}")
    logger.info(f"  Features: {len(result.feature_names)}")

    save_path = "models/latest.pkl"
    trainer.save(save_path)
    logger.info(f"Model saved to {save_path}/model.pkl")

    for fold in result.fold_metrics:
        logger.info(f"  Fold {fold['fold']}: RPS={fold['rps']:.4f}, Acc={fold['accuracy']:.3f}, "
                     f"train={fold['train_size']}, val={fold['val_size']}")
