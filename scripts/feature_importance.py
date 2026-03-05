"""Analyze feature importance and identify weak features for pruning."""
import logging
import sys

import numpy as np
import pandas as pd
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from sharpedge.db.engine import get_session
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


if __name__ == "__main__":
    logger.info("Loading data and training model for feature importance...")
    matches_df, elo_df, xg_df, predictions_df = load_training_data()

    trainer = ModelTrainer(n_splits=2, min_train_seasons=3)
    result = trainer.train(
        matches_df,
        elo_df=elo_df if len(elo_df) > 0 else None,
        xg_df=xg_df if len(xg_df) > 0 else None,
        predictions_df=predictions_df if len(predictions_df) > 0 else None,
    )

    # Get feature importances from XGBoost
    xgb_model = trainer.xgb_model
    feature_names = result.feature_names

    if hasattr(xgb_model, 'model_1x2') and xgb_model.model_1x2 is not None:
        importances = xgb_model.model_1x2.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        logger.info(f"\n{'='*60}")
        logger.info("FEATURE IMPORTANCE RANKING (XGBoost 1X2 model)")
        logger.info(f"{'='*60}")
        for _, row in importance_df.iterrows():
            bar = "#" * int(row["importance"] * 200)
            logger.info(f"  {row['feature']:<35} {row['importance']:.4f} {bar}")

        logger.info(f"\n{'='*60}")
        logger.info("TOP 15 FEATURES")
        logger.info(f"{'='*60}")
        for _, row in importance_df.head(15).iterrows():
            logger.info(f"  {row['feature']:<35} {row['importance']:.4f}")

        logger.info(f"\n{'='*60}")
        logger.info("BOTTOM 10 FEATURES (candidates for pruning)")
        logger.info(f"{'='*60}")
        for _, row in importance_df.tail(10).iterrows():
            logger.info(f"  {row['feature']:<35} {row['importance']:.4f}")

        # Check for zero-importance features
        zero_features = importance_df[importance_df["importance"] == 0]["feature"].tolist()
        if zero_features:
            logger.info(f"\nZERO-IMPORTANCE FEATURES: {zero_features}")

        # NaN analysis
        logger.info(f"\n{'='*60}")
        logger.info("NaN COVERAGE (% of values that are NaN per feature)")
        logger.info(f"{'='*60}")

        from sharpedge.ml.features.pipeline import FeaturePipeline
        pipe = FeaturePipeline()
        feature_matrix = pipe.build(
            matches_df,
            elo_df=elo_df if len(elo_df) > 0 else None,
            xg_df=xg_df if len(xg_df) > 0 else None,
            predictions_df=predictions_df if len(predictions_df) > 0 else None,
        )
        for col in feature_matrix.columns:
            nan_pct = feature_matrix[col].isna().mean() * 100
            if nan_pct > 50:
                logger.info(f"  {col:<35} {nan_pct:.1f}% NaN  *** HIGH ***")
            elif nan_pct > 20:
                logger.info(f"  {col:<35} {nan_pct:.1f}% NaN")
    else:
        logger.error("XGBoost 1X2 model not available for importance analysis")
