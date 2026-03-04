#!/usr/bin/env python3
"""CLI script to train SharpEdge ML models.

Usage:
    python scripts/train.py --leagues "Premier League" --output models/latest
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sharpedge.ml.training.trainer import ModelTrainer
from sharpedge.ml.data_loader import (
    load_historical_matches,
    load_elo_ratings,
    load_xg_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train SharpEdge ML models")
    parser.add_argument(
        "--leagues", nargs="+", default=None, help="Leagues to train on"
    )
    parser.add_argument(
        "--seasons", nargs="+", default=None, help="Seasons to include"
    )
    parser.add_argument(
        "--output", default="models/latest", help="Output directory"
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="XGBoost n_estimators"
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="XGBoost max_depth"
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    matches_df = load_historical_matches(
        leagues=args.leagues, seasons=args.seasons
    )
    logger.info(f"Loaded {len(matches_df)} matches")

    elo_df = load_elo_ratings()
    xg_df = load_xg_data()

    trainer = ModelTrainer(
        xgb_params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
    )

    logger.info("Starting training...")
    result = trainer.train(matches_df, elo_df=elo_df, xg_df=xg_df)

    logger.info(f"Mean RPS: {result.aggregate_metrics['mean_rps']:.4f}")
    logger.info(
        f"Mean Accuracy: {result.aggregate_metrics['mean_accuracy']:.3f}"
    )

    trainer.save(args.output)
    logger.info(f"Model saved to {args.output}")

    # Save metrics
    metrics_path = Path(args.output) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "fold_metrics": result.fold_metrics,
                "aggregate": result.aggregate_metrics,
                "ensemble_weights": result.ensemble_weights,
            },
            f,
            indent=2,
        )
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
