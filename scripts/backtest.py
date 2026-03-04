#!/usr/bin/env python3
"""CLI script to run SharpEdge backtests.

Usage:
    python scripts/backtest.py --output results/backtest_report.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sharpedge.backtest.engine import BacktestEngine, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run SharpEdge backtest")
    parser.add_argument(
        "--output",
        default="results/backtest_report.json",
        help="Output JSON",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll",
    )
    args = parser.parse_args()

    logger.info("Backtest CLI ready. Provide predictions and actuals data to run.")
    logger.info(
        "This script will be fully wired once Phase 3 integrates data loading."
    )

    # Placeholder: when wired, this will:
    # 1. Load trained model from models/latest
    # 2. Load test-season matches
    # 3. Generate predictions for each match
    # 4. Run backtest engine
    # 5. Save results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_path}")


if __name__ == "__main__":
    main()
