#!/usr/bin/env python3
"""Generate live picks for upcoming matches.

This is the PRODUCTION pipeline that finds real value bets by comparing
model probabilities against current bookmaker odds.

Usage:
    .venv/bin/python scripts/generate_live_picks.py
    .venv/bin/python scripts/generate_live_picks.py --days 3
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sharpedge.ml.data_loader import load_historical_matches
from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.ml.models.xgboost_model import XGBoostPredictor
from sharpedge.ml.models.bivariate_poisson import BivariatePoissonPredictor
from sharpedge.core.monte_carlo import monte_carlo_simulate
from sharpedge.core.consensus import compute_consensus

try:
    from sharpedge.ml.models.catboost_model import CatBoostPredictor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from sharpedge.ml.models.lightgbm_model import LightGBMPredictor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Logger
logger = logging.getLogger("live_picks")
logger.setLevel(logging.INFO)
logger.propagate = False
sh = logging.StreamHandler()
fh = logging.FileHandler(ROOT / "live_picks.log")
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh.setFormatter(fmt)
fh.setFormatter(fmt)
logger.addHandler(sh)
logger.addHandler(fh)


def fetch_upcoming_fixtures(days_ahead: int = 7) -> pd.DataFrame:
    """Fetch upcoming fixtures from football-data.org.

    The collector fetches ALL scheduled matches; we filter to the
    requested window here.
    """
    from sharpedge.collectors.football_data_org import FootballDataOrgCollector

    collector = FootballDataOrgCollector()
    df = collector.collect()
    if df is None or len(df) == 0:
        logger.warning("No upcoming fixtures found")
        return pd.DataFrame()

    # Filter to requested date window
    df["match_date"] = pd.to_datetime(df["match_date"], utc=True, errors="coerce")
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)
    df = df[(df["match_date"] >= now) & (df["match_date"] <= cutoff)]
    df = df.sort_values("match_date").reset_index(drop=True)

    logger.info(f"Found {len(df)} upcoming fixtures in next {days_ahead} days")
    return df


def fetch_current_odds(fixtures_df: pd.DataFrame) -> dict:
    """Fetch current odds for fixtures.

    Returns dict mapping (home_team, away_team) -> {home: odds, draw: odds, away: odds}
    """
    # Try The Odds API if available
    try:
        from sharpedge.collectors.odds_api import OddsAPICollector
        from sharpedge.config import settings
        if settings.odds_api_key:
            collector = OddsAPICollector()
            odds_df = collector.collect()
            if odds_df is not None and len(odds_df) > 0:
                odds_dict = {}
                for _, row in odds_df.iterrows():
                    key = (row.get("home_team", ""), row.get("away_team", ""))
                    odds_dict[key] = {
                        "home": row.get("home_odds", 0),
                        "draw": row.get("draw_odds", 0),
                        "away": row.get("away_odds", 0),
                        "bookmaker": row.get("bookmaker", "unknown"),
                    }
                logger.info(f"Fetched odds for {len(odds_dict)} matches from Odds API")
                return odds_dict
    except Exception as e:
        logger.warning(f"Odds API failed: {e}")

    logger.info("No live odds available -- will use model probabilities only")
    return {}


def train_production_model(df: pd.DataFrame):
    """Train all models on the FULL historical dataset."""
    logger.info(f"Training on {len(df)} matches...")

    pipeline = FeaturePipeline()
    features = pipeline.build(df)
    X = features.values.astype(np.float32)

    y_1x2 = df["FTR"].values
    y_ou = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int).values
    y_btts = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int).values

    # Build match dicts for Poisson models
    train_matches = [
        {
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "home_goals": int(row["FTHG"]),
            "away_goals": int(row["FTAG"]),
        }
        for _, row in df.iterrows()
    ]

    # Dixon-Coles
    dc_model = PoissonPredictor()
    dc_model.fit(train_matches)

    # Augment features with DC strengths
    n = len(df)
    dc_feats = np.full((n, 4), np.nan, dtype=np.float32)
    for k in range(n):
        h = df.iloc[k]["home_team_id"]
        a = df.iloc[k]["away_team_id"]
        dc_feats[k, 0] = dc_model.attack_strength.get(h, np.nan)
        dc_feats[k, 1] = dc_model.defence_strength.get(h, np.nan)
        dc_feats[k, 2] = dc_model.attack_strength.get(a, np.nan)
        dc_feats[k, 3] = dc_model.defence_strength.get(a, np.nan)
    X_aug = np.hstack([X, dc_feats])

    # XGBoost
    xgb = XGBoostPredictor()
    xgb.fit(X_aug, y_1x2, y_ou=y_ou, y_btts=y_btts)

    # CatBoost
    cat = None
    if HAS_CATBOOST:
        cat = CatBoostPredictor()
        cat.fit(X_aug, y_1x2, y_ou=y_ou, y_btts=y_btts)

    # LightGBM
    lgbm = None
    if HAS_LIGHTGBM:
        lgbm = LightGBMPredictor()
        lgbm.fit(X_aug, y_1x2, y_ou=y_ou, y_btts=y_btts)

    # Bivariate Poisson
    bvp = BivariatePoissonPredictor()
    bvp.fit(train_matches)

    logger.info("All models trained")

    return {
        "dc": dc_model, "xgb": xgb, "cat": cat, "lgbm": lgbm, "bvp": bvp,
        "pipeline": pipeline, "X_template": X,
    }


def predict_match(models, home_team, away_team, features_vector, league=""):
    """Generate prediction for a single upcoming match."""
    dc = models["dc"]
    xgb = models["xgb"]
    cat = models["cat"]
    lgbm = models["lgbm"]
    bvp = models["bvp"]

    # DC strengths for this match (use league average for unknown teams)
    avg_attack = float(np.nanmean(list(dc.attack_strength.values()))) if dc.attack_strength else 0.0
    avg_defence = float(np.nanmean(list(dc.defence_strength.values()))) if dc.defence_strength else 0.0

    dc_feats = np.array([
        dc.attack_strength.get(home_team, avg_attack),
        dc.defence_strength.get(home_team, avg_defence),
        dc.attack_strength.get(away_team, avg_attack),
        dc.defence_strength.get(away_team, avg_defence),
    ], dtype=np.float32)

    X = np.hstack([features_vector, dc_feats]).reshape(1, -1)

    # All model predictions — each must be (1, 3)
    model_probs = {}
    model_probs["xgb"] = xgb.predict_proba_1x2(X)  # already (1, 3)
    if cat:
        model_probs["catboost"] = cat.predict_proba_1x2(X)  # already (1, 3)
    if lgbm:
        model_probs["lgbm"] = lgbm.predict_proba_1x2(X)  # already (1, 3)
    # Poisson models return (3,) — reshape to (1, 3)
    model_probs["dc"] = dc.predict_proba_1x2(home_team, away_team).reshape(1, 3)
    model_probs["bvp"] = bvp.predict_proba_1x2(home_team, away_team).reshape(1, 3)

    # MC simulation
    mc_probs, mc_conf, mc_std = monte_carlo_simulate(model_probs, n_sims=5000)

    # Consensus
    consensus_pred, consensus_count = compute_consensus(model_probs)

    # Weighted blend (V2 best weights)
    weights = {"xgb": 0.30, "catboost": 0.20, "lgbm": 0.20, "dc": 0.20, "bvp": 0.10}
    blended = np.zeros(3)
    total_w = 0
    for name, probs in model_probs.items():
        w = weights.get(name, 0.1)
        blended += w * probs[0]
        total_w += w
    blended /= total_w

    # Combine with MC (70/30)
    final = 0.7 * blended + 0.3 * mc_probs[0]
    final = final / final.sum()

    # Draw floor
    if final[1] < 0.18:
        deficit = 0.18 - final[1]
        final[1] = 0.18
        ha = final[0] + final[2]
        if ha > 0:
            final[0] -= deficit * (final[0] / ha)
            final[2] -= deficit * (final[2] / ha)
        final = np.clip(final, 0.01, None)
        final /= final.sum()

    pred_idx = int(np.argmax(final))
    outcomes = ["Home", "Draw", "Away"]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "league": league,
        "probabilities": {"Home": round(float(final[0]), 3), "Draw": round(float(final[1]), 3), "Away": round(float(final[2]), 3)},
        "predicted_outcome": outcomes[pred_idx],
        "confidence": round(float(final[pred_idx]), 3),
        "consensus": int(consensus_count[0]),
        "n_models": len(model_probs),
        "mc_confidence": round(float(mc_conf[0]), 3),
        "mc_uncertainty": round(float(mc_std[0, pred_idx]), 4),
        "model_breakdown": {name: [round(float(p), 3) for p in probs[0]] for name, probs in model_probs.items()},
    }


def find_value_bets(predictions: list[dict], odds: dict, min_edge: float = 0.05) -> list[dict]:
    """Find bets where model probability exceeds implied probability by min_edge."""
    picks = []

    for pred in predictions:
        key = (pred["home_team"], pred["away_team"])
        match_odds = odds.get(key, {})

        if not match_odds:
            continue

        # Check each outcome for value
        for outcome, prob in pred["probabilities"].items():
            odds_key = outcome.lower()
            bet_odds = match_odds.get(odds_key, 0)

            if bet_odds <= 1.0:
                continue

            implied = 1.0 / bet_odds
            edge = prob - implied

            if edge >= min_edge and prob >= 0.50:
                # Kelly stake
                b = bet_odds - 1
                kelly = (prob * b - (1 - prob)) / b
                kelly_stake = max(0, kelly * 0.25)  # quarter Kelly

                picks.append({
                    **pred,
                    "market": f"1x2_{outcome.lower()}",
                    "selection": outcome,
                    "bet_odds": bet_odds,
                    "implied_prob": round(implied, 3),
                    "edge": round(edge, 3),
                    "edge_pct": round(edge * 100, 1),
                    "kelly_stake_pct": round(kelly_stake * 100, 2),
                    "bookmaker": match_odds.get("bookmaker", "unknown"),
                    "expected_value": round((prob * bet_odds - 1) * 100, 1),
                })

    # Sort by edge descending
    picks.sort(key=lambda x: x["edge"], reverse=True)
    return picks


def format_pick_message(pick: dict) -> str:
    """Format a pick for Telegram."""
    return (
        f"  {pick['home_team']} vs {pick['away_team']}\n"
        f"  {pick['league']}\n"
        f"  {pick['selection']} @ {pick['bet_odds']}\n"
        f"  Model: {pick['confidence']:.1%} | Implied: {pick['implied_prob']:.1%}\n"
        f"  Edge: {pick['edge_pct']}% | EV: {pick['expected_value']}%\n"
        f"  Stake: {pick['kelly_stake_pct']}% of bankroll\n"
        f"  Consensus: {pick['consensus']}/{pick['n_models']} models agree\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate live picks")
    parser.add_argument("--days", type=int, default=7, help="Days ahead to look")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge to bet")
    parser.add_argument("--output", type=str, default="live_picks.json", help="Output file")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SharpEdge Live Prediction Pipeline")
    logger.info("=" * 60)

    # 1. Load ALL historical data
    logger.info("Loading historical data...")
    df = load_historical_matches()
    logger.info(f"Loaded {len(df)} matches")

    # 2. Train production model
    models = train_production_model(df)

    # 3. Fetch upcoming fixtures
    logger.info(f"Fetching fixtures for next {args.days} days...")
    fixtures = fetch_upcoming_fixtures(days_ahead=args.days)

    if len(fixtures) == 0:
        logger.info("No upcoming fixtures. Exiting.")
        return

    # 4. Fetch current odds
    odds = fetch_current_odds(fixtures)

    # 5. Build features for upcoming matches
    logger.info("Building features for upcoming matches...")
    pipeline = models["pipeline"]

    # Upcoming matches need placeholder goal columns for the feature pipeline
    # (rolling features look back at historical data only)
    for col in ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]:
        if col not in fixtures.columns:
            if col == "FTR":
                fixtures[col] = "D"
            elif col == "HTR":
                fixtures[col] = "D"
            else:
                fixtures[col] = 0

    # Add season column if missing (needed by some feature groups)
    if "season" not in fixtures.columns:
        fixtures["season"] = "2024-25"

    # Combine historical + upcoming for feature context
    combined = pd.concat([df, fixtures], ignore_index=True)
    combined_features = pipeline.build(combined)
    upcoming_features = combined_features.iloc[len(df):].reset_index(drop=True)

    # 6. Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    for i, (_, fixture) in enumerate(fixtures.iterrows()):
        try:
            feat_vec = upcoming_features.iloc[i].values.astype(np.float32)
            pred = predict_match(
                models,
                fixture.get("home_team_id", fixture.get("home_team", "")),
                fixture.get("away_team_id", fixture.get("away_team", "")),
                feat_vec,
                league=fixture.get("league", ""),
            )
            pred["match_date"] = str(fixture.get("match_date", ""))
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Failed to predict {fixture.get('home_team', '?')} vs {fixture.get('away_team', '?')}: {e}")

    logger.info(f"Generated {len(predictions)} predictions")

    # 7. Find value bets
    picks = find_value_bets(predictions, odds, min_edge=args.min_edge)
    logger.info(f"Found {len(picks)} value bets (edge >= {args.min_edge:.0%})")

    # 8. Output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_fixtures": len(fixtures),
        "total_predictions": len(predictions),
        "total_picks": len(picks),
        "min_edge": args.min_edge,
        "predictions": predictions,
        "picks": picks,
    }

    output_path = ROOT / args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # 9. Print picks
    if picks:
        logger.info("\n" + "=" * 60)
        logger.info("  VALUE BETS FOUND")
        logger.info("=" * 60)
        for pick in picks[:10]:  # top 10
            logger.info(f"\n{format_pick_message(pick)}")
    else:
        logger.info("No value bets found at current odds.")

    # 10. Print all predictions summary
    logger.info("\n" + "=" * 60)
    logger.info("  ALL PREDICTIONS")
    logger.info("=" * 60)
    for pred in predictions[:20]:
        probs = pred["probabilities"]
        logger.info(
            f"{pred['match_date'][:10]} | {pred['league']:20s} | "
            f"{pred['home_team']:20s} vs {pred['away_team']:20s} | "
            f"H={probs['Home']:.0%} D={probs['Draw']:.0%} A={probs['Away']:.0%} | "
            f"{pred['predicted_outcome']} ({pred['confidence']:.0%})"
        )


if __name__ == "__main__":
    main()
