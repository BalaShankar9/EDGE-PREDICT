#!/usr/bin/env python3
"""Generate live picks for upcoming matches.

This is the PRODUCTION pipeline that finds real value bets by comparing
model probabilities against current bookmaker odds.

Uses SuperiorPredictor — the full ensemble with OpenSkill, TabPFN,
adaptive draw floors, AntifragileStaking, and Intelligence Bureau context.

Usage:
    .venv/bin/python scripts/generate_live_picks.py
    .venv/bin/python scripts/generate_live_picks.py --days 3
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sharpedge.ml.data_loader import load_historical_matches
from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.core.superior import SuperiorPredictor

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
    """Fetch upcoming fixtures from football-data.org."""
    from sharpedge.collectors.football_data_org import FootballDataOrgCollector

    collector = FootballDataOrgCollector()
    df = collector.collect()
    if df is None or len(df) == 0:
        logger.warning("No upcoming fixtures found")
        return pd.DataFrame()

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


def train_superior_predictor(df: pd.DataFrame):
    """Train SuperiorPredictor on the FULL historical dataset.

    Returns (predictor, pipeline, dc_model) so we can augment fixture features.
    """
    logger.info(f"Training SuperiorPredictor on {len(df)} matches...")

    pipeline = FeaturePipeline()
    features = pipeline.build(df)
    X = features.values.astype(np.float32)

    y_1x2 = df["FTR"].values
    y_ou = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int).values
    y_btts = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int).values

    # Build match dicts for Poisson models + OpenSkill
    train_matches = [
        {
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "home_goals": int(row["FTHG"]),
            "away_goals": int(row["FTAG"]),
        }
        for _, row in df.iterrows()
    ]

    # We need a temporary DC model to augment features with attack/defence strengths
    dc_temp = PoissonPredictor()
    dc_temp.fit(train_matches)

    # Augment features with DC strengths
    n = len(df)
    dc_feats = np.full((n, 4), np.nan, dtype=np.float32)
    for k in range(n):
        h = df.iloc[k]["home_team_id"]
        a = df.iloc[k]["away_team_id"]
        dc_feats[k, 0] = dc_temp.attack_strength.get(h, np.nan)
        dc_feats[k, 1] = dc_temp.defence_strength.get(h, np.nan)
        dc_feats[k, 2] = dc_temp.attack_strength.get(a, np.nan)
        dc_feats[k, 3] = dc_temp.defence_strength.get(a, np.nan)
    X_aug = np.hstack([X, dc_feats])

    # Train SuperiorPredictor — fits XGB, CatBoost, LightGBM, DC, BVP, TabPFN, OpenSkill
    predictor = SuperiorPredictor()
    predictor.fit(
        X_aug, y_1x2,
        train_df=df,
        train_matches=train_matches,
        y_ou=y_ou,
        y_btts=y_btts,
    )

    logger.info(
        "SuperiorPredictor ready: %d models, %d league draw rates, %d OpenSkill ratings",
        len(predictor._models),
        len(predictor._league_draw_rates),
        len(predictor._team_ratings),
    )

    return predictor, pipeline, dc_temp


def format_prediction(pred) -> dict:
    """Convert SuperiorPrediction to serializable dict."""
    return {
        "home_team": pred.home_team,
        "away_team": pred.away_team,
        "league": pred.league,
        "match_date": pred.match_date,
        "probabilities": {
            "Home": round(float(pred.probabilities[0]), 3),
            "Draw": round(float(pred.probabilities[1]), 3),
            "Away": round(float(pred.probabilities[2]), 3),
        },
        "predicted_outcome": pred.predicted_outcome,
        "confidence": round(pred.confidence, 3),
        "n_models": len(pred.model_probs),
        "model_agreement": round(pred.model_agreement, 2),
        "mc_uncertainty": round(pred.mc_std, 4),
        "prediction_entropy": round(pred.prediction_entropy, 3),
        "edge": round(pred.edge, 3),
        "is_value_bet": pred.is_value_bet,
        "expected_value": round(pred.expected_value, 1),
        "kelly_stake_pct": round(pred.kelly_stake, 2),
        "reasoning": pred.reasoning,
        "confidence_factors": pred.confidence_factors,
        "risk_factors": pred.risk_factors,
        "model_breakdown": pred.model_probs,
        "implied_probs": pred.implied_probs,
        "goto_probs": pred.goto_probs,
    }


def format_pick_message(pick: dict) -> str:
    """Format a pick for display."""
    msg = (
        f"  {pick['home_team']} vs {pick['away_team']}\n"
        f"  {pick['league']}\n"
        f"  {pick['predicted_outcome']} @ {pick['confidence']:.1%}\n"
        f"  Edge: {pick['edge']:.1%} | EV: {pick['expected_value']}%\n"
        f"  Stake: {pick['kelly_stake_pct']}% | {pick['n_models']} models, {pick['model_agreement']:.0%} agree\n"
    )
    if pick.get("confidence_factors"):
        msg += f"  + {', '.join(pick['confidence_factors'])}\n"
    if pick.get("risk_factors"):
        msg += f"  ! {', '.join(pick['risk_factors'])}\n"
    return msg


def main():
    parser = argparse.ArgumentParser(description="Generate live picks")
    parser.add_argument("--days", type=int, default=7, help="Days ahead to look")
    parser.add_argument("--min-edge", type=float, default=0.05, help="Minimum edge to bet")
    parser.add_argument("--output", type=str, default="live_picks.json", help="Output file")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  SharpEdge Live Prediction Pipeline (SuperiorPredictor)")
    logger.info("=" * 60)

    # 1. Load ALL historical data
    logger.info("Loading historical data...")
    df = load_historical_matches()
    logger.info(f"Loaded {len(df)} matches")

    # 2. Train SuperiorPredictor (all models + OpenSkill + adaptive draw floors)
    predictor, pipeline, dc_temp = train_superior_predictor(df)

    # 3. Fetch upcoming fixtures
    logger.info(f"Fetching fixtures for next {args.days} days...")
    fixtures = fetch_upcoming_fixtures(days_ahead=args.days)

    if len(fixtures) == 0:
        logger.info("No upcoming fixtures. Exiting.")
        return

    # 4. Fetch current odds
    odds_dict = fetch_current_odds(fixtures)

    # 5. Build features for upcoming matches
    logger.info("Building features for upcoming matches...")

    for col in ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR"]:
        if col not in fixtures.columns:
            if col in ("FTR", "HTR"):
                fixtures[col] = "D"
            else:
                fixtures[col] = 0

    if "season" not in fixtures.columns:
        fixtures["season"] = "2024-25"

    combined = pd.concat([df, fixtures], ignore_index=True)
    combined_features = pipeline.build(combined)
    upcoming_features = combined_features.iloc[len(df):].reset_index(drop=True)

    # DC strength averages for unknown teams
    avg_attack = float(np.nanmean(list(dc_temp.attack_strength.values()))) if dc_temp.attack_strength else 0.0
    avg_defence = float(np.nanmean(list(dc_temp.defence_strength.values()))) if dc_temp.defence_strength else 0.0

    # 6. Generate predictions using SuperiorPredictor
    logger.info("Generating predictions...")
    predictions = []
    for i, (_, fixture) in enumerate(fixtures.iterrows()):
        try:
            feat_vec = upcoming_features.iloc[i].values.astype(np.float32)
            home_team = fixture.get("home_team_id", fixture.get("home_team", ""))
            away_team = fixture.get("away_team_id", fixture.get("away_team", ""))
            league = fixture.get("league", "")

            # Augment feature vector with DC strengths
            dc_feats = np.array([
                dc_temp.attack_strength.get(home_team, avg_attack),
                dc_temp.defence_strength.get(home_team, avg_defence),
                dc_temp.attack_strength.get(away_team, avg_attack),
                dc_temp.defence_strength.get(away_team, avg_defence),
            ], dtype=np.float32)
            feat_aug = np.hstack([feat_vec, dc_feats])

            # Look up odds for this match
            match_odds = odds_dict.get((home_team, away_team))

            pred = predictor.predict(
                X=feat_aug,
                home_team=home_team,
                away_team=away_team,
                league=league,
                odds=match_odds,
                match_date=str(fixture.get("match_date", "")),
            )

            predictions.append(format_prediction(pred))
        except Exception as e:
            logger.warning(
                f"Failed to predict {fixture.get('home_team', '?')} vs "
                f"{fixture.get('away_team', '?')}: {e}"
            )

    logger.info(f"Generated {len(predictions)} predictions")

    # 7. Identify value bets (already computed by SuperiorPredictor)
    value_picks = [p for p in predictions if p["is_value_bet"]]
    value_picks.sort(key=lambda x: x["edge"], reverse=True)
    logger.info(f"Found {len(value_picks)} value bets")

    # 8. Output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "SuperiorPredictor",
        "total_fixtures": len(fixtures),
        "total_predictions": len(predictions),
        "total_value_bets": len(value_picks),
        "min_edge": args.min_edge,
        "models_used": list(predictor._models.keys()),
        "league_draw_rates": {k: round(v, 3) for k, v in predictor._league_draw_rates.items()},
        "openskill_teams": len(predictor._team_ratings),
        "predictions": predictions,
        "value_picks": value_picks,
    }

    output_path = ROOT / args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    # 9. Print value bets
    if value_picks:
        logger.info("\n" + "=" * 60)
        logger.info("  VALUE BETS FOUND")
        logger.info("=" * 60)
        for pick in value_picks[:10]:
            logger.info(f"\n{format_pick_message(pick)}")
    else:
        logger.info("No value bets found at current odds.")

    # 10. Print all predictions summary
    logger.info("\n" + "=" * 60)
    logger.info("  ALL PREDICTIONS")
    logger.info("=" * 60)

    # Verify draw probabilities are NOT all the same
    draw_probs = [p["probabilities"]["Draw"] for p in predictions]
    unique_draws = len(set(draw_probs))
    logger.info(f"Draw probability diversity: {unique_draws} unique values across {len(predictions)} predictions")
    if unique_draws <= 3 and len(predictions) > 10:
        logger.warning("LOW DRAW DIVERSITY — adaptive floor may not be working!")

    for pred in predictions[:20]:
        probs = pred["probabilities"]
        logger.info(
            f"{pred['match_date'][:10]} | {pred['league']:20s} | "
            f"{pred['home_team']:20s} vs {pred['away_team']:20s} | "
            f"H={probs['Home']:.0%} D={probs['Draw']:.0%} A={probs['Away']:.0%} | "
            f"{pred['predicted_outcome']} ({pred['confidence']:.0%}) | "
            f"{pred['n_models']} models"
        )


if __name__ == "__main__":
    main()
