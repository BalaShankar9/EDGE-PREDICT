#!/usr/bin/env python3
"""SharpEdge Optimizer V4 — Profitability Edition.

Strategy: combine V2's focused approach with V3's intelligence.

Key changes from V2:
  - Train on ALL 12 leagues, but only BET on Big 5 (efficient markets)
  - MAPIE conformal prediction: skip ambiguous matches (prediction set > 1)
  - Adaptive per-league draw floor (not a fixed value)
  - Per-league ROI tracking in results
  - Tighter MC uncertainty threshold (0.12 vs 0.15)
  - Starting config based on V2's best parameters

Usage:
    .venv/bin/python scripts/optimize_v4_profitable.py
    .venv/bin/python scripts/optimize_v4_profitable.py --iterations 50
    .venv/bin/python scripts/optimize_v4_profitable.py --iterations 2 --patience 2 --mc-sims 500
"""
import argparse
import copy
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root & path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sharpedge.core.conformal import ConformalPredictor
from sharpedge.core.consensus import compute_consensus
from sharpedge.core.monte_carlo import monte_carlo_simulate
from sharpedge.ml.data_loader import load_historical_matches
from sharpedge.ml.features.pipeline import FeaturePipeline
from sharpedge.ml.models.bivariate_poisson import BivariatePoissonPredictor
from sharpedge.ml.models.poisson_model import PoissonPredictor
from sharpedge.ml.models.xgboost_model import XGBoostPredictor

# Optional gradient-boosting alternatives
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

# ---------------------------------------------------------------------------
# Logging — dedicated V4 logger (force own file handler)
# ---------------------------------------------------------------------------
v4_logger = logging.getLogger("optimize_v4")
v4_logger.setLevel(logging.INFO)
v4_logger.handlers.clear()
v4_logger.addHandler(logging.StreamHandler())
v4_logger.addHandler(logging.FileHandler(ROOT / "optimization_v4.log", mode="w"))
v4_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
for h in v4_logger.handlers:
    h.setFormatter(v4_formatter)
logger = v4_logger

# ---------------------------------------------------------------------------
# Betting leagues — Big 5 by default, expansion leagues added dynamically
# ---------------------------------------------------------------------------
BETTING_LEAGUES = {
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
}


# ---------------------------------------------------------------------------
# Configuration dataclass — V4 with conformal + league filter additions
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """All tunable parameters for the V4 betting pipeline."""

    # Filter params
    min_confidence_home: float = 0.55
    min_confidence_away: float = 0.48
    min_confidence_over: float = 0.55
    min_edge: float = 0.04
    max_odds: float = 2.75  # slightly higher than V2's 2.50

    # XGBoost params (start from V2's best)
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.01
    xgb_n_estimators: int = 300
    xgb_min_child_weight: int = 8
    xgb_subsample: float = 0.8
    xgb_reg_alpha: float = 1.0
    xgb_reg_lambda: float = 1.0

    # Draw floor (fallback if adaptive is disabled)
    draw_floor: float = 0.20

    # Staking
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03

    # 5-model blend weights
    w_xgb: float = 0.30
    w_catboost: float = 0.20
    w_lgbm: float = 0.20
    w_dc: float = 0.20
    w_bvp: float = 0.10

    # Consensus filter
    min_consensus: int = 3

    # Monte Carlo
    mc_sims: int = 2000

    # V4 additions
    use_conformal: bool = True
    conformal_confidence: float = 0.80  # lower than 0.90 — 3-class sets are wide at high confidence
    max_conformal_set_size: int = 2  # allow up to 2 (skip only fully ambiguous size=3)
    league_filter: bool = True  # only bet on Big 5
    adaptive_draw_floor: bool = True
    max_mc_std: float = 0.12  # tighter than V2's 0.15


# Per-parameter mutation rules: (step, min, max, is_int)
PARAM_SPECS: dict[str, tuple[float, float, float, bool]] = {
    "min_confidence_home": (0.02, 0.40, 0.75, False),
    "min_confidence_away": (0.02, 0.35, 0.70, False),
    "min_confidence_over": (0.02, 0.40, 0.70, False),
    "min_edge": (0.01, 0.00, 0.15, False),
    "max_odds": (0.25, 1.50, 5.00, False),
    "xgb_max_depth": (1, 2, 8, True),
    "xgb_learning_rate": (0.005, 0.005, 0.10, False),
    "xgb_n_estimators": (50, 100, 800, True),
    "xgb_min_child_weight": (2, 1, 30, True),
    "xgb_subsample": (0.05, 0.5, 1.0, False),
    "xgb_reg_alpha": (0.2, 0.0, 5.0, False),
    "xgb_reg_lambda": (0.2, 0.0, 5.0, False),
    "draw_floor": (0.02, 0.10, 0.30, False),
    "kelly_fraction": (0.05, 0.05, 0.50, False),
    "max_stake_pct": (0.005, 0.01, 0.10, False),
    # All 5 blend weights
    "w_xgb": (0.05, 0.05, 0.60, False),
    "w_catboost": (0.05, 0.00, 0.50, False),
    "w_lgbm": (0.05, 0.00, 0.50, False),
    "w_dc": (0.05, 0.05, 0.50, False),
    "w_bvp": (0.05, 0.00, 0.40, False),
    # Consensus
    "min_consensus": (1, 2, 5, True),
    # V4 additions
    "conformal_confidence": (0.05, 0.50, 0.95, False),
    "max_conformal_set_size": (1, 1, 3, True),
    "max_mc_std": (0.01, 0.05, 0.20, False),
}


def mutate(cfg: Config, n_mutations: int | None = None) -> tuple[Config, list[str]]:
    """Create a mutated copy of *cfg*, changing 1-3 random parameters."""
    new = copy.deepcopy(cfg)
    if n_mutations is None:
        n_mutations = random.randint(1, 3)

    available = list(PARAM_SPECS.keys())
    # Remove catboost/lgbm weights if not available
    if not HAS_CATBOOST:
        available = [p for p in available if p != "w_catboost"]
    if not HAS_LIGHTGBM:
        available = [p for p in available if p != "w_lgbm"]

    params = random.sample(available, k=min(n_mutations, len(available)))
    descriptions: list[str] = []

    for name in params:
        step, lo, hi, is_int = PARAM_SPECS[name]
        old_val = getattr(new, name)
        delta = random.choice([-1, 1]) * step
        new_val = old_val + delta
        new_val = max(lo, min(hi, new_val))
        if is_int:
            new_val = int(round(new_val))
        setattr(new, name, new_val)
        descriptions.append(f"{name}: {old_val:.4g} -> {new_val:.4g}")

    return new, descriptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def implied_prob(odds: float) -> float:
    """Decimal odds to implied probability."""
    return 1.0 / odds if odds > 1.0 else 1.0


def _build_match_dicts(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame rows to match dicts for Poisson models."""
    return [
        {
            "home_team_id": row["home_team_id"],
            "away_team_id": row["away_team_id"],
            "home_goals": int(row["FTHG"]),
            "away_goals": int(row["FTAG"]),
        }
        for _, row in df.iterrows()
    ]


def _augment_dc(
    X: np.ndarray,
    match_df: pd.DataFrame,
    model: PoissonPredictor,
) -> np.ndarray:
    """Add 4 Dixon-Coles strength columns to the feature matrix."""
    n = len(match_df)
    dc_feats = np.full((n, 4), np.nan, dtype=np.float32)
    for k in range(n):
        h = match_df.iloc[k]["home_team_id"]
        a = match_df.iloc[k]["away_team_id"]
        dc_feats[k, 0] = model.attack_strength.get(h, np.nan)
        dc_feats[k, 1] = model.defence_strength.get(h, np.nan)
        dc_feats[k, 2] = model.attack_strength.get(a, np.nan)
        dc_feats[k, 3] = model.defence_strength.get(a, np.nan)
    return np.hstack([X, dc_feats])


def _apply_draw_floor(probs: np.ndarray, floor: float) -> np.ndarray:
    """Raise draw probability to at least *floor*, redistributing from H/A."""
    out = probs.copy()
    for k in range(len(out)):
        if out[k, 1] < floor:
            deficit = floor - out[k, 1]
            out[k, 1] = floor
            ha_sum = out[k, 0] + out[k, 2]
            if ha_sum > 0:
                out[k, 0] -= deficit * (out[k, 0] / ha_sum)
                out[k, 2] -= deficit * (out[k, 2] / ha_sum)
            out[k] = np.clip(out[k], 0.01, None)
            out[k] /= out[k].sum()
    return out


def _apply_draw_floor_adaptive(
    probs: np.ndarray,
    val_df: pd.DataFrame,
    league_draw_rates: dict[str, float],
    fallback_floor: float,
) -> np.ndarray:
    """Apply per-league adaptive draw floor."""
    out = probs.copy()
    for k in range(len(out)):
        league = val_df.iloc[k].get("league", "")
        floor = league_draw_rates.get(league, fallback_floor)
        if out[k, 1] < floor:
            deficit = floor - out[k, 1]
            out[k, 1] = floor
            ha_sum = out[k, 0] + out[k, 2]
            if ha_sum > 0:
                out[k, 0] -= deficit * (out[k, 0] / ha_sum)
                out[k, 2] -= deficit * (out[k, 2] / ha_sum)
            out[k] = np.clip(out[k], 0.01, None)
            out[k] /= out[k].sum()
    return out


def _normalise_weights(cfg: Config) -> dict[str, float]:
    """Return normalised blend weights for all available models."""
    weights = {
        "xgb": cfg.w_xgb,
        "dc": cfg.w_dc,
        "bvp": cfg.w_bvp,
    }
    if HAS_CATBOOST:
        weights["catboost"] = cfg.w_catboost
    if HAS_LIGHTGBM:
        weights["lgbm"] = cfg.w_lgbm

    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Pre-computed fold data
# ---------------------------------------------------------------------------


@dataclass
class FoldData:
    """Pre-computed data for a single walk-forward fold."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    X_train: np.ndarray
    X_val: np.ndarray
    y_train_1x2: np.ndarray
    y_train_ou: np.ndarray
    y_train_btts: np.ndarray
    y_val_encoded: np.ndarray
    train_matches: list[dict]
    odds_h: np.ndarray
    odds_d: np.ndarray
    odds_a: np.ndarray
    val_season: str


def build_fold_cache(
    df: pd.DataFrame,
    seasons: list,
    min_train: int = 3,
) -> list[FoldData]:
    """Pre-compute all fold data. Called once before the optimisation loop."""
    folds: list[FoldData] = []
    pipeline = FeaturePipeline()

    for fold_idx in range(min_train, len(seasons)):
        train_seasons = seasons[:fold_idx]
        val_season = seasons[fold_idx]

        train_mask = df["season"].isin(train_seasons)
        val_mask = df["season"] == val_season

        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True)

        if len(train_df) == 0 or len(val_df) == 0:
            continue

        logger.info(
            f"  Fold {fold_idx - min_train + 1}: "
            f"train {train_seasons} ({len(train_df)}) -> val [{val_season}] ({len(val_df)})"
        )

        # Build features (train only)
        train_features = pipeline.build(train_df)
        # Build features (train + val combined, then slice val portion)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        combined_features = pipeline.build(combined_df)
        val_features = combined_features.iloc[len(train_df):].reset_index(drop=True)

        X_train = train_features.values.astype(np.float32)
        X_val = val_features.values.astype(np.float32)

        # Targets
        y_train_1x2 = train_df["FTR"].values
        y_train_ou = (
            (train_df["FTHG"] + train_df["FTAG"]) > 2.5
        ).astype(int).values
        y_train_btts = (
            (train_df["FTHG"] > 0) & (train_df["FTAG"] > 0)
        ).astype(int).values
        y_val_encoded = np.array(
            [{"H": 0, "D": 1, "A": 2}.get(r, 1) for r in val_df["FTR"].values]
        )

        # Match dicts for Poisson models
        train_matches = _build_match_dicts(train_df)

        # Odds
        odds_h = pd.to_numeric(val_df.get("B365H"), errors="coerce").values
        odds_d = pd.to_numeric(val_df.get("B365D"), errors="coerce").values
        odds_a = pd.to_numeric(val_df.get("B365A"), errors="coerce").values

        folds.append(
            FoldData(
                train_df=train_df,
                val_df=val_df,
                X_train=X_train,
                X_val=X_val,
                y_train_1x2=y_train_1x2,
                y_train_ou=y_train_ou,
                y_train_btts=y_train_btts,
                y_val_encoded=y_val_encoded,
                train_matches=train_matches,
                odds_h=odds_h,
                odds_d=odds_d,
                odds_a=odds_a,
                val_season=val_season,
            )
        )

    return folds


# ---------------------------------------------------------------------------
# Walk-forward backtest V4 — 5 models + conformal + league filter + adaptive
# ---------------------------------------------------------------------------


def run_backtest(cfg: Config, folds: list[FoldData]) -> dict:
    """Run walk-forward backtest with V4 enhancements.

    Returns
    -------
    dict with keys: roi, n_bets, win_pct, profit, avg_consensus,
                    avg_mc_confidence, league_breakdown, conformal_filtered
    """
    weights = _normalise_weights(cfg)
    xgb_params = {
        "n_estimators": cfg.xgb_n_estimators,
        "max_depth": cfg.xgb_max_depth,
        "learning_rate": cfg.xgb_learning_rate,
        "subsample": cfg.xgb_subsample,
        "colsample_bytree": 0.8,
        "min_child_weight": cfg.xgb_min_child_weight,
        "reg_alpha": cfg.xgb_reg_alpha,
        "reg_lambda": cfg.xgb_reg_lambda,
        "random_state": 42,
    }

    all_bets: list[dict] = []
    conformal_filtered_count = 0

    for fold in folds:
        # --- Compute per-league draw rates from training data ---
        league_draw_rates: dict[str, float] = {}
        if cfg.adaptive_draw_floor:
            for league in fold.train_df["league"].unique():
                league_mask = fold.train_df["league"] == league
                league_draw_rates[league] = (
                    fold.train_df.loc[league_mask, "FTR"] == "D"
                ).mean()

        # --- Dixon-Coles ---
        dc_model = PoissonPredictor()
        dc_model.fit(fold.train_matches)

        # Augment features with DC team strengths
        X_train_aug = _augment_dc(fold.X_train, fold.train_df, dc_model)
        X_val_aug = _augment_dc(fold.X_val, fold.val_df, dc_model)

        # --- XGBoost ---
        xgb_model = XGBoostPredictor(params=xgb_params)
        xgb_model.fit(
            X_train_aug,
            fold.y_train_1x2,
            y_ou=fold.y_train_ou,
            y_btts=fold.y_train_btts,
        )

        # --- CatBoost ---
        catboost_model = None
        if HAS_CATBOOST:
            catboost_model = CatBoostPredictor()
            catboost_model.fit(
                X_train_aug,
                fold.y_train_1x2,
                y_ou=fold.y_train_ou,
                y_btts=fold.y_train_btts,
            )

        # --- LightGBM ---
        lgbm_model = None
        if HAS_LIGHTGBM:
            lgbm_model = LightGBMPredictor()
            lgbm_model.fit(
                X_train_aug,
                fold.y_train_1x2,
                y_ou=fold.y_train_ou,
                y_btts=fold.y_train_btts,
            )

        # --- Bivariate Poisson ---
        bvp_model = BivariatePoissonPredictor()
        bvp_model.fit(fold.train_matches)

        # --- Conformal prediction ---
        conformal_fitted = False
        set_sizes = None
        if cfg.use_conformal:
            try:
                conformal = ConformalPredictor(
                    confidence_level=cfg.conformal_confidence
                )
                # Fit on training data using XGBoost's 1x2 model
                # Use a calibration split from training data
                n_train = len(X_train_aug)
                cal_size = max(200, n_train // 5)
                cal_start = n_train - cal_size
                X_cal = X_train_aug[cal_start:]
                y_cal = np.array(
                    [{"H": 0, "D": 1, "A": 2}.get(r, 1)
                     for r in fold.y_train_1x2[cal_start:]]
                )
                conformal.fit(X_cal, y_cal, base_model=xgb_model.model_1x2)
                # Get prediction sets for validation data
                _, pred_sets = conformal.predict_sets(X_val_aug)
                set_sizes = pred_sets.sum(axis=1)
                conformal_fitted = True
            except Exception as e:
                logger.warning(f"Conformal prediction failed: {e}")
                conformal_fitted = False

        # --- Validation predictions from ALL models ---
        n_val = len(fold.val_df)
        model_probs: dict[str, np.ndarray] = {}

        # XGBoost
        model_probs["xgb"] = xgb_model.predict_proba_1x2(X_val_aug)

        # CatBoost
        if catboost_model is not None:
            model_probs["catboost"] = catboost_model.predict_proba_1x2(X_val_aug)

        # LightGBM
        if lgbm_model is not None:
            model_probs["lgbm"] = lgbm_model.predict_proba_1x2(X_val_aug)

        # Dixon-Coles & BVP (per-match)
        dc_1x2 = np.zeros((n_val, 3))
        bvp_1x2 = np.zeros((n_val, 3))
        for k in range(n_val):
            h = fold.val_df.iloc[k]["home_team_id"]
            a = fold.val_df.iloc[k]["away_team_id"]
            dc_1x2[k] = dc_model.predict_proba_1x2(h, a)
            bvp_1x2[k] = bvp_model.predict_proba_1x2(h, a)

        model_probs["dc"] = dc_1x2
        model_probs["bvp"] = bvp_1x2

        # --- Model consensus ---
        consensus_pred, consensus_count = compute_consensus(model_probs)

        # --- Monte Carlo simulation ---
        mc_probs, mc_confidence, mc_std = monte_carlo_simulate(
            model_probs, n_sims=cfg.mc_sims
        )

        # --- Weighted blend (using optimiser weights) ---
        blended = np.zeros((n_val, 3))
        for model_name, model_pred in model_probs.items():
            w = weights.get(model_name, 0.0)
            blended += w * model_pred

        # Normalise
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        blended = blended / row_sums

        # --- Draw floor (adaptive or fixed) ---
        if cfg.adaptive_draw_floor and league_draw_rates:
            blended = _apply_draw_floor_adaptive(
                blended, fold.val_df, league_draw_rates, cfg.draw_floor
            )
        else:
            blended = _apply_draw_floor(blended, cfg.draw_floor)

        # --- Combine weighted blend with MC for final prediction ---
        # 70% weighted blend + 30% MC simulation (MC acts as regulariser)
        final_probs = 0.7 * blended + 0.3 * mc_probs
        # Re-normalise
        row_sums = final_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        final_probs = final_probs / row_sums

        # --- Bet simulation ---
        for k in range(n_val):
            if (
                np.isnan(fold.odds_h[k])
                or np.isnan(fold.odds_d[k])
                or np.isnan(fold.odds_a[k])
            ):
                continue

            # === LEAGUE FILTER ===
            # Only bet on qualifying leagues (Big 5 by default)
            if cfg.league_filter:
                match_league = fold.val_df.iloc[k].get("league", "")
                if match_league not in BETTING_LEAGUES:
                    continue

            # === CONFORMAL PREDICTION FILTER ===
            # Skip predictions where set_size exceeds threshold
            # set_size=1: very confident, set_size=2: somewhat confident, set_size=3: clueless
            if cfg.use_conformal and conformal_fitted and set_sizes is not None:
                if set_sizes[k] > cfg.max_conformal_set_size:
                    conformal_filtered_count += 1
                    continue

            probs = final_probs[k]
            pred = int(np.argmax(probs))
            pred_prob = float(probs[pred])
            pred_odds = [fold.odds_h[k], fold.odds_d[k], fold.odds_a[k]][pred]
            actual = fold.y_val_encoded[k]

            # Market label
            market_map = {0: "1x2_home", 1: "1x2_draw", 2: "1x2_away"}
            market = market_map[pred]

            # === CONSENSUS FILTER ===
            if consensus_count[k] < cfg.min_consensus:
                continue

            # === MC CONFIDENCE CHECK (tighter V4 threshold) ===
            if mc_std[k, pred] > cfg.max_mc_std:
                continue

            # Confidence threshold per market
            conf_thresholds = {
                0: cfg.min_confidence_home,
                1: cfg.min_confidence_home,  # draws use home threshold
                2: cfg.min_confidence_away,
            }
            if pred_prob < conf_thresholds[pred]:
                continue

            # Max odds filter
            if pred_odds > cfg.max_odds:
                continue

            # Edge filter
            edge = pred_prob - implied_prob(pred_odds)
            if edge < cfg.min_edge:
                continue

            won = pred == actual
            match_league = fold.val_df.iloc[k].get("league", "unknown")
            all_bets.append(
                {
                    "odds": float(pred_odds),
                    "prob": float(pred_prob),
                    "edge": float(edge),
                    "won": bool(won),
                    "market": market,
                    "season": fold.val_season,
                    "consensus": int(consensus_count[k]),
                    "mc_confidence": float(mc_confidence[k]),
                    "mc_std": float(mc_std[k, pred]),
                    "league": match_league,
                }
            )

    # --- Aggregate metrics ---
    n_bets = len(all_bets)
    if n_bets == 0:
        return {
            "roi": -100.0, "n_bets": 0, "win_pct": 0.0, "profit": 0.0,
            "avg_consensus": 0.0, "avg_mc_confidence": 0.0,
            "league_breakdown": {}, "conformal_filtered": conformal_filtered_count,
        }

    wins = sum(1 for b in all_bets if b["won"])
    total_staked = float(n_bets)  # flat 1-unit stake
    total_return = sum(b["odds"] for b in all_bets if b["won"])
    profit = total_return - total_staked
    roi = (profit / total_staked) * 100
    win_pct = (wins / n_bets) * 100
    avg_consensus = np.mean([b["consensus"] for b in all_bets])
    avg_mc_conf = np.mean([b["mc_confidence"] for b in all_bets])

    # Per-league ROI tracking
    league_breakdown: dict[str, dict] = {}
    for bet in all_bets:
        league = bet.get("league", "unknown")
        if league not in league_breakdown:
            league_breakdown[league] = {"bets": 0, "wins": 0, "profit": 0.0}
        league_breakdown[league]["bets"] += 1
        if bet["won"]:
            league_breakdown[league]["wins"] += 1
            league_breakdown[league]["profit"] += bet["odds"] - 1.0
        else:
            league_breakdown[league]["profit"] -= 1.0

    # Compute per-league ROI
    for league, stats in league_breakdown.items():
        if stats["bets"] > 0:
            stats["roi"] = round((stats["profit"] / stats["bets"]) * 100, 2)
            stats["win_pct"] = round((stats["wins"] / stats["bets"]) * 100, 1)
            stats["profit"] = round(stats["profit"], 2)

    return {
        "roi": round(roi, 2),
        "n_bets": n_bets,
        "win_pct": round(win_pct, 2),
        "profit": round(profit, 2),
        "avg_consensus": round(avg_consensus, 2),
        "avg_mc_confidence": round(avg_mc_conf, 4),
        "league_breakdown": league_breakdown,
        "conformal_filtered": conformal_filtered_count,
    }


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SharpEdge V4 Profitability Optimizer"
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Max iterations (default 100)",
    )
    parser.add_argument(
        "--min-bets", type=int, default=20,
        help="Minimum bets for a valid config (default 20)",
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Stop after N iterations without improvement",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
    parser.add_argument(
        "--mc-sims", type=int, default=2000,
        help="Monte Carlo simulations per match (default 2000)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("  SharpEdge Optimizer V4 — Profitability Edition")
    logger.info("=" * 70)
    logger.info(f"Strategy: Train ALL leagues, bet Big 5 only")
    logger.info(f"Max iterations: {args.iterations}")
    logger.info(f"Min bets:       {args.min_bets}")
    logger.info(f"Patience:       {args.patience}")
    logger.info(f"MC simulations: {args.mc_sims}")
    logger.info(f"CatBoost:       {HAS_CATBOOST}")
    logger.info(f"LightGBM:       {HAS_LIGHTGBM}")

    n_models = 3 + int(HAS_CATBOOST) + int(HAS_LIGHTGBM)
    logger.info(f"Active models:  {n_models}")

    # ------------------------------------------------------------------
    # 1. Load data ONCE — ALL leagues for training
    # ------------------------------------------------------------------
    logger.info("Loading historical match data (ALL leagues for training)...")
    t0 = time.time()
    df = load_historical_matches()
    logger.info(
        f"Loaded {len(df)} matches across {df['league'].nunique()} leagues, "
        f"{df['season'].nunique()} seasons in {time.time() - t0:.1f}s"
    )

    # Verify 'league' column exists
    if "league" not in df.columns:
        logger.error("'league' column not found in data. Aborting.")
        return

    # Show league distribution
    league_counts = df["league"].value_counts()
    logger.info("League distribution:")
    for league, count in league_counts.items():
        marker = " [BET]" if league in BETTING_LEAGUES else ""
        logger.info(f"  {league}: {count} matches{marker}")

    seasons = sorted(df["season"].unique())
    min_train = 3
    if len(seasons) <= min_train:
        logger.error(
            f"Need >{min_train} seasons, only have {len(seasons)}. Aborting."
        )
        return

    logger.info(f"Seasons: {seasons}")

    # ------------------------------------------------------------------
    # 2. Build features & cache fold data ONCE (expensive)
    # ------------------------------------------------------------------
    logger.info(
        "Building features and caching fold data (this may take a few minutes)..."
    )
    t0 = time.time()
    folds = build_fold_cache(df, seasons, min_train)
    cache_time = time.time() - t0
    logger.info(f"Fold cache built: {len(folds)} folds in {cache_time:.1f}s")

    # ------------------------------------------------------------------
    # 3. Baseline evaluation
    # ------------------------------------------------------------------
    best_cfg = Config(mc_sims=args.mc_sims)
    logger.info("Evaluating baseline config (V4: conformal + league filter)...")
    t0 = time.time()
    best_metrics = run_backtest(best_cfg, folds)
    baseline_time = time.time() - t0
    logger.info(
        f"Baseline: ROI={best_metrics['roi']:.2f}%, "
        f"bets={best_metrics['n_bets']}, "
        f"win%={best_metrics['win_pct']:.1f}%, "
        f"profit={best_metrics['profit']:.1f}u, "
        f"consensus={best_metrics['avg_consensus']:.1f}, "
        f"mc_conf={best_metrics['avg_mc_confidence']:.3f}, "
        f"conformal_filtered={best_metrics['conformal_filtered']} "
        f"({baseline_time:.1f}s)"
    )

    # Log per-league breakdown
    if best_metrics["league_breakdown"]:
        logger.info("Per-league breakdown:")
        for league, stats in sorted(
            best_metrics["league_breakdown"].items(),
            key=lambda x: x[1].get("roi", -999),
            reverse=True,
        ):
            logger.info(
                f"  {league}: ROI={stats.get('roi', 0):.1f}%, "
                f"bets={stats['bets']}, "
                f"wins={stats['wins']}, "
                f"win%={stats.get('win_pct', 0):.1f}%"
            )

    iteration_log: list[dict] = [
        {
            "iteration": 0,
            "config": asdict(best_cfg),
            "metrics": best_metrics,
            "status": "baseline",
            "mutations": [],
        }
    ]

    # ------------------------------------------------------------------
    # 4. Optimisation loop
    # ------------------------------------------------------------------
    no_improvement = 0
    best_roi = (
        best_metrics["roi"]
        if best_metrics["n_bets"] >= args.min_bets
        else -999.0
    )

    for iteration in range(1, args.iterations + 1):
        t_iter = time.time()

        # Mutate
        candidate_cfg, mutations = mutate(best_cfg)
        candidate_cfg.mc_sims = args.mc_sims  # keep MC sims from args
        logger.info(f"\n--- Iteration {iteration}/{args.iterations} ---")
        for m in mutations:
            logger.info(f"  Mutation: {m}")

        # Evaluate
        metrics = run_backtest(candidate_cfg, folds)
        elapsed = time.time() - t_iter

        # Selection: prefer higher win% when ROI is close
        is_better = False
        if metrics["n_bets"] >= args.min_bets:
            if metrics["roi"] > best_roi:
                is_better = True
            elif (
                abs(metrics["roi"] - best_roi) < 1.0
                and metrics["win_pct"] > best_metrics["win_pct"]
            ):
                # Within 1% ROI — prefer higher win rate
                is_better = True

        if is_better:
            status = "IMPROVED"
            best_cfg = candidate_cfg
            best_metrics = metrics
            best_roi = metrics["roi"]
            no_improvement = 0
            logger.info(
                f"  >>> IMPROVED: ROI={metrics['roi']:.2f}%, "
                f"bets={metrics['n_bets']}, "
                f"win%={metrics['win_pct']:.1f}%, "
                f"profit={metrics['profit']:.1f}u, "
                f"consensus={metrics['avg_consensus']:.1f}, "
                f"conformal_filtered={metrics['conformal_filtered']} "
                f"({elapsed:.1f}s)"
            )
        else:
            status = "rejected"
            no_improvement += 1
            reason = (
                f"bets={metrics['n_bets']}<{args.min_bets}"
                if metrics["n_bets"] < args.min_bets
                else f"ROI={metrics['roi']:.2f}%/win%={metrics['win_pct']:.1f}% not better"
            )
            logger.info(
                f"  rejected ({reason}), "
                f"streak={no_improvement}/{args.patience} "
                f"({elapsed:.1f}s)"
            )

        iteration_log.append(
            {
                "iteration": iteration,
                "config": asdict(candidate_cfg),
                "metrics": metrics,
                "status": status,
                "mutations": mutations,
            }
        )

        # Convergence check
        if no_improvement >= args.patience:
            logger.info(
                f"\nConverged: no improvement for {args.patience} iterations."
            )
            break

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    output = {
        "version": "v4-profitability",
        "strategy": "train-all-bet-big5-conformal-adaptive",
        "best_config": asdict(best_cfg),
        "best_metrics": best_metrics,
        "n_models": n_models,
        "iterations": len(iteration_log) - 1,
        "log": iteration_log,
    }

    output_path = ROOT / "optimization_results_v4.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("  OPTIMIZATION V4 COMPLETE — Profitability Edition")
    logger.info("=" * 70)
    logger.info(f"Models used:         {n_models}")
    logger.info(f"Best ROI:            {best_metrics['roi']:.2f}%")
    logger.info(f"Best bets:           {best_metrics['n_bets']}")
    logger.info(f"Best win%:           {best_metrics['win_pct']:.1f}%")
    logger.info(f"Best profit:         {best_metrics['profit']:.1f}u")
    logger.info(f"Avg consensus:       {best_metrics['avg_consensus']:.1f}")
    logger.info(f"Avg MC confidence:   {best_metrics['avg_mc_confidence']:.3f}")
    logger.info(f"Conformal filtered:  {best_metrics['conformal_filtered']}")
    logger.info(f"Results saved to {output_path}")

    # Print league breakdown
    if best_metrics["league_breakdown"]:
        logger.info("\nPer-league ROI breakdown:")
        for league, stats in sorted(
            best_metrics["league_breakdown"].items(),
            key=lambda x: x[1].get("roi", -999),
            reverse=True,
        ):
            logger.info(
                f"  {league}: ROI={stats.get('roi', 0):.1f}%, "
                f"bets={stats['bets']}, "
                f"win%={stats.get('win_pct', 0):.1f}%, "
                f"profit={stats['profit']:.1f}u"
            )

    # Print best config
    logger.info("\nBest configuration:")
    for k, v in asdict(best_cfg).items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
