#!/usr/bin/env python3
"""Automated optimization loop for SharpEdge — V3 Agent Swarm + Bayesian Arbiter.

Upgrades over V2:
  - 7 agents: Statistical, Gradient, OvR, Form, H2H, Market, Contrarian
  - BayesianArbiter combines all agent predictions with performance weighting
  - Consensus based on agent agreement %, not raw model count
  - Per-agent accuracy tracking
  - Arbiter entropy and confidence metrics

Key design:
  - Data loading and feature building happen ONCE (expensive, ~2-5 min).
  - Per-fold feature matrices are pre-cached (reuses V2's FoldData/build_fold_cache).
  - Each iteration trains agents, runs arbiter, and simulates bets.

Usage:
    .venv/bin/python scripts/optimize_v3_agents.py
    .venv/bin/python scripts/optimize_v3_agents.py --iterations 100
    .venv/bin/python scripts/optimize_v3_agents.py --iterations 2 --patience 2 --mc-sims 500
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

from sharpedge.agents.arbiter import ArbiterResult, BayesianArbiter
from sharpedge.agents.base_agent import MatchContext
from sharpedge.agents.contrarian_agent import ContrarianAgent
from sharpedge.agents.form_momentum_agent import FormMomentumAgent
from sharpedge.agents.gradient_agent import GradientAgent
from sharpedge.agents.h2h_venue_agent import H2HVenueAgent
from sharpedge.agents.market_agent import MarketAgent
from sharpedge.agents.ovr_specialist_agent import OvRSpecialistAgent
from sharpedge.agents.statistical_agent import StatisticalAgent
from sharpedge.ml.data_loader import load_historical_matches
from sharpedge.ml.models.poisson_model import PoissonPredictor

# Reuse V2's fold cache infrastructure
from optimize_loop import (
    FoldData,
    _augment_dc,
    _apply_draw_floor,
    build_fold_cache,
    implied_prob,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "optimization_v3.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass — V3 with agent-specific params
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """All tunable parameters for the V3 agent-powered pipeline."""

    # Filter params
    min_confidence_home: float = 0.52
    min_confidence_away: float = 0.48
    min_confidence_over: float = 0.55
    min_edge: float = 0.05
    max_odds: float = 2.50

    # XGBoost params (used by GradientAgent internally)
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.01
    xgb_n_estimators: int = 300
    xgb_min_child_weight: int = 10
    xgb_subsample: float = 0.8
    xgb_reg_alpha: float = 1.0
    xgb_reg_lambda: float = 1.0

    # Draw floor
    draw_floor: float = 0.18

    # Staking
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03

    # Monte Carlo simulation count per match
    mc_sims: int = 5000

    # --- Agent arbiter params ---
    arbiter_decay: float = 0.95          # Bayesian weight decay
    min_consensus_pct: float = 0.40      # minimum % of agents that must agree (was 0.60 - too strict)
    min_arbiter_confidence: float = 0.45  # minimum arbiter confidence to bet (was 0.55 - too strict)

    # --- Agent toggles ---
    use_statistical: bool = True
    use_gradient: bool = True
    use_ovr: bool = True
    use_form: bool = True
    use_h2h: bool = True
    use_market: bool = True
    use_contrarian: bool = True


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
    # Agent arbiter params
    "arbiter_decay": (0.02, 0.80, 0.99, False),
    "min_consensus_pct": (0.05, 0.30, 0.90, False),
    "min_arbiter_confidence": (0.02, 0.45, 0.75, False),
}


def mutate(cfg: Config, n_mutations: int | None = None) -> tuple[Config, list[str]]:
    """Create a mutated copy of *cfg*, changing 1-3 random parameters."""
    new = copy.deepcopy(cfg)
    if n_mutations is None:
        n_mutations = random.randint(1, 3)

    available = list(PARAM_SPECS.keys())
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
# Build match dicts (same as V2)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Walk-forward backtest V3 — Agent Swarm + Bayesian Arbiter
# ---------------------------------------------------------------------------


def run_backtest(cfg: Config, folds: list[FoldData]) -> dict:
    """Run walk-forward backtest with Agent Swarm + Bayesian Arbiter.

    Returns
    -------
    dict with keys: roi, n_bets, win_pct, profit, avg_consensus_pct,
                    avg_entropy, avg_agents_contributing, agent_accuracy
    """
    all_bets: list[dict] = []
    # Per-agent accuracy tracking across all folds
    agent_accuracy: dict[str, dict[str, int]] = {}  # name -> {correct, total}

    for fold in folds:
        # --- 1. Train a Dixon-Coles model for DC features (augmentation) ---
        # Filter out matches with None team IDs
        clean_matches = [
            m for m in fold.train_matches
            if m["home_team_id"] is not None and m["away_team_id"] is not None
        ]
        dc_model = PoissonPredictor()
        dc_model.fit(clean_matches)

        # Augment features with DC team strengths
        X_train_aug = _augment_dc(fold.X_train, fold.train_df, dc_model)
        X_val_aug = _augment_dc(fold.X_val, fold.val_df, dc_model)

        # --- 2. Train all agents that need training ---
        agents_for_fold = []

        if cfg.use_statistical:
            stat_agent = StatisticalAgent()
            stat_agent.fit(clean_matches)
            agents_for_fold.append(stat_agent)

        if cfg.use_gradient:
            grad_agent = GradientAgent()
            grad_agent.fit(
                X_train_aug,
                fold.y_train_1x2,
                y_train_ou=fold.y_train_ou,
                y_train_btts=fold.y_train_btts,
            )
            agents_for_fold.append(grad_agent)

        if cfg.use_ovr:
            ovr_agent = OvRSpecialistAgent()
            ovr_agent.fit(X_train_aug, fold.y_train_1x2)
            agents_for_fold.append(ovr_agent)

        if cfg.use_form:
            form_agent = FormMomentumAgent()
            form_agent.fit(fold.train_df)
            agents_for_fold.append(form_agent)

        if cfg.use_h2h:
            h2h_agent = H2HVenueAgent()
            h2h_agent.fit(fold.train_df)
            agents_for_fold.append(h2h_agent)

        if cfg.use_market:
            market_agent = MarketAgent()
            market_agent.fit()
            agents_for_fold.append(market_agent)

        if cfg.use_contrarian:
            contrarian_agent = ContrarianAgent()
            contrarian_agent.fit()
            agents_for_fold.append(contrarian_agent)

        if not agents_for_fold:
            continue

        # --- 3. Create arbiter with all agents ---
        arbiter = BayesianArbiter(agents=agents_for_fold, decay=cfg.arbiter_decay)

        # --- 4. For each validation match, create MatchContext and run arbiter ---
        n_val = len(fold.val_df)
        y_val_encoded = np.array(
            [{"H": 0, "D": 1, "A": 2}.get(r, 1) for r in fold.val_df["FTR"].values]
        )

        for k in range(n_val):
            # Skip matches with missing odds
            if (
                np.isnan(fold.odds_h[k])
                or np.isnan(fold.odds_d[k])
                or np.isnan(fold.odds_a[k])
            ):
                continue

            row = fold.val_df.iloc[k]

            context = MatchContext(
                match_id=f"{fold.val_season}_{k}",
                sport="football",
                league=str(row.get("league", "unknown")),
                match_date=str(row.get("match_date", "")),
                home_team=str(row["home_team_id"]),
                away_team=str(row["away_team_id"]),
                features=X_val_aug[k],
                odds={
                    "home": float(fold.odds_h[k]),
                    "draw": float(fold.odds_d[k]),
                    "away": float(fold.odds_a[k]),
                },
                market="1x2",
                outcomes=("home", "draw", "away"),
            )

            result = arbiter.predict(context)
            if result is None:
                continue

            # Map outcome names to indices
            outcome_map = {"home": 0, "draw": 1, "away": 2}

            # Use arbiter's blended probabilities
            probs = result.probabilities.copy()

            # Apply draw floor
            if probs[1] < cfg.draw_floor:
                deficit = cfg.draw_floor - probs[1]
                probs[1] = cfg.draw_floor
                ha_sum = probs[0] + probs[2]
                if ha_sum > 0:
                    probs[0] -= deficit * (probs[0] / ha_sum)
                    probs[2] -= deficit * (probs[2] / ha_sum)
                probs = np.clip(probs, 0.01, None)
                probs /= probs.sum()

            pred_idx = int(np.argmax(probs))
            pred_prob = float(probs[pred_idx])
            pred_odds = [fold.odds_h[k], fold.odds_d[k], fold.odds_a[k]][pred_idx]
            actual = y_val_encoded[k]

            # === CONSENSUS FILTER (agent agreement %) ===
            if result.consensus_score < cfg.min_consensus_pct:
                continue

            # === ARBITER CONFIDENCE CHECK ===
            if pred_prob < cfg.min_arbiter_confidence:
                continue

            # Confidence threshold per market
            conf_thresholds = {
                0: cfg.min_confidence_home,
                1: cfg.min_confidence_home,  # draws use home threshold
                2: cfg.min_confidence_away,
            }
            if pred_prob < conf_thresholds[pred_idx]:
                continue

            # Max odds filter
            if pred_odds > cfg.max_odds:
                continue

            # Edge filter
            edge = pred_prob - implied_prob(pred_odds)
            if edge < cfg.min_edge:
                continue

            won = pred_idx == actual

            # Track per-agent accuracy on this match
            for agent_pred in result.agent_predictions:
                aname = agent_pred.agent_name
                if aname not in agent_accuracy:
                    agent_accuracy[aname] = {"correct": 0, "total": 0}
                agent_accuracy[aname]["total"] += 1
                agent_pred_idx = outcome_map.get(agent_pred.predicted_outcome, -1)
                if agent_pred_idx == actual:
                    agent_accuracy[aname]["correct"] += 1

            # Record agent results for arbiter learning
            agent_results = {}
            for agent_pred in result.agent_predictions:
                agent_pred_idx = outcome_map.get(agent_pred.predicted_outcome, -1)
                agent_results[agent_pred.agent_name] = (agent_pred_idx == actual)
            arbiter.record_results(agent_results)

            market_map = {0: "1x2_home", 1: "1x2_draw", 2: "1x2_away"}
            all_bets.append(
                {
                    "odds": float(pred_odds),
                    "prob": float(pred_prob),
                    "edge": float(edge),
                    "won": bool(won),
                    "market": market_map[pred_idx],
                    "season": fold.val_season,
                    "consensus_pct": float(result.consensus_score),
                    "entropy": float(result.entropy),
                    "n_agents": int(result.n_agents_contributing),
                }
            )

    # --- Aggregate metrics ---
    n_bets = len(all_bets)
    if n_bets == 0:
        return {
            "roi": -100.0, "n_bets": 0, "win_pct": 0.0, "profit": 0.0,
            "avg_consensus_pct": 0.0, "avg_entropy": 0.0,
            "avg_agents_contributing": 0.0, "agent_accuracy": {},
        }

    wins = sum(1 for b in all_bets if b["won"])
    total_staked = float(n_bets)  # flat 1-unit stake
    total_return = sum(b["odds"] for b in all_bets if b["won"])
    profit = total_return - total_staked
    roi = (profit / total_staked) * 100
    win_pct = (wins / n_bets) * 100
    avg_consensus_pct = float(np.mean([b["consensus_pct"] for b in all_bets]))
    avg_entropy = float(np.mean([b["entropy"] for b in all_bets]))
    avg_agents = float(np.mean([b["n_agents"] for b in all_bets]))

    # Compute per-agent win rates
    agent_win_rates = {}
    for aname, counts in agent_accuracy.items():
        if counts["total"] > 0:
            agent_win_rates[aname] = {
                "correct": counts["correct"],
                "total": counts["total"],
                "win_rate": round(counts["correct"] / counts["total"] * 100, 2),
            }

    return {
        "roi": round(roi, 2),
        "n_bets": n_bets,
        "win_pct": round(win_pct, 2),
        "profit": round(profit, 2),
        "avg_consensus_pct": round(avg_consensus_pct, 4),
        "avg_entropy": round(avg_entropy, 4),
        "avg_agents_contributing": round(avg_agents, 2),
        "agent_accuracy": agent_win_rates,
    }


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SharpEdge optimizer V3 (Agent Swarm + Bayesian Arbiter)"
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Max iterations (default 100)",
    )
    parser.add_argument(
        "--min-bets", type=int, default=30,
        help="Minimum bets for a valid config (default 30)",
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Stop after N iterations without improvement",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed",
    )
    parser.add_argument(
        "--mc-sims", type=int, default=5000,
        help="Monte Carlo simulations per match (default 5000)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("  SharpEdge Optimizer V3 — Agent Swarm + Bayesian Arbiter")
    logger.info("=" * 70)
    logger.info(f"Max iterations: {args.iterations}")
    logger.info(f"Min bets:       {args.min_bets}")
    logger.info(f"Patience:       {args.patience}")
    logger.info(f"MC simulations: {args.mc_sims}")

    agent_names = [
        "StatisticalAgent (DC + BVP)",
        "GradientAgent (XGB + CatBoost + LightGBM)",
        "OvRSpecialistAgent (3 binary classifiers)",
        "FormMomentumAgent (rolling form)",
        "H2HVenueAgent (head-to-head)",
        "MarketAgent (odds-only devigged)",
        "ContrarianAgent (fade favorites)",
    ]
    logger.info(f"Active agents:  {len(agent_names)}")
    for a in agent_names:
        logger.info(f"  - {a}")

    # ------------------------------------------------------------------
    # 1. Load data ONCE
    # ------------------------------------------------------------------
    logger.info("Loading historical match data...")
    t0 = time.time()
    df = load_historical_matches()
    logger.info(
        f"Loaded {len(df)} matches across {df['league'].nunique()} leagues, "
        f"{df['season'].nunique()} seasons in {time.time() - t0:.1f}s"
    )

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
    logger.info("Evaluating baseline config (Agent Swarm + Arbiter)...")
    t0 = time.time()
    best_metrics = run_backtest(best_cfg, folds)
    elapsed_baseline = time.time() - t0
    logger.info(
        f"Baseline: ROI={best_metrics['roi']:.2f}%, "
        f"bets={best_metrics['n_bets']}, "
        f"win%={best_metrics['win_pct']:.1f}%, "
        f"profit={best_metrics['profit']:.1f}u, "
        f"consensus={best_metrics['avg_consensus_pct']:.1%}, "
        f"entropy={best_metrics['avg_entropy']:.3f}, "
        f"agents/bet={best_metrics['avg_agents_contributing']:.1f} "
        f"({elapsed_baseline:.1f}s)"
    )

    # Log per-agent accuracy
    if best_metrics["agent_accuracy"]:
        logger.info("Per-agent accuracy (baseline):")
        for aname, stats in best_metrics["agent_accuracy"].items():
            logger.info(
                f"  {aname}: {stats['win_rate']:.1f}% "
                f"({stats['correct']}/{stats['total']})"
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
                f"consensus={metrics['avg_consensus_pct']:.1%}, "
                f"entropy={metrics['avg_entropy']:.3f} "
                f"({elapsed:.1f}s)"
            )
            # Log per-agent accuracy on improvement
            if metrics["agent_accuracy"]:
                for aname, stats in metrics["agent_accuracy"].items():
                    logger.info(
                        f"    {aname}: {stats['win_rate']:.1f}%"
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
        "version": "v3-agent-swarm-arbiter",
        "best_config": asdict(best_cfg),
        "best_metrics": best_metrics,
        "n_agents": len(agent_names),
        "agent_names": agent_names,
        "iterations": len(iteration_log) - 1,
        "log": iteration_log,
    }

    output_path = ROOT / "optimization_results_v3.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("  OPTIMIZATION V3 COMPLETE — Agent Swarm + Arbiter")
    logger.info("=" * 70)
    logger.info(f"Agents:              {len(agent_names)}")
    logger.info(f"Best ROI:            {best_metrics['roi']:.2f}%")
    logger.info(f"Best bets:           {best_metrics['n_bets']}")
    logger.info(f"Best win%:           {best_metrics['win_pct']:.1f}%")
    logger.info(f"Best profit:         {best_metrics['profit']:.1f}u")
    logger.info(f"Avg consensus:       {best_metrics['avg_consensus_pct']:.1%}")
    logger.info(f"Avg entropy:         {best_metrics['avg_entropy']:.3f}")
    logger.info(f"Avg agents/bet:      {best_metrics['avg_agents_contributing']:.1f}")
    logger.info(f"Results saved to {output_path}")

    # Print per-agent accuracy
    if best_metrics["agent_accuracy"]:
        logger.info("\nPer-agent accuracy (best config):")
        for aname, stats in best_metrics["agent_accuracy"].items():
            logger.info(
                f"  {aname}: {stats['win_rate']:.1f}% "
                f"({stats['correct']}/{stats['total']})"
            )

    # Print best config
    logger.info("\nBest configuration:")
    for k, v in asdict(best_cfg).items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
