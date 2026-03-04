"""Historical backtesting engine.

Simulates the full prediction pipeline on historical data:
1. For each matchday in the test period:
   a. Build features using only data available BEFORE that matchday
   b. Generate predictions from trained models
   c. Apply banker filter
   d. Record picks + actual results
2. Compute cumulative P&L, win rate by tier, CLV, drawdown
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sharpedge.ml.banker.filter import BankerFilter, Pick
from sharpedge.ml.banker.tiers import TierAssigner
from sharpedge.ml.banker.staking import StakingCalculator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest results."""

    total_picks: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    picks_by_tier: dict = field(default_factory=dict)
    monthly_results: list = field(default_factory=list)
    pick_log: list = field(default_factory=list)


class BacktestEngine:
    """Simulates predictions on historical data."""

    def __init__(
        self,
        banker_filter: BankerFilter | None = None,
        tier_assigner: TierAssigner | None = None,
        staking: StakingCalculator | None = None,
        initial_bankroll: float = 1000.0,
    ):
        self.banker_filter = banker_filter or BankerFilter()
        self.tier_assigner = tier_assigner or TierAssigner()
        self.staking = staking or StakingCalculator()
        self.initial_bankroll = initial_bankroll

    def run(
        self,
        predictions: list[dict],
        actuals: pd.DataFrame,
    ) -> BacktestResult:
        """Run backtest on pre-computed predictions.

        Parameters
        ----------
        predictions : list of prediction dicts (same format as BankerFilter input)
            Each must include: match_id, model_prob, model_spread, best_odds,
            bookmaker, market, meta_agreement, risk_flags,
            home_team, away_team, league, match_date
        actuals : DataFrame with match_id, FTR (H/D/A), FTHG, FTAG columns
            Used to determine if picks won or lost.

        Returns
        -------
        BacktestResult with full P&L analysis
        """
        result = BacktestResult()
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0

        # Apply banker filter
        picks = self.banker_filter.filter(predictions)
        picks = self.tier_assigner.assign(picks)

        if not picks:
            logger.warning("No picks survived the banker filter.")
            return result

        # Create actuals lookup
        actual_map = {}
        for _, row in actuals.iterrows():
            actual_map[row["match_id"]] = {
                "result": row.get("FTR", ""),
                "home_goals": row.get("FTHG", 0),
                "away_goals": row.get("FTAG", 0),
            }

        wins = 0
        losses = 0
        total_staked = 0.0
        total_return = 0.0
        tier_stats: dict[str, dict] = {}

        for pick in picks:
            actual = actual_map.get(pick.match_id)
            if not actual:
                continue

            # Determine if pick won
            won = False
            if pick.market == "1x2_home" and actual["result"] == "H":
                won = True
            elif pick.market == "1x2_draw" and actual["result"] == "D":
                won = True
            elif pick.market == "1x2_away" and actual["result"] == "A":
                won = True
            elif pick.market == "over_25" and (actual["home_goals"] + actual["away_goals"]) > 2.5:
                won = True
            elif pick.market == "btts_yes" and actual["home_goals"] > 0 and actual["away_goals"] > 0:
                won = True

            # Stake calculation
            stake = self.staking.flat_stake(bankroll)
            total_staked += stake

            if won:
                wins += 1
                profit = stake * (pick.best_odds - 1)
                total_return += stake * pick.best_odds
                bankroll += profit
            else:
                losses += 1
                profit = -stake
                bankroll -= stake

            # Track drawdown
            peak_bankroll = max(peak_bankroll, bankroll)
            current_drawdown = (
                (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            )
            max_drawdown = max(max_drawdown, current_drawdown)

            # Tier stats
            tier = pick.tier or "unknown"
            if tier not in tier_stats:
                tier_stats[tier] = {
                    "count": 0,
                    "wins": 0,
                    "total_staked": 0.0,
                    "total_return": 0.0,
                }
            tier_stats[tier]["count"] += 1
            if won:
                tier_stats[tier]["wins"] += 1
                tier_stats[tier]["total_return"] += stake * pick.best_odds
            tier_stats[tier]["total_staked"] += stake

            # Log
            result.pick_log.append(
                {
                    "match_id": pick.match_id,
                    "market": pick.market,
                    "tier": pick.tier,
                    "model_prob": pick.model_prob,
                    "best_odds": pick.best_odds,
                    "edge": pick.edge,
                    "won": won,
                    "stake": stake,
                    "profit": profit,
                    "bankroll_after": bankroll,
                }
            )

        # Compile results
        total = wins + losses
        result.total_picks = total
        result.win_rate = wins / total if total > 0 else 0.0
        result.roi = (
            (total_return - total_staked) / total_staked if total_staked > 0 else 0.0
        )
        result.total_profit = bankroll - self.initial_bankroll
        result.max_drawdown = max_drawdown

        # Tier breakdown
        for tier, stats in tier_stats.items():
            result.picks_by_tier[tier] = {
                "count": stats["count"],
                "wins": stats["wins"],
                "win_rate": (
                    stats["wins"] / stats["count"] if stats["count"] > 0 else 0
                ),
                "roi": (
                    (stats["total_return"] - stats["total_staked"])
                    / stats["total_staked"]
                    if stats["total_staked"] > 0
                    else 0
                ),
            }

        logger.info(
            f"Backtest: {total} picks, {result.win_rate:.1%} win rate, "
            f"{result.roi:.1%} ROI"
        )
        return result
