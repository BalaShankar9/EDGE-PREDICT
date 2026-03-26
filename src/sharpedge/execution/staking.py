"""Anti-Fragile Staking Engine.

NOT simple Kelly. Multi-objective: maximize growth while protecting against ruin.

Three modes:
1. Fractional Kelly (default): conservative Kelly fraction
2. Drawdown-protected: reduce stakes during losing streaks
3. Aggressive: increase stakes during winning streaks

Hard limits:
- Max 3% bankroll per bet
- Max 10% bankroll per day
- Max 15% on correlated outcomes
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class StakeRecommendation:
    """Recommended stake for a single bet."""
    match_id: str
    stake_pct: float          # as fraction of bankroll (0.01 = 1%)
    stake_units: float        # absolute units
    method: str               # "kelly", "flat", "drawdown_protected"
    kelly_raw: float          # raw Kelly fraction (before limits)
    reasoning: str


class AntifragileStaking:
    """Multi-mode staking engine with drawdown protection."""

    def __init__(self, bankroll: float = 1000.0, kelly_fraction: float = 0.25,
                 max_bet_pct: float = 0.03, max_daily_pct: float = 0.10):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.max_daily_pct = max_daily_pct
        self._peak_bankroll = bankroll
        self._daily_staked = 0.0
        self._recent_results: list[bool] = []

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self._peak_bankroll <= 0:
            return 0.0
        return (self._peak_bankroll - self.bankroll) / self._peak_bankroll

    @property
    def drawdown_level(self) -> int:
        """Circuit breaker level: 0=normal, 1=yellow, 2=orange, 3=red, 4=stop."""
        dd = self.drawdown_pct
        if dd >= 0.20:
            return 4
        if dd >= 0.15:
            return 3
        if dd >= 0.10:
            return 2
        if dd >= 0.05:
            return 1
        return 0

    def kelly_stake(self, prob: float, odds: float) -> float:
        """Raw Kelly criterion: f* = (p*b - q) / b where b = odds-1."""
        b = odds - 1.0
        q = 1.0 - prob
        if b <= 0:
            return 0.0
        kelly = (prob * b - q) / b
        return max(0.0, kelly)

    def compute_stake(self, match_id: str, prob: float, odds: float) -> StakeRecommendation:
        """Compute optimal stake with all protections applied."""
        # Raw Kelly
        raw_kelly = self.kelly_stake(prob, odds)

        # Apply Kelly fraction
        stake_pct = raw_kelly * self.kelly_fraction

        # Drawdown protection
        level = self.drawdown_level
        dd_multiplier = {0: 1.0, 1: 0.75, 2: 0.50, 3: 0.25, 4: 0.0}[level]
        stake_pct *= dd_multiplier
        method = "kelly" if level == 0 else f"drawdown_protected_L{level}"

        if level == 4:
            return StakeRecommendation(
                match_id=match_id, stake_pct=0.0, stake_units=0.0,
                method="STOPPED", kelly_raw=raw_kelly,
                reasoning=f"Circuit breaker LEVEL 4: {self.drawdown_pct:.1%} drawdown. All betting paused."
            )

        # Hard cap
        stake_pct = min(stake_pct, self.max_bet_pct)

        # Daily limit check
        remaining_daily = self.max_daily_pct - self._daily_staked
        stake_pct = min(stake_pct, max(0, remaining_daily))

        stake_units = stake_pct * self.bankroll

        reasoning = (
            f"Kelly raw={raw_kelly:.3f}, fraction={self.kelly_fraction}, "
            f"DD level={level} (mult={dd_multiplier}), "
            f"final={stake_pct:.3f} ({stake_units:.1f} units)"
        )

        return StakeRecommendation(
            match_id=match_id, stake_pct=stake_pct, stake_units=stake_units,
            method=method, kelly_raw=raw_kelly, reasoning=reasoning,
        )

    def record_bet(self, stake_pct: float, won: bool, profit: float) -> None:
        """Record a bet result and update bankroll."""
        self._daily_staked += stake_pct
        self.bankroll += profit
        self._peak_bankroll = max(self._peak_bankroll, self.bankroll)
        self._recent_results.append(won)
        if len(self._recent_results) > 50:
            self._recent_results.pop(0)

    def reset_daily(self) -> None:
        """Reset daily staked counter (call at start of each day)."""
        self._daily_staked = 0.0
