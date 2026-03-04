"""Staking calculators for bankroll management.

Provides flat stake, Kelly criterion, and controlled martingale.
"""
import math


class StakingCalculator:
    """Calculates optimal stake sizes."""

    def flat_stake(self, bankroll: float, pct: float = 0.03) -> float:
        """Flat stake: fixed percentage of bankroll.

        Parameters
        ----------
        bankroll : current bankroll
        pct : fraction to stake (default 3%)
        """
        return round(bankroll * pct, 2)

    def kelly(self, bankroll: float, prob: float, odds: float) -> float:
        """Kelly criterion stake.

        Kelly fraction = (p * (odds - 1) - (1 - p)) / (odds - 1)
        We use fractional Kelly (25%) to reduce variance.

        Parameters
        ----------
        bankroll : current bankroll
        prob : estimated win probability
        odds : decimal odds
        """
        if odds <= 1 or prob <= 0 or prob >= 1:
            return 0.0

        kelly_fraction = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        if kelly_fraction <= 0:
            return 0.0

        # Fractional Kelly (25%) for safety
        safe_fraction = kelly_fraction * 0.25
        # Cap at 5% of bankroll
        safe_fraction = min(safe_fraction, 0.05)

        return round(bankroll * safe_fraction, 2)

    def martingale(
        self,
        bankroll: float,
        step: int,
        base_pct: float = 0.02,
        progression: float = 1.5,
        max_steps: int = 4,
    ) -> float:
        """Controlled martingale with progression cap.

        Parameters
        ----------
        bankroll : current bankroll
        step : current losing streak step (0 = first bet)
        base_pct : base stake as fraction of bankroll
        progression : multiplier per step
        max_steps : maximum progression steps
        """
        effective_step = min(step, max_steps)
        multiplier = progression ** effective_step
        stake = bankroll * base_pct * multiplier
        # Never exceed 10% of bankroll
        max_stake = bankroll * 0.10
        return round(min(stake, max_stake), 2)
