import pytest
from sharpedge.ml.banker.staking import StakingCalculator


def test_flat_stake():
    sc = StakingCalculator()
    stake = sc.flat_stake(1000, pct=0.03)
    assert stake == 30.0


def test_kelly_positive_edge():
    sc = StakingCalculator()
    stake = sc.kelly(bankroll=1000, prob=0.60, odds=2.0)
    assert stake > 0
    assert stake <= 50  # Max 5% of bankroll


def test_kelly_no_edge():
    sc = StakingCalculator()
    stake = sc.kelly(bankroll=1000, prob=0.40, odds=2.0)
    assert stake == 0  # Negative Kelly = no bet


def test_martingale_step_0():
    sc = StakingCalculator()
    stake = sc.martingale(bankroll=1000, step=0)
    assert stake == 20.0  # 2% of 1000


def test_martingale_step_4_cap():
    sc = StakingCalculator()
    stake = sc.martingale(bankroll=1000, step=10)  # Beyond max_steps
    # Should be capped
    assert stake <= 100  # Max 10% of bankroll


def test_martingale_progression():
    sc = StakingCalculator()
    s0 = sc.martingale(bankroll=1000, step=0)
    s1 = sc.martingale(bankroll=1000, step=1)
    s2 = sc.martingale(bankroll=1000, step=2)
    assert s1 > s0
    assert s2 > s1
