"""Tests for AntifragileStaking engine."""
import pytest
from sharpedge.execution.staking import AntifragileStaking, StakeRecommendation


class TestKellyStake:
    def test_positive_edge(self):
        s = AntifragileStaking()
        # prob=0.6, odds=2.0 => b=1, kelly = (0.6*1 - 0.4)/1 = 0.2
        assert pytest.approx(s.kelly_stake(0.6, 2.0), abs=1e-6) == 0.2

    def test_negative_edge_returns_zero(self):
        s = AntifragileStaking()
        # prob=0.3, odds=2.0 => kelly = (0.3 - 0.7)/1 = -0.4 -> 0
        assert s.kelly_stake(0.3, 2.0) == 0.0

    def test_odds_at_one_returns_zero(self):
        s = AntifragileStaking()
        assert s.kelly_stake(0.9, 1.0) == 0.0

    def test_fair_odds_returns_zero(self):
        s = AntifragileStaking()
        # prob=0.5, odds=2.0 => kelly = (0.5 - 0.5)/1 = 0
        assert s.kelly_stake(0.5, 2.0) == 0.0


class TestComputeStake:
    def test_normal_stake_within_max(self):
        s = AntifragileStaking(bankroll=1000.0)
        rec = s.compute_stake("m1", 0.6, 2.0)
        assert isinstance(rec, StakeRecommendation)
        assert 0 <= rec.stake_pct <= s.max_bet_pct
        assert rec.method == "kelly"

    def test_stake_units_match_pct(self):
        s = AntifragileStaking(bankroll=1000.0)
        rec = s.compute_stake("m1", 0.7, 2.5)
        assert pytest.approx(rec.stake_units, abs=0.01) == rec.stake_pct * 1000.0

    def test_max_bet_pct_enforced(self):
        s = AntifragileStaking(bankroll=1000.0, kelly_fraction=1.0, max_bet_pct=0.03)
        # High edge => raw Kelly high, but capped at 3%
        rec = s.compute_stake("m1", 0.9, 2.0)
        assert rec.stake_pct <= 0.03

    def test_daily_limit_enforced(self):
        s = AntifragileStaking(bankroll=1000.0, max_daily_pct=0.10)
        # Simulate 9% already staked today
        s._daily_staked = 0.09
        rec = s.compute_stake("m1", 0.7, 2.5)
        assert rec.stake_pct <= 0.01 + 1e-9  # only 1% remaining

    def test_negative_edge_zero_stake(self):
        s = AntifragileStaking()
        rec = s.compute_stake("m1", 0.3, 2.0)
        assert rec.stake_pct == 0.0
        assert rec.stake_units == 0.0


class TestDrawdownProtection:
    def test_level_0_no_drawdown(self):
        s = AntifragileStaking(bankroll=1000.0)
        assert s.drawdown_level == 0

    def test_level_1_at_5pct(self):
        s = AntifragileStaking(bankroll=950.0)
        s._peak_bankroll = 1000.0
        assert s.drawdown_level == 1

    def test_level_2_at_10pct(self):
        s = AntifragileStaking(bankroll=900.0)
        s._peak_bankroll = 1000.0
        assert s.drawdown_level == 2

    def test_level_4_stops_betting(self):
        s = AntifragileStaking(bankroll=800.0)
        s._peak_bankroll = 1000.0
        assert s.drawdown_level == 4
        rec = s.compute_stake("m1", 0.7, 2.5)
        assert rec.stake_pct == 0.0
        assert rec.method == "STOPPED"

    def test_drawdown_reduces_stake(self):
        s_normal = AntifragileStaking(bankroll=1000.0)
        rec_normal = s_normal.compute_stake("m1", 0.6, 2.0)

        s_dd = AntifragileStaking(bankroll=900.0)
        s_dd._peak_bankroll = 1000.0  # 10% drawdown => level 2, mult=0.5
        rec_dd = s_dd.compute_stake("m1", 0.6, 2.0)

        assert rec_dd.stake_pct < rec_normal.stake_pct


class TestRecordBet:
    def test_bankroll_updates(self):
        s = AntifragileStaking(bankroll=1000.0)
        s.record_bet(0.02, True, 20.0)
        assert s.bankroll == 1020.0
        assert s._peak_bankroll == 1020.0

    def test_daily_staked_accumulates(self):
        s = AntifragileStaking(bankroll=1000.0)
        s.record_bet(0.02, True, 20.0)
        s.record_bet(0.03, False, -30.0)
        assert pytest.approx(s._daily_staked) == 0.05

    def test_reset_daily(self):
        s = AntifragileStaking(bankroll=1000.0)
        s._daily_staked = 0.08
        s.reset_daily()
        assert s._daily_staked == 0.0

    def test_recent_results_capped_at_50(self):
        s = AntifragileStaking()
        for _ in range(60):
            s.record_bet(0.01, True, 5.0)
        assert len(s._recent_results) == 50
