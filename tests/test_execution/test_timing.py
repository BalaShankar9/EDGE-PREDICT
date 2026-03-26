"""Tests for BetTimingOptimizer."""
import pytest
from sharpedge.execution.timing import BetTimingOptimizer


class TestOptimalBetTime:
    def test_football_48h(self):
        t = BetTimingOptimizer()
        assert t.optimal_bet_time("football") == 48

    def test_tennis_4h(self):
        t = BetTimingOptimizer()
        assert t.optimal_bet_time("tennis") == 4

    def test_basketball_1h(self):
        t = BetTimingOptimizer()
        assert t.optimal_bet_time("basketball") == 1

    def test_american_football_72h(self):
        t = BetTimingOptimizer()
        assert t.optimal_bet_time("american_football") == 72

    def test_unknown_sport_defaults_24h(self):
        t = BetTimingOptimizer()
        assert t.optimal_bet_time("curling") == 24

    def test_league_param_accepted(self):
        t = BetTimingOptimizer()
        assert t.optimal_bet_time("football", "EPL") == 48


class TestComputeFeatures:
    def test_perfect_timing_score_one(self):
        t = BetTimingOptimizer()
        feats = t.compute_features("football", 48.0)
        assert feats["bt_timing_score"] == pytest.approx(1.0)

    def test_zero_hours_timing_score(self):
        t = BetTimingOptimizer()
        feats = t.compute_features("football", 0.0)
        assert feats["bt_timing_score"] == pytest.approx(0.0)

    def test_early_bet_flag(self):
        t = BetTimingOptimizer()
        # football optimal=48, early threshold=72
        feats = t.compute_features("football", 80.0)
        assert feats["bt_early_bet"] == 1.0
        assert feats["bt_late_bet"] == 0.0

    def test_late_bet_flag(self):
        t = BetTimingOptimizer()
        # football optimal=48, late threshold=14.4
        feats = t.compute_features("football", 5.0)
        assert feats["bt_late_bet"] == 1.0
        assert feats["bt_early_bet"] == 0.0

    def test_all_keys_present(self):
        t = BetTimingOptimizer()
        feats = t.compute_features("tennis", 3.0)
        expected_keys = {"bt_hours_to_match", "bt_optimal_hours", "bt_timing_score",
                         "bt_early_bet", "bt_late_bet"}
        assert set(feats.keys()) == expected_keys

    def test_timing_score_non_negative(self):
        t = BetTimingOptimizer()
        # Very far from optimal
        feats = t.compute_features("basketball", 100.0)
        assert feats["bt_timing_score"] >= 0.0
