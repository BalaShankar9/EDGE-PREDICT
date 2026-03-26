"""Tests for LineMovementTracker."""
from datetime import datetime, timedelta

import pytest

from sharpedge.intelligence.line_movement import (
    LineMovementTracker, LineSnapshot, LineMovement,
)


@pytest.fixture
def tracker():
    return LineMovementTracker()


def _snap(match_id, ts, odds, bookmaker="bet365", market="1X2"):
    return LineSnapshot(match_id=match_id, timestamp=ts, bookmaker=bookmaker, market=market, odds=odds)


class TestLineMovementDefaults:
    def test_no_snapshots_returns_defaults(self, tracker):
        feats = tracker.compute_features("unknown_match")
        assert feats["lm_home_movement"] == 0.0
        assert feats["lm_away_movement"] == 0.0
        assert feats["lm_steam_move"] == 0.0
        assert feats["lm_reverse_line_move"] == 0.0
        assert feats["lm_total_movement"] == 0.0

    def test_single_snapshot_returns_defaults(self, tracker):
        tracker.record_snapshot(_snap("m1", datetime(2026, 1, 1), {"home": 2.0, "away": 2.0}))
        feats = tracker.compute_features("m1")
        assert feats["lm_home_movement"] == 0.0


class TestLineMovementNormal:
    def test_movement_computed_correctly(self, tracker):
        t0 = datetime(2026, 1, 1, 12, 0)
        tracker.record_snapshot(_snap("m1", t0, {"home": 2.0, "draw": 3.5, "away": 4.0}))
        tracker.record_snapshot(_snap("m1", t0 + timedelta(hours=2), {"home": 1.8, "draw": 3.5, "away": 4.5}))
        feats = tracker.compute_features("m1")
        # home went from 2.0 (0.5 impl) to 1.8 (0.556 impl) => +0.056
        assert feats["lm_home_movement"] == pytest.approx(1/1.8 - 1/2.0, abs=1e-4)
        assert feats["lm_total_movement"] > 0

    def test_steam_move_detected(self, tracker):
        t0 = datetime(2026, 1, 1)
        tracker.record_snapshot(_snap("m1", t0, {"home": 2.5, "away": 2.5}))
        tracker.record_snapshot(_snap("m1", t0 + timedelta(hours=1), {"home": 2.0, "away": 3.0}))
        feats = tracker.compute_features("m1")
        assert feats["lm_steam_move"] == 1.0

    def test_no_steam_move_when_small_change(self, tracker):
        t0 = datetime(2026, 1, 1)
        tracker.record_snapshot(_snap("m1", t0, {"home": 2.00, "away": 2.00}))
        tracker.record_snapshot(_snap("m1", t0 + timedelta(hours=1), {"home": 1.98, "away": 2.02}))
        feats = tracker.compute_features("m1")
        assert feats["lm_steam_move"] == 0.0


class TestLineMovementEdgeCases:
    def test_reverse_line_move(self, tracker):
        t0 = datetime(2026, 1, 1)
        # Home is favorite (lower odds = higher impl prob), then becomes less favored
        tracker.record_snapshot(_snap("m1", t0, {"home": 1.5, "away": 3.0}))
        tracker.record_snapshot(_snap("m1", t0 + timedelta(hours=1), {"home": 1.8, "away": 2.4}))
        mv = tracker.get_movement("m1")
        assert mv is not None
        assert mv.reverse_line_move is True

    def test_multiple_snapshots_uses_first_and_last(self, tracker):
        t0 = datetime(2026, 1, 1)
        tracker.record_snapshot(_snap("m1", t0, {"home": 2.0, "away": 2.0}))
        tracker.record_snapshot(_snap("m1", t0 + timedelta(hours=1), {"home": 1.9, "away": 2.1}))
        tracker.record_snapshot(_snap("m1", t0 + timedelta(hours=2), {"home": 1.7, "away": 2.5}))
        mv = tracker.get_movement("m1")
        assert mv.opening_odds == {"home": 2.0, "away": 2.0}
        assert mv.current_odds == {"home": 1.7, "away": 2.5}
