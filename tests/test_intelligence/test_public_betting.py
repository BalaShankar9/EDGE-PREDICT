"""Tests for PublicBettingTracker."""
import pytest

from sharpedge.intelligence.public_betting import PublicBettingTracker, PublicBettingData


@pytest.fixture
def tracker():
    return PublicBettingTracker()


class TestPublicBettingDefaults:
    def test_unknown_match_returns_defaults(self, tracker):
        feats = tracker.compute_features("unknown")
        assert feats["pb_public_pct_home"] == 0.5
        assert feats["pb_public_pct_away"] == 0.5
        assert feats["pb_public_heavy_side"] == 0.0
        assert feats["pb_contrarian_signal"] == 0.0


class TestPublicBettingNormal:
    def test_normal_betting_percentages(self, tracker):
        tracker.record(PublicBettingData("m1", {"home": 0.60, "draw": 0.20, "away": 0.20}))
        feats = tracker.compute_features("m1")
        assert feats["pb_public_pct_home"] == 0.60
        assert feats["pb_public_pct_away"] == 0.20
        assert feats["pb_public_heavy_side"] == 0.60
        assert feats["pb_contrarian_signal"] == 0.0  # not > 0.70

    def test_contrarian_signal_triggered(self, tracker):
        tracker.record(PublicBettingData("m1", {"home": 0.80, "draw": 0.10, "away": 0.10}))
        feats = tracker.compute_features("m1")
        assert feats["pb_contrarian_signal"] == 1.0

    def test_contrarian_at_boundary(self, tracker):
        tracker.record(PublicBettingData("m1", {"home": 0.70, "draw": 0.15, "away": 0.15}))
        feats = tracker.compute_features("m1")
        assert feats["pb_contrarian_signal"] == 0.0  # boundary: not > 0.70


class TestPublicBettingEdgeCases:
    def test_record_overwrites(self, tracker):
        tracker.record(PublicBettingData("m1", {"home": 0.50, "away": 0.50}))
        tracker.record(PublicBettingData("m1", {"home": 0.80, "away": 0.20}))
        feats = tracker.compute_features("m1")
        assert feats["pb_public_pct_home"] == 0.80

    def test_missing_outcome_key(self, tracker):
        tracker.record(PublicBettingData("m1", {"home": 0.90}))
        feats = tracker.compute_features("m1")
        assert feats["pb_public_pct_away"] == 0.5  # default
