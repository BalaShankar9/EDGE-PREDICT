"""Tests for TravelFatigueModeler."""
import pytest

from sharpedge.intelligence.travel_fatigue import TravelFatigueModeler


@pytest.fixture
def modeler():
    return TravelFatigueModeler()


class TestTravelDefaults:
    def test_no_travel_zero_fatigue(self, modeler):
        feats = modeler.compute_features("football")
        assert feats["tf_fatigue_index"] == 0.0
        assert feats["tf_timezone_diff"] == 0
        assert feats["tf_back_to_back"] == 0.0
        assert feats["tf_days_rest"] == 3


class TestTravelNormal:
    def test_timezone_increases_fatigue(self, modeler):
        feats = modeler.compute_features("football", timezone_diff=3)
        assert feats["tf_fatigue_index"] > 0
        assert feats["tf_timezone_diff"] == 3

    def test_back_to_back_increases_fatigue(self, modeler):
        feats = modeler.compute_features("basketball", back_to_back=True)
        assert feats["tf_fatigue_index"] >= 0.3

    def test_low_rest_increases_fatigue(self, modeler):
        feats = modeler.compute_features("football", days_rest=1)
        assert feats["tf_fatigue_index"] > 0

    def test_long_travel_increases_fatigue(self, modeler):
        feats = modeler.compute_features("football", travel_km=5000)
        assert feats["tf_fatigue_index"] > 0


class TestTravelEdgeCases:
    def test_max_fatigue_capped(self, modeler):
        feats = modeler.compute_features("football", timezone_diff=10, back_to_back=True,
                                         days_rest=0, travel_km=10000)
        assert feats["tf_fatigue_index"] <= 1.0

    def test_negative_timezone_handled(self, modeler):
        feats = modeler.compute_features("football", timezone_diff=-3)
        assert feats["tf_timezone_diff"] == 3  # abs value
