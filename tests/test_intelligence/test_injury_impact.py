"""Tests for InjuryImpactModeler."""
import pytest

from sharpedge.intelligence.injury_impact import InjuryImpactModeler


@pytest.fixture
def modeler():
    return InjuryImpactModeler()


class TestInjuryDefaults:
    def test_no_injuries_returns_zeros(self, modeler):
        feats = modeler.compute_features()
        assert feats["inj_home_impact"] == 0.0
        assert feats["inj_away_impact"] == 0.0
        assert feats["inj_impact_diff"] == 0.0
        assert feats["inj_home_key_player_out"] == 0.0
        assert feats["inj_away_key_player_out"] == 0.0

    def test_empty_lists_returns_zeros(self, modeler):
        feats = modeler.compute_features([], [])
        assert feats["inj_home_impact"] == 0.0


class TestInjuryNormal:
    def test_single_injury(self, modeler):
        home_inj = [{"player": "Star", "minutes_share": 0.20, "quality_rating": 80, "status": "out"}]
        feats = modeler.compute_features(home_injuries=home_inj)
        # 0.20 * 0.80 * 1.0 = 0.16
        assert feats["inj_home_impact"] == pytest.approx(0.16)
        assert feats["inj_home_key_player_out"] == 1.0

    def test_doubtful_reduced_impact(self, modeler):
        inj = [{"player": "Star", "minutes_share": 0.20, "quality_rating": 80, "status": "doubtful"}]
        feats = modeler.compute_features(home_injuries=inj)
        # 0.20 * 0.80 * 0.7 = 0.112
        assert feats["inj_home_impact"] == pytest.approx(0.112)
        assert feats["inj_home_key_player_out"] == 0.0  # only "out" triggers

    def test_impact_diff_positive_means_home_advantage(self, modeler):
        away_inj = [{"player": "Star", "minutes_share": 0.30, "quality_rating": 90, "status": "out"}]
        feats = modeler.compute_features(away_injuries=away_inj)
        assert feats["inj_impact_diff"] > 0  # away impact > home impact


class TestInjuryEdgeCases:
    def test_impact_capped_at_one(self, modeler):
        # Many high-impact injuries
        injuries = [
            {"player": f"Player{i}", "minutes_share": 0.20, "quality_rating": 90, "status": "out"}
            for i in range(10)
        ]
        feats = modeler.compute_features(home_injuries=injuries)
        assert feats["inj_home_impact"] <= 1.0

    def test_questionable_low_impact(self, modeler):
        inj = [{"player": "P1", "minutes_share": 0.10, "quality_rating": 50, "status": "questionable"}]
        feats = modeler.compute_features(home_injuries=inj)
        # 0.10 * 0.50 * 0.3 = 0.015
        assert feats["inj_home_impact"] == pytest.approx(0.015)

    def test_key_player_threshold(self, modeler):
        # minutes_share exactly 0.15 should NOT trigger (need > 0.15)
        inj = [{"player": "P1", "minutes_share": 0.15, "quality_rating": 80, "status": "out"}]
        feats = modeler.compute_features(home_injuries=inj)
        assert feats["inj_home_key_player_out"] == 0.0
