"""Tests for MotivationScorer."""
import pytest

from sharpedge.intelligence.motivation_context import MotivationScorer


@pytest.fixture
def scorer():
    return MotivationScorer()


class TestMotivationDefaults:
    def test_baseline_motivation(self, scorer):
        feats = scorer.compute_features("football")
        assert feats["mot_home_motivation"] == 0.5
        assert feats["mot_away_motivation"] == 0.5
        assert feats["mot_motivation_diff"] == 0.0
        assert feats["mot_derby_flag"] == 0.0
        assert feats["mot_dead_rubber_flag"] == 0.0


class TestMotivationNormal:
    def test_derby_boosts_both(self, scorer):
        feats = scorer.compute_features("football", is_derby=True)
        assert feats["mot_home_motivation"] == 0.6
        assert feats["mot_away_motivation"] == 0.6
        assert feats["mot_derby_flag"] == 1.0

    def test_dead_rubber_reduces_both(self, scorer):
        feats = scorer.compute_features("football", is_dead_rubber=True)
        assert feats["mot_home_motivation"] == 0.35
        assert feats["mot_away_motivation"] == 0.35
        assert feats["mot_dead_rubber_flag"] == 1.0

    def test_relegation_battle_position_dependent(self, scorer):
        feats = scorer.compute_features("football", is_relegation_battle=True,
                                        league_position_home=18, league_position_away=5)
        assert feats["mot_home_motivation"] > 0.5
        assert feats["mot_away_motivation"] == 0.5  # position 5 not > 15

    def test_title_race(self, scorer):
        feats = scorer.compute_features("football", is_title_race=True,
                                        league_position_home=1, league_position_away=2)
        assert feats["mot_home_motivation"] == 0.6
        assert feats["mot_away_motivation"] == 0.6


class TestMotivationEdgeCases:
    def test_motivation_clamped_above_minimum(self, scorer):
        # dead rubber with no other flags => 0.35, check it stays >= 0.1
        feats = scorer.compute_features("football", is_dead_rubber=True)
        assert feats["mot_home_motivation"] >= 0.1
        assert feats["mot_away_motivation"] >= 0.1

    def test_combined_flags(self, scorer):
        feats = scorer.compute_features("football", is_derby=True, is_playoff=True,
                                        is_title_race=True, league_position_home=1,
                                        league_position_away=2)
        assert feats["mot_home_motivation"] <= 1.0
        assert feats["mot_playoff_flag"] == 1.0

    def test_revenge_game_boosts_away(self, scorer):
        feats = scorer.compute_features("football", is_revenge_game=True)
        assert feats["mot_away_motivation"] > feats["mot_home_motivation"]
