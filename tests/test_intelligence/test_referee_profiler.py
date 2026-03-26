"""Tests for RefereeProfiler."""
import pytest

from sharpedge.intelligence.referee_profiler import RefereeProfiler


@pytest.fixture
def profiler():
    return RefereeProfiler()


class TestRefereeDefaults:
    def test_none_referee_returns_defaults(self, profiler):
        feats = profiler.compute_features(None)
        assert feats["ref_home_bias"] == 0.5
        assert feats["ref_cards_per_match"] == 4.0
        assert feats["ref_fouls_per_match"] == 22.0
        assert feats["ref_penalty_rate"] == 0.0
        assert feats["ref_strictness"] == 0.5

    def test_unknown_referee_returns_defaults(self, profiler):
        feats = profiler.compute_features("Unknown Ref")
        assert feats["ref_home_bias"] == 0.5


class TestRefereeNormal:
    def test_single_match_profile(self, profiler):
        profiler.update("Mike Dean", home_win=True, cards=6, fouls=24, penalties=1)
        feats = profiler.compute_features("Mike Dean")
        assert feats["ref_home_bias"] == 1.0
        assert feats["ref_cards_per_match"] == 6.0
        assert feats["ref_fouls_per_match"] == 24.0
        assert feats["ref_penalty_rate"] == 1.0
        assert feats["ref_strictness"] == 1.0  # 6/6 = 1.0

    def test_multiple_matches_averaged(self, profiler):
        profiler.update("Mike Dean", home_win=True, cards=4, fouls=20, penalties=0)
        profiler.update("Mike Dean", home_win=False, cards=6, fouls=24, penalties=1)
        feats = profiler.compute_features("Mike Dean")
        assert feats["ref_home_bias"] == pytest.approx(0.5)
        assert feats["ref_cards_per_match"] == pytest.approx(5.0)
        assert feats["ref_penalty_rate"] == pytest.approx(0.5)


class TestRefereeEdgeCases:
    def test_strictness_capped_at_one(self, profiler):
        profiler.update("Strict Ref", home_win=True, cards=12)
        feats = profiler.compute_features("Strict Ref")
        assert feats["ref_strictness"] == 1.0  # capped

    def test_lenient_ref(self, profiler):
        profiler.update("Lenient Ref", home_win=False, cards=1, fouls=10)
        feats = profiler.compute_features("Lenient Ref")
        assert feats["ref_strictness"] == pytest.approx(1.0 / 6.0, abs=1e-4)
