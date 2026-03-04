"""Tests for result resolution logic."""
import pytest
from sharpedge.pipeline.resolver import resolve_pick


def test_resolve_home_win_correct():
    result = resolve_pick(pick_market="1x2_home", home_goals=2, away_goals=1, best_odds=1.85, stake=1.0)
    assert result["result"] == "win"
    assert result["profit_loss"] == pytest.approx(0.85)


def test_resolve_home_win_incorrect():
    result = resolve_pick(pick_market="1x2_home", home_goals=1, away_goals=2, best_odds=1.85, stake=1.0)
    assert result["result"] == "loss"
    assert result["profit_loss"] == pytest.approx(-1.0)


def test_resolve_draw():
    result = resolve_pick(pick_market="1x2_draw", home_goals=1, away_goals=1, best_odds=3.20, stake=1.0)
    assert result["result"] == "win"
    assert result["profit_loss"] == pytest.approx(2.20)


def test_resolve_over_25_win():
    result = resolve_pick(pick_market="over_25", home_goals=2, away_goals=1, best_odds=1.90, stake=1.0)
    assert result["result"] == "win"
    assert result["profit_loss"] == pytest.approx(0.90)


def test_resolve_over_25_loss():
    result = resolve_pick(pick_market="over_25", home_goals=1, away_goals=1, best_odds=1.90, stake=1.0)
    assert result["result"] == "loss"


def test_resolve_btts_yes():
    result = resolve_pick(pick_market="btts_yes", home_goals=2, away_goals=1, best_odds=1.75, stake=1.0)
    assert result["result"] == "win"


def test_resolve_btts_no():
    result = resolve_pick(pick_market="btts_yes", home_goals=2, away_goals=0, best_odds=1.75, stake=1.0)
    assert result["result"] == "loss"


def test_resolve_void_when_no_goals():
    result = resolve_pick(pick_market="1x2_home", home_goals=None, away_goals=None, best_odds=1.85, stake=1.0)
    assert result["result"] == "void"
    assert result["profit_loss"] == 0.0
