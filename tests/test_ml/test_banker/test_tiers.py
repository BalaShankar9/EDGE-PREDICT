import pytest
from sharpedge.ml.banker.filter import Pick
from sharpedge.ml.banker.tiers import TierAssigner


def make_pick(**kwargs):
    defaults = {
        "match_id": "m1", "home_team": "a", "away_team": "b",
        "league": "PL", "match_date": "2025-01-01", "market": "1x2_home",
        "model_prob": 0.80, "model_spread": 0.03, "best_odds": 2.0,
        "bookmaker": "b365", "implied_prob": 0.5, "edge": 0.30,
        "meta_agreement": 3, "risk_flags": [], "confidence_factors": [],
    }
    defaults.update(kwargs)
    return Pick(**defaults)


def test_platinum_tier():
    pick = make_pick(model_prob=0.90, edge=0.15, risk_flags=[])
    ta = TierAssigner()
    [result] = ta.assign([pick])
    assert result.tier == "platinum"


def test_gold_tier():
    pick = make_pick(model_prob=0.80, edge=0.08, risk_flags=["minor_fatigue"])
    ta = TierAssigner()
    [result] = ta.assign([pick])
    assert result.tier == "gold"


def test_silver_tier():
    pick = make_pick(model_prob=0.72, edge=0.06, risk_flags=["minor_a", "minor_b"])
    ta = TierAssigner()
    [result] = ta.assign([pick])
    assert result.tier == "silver"
