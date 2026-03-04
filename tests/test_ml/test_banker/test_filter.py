import pytest
from sharpedge.ml.banker.filter import BankerFilter, Pick


@pytest.fixture
def sample_predictions():
    """50 predictions, only a few should pass all 5 stages."""
    preds = []
    for i in range(50):
        preds.append({
            "match_id": f"match_{i}",
            "home_team": f"team_h_{i}",
            "away_team": f"team_a_{i}",
            "league": "Premier League",
            "match_date": "2025-01-01",
            "market": "1x2_home",
            "model_prob": 0.50 + i * 0.01,  # 0.50 to 0.99
            "model_spread": 0.10 - i * 0.002,  # 0.10 to 0.002
            "best_odds": 1.5 + i * 0.02,  # 1.5 to 2.48
            "bookmaker": "bet365",
            "meta_agreement": 2 if i > 25 else 1,
            "risk_flags": ["CRITICAL_injury"] if i == 49 else [],
        })
    return preds


def test_filter_reduces_predictions(sample_predictions):
    bf = BankerFilter()
    picks = bf.filter(sample_predictions)
    assert len(picks) < len(sample_predictions)
    assert len(picks) >= 1


def test_filter_all_picks_meet_criteria(sample_predictions):
    bf = BankerFilter()
    picks = bf.filter(sample_predictions)
    for pick in picks:
        assert pick.model_prob >= 0.70
        assert pick.model_spread <= 0.08
        assert pick.edge >= 0.05
        assert pick.meta_agreement >= 2
        assert not any(f.startswith("CRITICAL") for f in pick.risk_flags)


def test_filter_rejects_critical_flags():
    preds = [{
        "match_id": "m1", "home_team": "a", "away_team": "b",
        "league": "PL", "match_date": "2025-01-01", "market": "1x2_home",
        "model_prob": 0.90, "model_spread": 0.02, "best_odds": 1.2,
        "bookmaker": "b365", "meta_agreement": 3,
        "risk_flags": ["CRITICAL_red_card"],
    }]
    bf = BankerFilter()
    picks = bf.filter(preds)
    assert len(picks) == 0
