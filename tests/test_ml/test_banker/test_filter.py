import pytest
from sharpedge.ml.banker.filter import (
    BankerFilter, Pick, PROFITABLE_MARKETS, MARKET_THRESHOLDS, MAX_ODDS,
)


@pytest.fixture
def sample_predictions():
    """50 predictions with varying confidence levels."""
    preds = []
    for i in range(50):
        preds.append({
            "match_id": f"match_{i}",
            "home_team": f"team_h_{i}",
            "away_team": f"team_a_{i}",
            "league": "Premier League",
            "match_date": "2025-01-01",
            "market": "1x2_home",
            "model_prob": 0.40 + i * 0.012,  # 0.40 to 0.988
            "model_spread": 0.10 - i * 0.002,
            "best_odds": 1.5 + i * 0.02,
            "bookmaker": "bet365",
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
    home_threshold = MARKET_THRESHOLDS["1x2_home"]
    for pick in picks:
        assert pick.model_prob >= home_threshold
        assert not any(f.startswith("CRITICAL") for f in pick.risk_flags)


def test_filter_rejects_critical_flags():
    preds = [{
        "match_id": "m1", "home_team": "a", "away_team": "b",
        "league": "PL", "match_date": "2025-01-01", "market": "1x2_home",
        "model_prob": 0.90, "model_spread": 0.02, "best_odds": 1.2,
        "bookmaker": "b365",
        "risk_flags": ["CRITICAL_red_card"],
    }]
    bf = BankerFilter()
    picks = bf.filter(preds)
    assert len(picks) == 0


def test_filter_custom_threshold():
    preds = [{
        "match_id": "m1", "home_team": "a", "away_team": "b",
        "league": "PL", "match_date": "2025-01-01", "market": "1x2_home",
        "model_prob": 0.72, "best_odds": 1.5, "bookmaker": "b365",
    }]
    # Should pass at 0.60 threshold (prob=0.72 >= max(0.58, 0.60)=0.60, edge=0.053 >= 0.05)
    bf_low = BankerFilter(min_confidence=0.60)
    assert len(bf_low.filter(preds)) == 1
    # Should fail at 0.75 threshold (prob=0.72 < max(0.58, 0.75)=0.75)
    bf_high = BankerFilter(min_confidence=0.75)
    assert len(bf_high.filter(preds)) == 0


# ---------------------------------------------------------------------------
# New backtest-proven rule tests
# ---------------------------------------------------------------------------


def _make_pred(market="1x2_home", prob=0.65, odds=1.80, risk_flags=None):
    """Helper to build a single prediction dict."""
    return {
        "match_id": "m1",
        "home_team": "Team A",
        "away_team": "Team B",
        "league": "PL",
        "match_date": "2025-01-01",
        "market": market,
        "model_prob": prob,
        "model_spread": 0.10,
        "best_odds": odds,
        "bookmaker": "bet365",
        "risk_flags": risk_flags or [],
    }


def test_filter_rejects_longshot_odds():
    """Odds > MAX_ODDS (2.50) should be rejected (backtest: -11.3% ROI)."""
    pred = _make_pred(market="1x2_away", prob=0.60, odds=3.50)
    bf = BankerFilter()
    picks = bf.filter([pred])
    assert len(picks) == 0


def test_filter_accepts_short_odds():
    """Odds within range with sufficient prob/edge should be accepted."""
    pred = _make_pred(market="1x2_away", prob=0.60, odds=2.00)
    bf = BankerFilter()
    picks = bf.filter([pred])
    assert len(picks) == 1


def test_filter_higher_threshold_for_home():
    """1x2_home needs 0.58 confidence; prob=0.55 should be rejected."""
    pred = _make_pred(market="1x2_home", prob=0.55, odds=1.80)
    bf = BankerFilter()
    picks = bf.filter([pred])
    assert len(picks) == 0


def test_filter_lower_threshold_for_away():
    """1x2_away only needs 0.50; prob=0.52 should pass confidence check."""
    # Use min_edge=0.0 to isolate the confidence-threshold behaviour
    pred = _make_pred(market="1x2_away", prob=0.52, odds=2.00)
    bf = BankerFilter(min_edge=0.0)
    picks = bf.filter([pred])
    assert len(picks) == 1


def test_filter_rejects_dead_markets():
    """Markets not in PROFITABLE_MARKETS are rejected regardless of prob."""
    dead_markets = [
        "btts_yes", "btts_no", "under_25",
        "dc_1x", "dc_12", "asian_handicap",
    ]
    bf = BankerFilter()
    for mkt in dead_markets:
        pred = _make_pred(market=mkt, prob=0.80, odds=1.50)
        picks = bf.filter([pred])
        assert len(picks) == 0, f"Expected {mkt} to be rejected"


def test_filter_accepts_profitable_markets():
    """All PROFITABLE_MARKETS should pass when prob/edge are sufficient."""
    good_markets = ["1x2_home", "1x2_away", "over_25", "dc_x2"]
    bf = BankerFilter()
    for mkt in good_markets:
        pred = _make_pred(market=mkt, prob=0.65, odds=1.80)
        picks = bf.filter([pred])
        assert len(picks) == 1, f"Expected {mkt} to be accepted"


def test_filter_edge_floor_for_value():
    """Edge exactly at min_edge passes; just below fails."""
    # prob=0.55, odds=2.00 -> implied=0.50, edge=0.05
    pred = _make_pred(market="1x2_away", prob=0.55, odds=2.00)

    bf_pass = BankerFilter(min_edge=0.05)
    assert len(bf_pass.filter([pred])) == 1

    bf_fail = BankerFilter(min_edge=0.06)
    assert len(bf_fail.filter([pred])) == 0
