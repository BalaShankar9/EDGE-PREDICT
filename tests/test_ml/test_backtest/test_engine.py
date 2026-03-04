import pandas as pd
import numpy as np
import pytest

from sharpedge.backtest.engine import BacktestEngine, BacktestResult


@pytest.fixture
def sample_predictions():
    """Generate predictions that will pass the banker filter."""
    preds = []
    for i in range(10):
        preds.append(
            {
                "match_id": f"match_{i}",
                "home_team": f"team_h_{i}",
                "away_team": f"team_a_{i}",
                "league": "Premier League",
                "match_date": f"2025-01-{i + 1:02d}",
                "market": "1x2_home",
                "model_prob": 0.80,
                "model_spread": 0.03,
                "best_odds": 1.5,  # implied=0.667, edge=0.133
                "bookmaker": "bet365",
                "meta_agreement": 3,
                "risk_flags": [],
            }
        )
    return preds


@pytest.fixture
def sample_actuals():
    """Actuals where 6/10 are home wins."""
    rows = []
    for i in range(10):
        ftr = "H" if i < 6 else "A"
        rows.append(
            {
                "match_id": f"match_{i}",
                "FTR": ftr,
                "FTHG": 2 if ftr == "H" else 0,
                "FTAG": 0 if ftr == "H" else 2,
            }
        )
    return pd.DataFrame(rows)


def test_backtest_runs(sample_predictions, sample_actuals):
    engine = BacktestEngine()
    result = engine.run(sample_predictions, sample_actuals)
    assert isinstance(result, BacktestResult)
    assert result.total_picks > 0


def test_backtest_win_rate(sample_predictions, sample_actuals):
    engine = BacktestEngine()
    result = engine.run(sample_predictions, sample_actuals)
    assert 0 <= result.win_rate <= 1
    assert result.win_rate == pytest.approx(0.6, abs=0.01)


def test_backtest_pick_log(sample_predictions, sample_actuals):
    engine = BacktestEngine()
    result = engine.run(sample_predictions, sample_actuals)
    assert len(result.pick_log) == result.total_picks
    for entry in result.pick_log:
        assert "won" in entry
        assert "stake" in entry
        assert "profit" in entry


def test_backtest_tiers(sample_predictions, sample_actuals):
    engine = BacktestEngine()
    result = engine.run(sample_predictions, sample_actuals)
    assert len(result.picks_by_tier) > 0
    for tier, stats in result.picks_by_tier.items():
        assert stats["count"] > 0
        assert 0 <= stats["win_rate"] <= 1


def test_backtest_empty_predictions():
    engine = BacktestEngine()
    result = engine.run([], pd.DataFrame())
    assert result.total_picks == 0
    assert result.win_rate == 0.0
