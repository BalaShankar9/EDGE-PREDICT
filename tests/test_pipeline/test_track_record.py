"""Tests for track record calculation."""
import pytest
from sharpedge.pipeline.track_record import calculate_track_record


def _make_picks():
    return [
        {"tier": "platinum", "league": "Premier League", "result": "win",
         "profit_loss": 0.85, "best_odds": 1.85, "match_date": "2026-03-01"},
        {"tier": "platinum", "league": "La Liga", "result": "win",
         "profit_loss": 0.65, "best_odds": 1.65, "match_date": "2026-03-01"},
        {"tier": "gold", "league": "Premier League", "result": "loss",
         "profit_loss": -1.0, "best_odds": 2.10, "match_date": "2026-03-02"},
        {"tier": "gold", "league": "Bundesliga", "result": "win",
         "profit_loss": 0.90, "best_odds": 1.90, "match_date": "2026-03-02"},
        {"tier": "silver", "league": "Serie A", "result": "loss",
         "profit_loss": -1.0, "best_odds": 1.75, "match_date": "2026-03-03"},
    ]


def test_overall_record():
    record = calculate_track_record(_make_picks())
    assert record["total_picks"] == 5
    assert record["wins"] == 3
    assert record["losses"] == 2
    assert record["win_rate"] == pytest.approx(0.60)
    assert record["total_profit"] == pytest.approx(0.40)


def test_roi():
    record = calculate_track_record(_make_picks())
    assert record["roi"] == pytest.approx(8.0)


def test_by_tier():
    record = calculate_track_record(_make_picks())
    tiers = record["by_tier"]
    assert tiers["platinum"]["wins"] == 2
    assert tiers["platinum"]["losses"] == 0
    assert tiers["platinum"]["win_rate"] == pytest.approx(1.0)
    assert tiers["gold"]["wins"] == 1
    assert tiers["gold"]["losses"] == 1


def test_by_league():
    record = calculate_track_record(_make_picks())
    leagues = record["by_league"]
    assert leagues["Premier League"]["total_picks"] == 2
    assert leagues["Premier League"]["wins"] == 1


def test_empty_picks():
    record = calculate_track_record([])
    assert record["total_picks"] == 0
    assert record["win_rate"] == 0.0
    assert record["roi"] == 0.0


def test_max_drawdown():
    record = calculate_track_record(_make_picks())
    assert "max_drawdown" in record
    assert record["max_drawdown"] <= 0.0


def test_avg_odds():
    record = calculate_track_record(_make_picks())
    expected_avg = (1.85 + 1.65 + 2.10 + 1.90 + 1.75) / 5
    assert record["avg_odds"] == pytest.approx(expected_avg)
