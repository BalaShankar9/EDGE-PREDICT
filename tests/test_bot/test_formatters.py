"""Tests for Telegram message formatting."""
import pytest
from sharpedge.bot.formatters import format_pick_message, format_results_message, format_record_message


def test_format_pick_message():
    pick = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "league": "Premier League", "match_date": "2026-03-04",
        "pick_selection": "Home Win", "pick_market": "1x2_home",
        "best_odds": 1.85, "bookmaker": "Bet365",
        "edge": 0.082, "tier": "platinum",
        "model_prob": 0.832, "meta_agreement": 3, "risk_flags": [],
    }
    msg = format_pick_message(pick)
    assert "Arsenal" in msg
    assert "Chelsea" in msg
    assert "PLATINUM" in msg
    assert "1.85" in msg
    assert "Bet365" in msg


def test_format_results_message():
    results = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "pick_selection": "Home Win", "result": "win",
         "best_odds": 1.85, "profit_loss": 0.85,
         "home_goals": 2, "away_goals": 1},
        {"home_team": "Bayern", "away_team": "Dortmund",
         "pick_selection": "Home Win", "result": "loss",
         "best_odds": 1.65, "profit_loss": -1.0,
         "home_goals": 0, "away_goals": 1},
    ]
    summary = {"wins": 1, "losses": 1, "profit": -0.15,
               "month_wins": 18, "month_losses": 5, "month_profit": 6.2,
               "month_roi": 8.5, "all_wins": 142, "all_losses": 38,
               "all_profit": 42.1}
    msg = format_results_message(results, summary, "Mar 3")
    assert "Arsenal" in msg
    assert "Bayern" in msg
    assert "1W 1L" in msg


def test_format_record_message():
    record = {
        "total_picks": 180, "wins": 142, "losses": 38,
        "win_rate": 0.789, "total_profit": 42.1,
        "roi": 8.5, "avg_odds": 1.82,
        "by_tier": {
            "platinum": {"total_picks": 50, "wins": 45, "losses": 5,
                        "win_rate": 0.90, "total_profit": 25.0, "roi": 12.5},
        },
    }
    msg = format_record_message(record)
    assert "180" in msg
    assert "78.9%" in msg
    assert "42.1" in msg


def test_format_pick_message_no_risk_flags():
    pick = {
        "home_team": "Arsenal", "away_team": "Chelsea",
        "league": "Premier League", "match_date": "2026-03-04",
        "pick_selection": "Home Win", "pick_market": "1x2_home",
        "best_odds": 1.85, "bookmaker": "Bet365",
        "edge": 0.082, "tier": "gold",
        "model_prob": 0.80, "meta_agreement": 2, "risk_flags": [],
    }
    msg = format_pick_message(pick)
    assert "None" in msg or "Risk Flags: None" in msg
