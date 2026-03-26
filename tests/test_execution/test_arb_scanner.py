"""Tests for ValueScanner."""
import pytest
from sharpedge.execution.arb_scanner import ValueScanner


class TestScan:
    def test_single_book_returns_empty(self):
        vs = ValueScanner()
        odds = {"bet365": {"home": 1.8, "draw": 3.5, "away": 4.2}}
        assert vs.scan(odds) == []

    def test_uniform_odds_no_value(self):
        vs = ValueScanner()
        odds = {
            "bet365": {"home": 2.0, "draw": 3.0, "away": 4.0},
            "pinnacle": {"home": 2.0, "draw": 3.0, "away": 4.0},
        }
        assert vs.scan(odds) == []

    def test_finds_value_when_book_is_outlier(self):
        vs = ValueScanner()
        # bet365 offers much higher away odds than market
        odds = {
            "bet365": {"home": 1.8, "draw": 3.5, "away": 5.0},
            "pinnacle": {"home": 1.8, "draw": 3.5, "away": 3.5},
            "betfair": {"home": 1.8, "draw": 3.5, "away": 3.6},
        }
        opps = vs.scan(odds)
        assert len(opps) >= 1
        # The outlier should be bet365 on away
        assert opps[0]["bookmaker"] == "bet365"
        assert opps[0]["outcome"] == "away"

    def test_value_edge_positive(self):
        vs = ValueScanner()
        odds = {
            "bet365": {"home": 2.5, "draw": 3.5, "away": 3.0},
            "pinnacle": {"home": 2.0, "draw": 3.5, "away": 3.0},
        }
        opps = vs.scan(odds)
        for opp in opps:
            assert opp["value_edge_pct"] > 0

    def test_sorted_by_value_desc(self):
        vs = ValueScanner()
        odds = {
            "book_a": {"home": 3.0, "draw": 5.0, "away": 2.0},
            "book_b": {"home": 2.0, "draw": 3.0, "away": 2.0},
            "book_c": {"home": 2.1, "draw": 3.1, "away": 2.0},
        }
        opps = vs.scan(odds)
        for i in range(len(opps) - 1):
            assert opps[i]["value_edge_pct"] >= opps[i + 1]["value_edge_pct"]

    def test_min_edge_threshold(self):
        vs = ValueScanner()
        vs.MIN_VALUE_EDGE = 0.10  # require 10% edge
        odds = {
            "bet365": {"home": 2.1, "draw": 3.5, "away": 3.0},
            "pinnacle": {"home": 2.0, "draw": 3.5, "away": 3.0},
        }
        opps = vs.scan(odds)
        for opp in opps:
            assert opp["value_edge_pct"] > 10.0

    def test_empty_books_returns_empty(self):
        vs = ValueScanner()
        assert vs.scan({}) == []

    def test_result_fields(self):
        vs = ValueScanner()
        odds = {
            "bet365": {"home": 3.0, "draw": 3.5, "away": 2.0},
            "pinnacle": {"home": 2.0, "draw": 3.5, "away": 2.0},
        }
        opps = vs.scan(odds)
        if opps:
            required_keys = {"bookmaker", "outcome", "odds", "book_implied",
                             "market_implied", "value_edge_pct"}
            assert set(opps[0].keys()) == required_keys
