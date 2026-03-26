"""Cross-Book Value Scanner.

Finds bets where one bookmaker's odds are significantly better than the market.
"""
import numpy as np


class ValueScanner:
    """Finds cross-book value opportunities."""

    MIN_VALUE_EDGE = 0.03  # minimum 3% value over market average

    def scan(self, odds_by_book: dict[str, dict[str, float]]) -> list[dict]:
        """
        Parameters
        ----------
        odds_by_book : {"bet365": {"home": 1.8, "draw": 3.5, "away": 4.2},
                        "pinnacle": {"home": 1.75, "draw": 3.4, "away": 4.0}, ...}

        Returns list of value opportunities.
        """
        if len(odds_by_book) < 2:
            return []

        # Compute market average implied prob per outcome
        outcomes = list(next(iter(odds_by_book.values())).keys())
        market_implied: dict[str, float] = {}

        for outcome in outcomes:
            implied_probs = []
            for book_odds in odds_by_book.values():
                odd = book_odds.get(outcome, 0)
                if odd > 1.0:
                    implied_probs.append(1.0 / odd)
            if implied_probs:
                market_implied[outcome] = float(np.mean(implied_probs))

        # Find outlier value
        opportunities: list[dict] = []
        for book, book_odds in odds_by_book.items():
            for outcome in outcomes:
                odd = book_odds.get(outcome, 0)
                if odd <= 1.0:
                    continue
                book_implied = 1.0 / odd
                market_avg = market_implied.get(outcome, book_implied)

                if market_avg > 0:
                    value_edge = (market_avg / book_implied) - 1.0
                    if value_edge > self.MIN_VALUE_EDGE:
                        opportunities.append({
                            "bookmaker": book,
                            "outcome": outcome,
                            "odds": odd,
                            "book_implied": book_implied,
                            "market_implied": market_avg,
                            "value_edge_pct": value_edge * 100,
                        })

        opportunities.sort(key=lambda x: x["value_edge_pct"], reverse=True)
        return opportunities
