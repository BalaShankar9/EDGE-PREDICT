"""Correlation-Aware Portfolio Manager.

Tracks correlations between active bets and limits exposure.
"""
from dataclasses import dataclass


@dataclass
class PortfolioBet:
    match_id: str
    sport: str
    league: str
    market: str
    outcome: str
    stake_pct: float


class PortfolioManager:
    """Manages bet correlations and exposure limits."""

    MAX_SAME_LEAGUE = 3        # max bets in same league per day
    MAX_SPORT_EXPOSURE = 0.40  # max 40% of daily allocation per sport

    def __init__(self):
        self._active_bets: list[PortfolioBet] = []

    def can_add(self, bet: PortfolioBet) -> tuple[bool, str]:
        """Check if a bet can be added without violating portfolio rules."""
        # Same league limit
        same_league = [b for b in self._active_bets if b.league == bet.league]
        if len(same_league) >= self.MAX_SAME_LEAGUE:
            return False, f"Max {self.MAX_SAME_LEAGUE} bets in {bet.league}"

        # Sport exposure limit
        sport_stake = sum(b.stake_pct for b in self._active_bets if b.sport == bet.sport)
        if sport_stake + bet.stake_pct > self.MAX_SPORT_EXPOSURE:
            return False, f"Sport exposure {sport_stake + bet.stake_pct:.1%} > {self.MAX_SPORT_EXPOSURE:.0%}"

        return True, "OK"

    def add_bet(self, bet: PortfolioBet) -> bool:
        ok, _ = self.can_add(bet)
        if ok:
            self._active_bets.append(bet)
        return ok

    def reset_daily(self) -> None:
        self._active_bets.clear()

    @property
    def diversification_score(self) -> float:
        """Score 0-1: higher = more diversified portfolio."""
        if not self._active_bets:
            return 0.0
        sports = set(b.sport for b in self._active_bets)
        leagues = set(b.league for b in self._active_bets)
        return min(1.0, (len(sports) * 0.3 + len(leagues) * 0.1))

    @property
    def total_exposure(self) -> float:
        return sum(b.stake_pct for b in self._active_bets)
