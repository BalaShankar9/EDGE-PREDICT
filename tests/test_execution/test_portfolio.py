"""Tests for PortfolioManager."""
import pytest
from sharpedge.execution.portfolio import PortfolioManager, PortfolioBet


def _make_bet(league="EPL", sport="football", stake=0.02, match_id="m1",
              market="1x2", outcome="home"):
    return PortfolioBet(match_id=match_id, sport=sport, league=league,
                        market=market, outcome=outcome, stake_pct=stake)


class TestCanAdd:
    def test_first_bet_allowed(self):
        pm = PortfolioManager()
        ok, msg = pm.can_add(_make_bet())
        assert ok is True
        assert msg == "OK"

    def test_same_league_limit(self):
        pm = PortfolioManager()
        for i in range(3):
            pm.add_bet(_make_bet(match_id=f"m{i}"))
        ok, msg = pm.can_add(_make_bet(match_id="m4"))
        assert ok is False
        assert "Max 3" in msg

    def test_different_leagues_ok(self):
        pm = PortfolioManager()
        for i, league in enumerate(["EPL", "LaLiga", "SerieA", "Bundesliga"]):
            assert pm.add_bet(_make_bet(league=league, match_id=f"m{i}")) is True

    def test_sport_exposure_limit(self):
        pm = PortfolioManager()
        # Add bets totaling 0.38 in football
        pm.add_bet(_make_bet(league="EPL", stake=0.15, match_id="m1"))
        pm.add_bet(_make_bet(league="LaLiga", stake=0.15, match_id="m2"))
        pm.add_bet(_make_bet(league="SerieA", stake=0.08, match_id="m3"))
        # Next 0.05 would exceed 0.40
        ok, msg = pm.can_add(_make_bet(league="Bundesliga", stake=0.05, match_id="m4"))
        assert ok is False
        assert "Sport exposure" in msg

    def test_different_sports_independent(self):
        pm = PortfolioManager()
        pm.add_bet(_make_bet(sport="football", league="EPL", stake=0.35, match_id="m1"))
        ok, _ = pm.can_add(_make_bet(sport="tennis", league="ATP", stake=0.35, match_id="m2"))
        assert ok is True


class TestDiversification:
    def test_empty_portfolio_zero(self):
        pm = PortfolioManager()
        assert pm.diversification_score == 0.0

    def test_single_sport_single_league(self):
        pm = PortfolioManager()
        pm.add_bet(_make_bet())
        score = pm.diversification_score
        assert score == pytest.approx(0.4)  # 1*0.3 + 1*0.1

    def test_increases_with_variety(self):
        pm = PortfolioManager()
        pm.add_bet(_make_bet(sport="football", league="EPL", match_id="m1"))
        score1 = pm.diversification_score

        pm.add_bet(_make_bet(sport="tennis", league="ATP", match_id="m2"))
        score2 = pm.diversification_score
        assert score2 > score1

    def test_capped_at_one(self):
        pm = PortfolioManager()
        for i, (sport, league) in enumerate([
            ("football", "EPL"), ("tennis", "ATP"), ("basketball", "NBA"),
            ("baseball", "MLB"), ("ice_hockey", "NHL"),
        ]):
            pm.add_bet(_make_bet(sport=sport, league=league, match_id=f"m{i}", stake=0.01))
        assert pm.diversification_score <= 1.0


class TestResetAndExposure:
    def test_total_exposure(self):
        pm = PortfolioManager()
        pm.add_bet(_make_bet(stake=0.02, match_id="m1"))
        pm.add_bet(_make_bet(stake=0.03, league="LaLiga", match_id="m2"))
        assert pytest.approx(pm.total_exposure) == 0.05

    def test_reset_clears(self):
        pm = PortfolioManager()
        pm.add_bet(_make_bet())
        pm.reset_daily()
        assert pm.total_exposure == 0.0
        assert pm.diversification_score == 0.0
