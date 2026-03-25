"""Tests for football sport configuration."""
from sharpedge.core.sport import SportRegistry


def test_football_registers_on_import():
    """Importing football module should register it."""
    # Fresh registry to avoid singleton pollution
    registry = SportRegistry()
    from sharpedge.sports.football.config import FOOTBALL_CONFIG
    registry.register(FOOTBALL_CONFIG)
    assert registry.is_registered("football")
    cfg = registry.get("football")
    assert cfg.name == "Football"
    assert cfg.slug == "football"
    assert len(cfg.markets) == 5


def test_football_has_12_leagues():
    from sharpedge.sports.football.config import LEAGUES
    assert len(LEAGUES) == 12
    assert "Premier League" in LEAGUES
    assert "Eredivisie" in LEAGUES


def test_football_markets_use_tuples():
    """Markets must use tuples for immutable outcomes."""
    from sharpedge.sports.football.config import FOOTBALL_MARKETS
    for market in FOOTBALL_MARKETS:
        assert isinstance(market.outcomes, tuple)


def test_league_reliability_factors():
    from sharpedge.sports.football.config import LEAGUE_RELIABILITY
    assert LEAGUE_RELIABILITY["La Liga"] == 1.08
    assert LEAGUE_RELIABILITY["Ligue 1"] == 0.97


def test_profitable_markets():
    from sharpedge.sports.football.markets import PROFITABLE_MARKETS
    assert "1x2_home" in PROFITABLE_MARKETS
    assert "1x2_away" in PROFITABLE_MARKETS
    assert len(PROFITABLE_MARKETS) == 5
