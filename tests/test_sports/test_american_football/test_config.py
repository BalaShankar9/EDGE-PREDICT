"""Tests for american football sport configuration and registry."""

from sharpedge.core.sport import SportRegistry
from sharpedge.sports.american_football.config import (
    NFL_CONFIG,
    NFL_MARKETS,
    LEAGUES,
)


def test_nfl_registers_on_import():
    registry = SportRegistry()
    registry.register(NFL_CONFIG)
    assert registry.is_registered("american_football")
    assert registry.get("american_football").name == "American Football"


def test_nfl_config_slug():
    assert NFL_CONFIG.slug == "american_football"


def test_nfl_has_3_markets():
    assert len(NFL_MARKETS) == 3
    assert NFL_MARKETS[0].name == "moneyline"
    assert NFL_MARKETS[0].outcomes == ("home", "away")


def test_nfl_markets_are_2_outcome():
    for m in NFL_MARKETS:
        assert len(m.outcomes) == 2


def test_nfl_market_names():
    names = [m.name for m in NFL_MARKETS]
    assert "moneyline" in names
    assert "spread" in names
    assert "totals" in names


def test_nfl_min_train_seasons():
    assert NFL_CONFIG.min_train_seasons == 3


def test_nfl_mc_sims():
    assert NFL_CONFIG.default_mc_sims == 5000


def test_leagues_defined():
    assert "NFL" in LEAGUES
    assert "NCAAF" in LEAGUES
    assert LEAGUES["NFL"]["country"] == "USA"
    assert LEAGUES["NFL"]["season_start"] == 9
    assert LEAGUES["NFL"]["season_end"] == 2


def test_register_idempotent():
    """Calling register() twice should not raise."""
    from sharpedge.sports.american_football.config import register

    register()
    register()
