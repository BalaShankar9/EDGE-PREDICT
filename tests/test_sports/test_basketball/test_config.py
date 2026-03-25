"""Tests for basketball sport configuration and registry."""

from sharpedge.core.sport import SportRegistry
from sharpedge.sports.basketball.config import (
    BASKETBALL_CONFIG,
    BASKETBALL_MARKETS,
    LEAGUES,
)


def test_basketball_registers_on_import():
    registry = SportRegistry()
    registry.register(BASKETBALL_CONFIG)
    assert registry.is_registered("basketball")
    assert registry.get("basketball").name == "Basketball"


def test_basketball_config_slug():
    assert BASKETBALL_CONFIG.slug == "basketball"


def test_basketball_has_3_markets():
    assert len(BASKETBALL_MARKETS) == 3
    assert BASKETBALL_MARKETS[0].name == "moneyline"
    assert BASKETBALL_MARKETS[0].outcomes == ("home", "away")


def test_basketball_markets_are_2_outcome():
    for m in BASKETBALL_MARKETS:
        assert len(m.outcomes) == 2


def test_basketball_market_names():
    names = [m.name for m in BASKETBALL_MARKETS]
    assert "moneyline" in names
    assert "spread" in names
    assert "totals" in names


def test_basketball_min_train_seasons():
    assert BASKETBALL_CONFIG.min_train_seasons == 2


def test_basketball_mc_sims():
    assert BASKETBALL_CONFIG.default_mc_sims == 5000


def test_leagues_defined():
    assert "NBA" in LEAGUES
    assert "EuroLeague" in LEAGUES
    assert LEAGUES["NBA"]["country"] == "USA"
    assert LEAGUES["EuroLeague"]["country"] == "Europe"
    assert LEAGUES["NBA"]["season_start"] == 10
    assert LEAGUES["NBA"]["season_end"] == 6


def test_register_idempotent():
    """Calling register() twice should not raise."""
    from sharpedge.sports.basketball.config import register

    register()
    register()  # should not raise
