"""Tests for ice hockey sport configuration and registry."""

from sharpedge.core.sport import SportRegistry
from sharpedge.sports.ice_hockey.config import (
    HOCKEY_CONFIG,
    HOCKEY_MARKETS,
    LEAGUES,
)


def test_hockey_registers_on_import():
    registry = SportRegistry()
    registry.register(HOCKEY_CONFIG)
    assert registry.is_registered("ice_hockey")
    assert registry.get("ice_hockey").name == "Ice Hockey"


def test_hockey_config_slug():
    assert HOCKEY_CONFIG.slug == "ice_hockey"


def test_hockey_has_3_markets():
    assert len(HOCKEY_MARKETS) == 3
    assert HOCKEY_MARKETS[0].name == "moneyline"
    assert HOCKEY_MARKETS[0].outcomes == ("home", "away")


def test_hockey_markets_are_2_outcome():
    for m in HOCKEY_MARKETS:
        assert len(m.outcomes) == 2


def test_hockey_market_names():
    names = [m.name for m in HOCKEY_MARKETS]
    assert "moneyline" in names
    assert "puck_line" in names
    assert "totals" in names


def test_hockey_min_train_seasons():
    assert HOCKEY_CONFIG.min_train_seasons == 3


def test_hockey_mc_sims():
    assert HOCKEY_CONFIG.default_mc_sims == 5000


def test_leagues_defined():
    assert "NHL" in LEAGUES
    assert "KHL" in LEAGUES
    assert LEAGUES["NHL"]["country"] == "USA/Canada"
    assert LEAGUES["NHL"]["season_start"] == 10
    assert LEAGUES["NHL"]["season_end"] == 6


def test_register_idempotent():
    """Calling register() twice should not raise."""
    from sharpedge.sports.ice_hockey.config import register

    register()
    register()  # should not raise
