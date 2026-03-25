"""Tests for tennis sport configuration and registry."""

from sharpedge.core.sport import SportRegistry
from sharpedge.sports.tennis.config import (
    TENNIS_CONFIG,
    TENNIS_MARKETS,
    TOURS,
    SURFACES,
)


def test_tennis_registers_on_import():
    registry = SportRegistry()
    registry.register(TENNIS_CONFIG)
    assert registry.is_registered("tennis")
    assert registry.get("tennis").name == "Tennis"


def test_tennis_config_slug():
    assert TENNIS_CONFIG.slug == "tennis"


def test_tennis_has_3_markets():
    assert len(TENNIS_MARKETS) == 3
    assert TENNIS_MARKETS[0].name == "match_winner"
    assert TENNIS_MARKETS[0].outcomes == ("player1", "player2")


def test_tennis_markets_are_2_outcome():
    for m in TENNIS_MARKETS:
        assert len(m.outcomes) == 2


def test_tennis_min_train_seasons():
    assert TENNIS_CONFIG.min_train_seasons == 2


def test_tours_defined():
    assert "ATP" in TOURS
    assert "WTA" in TOURS
    assert TOURS["ATP"]["gender"] == "male"
    assert TOURS["WTA"]["gender"] == "female"


def test_surfaces_defined():
    assert "Hard" in SURFACES
    assert "Clay" in SURFACES
    assert "Grass" in SURFACES
    assert "Carpet" in SURFACES
    assert len(SURFACES) == 4


def test_register_idempotent():
    """Calling register() twice should not raise."""
    from sharpedge.sports.tennis.config import register

    # First call may or may not have already happened via import
    register()
    register()  # should not raise
