"""Tests for baseball sport configuration and registry."""

from sharpedge.core.sport import SportRegistry
from sharpedge.sports.baseball.config import (
    BASEBALL_CONFIG,
    BASEBALL_MARKETS,
    LEAGUES,
)


def test_baseball_registers_on_import():
    registry = SportRegistry()
    registry.register(BASEBALL_CONFIG)
    assert registry.is_registered("baseball")
    assert registry.get("baseball").name == "Baseball"


def test_baseball_config_slug():
    assert BASEBALL_CONFIG.slug == "baseball"


def test_baseball_has_3_markets():
    assert len(BASEBALL_MARKETS) == 3
    assert BASEBALL_MARKETS[0].name == "moneyline"
    assert BASEBALL_MARKETS[0].outcomes == ("home", "away")


def test_baseball_markets_are_2_outcome():
    for m in BASEBALL_MARKETS:
        assert len(m.outcomes) == 2


def test_baseball_market_names():
    names = [m.name for m in BASEBALL_MARKETS]
    assert "moneyline" in names
    assert "run_line" in names
    assert "totals" in names


def test_baseball_min_train_seasons():
    assert BASEBALL_CONFIG.min_train_seasons == 3


def test_baseball_mc_sims():
    assert BASEBALL_CONFIG.default_mc_sims == 5000


def test_leagues_defined():
    assert "MLB" in LEAGUES
    assert "NPB" in LEAGUES
    assert LEAGUES["MLB"]["country"] == "USA"
    assert LEAGUES["MLB"]["season_start"] == 3
    assert LEAGUES["MLB"]["season_end"] == 10


def test_register_idempotent():
    """Calling register() twice should not raise."""
    from sharpedge.sports.baseball.config import register

    register()
    register()
