"""Tests for sport-agnostic configuration and registry."""

import pytest

from sharpedge.core.sport import Market, SportConfig, SportRegistry


class TestMarket:
    def test_market_creation(self):
        m = Market(name="1x2", outcomes=("home", "draw", "away"), description="Match result")
        assert m.name == "1x2"
        assert m.outcomes == ("home", "draw", "away")
        assert m.description == "Match result"

    def test_market_immutable_outcomes(self):
        m = Market(name="btts", outcomes=("yes", "no"))
        assert isinstance(m.outcomes, tuple)

    def test_market_is_frozen(self):
        m = Market(name="btts", outcomes=("yes", "no"))
        with pytest.raises(AttributeError):
            m.name = "other"


class TestSportConfig:
    def test_creation_with_defaults(self):
        cfg = SportConfig(name="Football", slug="football")
        assert cfg.name == "Football"
        assert cfg.slug == "football"
        assert cfg.markets == []
        assert cfg.min_train_seasons == 3
        assert cfg.default_mc_sims == 5000
        assert cfg.default_min_consensus == 3

    def test_creation_with_markets(self):
        m = Market(name="1x2", outcomes=("home", "draw", "away"))
        cfg = SportConfig(name="Football", slug="football", markets=[m])
        assert len(cfg.markets) == 1
        assert cfg.markets[0].name == "1x2"


class TestSportRegistry:
    def test_register_and_get(self):
        reg = SportRegistry()
        cfg = SportConfig(name="Football", slug="football")
        reg.register(cfg)
        assert reg.get("football") is cfg

    def test_duplicate_slug_raises(self):
        reg = SportRegistry()
        cfg = SportConfig(name="Football", slug="football")
        reg.register(cfg)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(SportConfig(name="Football2", slug="football"))

    def test_get_unknown_raises(self):
        reg = SportRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("unknown")

    def test_list_sports(self):
        reg = SportRegistry()
        reg.register(SportConfig(name="Football", slug="football"))
        reg.register(SportConfig(name="Tennis", slug="tennis"))
        assert sorted(reg.list_sports()) == ["football", "tennis"]

    def test_is_registered(self):
        reg = SportRegistry()
        assert reg.is_registered("football") is False
        reg.register(SportConfig(name="Football", slug="football"))
        assert reg.is_registered("football") is True
