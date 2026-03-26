"""Ice hockey sport configuration.

Registers ice hockey with the global sport registry, defining all supported
leagues, markets, and sport-specific settings.
"""
from sharpedge.core.sport import SportConfig, Market, sport_registry

# Hockey markets — all 2-outcome
HOCKEY_MARKETS = [
    Market("moneyline", ("home", "away"), "Match winner (inc. OT/SO)"),
    Market("puck_line", ("home", "away"), "Puck line (+/- 1.5)"),
    Market("totals", ("over", "under"), "Total goals over/under"),
]

HOCKEY_CONFIG = SportConfig(
    name="Ice Hockey",
    slug="ice_hockey",
    markets=HOCKEY_MARKETS,
    min_train_seasons=3,
    default_mc_sims=5000,
    default_min_consensus=3,
)

# Supported leagues
LEAGUES = {
    "NHL": {"country": "USA/Canada", "season_start": 10, "season_end": 6},
    "KHL": {"country": "Russia", "season_start": 9, "season_end": 4},
}


def register():
    """Register ice hockey with the global sport registry."""
    if not sport_registry.is_registered("ice_hockey"):
        sport_registry.register(HOCKEY_CONFIG)
