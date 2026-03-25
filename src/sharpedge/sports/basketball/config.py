"""Basketball sport configuration.

Registers basketball with the global sport registry, defining all supported
leagues, markets, and sport-specific settings.
"""
from sharpedge.core.sport import SportConfig, Market, sport_registry

# Basketball markets — all 2-outcome
BASKETBALL_MARKETS = [
    Market("moneyline", ("home", "away"), "Match winner"),
    Market("spread", ("home", "away"), "Point spread"),
    Market("totals", ("over", "under"), "Total points over/under"),
]

BASKETBALL_CONFIG = SportConfig(
    name="Basketball",
    slug="basketball",
    markets=BASKETBALL_MARKETS,
    min_train_seasons=2,
    default_mc_sims=5000,
    default_min_consensus=3,
)

# Supported leagues
LEAGUES = {
    "NBA": {"country": "USA", "season_start": 10, "season_end": 6},
    "EuroLeague": {"country": "Europe", "season_start": 10, "season_end": 5},
}


def register():
    """Register basketball with the global sport registry."""
    if not sport_registry.is_registered("basketball"):
        sport_registry.register(BASKETBALL_CONFIG)
