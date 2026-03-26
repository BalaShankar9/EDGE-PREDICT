"""American football sport configuration.

Registers american football with the global sport registry, defining all supported
leagues, markets, and sport-specific settings.
"""
from sharpedge.core.sport import SportConfig, Market, sport_registry

# NFL markets — all 2-outcome
NFL_MARKETS = [
    Market("moneyline", ("home", "away"), "Match winner"),
    Market("spread", ("home", "away"), "Point spread"),
    Market("totals", ("over", "under"), "Total points over/under"),
]

NFL_CONFIG = SportConfig(
    name="American Football",
    slug="american_football",
    markets=NFL_MARKETS,
    min_train_seasons=3,
    default_mc_sims=5000,
    default_min_consensus=3,
)

# Supported leagues
LEAGUES = {
    "NFL": {"country": "USA", "season_start": 9, "season_end": 2},
    "NCAAF": {"country": "USA", "season_start": 9, "season_end": 1},
}


def register():
    """Register american football with the global sport registry."""
    if not sport_registry.is_registered("american_football"):
        sport_registry.register(NFL_CONFIG)
