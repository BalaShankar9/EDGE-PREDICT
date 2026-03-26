"""Baseball sport configuration.

Registers baseball with the global sport registry, defining all supported
leagues, markets, and sport-specific settings.
"""
from sharpedge.core.sport import SportConfig, Market, sport_registry

# Baseball markets — all 2-outcome
BASEBALL_MARKETS = [
    Market("moneyline", ("home", "away"), "Match winner"),
    Market("run_line", ("home", "away"), "Run line (+/- 1.5)"),
    Market("totals", ("over", "under"), "Total runs over/under"),
]

BASEBALL_CONFIG = SportConfig(
    name="Baseball",
    slug="baseball",
    markets=BASEBALL_MARKETS,
    min_train_seasons=3,
    default_mc_sims=5000,
    default_min_consensus=3,
)

# Supported leagues
LEAGUES = {
    "MLB": {"country": "USA", "season_start": 3, "season_end": 10},
    "NPB": {"country": "Japan", "season_start": 3, "season_end": 10},
}


def register():
    """Register baseball with the global sport registry."""
    if not sport_registry.is_registered("baseball"):
        sport_registry.register(BASEBALL_CONFIG)
