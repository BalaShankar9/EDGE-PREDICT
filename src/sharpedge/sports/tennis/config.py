"""Tennis sport configuration.

Registers tennis with the global sport registry, defining all supported
tours, surfaces, markets, and sport-specific settings.
"""
from sharpedge.core.sport import SportConfig, Market, sport_registry

# Tennis markets
TENNIS_MARKETS = [
    Market("match_winner", ("player1", "player2"), "Match winner"),
    Market("total_games", ("over", "under"), "Total games over/under"),
    Market("set_handicap", ("player1", "player2"), "Set handicap"),
]

TENNIS_CONFIG = SportConfig(
    name="Tennis",
    slug="tennis",
    markets=TENNIS_MARKETS,
    min_train_seasons=2,  # tennis has more data per year
    default_mc_sims=5000,
    default_min_consensus=3,
)

# Supported tours
TOURS = {
    "ATP": {"gender": "male", "url_prefix": "atp"},
    "WTA": {"gender": "female", "url_prefix": "wta"},
}

# Court surfaces
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


def register():
    """Register tennis with the global sport registry."""
    if not sport_registry.is_registered("tennis"):
        sport_registry.register(TENNIS_CONFIG)
