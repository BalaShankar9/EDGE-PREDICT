"""Football sport configuration.

Registers football with the global sport registry, defining all supported
leagues, markets, and sport-specific settings.
"""
from sharpedge.core.sport import SportConfig, Market, sport_registry

# Football markets
FOOTBALL_MARKETS = [
    Market("1x2", ("home", "draw", "away"), "Match result"),
    Market("over_under_2.5", ("over", "under"), "Total goals over/under 2.5"),
    Market("btts", ("yes", "no"), "Both teams to score"),
    Market("double_chance", ("1x", "12", "x2"), "Double chance"),
    Market("asian_handicap", ("home", "away"), "Asian handicap"),
]

FOOTBALL_CONFIG = SportConfig(
    name="Football",
    slug="football",
    markets=FOOTBALL_MARKETS,
    min_train_seasons=3,
    default_mc_sims=5000,
    default_min_consensus=3,
)

# 12 supported leagues with football-data.co.uk codes
LEAGUES = {
    # Big 5
    "Premier League": {"country": "England", "code": "E0", "tier": 1},
    "La Liga": {"country": "Spain", "code": "SP1", "tier": 1},
    "Bundesliga": {"country": "Germany", "code": "D1", "tier": 1},
    "Serie A": {"country": "Italy", "code": "I1", "tier": 1},
    "Ligue 1": {"country": "France", "code": "F1", "tier": 1},
    # Expansion
    "Eredivisie": {"country": "Netherlands", "code": "N1", "tier": 2},
    "Liga Portugal": {"country": "Portugal", "code": "P1", "tier": 2},
    "Belgian Pro League": {"country": "Belgium", "code": "B1", "tier": 2},
    "Turkish Super Lig": {"country": "Turkey", "code": "T1", "tier": 2},
    "Scottish Premiership": {"country": "Scotland", "code": "SC0", "tier": 2},
    "Super League Greece": {"country": "Greece", "code": "G1", "tier": 2},
    "Championship": {"country": "England", "code": "EC", "tier": 2},
}

# League-specific reliability factors for probability calibration
LEAGUE_RELIABILITY = {
    "La Liga": 1.08,
    "Premier League": 0.99,
    "Serie A": 0.99,
    "Bundesliga": 0.99,
    "Ligue 1": 0.97,
    # Expansion leagues default to 1.0 (neutral)
}


def register():
    """Register football with the global sport registry."""
    if not sport_registry.is_registered("football"):
        sport_registry.register(FOOTBALL_CONFIG)
