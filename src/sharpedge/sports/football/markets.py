"""Football betting market definitions and helpers."""

# Market names used throughout the system
MARKET_1X2 = "1x2"
MARKET_OVER_UNDER = "over_under_2.5"
MARKET_BTTS = "btts"
MARKET_DOUBLE_CHANCE = "double_chance"
MARKET_ASIAN_HANDICAP = "asian_handicap"
MARKET_CORRECT_SCORE = "correct_score"

# Markets proven profitable in walk-forward backtest
PROFITABLE_MARKETS = frozenset({
    "1x2_home", "1x2_away", "1x2_draw",
    "over_25", "dc_x2",
})

# Per-market confidence thresholds (backtest-optimized)
MARKET_THRESHOLDS = {
    "1x2_home": 0.58,
    "1x2_away": 0.50,
    "1x2_draw": 0.50,
    "over_25": 0.55,
    "dc_x2": 0.65,
}

# Maximum odds (backtest: odds > 2.50 -> -11.3% ROI)
MAX_ODDS = 2.50
