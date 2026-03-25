"""Market definitions for all supported sports."""

FOOTBALL_MARKETS = (
    ("1x2", ("home", "draw", "away"), "Match result"),
    ("over_under_2.5", ("over", "under"), "Total goals over/under 2.5"),
    ("btts", ("yes", "no"), "Both teams to score"),
    ("double_chance", ("1x", "12", "x2"), "Double chance"),
    ("asian_handicap", ("home", "away"), "Asian handicap"),
    ("correct_score", (), "Correct score"),  # dynamic outcomes
)

TENNIS_MARKETS = (
    ("match_winner", ("player1", "player2"), "Match winner"),
    ("set_betting", (), "Exact set score"),  # dynamic
    ("total_games", ("over", "under"), "Total games over/under"),
    ("set_handicap", ("player1", "player2"), "Set handicap"),
)

BASKETBALL_MARKETS = (
    ("moneyline", ("home", "away"), "Match winner"),
    ("spread", ("home", "away"), "Point spread"),
    ("totals", ("over", "under"), "Total points over/under"),
)

NHL_MARKETS = (
    ("moneyline", ("home", "away"), "Match winner"),
    ("puck_line", ("home", "away"), "Puck line +/- 1.5"),
    ("totals", ("over", "under"), "Total goals over/under"),
)

NFL_MARKETS = (
    ("moneyline", ("home", "away"), "Match winner"),
    ("spread", ("home", "away"), "Point spread"),
    ("totals", ("over", "under"), "Total points over/under"),
)

MLB_MARKETS = (
    ("moneyline", ("home", "away"), "Match winner"),
    ("run_line", ("home", "away"), "Run line +/- 1.5"),
    ("totals", ("over", "under"), "Total runs over/under"),
)
