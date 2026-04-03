"""Per-league confidence adjustment.

Applies empirical reliability factors per league to model probabilities.
Leagues where our model over-predicts get dampened; leagues where
it under-predicts get amplified.

Factors derived from walk-forward accuracy at >= 0.60 confidence.
"""

# Reliability factors: >1.0 = model outperforms, <1.0 = model underperforms
LEAGUE_RELIABILITY = {
    "La Liga": 1.08,
    "Premier League": 0.99,
    "Serie A": 0.99,
    "Bundesliga": 0.99,
    "Ligue 1": 0.97,
    "Eredivisie": 0.95,
    "Liga Portugal": 0.96,
    "Belgian Pro League": 0.94,
    "Turkish Super Lig": 0.93,
    "Scottish Premiership": 0.95,
    "Super League Greece": 0.92,
    "Championship": 0.96,
}


def adjust_confidence(prob: float, league: str) -> float:
    """Adjust model probability by league reliability factor.

    Parameters
    ----------
    prob : raw model probability (0-1)
    league : league name

    Returns
    -------
    adjusted probability, clamped to [0.01, 0.99]
    """
    factor = LEAGUE_RELIABILITY.get(league, 1.0)
    adjusted = 0.5 + (prob - 0.5) * factor
    return max(0.01, min(0.99, adjusted))
