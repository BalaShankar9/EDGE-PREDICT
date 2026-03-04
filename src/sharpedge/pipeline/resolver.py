"""Result resolution logic for daily picks."""
from typing import Optional


def resolve_pick(
    pick_market: str,
    home_goals: Optional[int],
    away_goals: Optional[int],
    best_odds: float,
    stake: float = 1.0,
) -> dict:
    """Resolve a single pick against actual match result.

    Returns dict with 'result' ('win'/'loss'/'void') and 'profit_loss'.
    """
    if home_goals is None or away_goals is None:
        return {"result": "void", "profit_loss": 0.0}

    total_goals = home_goals + away_goals
    both_scored = home_goals > 0 and away_goals > 0

    market_outcomes = {
        "1x2_home": home_goals > away_goals,
        "1x2_draw": home_goals == away_goals,
        "1x2_away": away_goals > home_goals,
        "over_25": total_goals > 2.5,
        "under_25": total_goals < 2.5,
        "btts_yes": both_scored,
        "btts_no": not both_scored,
    }

    won = market_outcomes.get(pick_market, False)
    if won:
        profit_loss = (best_odds - 1) * stake
    else:
        profit_loss = -stake

    return {"result": "win" if won else "loss", "profit_loss": round(profit_loss, 4)}
