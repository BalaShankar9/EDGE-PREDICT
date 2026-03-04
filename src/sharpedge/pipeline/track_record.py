"""Track record calculator — aggregates pick history into performance metrics."""
from collections import defaultdict


def calculate_track_record(picks: list[dict]) -> dict:
    """Calculate track record from resolved pick dicts.

    Each pick must have: tier, league, result, profit_loss, best_odds, match_date.
    Returns: total_picks, wins, losses, win_rate, total_profit, roi,
    max_drawdown, avg_odds, longest_win_streak, longest_loss_streak, by_tier, by_league.
    """
    if not picks:
        return {
            "total_picks": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_profit": 0.0, "roi": 0.0,
            "max_drawdown": 0.0, "avg_odds": 0.0,
            "longest_win_streak": 0, "longest_loss_streak": 0,
            "by_tier": {}, "by_league": {},
        }

    wins = sum(1 for p in picks if p["result"] == "win")
    losses = sum(1 for p in picks if p["result"] == "loss")
    total = len(picks)
    total_profit = sum(p["profit_loss"] for p in picks)
    total_staked = total * 1.0
    avg_odds = sum(p["best_odds"] for p in picks) / total

    # Drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in picks:
        cumulative += p["profit_loss"]
        if cumulative > peak:
            peak = cumulative
        dd = cumulative - peak
        if dd < max_dd:
            max_dd = dd

    # Streaks
    win_streak = loss_streak = max_win = max_loss = 0
    for p in picks:
        if p["result"] == "win":
            win_streak += 1
            loss_streak = 0
        else:
            loss_streak += 1
            win_streak = 0
        max_win = max(max_win, win_streak)
        max_loss = max(max_loss, loss_streak)

    return {
        "total_picks": total,
        "wins": wins, "losses": losses,
        "win_rate": wins / total if total > 0 else 0.0,
        "total_profit": round(total_profit, 4),
        "roi": round((total_profit / total_staked) * 100, 4) if total_staked > 0 else 0.0,
        "max_drawdown": round(max_dd, 4),
        "avg_odds": round(avg_odds, 4),
        "longest_win_streak": max_win,
        "longest_loss_streak": max_loss,
        "by_tier": _group_stats(picks, "tier"),
        "by_league": _group_stats(picks, "league"),
    }


def _group_stats(picks: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        groups[p[key]].append(p)
    result = {}
    for name, group in groups.items():
        w = sum(1 for p in group if p["result"] == "win")
        t = len(group)
        profit = sum(p["profit_loss"] for p in group)
        result[name] = {
            "total_picks": t, "wins": w, "losses": t - w,
            "win_rate": w / t if t > 0 else 0.0,
            "total_profit": round(profit, 4),
            "roi": round((profit / t) * 100, 4) if t > 0 else 0.0,
        }
    return result
