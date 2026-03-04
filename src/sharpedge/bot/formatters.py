"""Telegram message formatting for picks, results, and track record."""


def format_pick_message(pick: dict) -> str:
    """Format a single pick for Telegram channel broadcast."""
    tier_upper = pick["tier"].upper()
    edge_pct = f"{pick['edge'] * 100:.1f}" if pick["edge"] < 1 else f"{pick['edge']:.1f}"
    prob_pct = f"{pick['model_prob'] * 100:.1f}" if pick["model_prob"] <= 1 else f"{pick['model_prob']:.1f}"
    flags = ", ".join(pick.get("risk_flags", [])) or "None"
    league_tag = pick["league"].replace(" ", "")

    return (
        f"\U0001f3af SharpEdge Banker Pick\n\n"
        f"\u26bd {pick['home_team']} vs {pick['away_team']}\n"
        f"\U0001f3c6 {pick['league']} | {pick['match_date']}\n\n"
        f"\U0001f4ca Pick: {pick['pick_selection']} ({pick['pick_market'].upper().replace('_', ' ')})\n"
        f"\U0001f4b0 Best Odds: {pick['best_odds']:.2f} @ {pick['bookmaker']}\n"
        f"\U0001f4c8 Edge: +{edge_pct}% vs market\n"
        f"\U0001f3c5 Tier: {tier_upper}\n\n"
        f"Model: {prob_pct}% confidence\n"
        f"Meta: {pick['meta_agreement']}/4 sites agree\n"
        f"Risk Flags: {flags}\n\n"
        f"#{league_tag} #{tier_upper.capitalize()}"
    )


def format_results_message(results: list[dict], summary: dict, date_label: str) -> str:
    """Format daily results summary for Telegram channel."""
    lines = [f"\U0001f4ca Yesterday's Results ({date_label})\n"]
    for r in results:
        icon = "\u2705" if r["result"] == "win" else "\u274c"
        score = f"{r.get('home_goals', '?')}-{r.get('away_goals', '?')}"
        profit = f"+{r['profit_loss']:.2f}u" if r["profit_loss"] >= 0 else f"{r['profit_loss']:.2f}u"
        lines.append(
            f"{icon} {r['home_team']} {score} {r['away_team']} "
            f"\u2014 {r['pick_selection']} @ {r['best_odds']:.2f} \u2192 {profit}"
        )
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    profit = summary.get("profit", 0)
    profit_str = f"+{profit:.2f}u" if profit >= 0 else f"{profit:.2f}u"
    lines.append(f"\nDay: {wins}W {losses}L | {profit_str}")
    m_wins = summary.get("month_wins", 0)
    m_losses = summary.get("month_losses", 0)
    m_total = m_wins + m_losses
    m_pct = f"{m_wins / m_total * 100:.0f}%" if m_total > 0 else "0%"
    m_profit = summary.get("month_profit", 0)
    m_roi = summary.get("month_roi", 0)
    lines.append(f"Month: {m_wins}W {m_losses}L ({m_pct}) | +{m_profit:.1f}u | ROI: +{m_roi:.1f}%")
    a_wins = summary.get("all_wins", 0)
    a_losses = summary.get("all_losses", 0)
    a_total = a_wins + a_losses
    a_pct = f"{a_wins / a_total * 100:.0f}%" if a_total > 0 else "0%"
    a_profit = summary.get("all_profit", 0)
    lines.append(f"All Time: {a_wins}W {a_losses}L ({a_pct}) | +{a_profit:.1f}u")
    return "\n".join(lines)


def format_record_message(record: dict) -> str:
    """Format track record summary for /record command."""
    win_pct = f"{record['win_rate'] * 100:.1f}%"
    profit = record["total_profit"]
    profit_str = f"+{profit:.1f}u" if profit >= 0 else f"{profit:.1f}u"
    lines = [
        "\U0001f4ca SharpEdge Track Record\n",
        f"Total Picks: {record['total_picks']}",
        f"Record: {record['wins']}W {record['losses']}L ({win_pct})",
        f"Profit: {profit_str}",
        f"ROI: +{record['roi']:.1f}%",
        f"Avg Odds: {record['avg_odds']:.2f}",
    ]
    tiers = record.get("by_tier", {})
    if tiers:
        lines.append("\n\U0001f3c5 By Tier:")
        for tier_name, stats in tiers.items():
            t_pct = f"{stats['win_rate'] * 100:.1f}%"
            t_profit = stats["total_profit"]
            t_str = f"+{t_profit:.1f}u" if t_profit >= 0 else f"{t_profit:.1f}u"
            lines.append(f"  {tier_name.upper()}: {stats['wins']}W {stats['losses']}L ({t_pct}) | {t_str}")
    return "\n".join(lines)
