"""Telegram bot command handlers."""
import logging
from datetime import date
from telegram import Update
from telegram.ext import ContextTypes
from sharpedge.db.engine import get_session
from sharpedge.db.models import DailyPick
from sharpedge.bot.formatters import format_pick_message, format_record_message
from sharpedge.pipeline.track_record import calculate_track_record

logger = logging.getLogger(__name__)


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "\U0001f3af Welcome to SharpEdge AI!\n\n"
        "AI-powered football predictions with verified edge.\n\n"
        "Commands:\n"
        "/today \u2014 Today's picks\n"
        "/platinum \u2014 Platinum picks only\n"
        "/gold \u2014 Gold + Platinum picks\n"
        "/record \u2014 Track record\n"
        "/leagues \u2014 Available leagues\n"
        "/help \u2014 Help & about"
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "\U0001f4d6 SharpEdge AI Commands\n\n"
        "/today \u2014 All picks for today\n"
        "/platinum \u2014 Platinum tier picks (85%+ confidence)\n"
        "/gold \u2014 Gold + Platinum picks (78%+)\n"
        "/record \u2014 Overall track record\n"
        "/leagues \u2014 Big 5 leagues covered\n\n"
        "Picks are posted daily at ~08:00 UTC.\n"
        "Results posted at ~23:00 UTC."
    )


async def handle_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = _get_todays_picks()
    if not picks:
        await update.message.reply_text("No picks for today yet. Picks are generated at 08:00 UTC.")
        return
    for pick in picks:
        await update.message.reply_text(format_pick_message(pick))


async def handle_platinum(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = _get_todays_picks(tier="platinum")
    if not picks:
        await update.message.reply_text("No Platinum picks for today.")
        return
    for pick in picks:
        await update.message.reply_text(format_pick_message(pick))


async def handle_gold(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    picks = _get_todays_picks(min_tier="gold")
    if not picks:
        await update.message.reply_text("No Gold/Platinum picks for today.")
        return
    for pick in picks:
        await update.message.reply_text(format_pick_message(pick))


async def handle_record(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    record = _get_track_record()
    msg = format_record_message(record)
    await update.message.reply_text(msg)


async def handle_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "\U0001f30d Leagues Covered\n\n"
        "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f Premier League\n"
        "\U0001f1ea\U0001f1f8 La Liga\n"
        "\U0001f1e9\U0001f1ea Bundesliga\n"
        "\U0001f1ee\U0001f1f9 Serie A\n"
        "\U0001f1eb\U0001f1f7 Ligue 1"
    )


def _get_todays_picks(tier: str = None, min_tier: str = None) -> list[dict]:
    session = get_session()
    try:
        query = session.query(DailyPick).filter(DailyPick.match_date == date.today())
        if tier:
            query = query.filter(DailyPick.tier == tier)
        elif min_tier == "gold":
            query = query.filter(DailyPick.tier.in_(["platinum", "gold"]))
        picks = query.all()
        return [
            {"home_team": p.home_team, "away_team": p.away_team,
             "league": p.league, "match_date": str(p.match_date),
             "pick_selection": p.pick_selection, "pick_market": p.pick_market,
             "best_odds": p.best_odds, "bookmaker": p.bookmaker,
             "edge": p.edge, "tier": p.tier,
             "model_prob": p.model_prob, "meta_agreement": p.meta_agreement,
             "risk_flags": p.risk_flags or []}
            for p in picks
        ]
    finally:
        session.close()


def _get_track_record() -> dict:
    session = get_session()
    try:
        picks = session.query(DailyPick).filter(DailyPick.result.isnot(None)).all()
        pick_dicts = [
            {"tier": p.tier, "league": p.league, "result": p.result,
             "profit_loss": p.profit_loss, "best_odds": p.best_odds,
             "match_date": str(p.match_date)}
            for p in picks
        ]
        return calculate_track_record(pick_dicts)
    finally:
        session.close()
