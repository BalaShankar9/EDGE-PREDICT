"""Channel broadcast logic -- polls DB for unbroadcasted picks."""
import logging
from datetime import datetime
from telegram import Bot
from sharpedge.db.engine import get_session
from sharpedge.db.models import DailyPick
from sharpedge.bot.formatters import format_pick_message
from sharpedge.config import settings

logger = logging.getLogger(__name__)


async def broadcast_new_picks(bot: Bot) -> int:
    if not settings.telegram_channel_id:
        logger.warning("No channel ID configured, skipping broadcast")
        return 0
    session = get_session()
    try:
        picks = (session.query(DailyPick)
                 .filter(DailyPick.broadcasted_at.is_(None))
                 .order_by(DailyPick.tier.asc()).all())
        count = 0
        for pick in picks:
            pick_dict = {
                "home_team": pick.home_team, "away_team": pick.away_team,
                "league": pick.league, "match_date": str(pick.match_date),
                "pick_selection": pick.pick_selection, "pick_market": pick.pick_market,
                "best_odds": pick.best_odds, "bookmaker": pick.bookmaker,
                "edge": pick.edge, "tier": pick.tier,
                "model_prob": pick.model_prob, "meta_agreement": pick.meta_agreement,
                "risk_flags": pick.risk_flags or [],
            }
            try:
                await bot.send_message(chat_id=settings.telegram_channel_id, text=format_pick_message(pick_dict))
                pick.broadcasted_at = datetime.now()
                session.commit()
                count += 1
            except Exception as e:
                logger.error(f"Failed to broadcast pick {pick.id}: {e}")
        return count
    finally:
        session.close()
