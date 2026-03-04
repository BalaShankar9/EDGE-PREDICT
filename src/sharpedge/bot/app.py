"""Telegram bot application -- entry point. Run with: python -m sharpedge.bot"""
import asyncio
import logging
from telegram.ext import ApplicationBuilder, CommandHandler
from sharpedge.config import settings
from sharpedge.bot.commands import (
    handle_start, handle_help, handle_today,
    handle_platinum, handle_gold, handle_record, handle_leagues,
)
from sharpedge.bot.broadcast import broadcast_new_picks

logger = logging.getLogger(__name__)


def create_bot_app():
    if not settings.telegram_bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN not configured")
    app = ApplicationBuilder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("today", handle_today))
    app.add_handler(CommandHandler("platinum", handle_platinum))
    app.add_handler(CommandHandler("gold", handle_gold))
    app.add_handler(CommandHandler("record", handle_record))
    app.add_handler(CommandHandler("leagues", handle_leagues))
    return app


async def _poll_broadcasts(app) -> None:
    while True:
        try:
            count = await broadcast_new_picks(app.bot)
            if count > 0:
                logger.info(f"Broadcasted {count} picks")
        except Exception as e:
            logger.error(f"Broadcast poll error: {e}")
        await asyncio.sleep(settings.telegram_poll_interval)


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting SharpEdge Telegram Bot...")
    app = create_bot_app()
    loop = asyncio.get_event_loop()
    app.post_init = lambda _app: loop.create_task(_poll_broadcasts(_app))
    app.run_polling()


if __name__ == "__main__":
    main()
