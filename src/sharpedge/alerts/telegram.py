import logging
from sharpedge.config import settings

logger = logging.getLogger(__name__)


async def send_alert(message: str) -> None:
    """Send alert to Telegram. Falls back to logging if not configured."""
    if not settings.telegram_bot_token or not settings.telegram_alert_chat_id:
        logger.warning(f"ALERT (Telegram not configured): {message}")
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        async with httpx.AsyncClient() as client:
            await client.post(url, json={
                "chat_id": settings.telegram_alert_chat_id,
                "text": f"SharpEdge Alert\n\n{message}",
                "parse_mode": "HTML",
            })
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def send_alert_sync(message: str) -> None:
    """Synchronous version for use in collectors."""
    if not settings.telegram_bot_token or not settings.telegram_alert_chat_id:
        logger.warning(f"ALERT: {message}")
        return
    try:
        import httpx
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        with httpx.Client() as client:
            client.post(url, json={
                "chat_id": settings.telegram_alert_chat_id,
                "text": f"SharpEdge Alert\n\n{message}",
            })
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
