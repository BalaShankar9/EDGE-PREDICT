"""Tests for Telegram bot command handlers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sharpedge.bot.commands import handle_start, handle_help, handle_today, handle_record


@pytest.fixture
def mock_update():
    update = MagicMock()
    update.effective_chat.id = 12345
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    return MagicMock()


@pytest.mark.asyncio
async def test_start_command(mock_update, mock_context):
    await handle_start(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once()
    msg = mock_update.message.reply_text.call_args[0][0]
    assert "SharpEdge" in msg


@pytest.mark.asyncio
async def test_help_command(mock_update, mock_context):
    await handle_help(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once()
    msg = mock_update.message.reply_text.call_args[0][0]
    assert "/today" in msg


@pytest.mark.asyncio
async def test_today_command(mock_update, mock_context):
    with patch("sharpedge.bot.commands._get_todays_picks") as mock_picks:
        mock_picks.return_value = []
        await handle_today(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()


@pytest.mark.asyncio
async def test_record_command(mock_update, mock_context):
    with patch("sharpedge.bot.commands._get_track_record") as mock_record:
        mock_record.return_value = {
            "total_picks": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_profit": 0.0, "roi": 0.0,
            "avg_odds": 0.0, "by_tier": {},
        }
        await handle_record(mock_update, mock_context)
        mock_update.message.reply_text.assert_called_once()
