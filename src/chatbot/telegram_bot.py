"""
telegram_bot.py – Telegram bot interface with /ask, /status, and /help commands.
"""

import os
import logging
import asyncio
from typing import Any

logger = logging.getLogger("cctv_assistant")


class TelegramBot:
    """
    Wraps ``python-telegram-bot`` to expose the CCTV assistant over Telegram.

    Commands
    --------
    * ``/ask <question>``  – Ask a natural-language question.
    * ``/status``          – Show system status.
    * ``/today``           – List today's events.
    * ``/help``            – Show available commands.

    Parameters
    ----------
    query_engine
        ``src.chatbot.query_engine.QueryEngine`` instance.
    db_manager : optional
        ``database.db_manager.DatabaseManager`` for ``/today``.
    bot_token : str | None
        Override for the Telegram bot token (defaults to env var).
    """

    def __init__(self, query_engine, db_manager=None, bot_token: str | None = None):
        self.query_engine = query_engine
        self.db = db_manager
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.application: Any = None

    # ── handlers ───────────────────────────────────────────────────────
    async def _cmd_start(self, update, context) -> None:
        await update.message.reply_text(
            "👋 Welcome to the Smart AI CCTV Assistant!\n\n"
            "Use /ask <question> to query your surveillance logs.\n"
            "Type /help for more commands."
        )

    async def _cmd_help(self, update, context) -> None:
        await update.message.reply_text(
            "📖 *Available Commands*\n\n"
            "/ask <question> – Ask about events\n"
            "/today – Show today's events\n"
            "/status – System status\n"
            "/help – Show this message",
            parse_mode="Markdown",
        )

    async def _cmd_ask(self, update, context) -> None:
        question = " ".join(context.args) if context.args else ""
        if not question:
            await update.message.reply_text("❓ Please provide a question. Example:\n/ask Did anyone come home?")
            return

        await update.message.reply_text("🔍 Searching…")
        try:
            answer = self.query_engine.answer(question)
            await update.message.reply_text(answer)
        except Exception as exc:
            logger.error("Error answering query: %s", exc)
            await update.message.reply_text("⚠️ Sorry, an error occurred while processing your question.")

    async def _cmd_today(self, update, context) -> None:
        if not self.db:
            await update.message.reply_text("Database not available.")
            return
        events = self.db.get_today_events(limit=20)
        if not events:
            await update.message.reply_text("📭 No events recorded today.")
            return

        lines = ["📋 *Today's Events*\n"]
        for e in events:
            lines.append(f"• {e.get('summary', 'N/A')}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _cmd_status(self, update, context) -> None:
        total = 0
        if self.db:
            total = len(self.db.get_events(limit=99999))
        await update.message.reply_text(
            f"✅ System is running.\n"
            f"📊 Total events logged: {total}"
        )

    # ── bot lifecycle ──────────────────────────────────────────────────
    def build(self) -> Any:
        """Build the ``Application`` object with all command handlers."""
        from telegram.ext import Application, CommandHandler

        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN is not set – bot will not start")
            return None

        self.application = (
            Application.builder().token(self.bot_token).build()
        )
        self.application.add_handler(CommandHandler("start", self._cmd_start))
        self.application.add_handler(CommandHandler("help", self._cmd_help))
        self.application.add_handler(CommandHandler("ask", self._cmd_ask))
        self.application.add_handler(CommandHandler("today", self._cmd_today))
        self.application.add_handler(CommandHandler("status", self._cmd_status))

        logger.info("Telegram bot built successfully")
        return self.application

    def run(self) -> None:
        """Start the bot in long-polling mode (blocking).

        Uses a manually created event loop so the bot can run safely
        in a daemon thread (``run_polling`` calls ``set_wakeup_fd``
        which is restricted to the main thread).
        """
        app = self.build()
        if app is None:
            return
        logger.info("Starting Telegram bot polling…")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(app.initialize())
            loop.run_until_complete(
                app.updater.start_polling(drop_pending_updates=True)
            )
            loop.run_until_complete(app.start())
            loop.run_forever()
        except Exception as exc:
            logger.error("Telegram bot polling error: %s", exc)
        finally:
            try:
                loop.run_until_complete(app.updater.stop())
                loop.run_until_complete(app.stop())
                loop.run_until_complete(app.shutdown())
            except Exception:
                pass
            loop.close()

    async def send_alert(self, message: str) -> None:
        """Send a proactive alert message to the configured chat."""
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not chat_id or not self.bot_token:
            logger.warning("Cannot send alert – TELEGRAM_CHAT_ID or token not set")
            return

        try:
            from telegram import Bot
            bot = Bot(token=self.bot_token)
            await bot.send_message(chat_id=chat_id, text=message)
            logger.info("Alert sent: %s", message[:80])
        except Exception as exc:
            logger.error("Failed to send Telegram alert: %s", exc)
