"""
alert_manager.py – Rule engine for proactive Telegram alerts.

Triggers alerts when:
  • Unknown person detected
  • Movement detected at night
  • Door / entrance activity detected
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.utils.time_utils import is_night

logger = logging.getLogger("cctv_assistant")


class AlertManager:
    """
    Evaluates events against alert rules and sends notifications
    via the Telegram bot.

    Parameters
    ----------
    telegram_bot
        ``src.chatbot.telegram_bot.TelegramBot`` instance (needs ``send_alert``).
    config : dict
        The ``alerts`` section of ``config.yaml``.
    """

    def __init__(self, telegram_bot: Any = None, config: dict | None = None):
        self.bot = telegram_bot
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.alert_unknown = cfg.get("unknown_person", True)
        self.alert_night = cfg.get("night_movement", True)
        self.alert_door = cfg.get("door_activity", True)
        self.night_start = cfg.get("night_start_hour", 22)
        self.night_end = cfg.get("night_end_hour", 6)

        # Cooldown: suppress repeated alerts for the same type within N seconds
        self._cooldown: dict[str, datetime] = {}
        self._cooldown_seconds = 120

    # ── cooldown ───────────────────────────────────────────────────────
    def _on_cooldown(self, key: str) -> bool:
        now = datetime.now()
        if key in self._cooldown:
            elapsed = (now - self._cooldown[key]).total_seconds()
            if elapsed < self._cooldown_seconds:
                return True
        self._cooldown[key] = now
        return False

    # ── evaluation ─────────────────────────────────────────────────────
    def evaluate(self, events: list[dict]) -> list[str]:
        """
        Check events against rules and return a list of alert messages
        (empty list = no alerts).
        """
        if not self.enabled:
            return []

        alerts: list[str] = []
        now_hour = datetime.now().hour

        for event in events:
            person = event.get("person", "Unknown")
            action = event.get("action", "")
            location = event.get("location", "")
            ts_str = event.get("timestamp", "")

            # Rule 1: Unknown person
            if self.alert_unknown and person == "Unknown":
                key = f"unknown_{location}"
                if not self._on_cooldown(key):
                    time_part = datetime.now().strftime("%-I:%M %p")
                    alerts.append(
                        f"⚠️ Alert: Unknown person detected in the {location} at {time_part}."
                    )

            # Rule 2: Night movement
            if self.alert_night and is_night(now_hour, self.night_start, self.night_end):
                key = f"night_{person}"
                if not self._on_cooldown(key):
                    time_part = datetime.now().strftime("%-I:%M %p")
                    alerts.append(
                        f"🌙 Alert: Movement detected at night ({time_part}). Person: {person}."
                    )

            # Rule 3: Door / entrance activity
            if self.alert_door and location in ("entrance", "door"):
                key = f"door_{person}"
                if not self._on_cooldown(key):
                    time_part = datetime.now().strftime("%-I:%M %p")
                    alerts.append(
                        f"🚪 Alert: {person} detected at the entrance at {time_part}."
                    )

        return alerts

    # ── send ───────────────────────────────────────────────────────────
    async def send_alerts(self, events: list[dict]) -> None:
        """Evaluate events and send any triggered alerts via Telegram."""
        messages = self.evaluate(events)
        for msg in messages:
            logger.warning(msg)
            if self.bot:
                try:
                    await self.bot.send_alert(msg)
                except Exception as exc:
                    logger.error("Failed to send alert: %s", exc)

    def send_alerts_sync(self, events: list[dict]) -> None:
        """Synchronous wrapper for environments without a running event loop."""
        messages = self.evaluate(events)
        for msg in messages:
            logger.warning(msg)
            if self.bot:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self.bot.send_alert(msg))
                    else:
                        loop.run_until_complete(self.bot.send_alert(msg))
                except RuntimeError:
                    asyncio.run(self.bot.send_alert(msg))
