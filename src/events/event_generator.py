"""
event_generator.py – Convert detection + action results into structured event dicts,
with time-based deduplication to avoid redundant entries.
"""

import logging
from datetime import datetime, timedelta

from src.utils.time_utils import now_iso, now_readable

logger = logging.getLogger("cctv_assistant")


class EventGenerator:
    """
    Produces structured event records from action detection output.

    Deduplication: if the same *(person, action, object)* tuple was emitted
    within the last ``dedup_window`` seconds, the duplicate is suppressed.

    Parameters
    ----------
    dedup_window : int
        Seconds to suppress duplicate events.
    """

    def __init__(self, dedup_window: int = 30):
        self.dedup_window = dedup_window
        # key = (person, action, object) → last emission time
        self._recent: dict[tuple, datetime] = {}

    # ── dedup ──────────────────────────────────────────────────────────
    def _is_duplicate(self, key: tuple) -> bool:
        now = datetime.now()
        if key in self._recent:
            if (now - self._recent[key]).total_seconds() < self.dedup_window:
                return True
        self._recent[key] = now
        return False

    def _cleanup_old(self) -> None:
        """Remove entries older than 2× the dedup window."""
        cutoff = datetime.now() - timedelta(seconds=self.dedup_window * 2)
        self._recent = {k: v for k, v in self._recent.items() if v > cutoff}

    # ── main API ───────────────────────────────────────────────────────
    def generate_events(
        self,
        actions: list[dict],
        clip_path: str = "",
    ) -> list[dict]:
        """
        Convert a list of action dicts into structured event records.

        Parameters
        ----------
        actions : list[dict]
            Output from ``ActionDetector.detect_actions``.
        clip_path : str
            Path to the video clip associated with these events.

        Returns
        -------
        list[dict]
            Each dict::

                {
                    "timestamp": str,       # ISO-8601
                    "person": str,
                    "action": str,
                    "object": str,
                    "location": str,
                    "video_clip_path": str,
                    "summary": str,         # human-readable one-liner
                }
        """
        self._cleanup_old()

        events: list[dict] = []
        ts_iso = now_iso()
        ts_human = now_readable()

        for act in actions:
            key = (act["person"], act["action"], act.get("object", ""))
            if self._is_duplicate(key):
                continue

            summary = self._build_summary(act, ts_human)

            event = {
                "timestamp": ts_iso,
                "person": act["person"],
                "action": act["action"],
                "object": act.get("object", ""),
                "location": act.get("location", ""),
                "video_clip_path": clip_path,
                "summary": summary,
            }
            events.append(event)
            logger.info("EVENT: %s", summary)

        return events

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _build_summary(action: dict, time_str: str) -> str:
        """Build a natural-language one-liner for an event."""
        person = action.get("person", "Someone")
        act = action.get("action", "was detected")
        location = action.get("location", "")

        parts = [f"{person} {act}"]
        if location:
            parts.append(f"in the {location}")
        parts.append(f"at {time_str}")

        return " ".join(parts) + "."
