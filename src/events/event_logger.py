"""
event_logger.py – Persist structured events to the SQLite database
and optionally to the vector store for semantic search.
"""

import logging
from pathlib import Path

# Use a string import path so the module works regardless of how it is invoked.
import sys

logger = logging.getLogger("cctv_assistant")


class EventLogger:
    """
    Writes event dicts to the database and, when available, to the
    embedding / FAISS vector store.

    Parameters
    ----------
    db_manager
        An instance of ``database.db_manager.DatabaseManager``.
    event_embedder : optional
        An instance of ``src.memory.event_embedding.EventEmbedder``.
    """

    def __init__(self, db_manager, event_embedder=None):
        self.db = db_manager
        self.embedder = event_embedder

    def log_event(self, event: dict) -> int:
        """
        Insert a single event dict and return the database row ID.
        """
        row_id = self.db.insert_event(
            timestamp=event["timestamp"],
            person=event.get("person", "Unknown"),
            action=event.get("action", ""),
            obj=event.get("object", ""),
            location=event.get("location", ""),
            video_clip_path=event.get("video_clip_path", ""),
            summary=event.get("summary", ""),
        )
        logger.debug("Logged event id=%d: %s", row_id, event.get("summary", ""))

        # Optionally embed for semantic search
        if self.embedder and event.get("summary"):
            try:
                self.embedder.embed_and_store(event["summary"], metadata=event)
            except Exception as exc:
                logger.error("Failed to embed event: %s", exc)

        return row_id

    def log_events(self, events: list[dict]) -> list[int]:
        """Insert multiple events.  Returns a list of row IDs."""
        return [self.log_event(e) for e in events]
