"""
db_manager.py – SQLite database wrapper for the events table.
"""

import sqlite3
from pathlib import Path
from typing import Optional


class DatabaseManager:
    """Thin wrapper around SQLite for event storage and retrieval."""

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT    NOT NULL,
        person          TEXT    DEFAULT 'Unknown',
        action          TEXT    NOT NULL,
        object          TEXT    DEFAULT '',
        location        TEXT    DEFAULT '',
        video_clip_path TEXT    DEFAULT '',
        summary         TEXT    DEFAULT ''
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    # ── connection management ──────────────────────────────────────────
    def connect(self) -> None:
        """Open (or reopen) the database connection."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _ensure_conn(self) -> sqlite3.Connection:
        if self.conn is None:
            self.connect()
        return self.conn  # type: ignore[return-value]

    # ── schema ─────────────────────────────────────────────────────────
    def initialize(self) -> None:
        """Create the *events* table if it does not exist."""
        conn = self._ensure_conn()
        conn.execute(self.CREATE_TABLE_SQL)
        conn.commit()

    # ── write ──────────────────────────────────────────────────────────
    def insert_event(
        self,
        timestamp: str,
        person: str = "Unknown",
        action: str = "",
        obj: str = "",
        location: str = "",
        video_clip_path: str = "",
        summary: str = "",
    ) -> int:
        """
        Insert a new event and return its row ID.

        Parameters
        ----------
        timestamp : str   ISO-8601 timestamp.
        person    : str   Identified person or ``"Unknown"``.
        action    : str   Detected action description.
        obj       : str   Object involved.
        location  : str   Area / zone.
        video_clip_path : str  Path to the saved clip.
        summary   : str   Human-readable one-line summary.

        Returns
        -------
        int   The ``rowid`` of the inserted row.
        """
        conn = self._ensure_conn()
        cursor = conn.execute(
            """
            INSERT INTO events (timestamp, person, action, object, location,
                                video_clip_path, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, person, action, obj, location, video_clip_path, summary),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    # ── read ───────────────────────────────────────────────────────────
    def get_events(
        self,
        limit: int = 50,
        person: str | None = None,
        since: str | None = None,
    ) -> list[dict]:
        """
        Retrieve events, optionally filtered by *person* and/or *since* timestamp.

        Returns a list of dicts.
        """
        conn = self._ensure_conn()
        query = "SELECT * FROM events WHERE 1=1"
        params: list = []

        if person:
            query += " AND LOWER(person) = LOWER(?)"
            params.append(person)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_today_events(self, limit: int = 100) -> list[dict]:
        """Return all events from today."""
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_events(limit=limit, since=today)

    def get_all_summaries(self) -> list[str]:
        """Return a flat list of every event summary string."""
        conn = self._ensure_conn()
        rows = conn.execute(
            "SELECT summary FROM events WHERE summary != '' ORDER BY timestamp"
        ).fetchall()
        return [r["summary"] for r in rows]

    def search_events(self, keyword: str, limit: int = 20) -> list[dict]:
        """Full-text keyword search across summary, person, action, object."""
        conn = self._ensure_conn()
        like = f"%{keyword}%"
        rows = conn.execute(
            """
            SELECT * FROM events
            WHERE summary   LIKE ?
               OR person    LIKE ?
               OR action    LIKE ?
               OR object    LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (like, like, like, like, limit),
        ).fetchall()
        return [dict(r) for r in rows]
