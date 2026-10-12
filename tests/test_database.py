"""
test_database.py – Unit tests for DatabaseManager using an in-memory SQLite database.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Tests against an in-memory SQLite database (no files created)."""

    def setUp(self):
        self.db = DatabaseManager(":memory:")
        self.db.connect()
        self.db.initialize()

    def tearDown(self):
        self.db.close()

    def test_insert_and_retrieve(self):
        row_id = self.db.insert_event(
            timestamp="2026-03-15T19:12:00",
            person="Sister",
            action="entered the house",
            obj="",
            location="entrance",
            summary="Sister entered the house at 7:12 PM.",
        )
        self.assertIsNotNone(row_id)

        events = self.db.get_events(limit=10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["person"], "Sister")

    def test_filter_by_person(self):
        self.db.insert_event(
            timestamp="2026-03-15T19:12:00", person="Sister",
            action="entered", summary="Sister entered at 7:12 PM."
        )
        self.db.insert_event(
            timestamp="2026-03-15T19:15:00", person="Brother",
            action="entered", summary="Brother entered at 7:15 PM."
        )

        sister_events = self.db.get_events(person="Sister")
        self.assertEqual(len(sister_events), 1)
        self.assertEqual(sister_events[0]["person"], "Sister")

    def test_search_events(self):
        self.db.insert_event(
            timestamp="2026-03-15T19:14:00", person="Sister",
            action="opened the fridge", obj="refrigerator",
            location="kitchen", summary="Sister opened the fridge at 7:14 PM."
        )
        self.db.insert_event(
            timestamp="2026-03-15T19:18:00", person="Sister",
            action="watching TV", obj="tv",
            location="living room", summary="Sister watching TV at 7:18 PM."
        )

        results = self.db.search_events("fridge")
        self.assertEqual(len(results), 1)
        self.assertIn("fridge", results[0]["summary"])

    def test_get_all_summaries(self):
        self.db.insert_event(timestamp="2026-03-15T19:12:00", person="A",
                             action="a", summary="Event A.")
        self.db.insert_event(timestamp="2026-03-15T19:13:00", person="B",
                             action="b", summary="Event B.")

        summaries = self.db.get_all_summaries()
        self.assertEqual(len(summaries), 2)
        self.assertIn("Event A.", summaries)

    def test_empty_database(self):
        events = self.db.get_events()
        self.assertEqual(events, [])

    def test_today_events_returns_list(self):
        # Should simply return an empty list if nothing was inserted today
        events = self.db.get_today_events()
        self.assertIsInstance(events, list)


class TestEventGeneratorDedup(unittest.TestCase):
    """Tests for EventGenerator deduplication."""

    def test_dedup_suppresses_repeat(self):
        from src.events.event_generator import EventGenerator

        gen = EventGenerator(dedup_window=60)

        actions = [{"person": "Sister", "action": "entered the house",
                     "object": "", "location": "entrance", "confidence": 0.9}]

        events1 = gen.generate_events(actions)
        self.assertEqual(len(events1), 1)

        events2 = gen.generate_events(actions)
        self.assertEqual(len(events2), 0, "Duplicate event should be suppressed")


if __name__ == "__main__":
    unittest.main()
