"""
initialize_database.py – Create the SQLite events table.

Usage:
    python scripts/initialize_database.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, resolve_path
from database.db_manager import DatabaseManager


def main() -> None:
    config = load_config()
    db_path = resolve_path(config["database"]["path"])

    print(f"📦 Initializing database at: {db_path}")
    db = DatabaseManager(db_path)
    db.initialize()
    db.close()
    print("✅ Database initialized successfully!")
    print("   Table 'events' is ready.")


if __name__ == "__main__":
    main()
