"""
Migration script to set nationality to NULL where it equals 'Egyptian' in users and user_profiles.

Usage:
    python scripts/null_default_nationality.py          # Dry-run mode (default)
    python scripts/null_default_nationality.py --apply  # Apply changes
"""

import argparse
import os
import sqlite3

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/ra7al.db")


def null_default_nationality(db_path: str, apply_changes: bool = False) -> None:
    if not os.path.exists(db_path):
        print(f"Database file not found at: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    try:
        users_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE nationality = 'Egyptian'"
        ).fetchone()[0]

        profiles_count = conn.execute(
            "SELECT COUNT(*) FROM user_profiles WHERE nationality = 'Egyptian'"
        ).fetchone()[0]

        print(f"Database: {db_path}")
        print(f"Found {users_count} row(s) in 'users' with nationality = 'Egyptian'.")
        print(f"Found {profiles_count} row(s) in 'user_profiles' with nationality = 'Egyptian'.")

        if not apply_changes:
            print("DRY-RUN mode: No database changes were made. Pass --apply to execute updates.")
            return

        conn.execute("UPDATE users SET nationality = NULL WHERE nationality = 'Egyptian'")
        conn.execute("UPDATE user_profiles SET nationality = NULL WHERE nationality = 'Egyptian'")
        conn.commit()
        print("APPLIED: Updated nationality to NULL in matching rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set nationality to NULL where it equals 'Egyptian' in users and user_profiles."
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database file."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes to the database. Without this flag, script runs in dry-run mode."
    )
    args = parser.parse_args()
    null_default_nationality(args.db_path, apply_changes=args.apply)
