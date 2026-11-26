"""
User Repository Layer (SQLite & Interface)
Implements Repository Pattern for persistent user account storage.
Module: src/db/repositories/user_repo.py
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/ra7al.db")


class UserRecord(BaseModel):
    """Domain Entity representing a user record."""
    id: Optional[int] = None
    user_id: str
    username: str
    name: str
    email: Optional[str] = None
    hashed_password: str
    nationality: Optional[str] = None
    created_at: Optional[str] = None


class IUserRepository(ABC):
    """Abstract interface for User Repository."""

    @abstractmethod
    def create(self, user: UserRecord) -> UserRecord:
        """Persist a new user record."""
        pass

    @abstractmethod
    def get_by_user_id(self, user_id: str) -> Optional[UserRecord]:
        """Fetch user by unique user_id."""
        pass

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[UserRecord]:
        """Fetch user by username (case-insensitive)."""
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[UserRecord]:
        """Fetch user by email (case-insensitive)."""
        pass

    @abstractmethod
    def list_all(self) -> List[UserRecord]:
        """List all users."""
        pass

    @abstractmethod
    def delete(self, user_id: str) -> bool:
        """Delete user by user_id."""
        pass


class SQLiteUserRepository(IUserRepository):
    """Concrete SQLite implementation of IUserRepository."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        # Ensure directory exists if path has directory components
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Legacy helper for raw connection access if needed."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _init_db(self) -> None:
        """Creates table and indices if they do not exist."""
        with self._connect() as conn:
            cursor = conn.cursor()
            # Note: fresh DBs use 'nationality TEXT' with no DEFAULT. Existing DBs keep their old DEFAULT,
            # which is harmless because inserts always pass an explicit value.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    hashed_password TEXT NOT NULL,
                    nationality TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    nationality TEXT,
                    dietary_json TEXT DEFAULT '[]',
                    budget_level TEXT DEFAULT 'moderate',
                    interests_json TEXT DEFAULT '[]',
                    avoided_interests_json TEXT DEFAULT '[]',
                    accessibility_needs_json TEXT DEFAULT '[]',
                    travel_habits_json TEXT DEFAULT '[]',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
            """)
            self._ensure_column(conn, "user_profiles", "avoided_interests_json", "TEXT DEFAULT '[]'")
            conn.commit()

    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not user_id:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM user_profiles WHERE LOWER(user_id) = LOWER(?)",
                (user_id.strip(),)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._profile_row_to_dict(row)

    def _write_profile(self, conn: sqlite3.Connection, profile: Dict[str, Any]) -> None:
        user_id = str(profile["user_id"]).strip().lower()
        nat = profile.get("nationality")
        if nat and str(nat).strip():
            nat = str(nat).strip()
        else:
            nat = None

        interests = profile.get("liked_interests") if "liked_interests" in profile else profile.get("interests", [])
        avoided_interests = profile.get("avoided_interests", [])

        conn.execute(
            """
            INSERT INTO user_profiles (
                user_id, name, nationality, dietary_json, budget_level,
                interests_json, avoided_interests_json, accessibility_needs_json, travel_habits_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                nationality = excluded.nationality,
                dietary_json = excluded.dietary_json,
                budget_level = excluded.budget_level,
                interests_json = excluded.interests_json,
                avoided_interests_json = excluded.avoided_interests_json,
                accessibility_needs_json = excluded.accessibility_needs_json,
                travel_habits_json = excluded.travel_habits_json,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                profile.get("name", "Traveler"),
                nat,
                json.dumps(profile.get("dietary", []), default=str),
                profile.get("budget_level", "moderate"),
                json.dumps(interests),
                json.dumps(avoided_interests),
                json.dumps(profile.get("accessibility_needs", [])),
                json.dumps(profile.get("travel_habits", [])),
                datetime.now(timezone.utc).isoformat(),
            )
        )

    def upsert_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(profile["user_id"]).strip().lower()
        with self._connect() as conn:
            self._write_profile(conn, profile)
            conn.commit()
        return self.get_profile(user_id) or profile

    def update_profile_atomic(self, user_id: str, mutate) -> Optional[Dict[str, Any]]:
        uid = user_id.strip().lower()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (uid,)).fetchone()
            if row is None:
                conn.rollback()
                return None
            new = mutate(self._profile_row_to_dict(row))
            self._write_profile(conn, {**new, "user_id": uid})
            conn.commit()
        return self.get_profile(uid)

    @staticmethod
    def _profile_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        nat = row["nationality"]
        if nat and str(nat).strip():
            nat = str(nat).strip()
        else:
            nat = None

        cols = row.keys()
        avoided = json.loads(row["avoided_interests_json"] or "[]") if "avoided_interests_json" in cols else []

        return {
            "user_id": row["user_id"],
            "name": row["name"],
            "nationality": nat,
            "dietary": json.loads(row["dietary_json"] or "[]"),
            "budget_level": row["budget_level"] or "moderate",
            "liked_interests": json.loads(row["interests_json"] or "[]"),
            "avoided_interests": avoided,
            "accessibility_needs": json.loads(row["accessibility_needs_json"] or "[]"),
            "travel_habits": json.loads(row["travel_habits_json"] or "[]"),
        }

    def create(self, user: UserRecord) -> UserRecord:
        """Inserts a new user record into SQLite."""
        nat = user.nationality.strip() if user.nationality and user.nationality.strip() else None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (user_id, username, name, email, hashed_password, nationality, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id.strip().lower(),
                    user.username.strip().lower(),
                    user.name.strip(),
                    user.email.strip().lower() if user.email else None,
                    user.hashed_password,
                    nat,
                    user.created_at or datetime.now(timezone.utc).isoformat(),
                )
            )
            conn.commit()
            user.id = cursor.lastrowid
            user.nationality = nat
            return user

    def get_by_user_id(self, user_id: str) -> Optional[UserRecord]:
        """Fetches a user by user_id."""
        if not user_id:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE LOWER(user_id) = LOWER(?)",
                (user_id.strip(),)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_by_username(self, username: str) -> Optional[UserRecord]:
        """Fetches a user by username."""
        if not username:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE LOWER(username) = LOWER(?)",
                (username.strip(),)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_by_email(self, email: str) -> Optional[UserRecord]:
        """Fetches a user by email."""
        if not email:
            return None
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE LOWER(email) = LOWER(?)",
                (email.strip(),)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def list_all(self) -> List[UserRecord]:
        """Returns all registered users."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users ORDER BY id ASC")
            rows = cursor.fetchall()
            return [self._row_to_record(r) for r in rows]

    def delete(self, user_id: str) -> bool:
        """Deletes user and user_profile by user_id in the same transaction."""
        if not user_id:
            return False
        uid = user_id.strip().lower()
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM user_profiles WHERE LOWER(user_id) = ?",
                (uid,)
            )
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM users WHERE LOWER(user_id) = ?",
                (uid,)
            )
            conn.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> UserRecord:
        nat = row["nationality"]
        if nat and str(nat).strip():
            nat = str(nat).strip()
        else:
            nat = None
        return UserRecord(
            id=row["id"],
            user_id=row["user_id"],
            username=row["username"],
            name=row["name"],
            email=row["email"],
            hashed_password=row["hashed_password"],
            nationality=nat,
            created_at=str(row["created_at"]) if row["created_at"] else None,
        )
