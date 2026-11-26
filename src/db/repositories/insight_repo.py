from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from src.db.repositories.user_repo import DEFAULT_DB_PATH


class SQLiteInsightRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
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

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    insight_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    polarity TEXT NOT NULL DEFAULT 'like',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    raw_text TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, insight_type, key)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_insights_user ON user_insights(user_id)")
            conn.commit()

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        return (user_id or "").strip().lower()

    @staticmethod
    def _key_token(token: str) -> str:
        normalized = token.strip().lower()
        return (
            normalized[:-1]
            if len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss")
            else normalized
        )

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {cls._key_token(token) for token in re.findall(r"\w+", text.lower())}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "insight_type": row["insight_type"],
            "key": row["key"],
            "value": row["value"],
            "polarity": row["polarity"],
            "tags": json.loads(row["tags_json"] or "[]"),
            "raw_text": row["raw_text"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert(self, user_id: str, insight: dict) -> Optional[Dict[str, Any]]:
        uid = self._normalize_user_id(user_id)
        key = str(insight.get("key") or "").strip().lower()
        insight_type = str(insight.get("insight_type") or "").strip()
        if not uid or not key or not insight_type:
            return None

        value = insight.get("value", "")
        value_text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, sort_keys=True
        )
        tags = insight.get("tags") or []
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_insights (
                    user_id, insight_type, key, value, polarity, tags_json,
                    raw_text, confidence, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, insight_type, key) DO UPDATE SET
                    value = excluded.value,
                    polarity = excluded.polarity,
                    tags_json = excluded.tags_json,
                    raw_text = excluded.raw_text,
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    uid,
                    insight_type,
                    key,
                    value_text,
                    str(insight.get("polarity") or "like"),
                    json.dumps(tags, ensure_ascii=False),
                    str(insight.get("raw_text") or ""),
                    float(insight.get("confidence", 1.0)),
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT * FROM user_insights
                WHERE user_id = ? AND insight_type = ? AND key = ?
                """,
                (uid, insight_type, key),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        uid = self._normalize_user_id(user_id)
        if not uid or limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM user_insights
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (uid, limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete(self, user_id: str, insight_type: str, key: str) -> bool:
        uid = self._normalize_user_id(user_id)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM user_insights
                WHERE user_id = ? AND insight_type = ? AND key = ?
                """,
                (uid, insight_type, str(key or "").strip().lower()),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_all(self, user_id: str) -> int:
        uid = self._normalize_user_id(user_id)
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM user_insights WHERE user_id = ?", (uid,))
            conn.commit()
            return cursor.rowcount

    def search(self, user_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query or "")
        uid = self._normalize_user_id(user_id)
        if not uid or not query_tokens or top_k <= 0:
            return []

        scored = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM user_insights WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
                (uid,),
            ).fetchall()
        for insight in (self._row_to_dict(row) for row in rows):
            doc_text = " ".join([
                insight["key"],
                insight["value"],
                " ".join(insight["tags"]),
                insight["raw_text"],
            ])
            doc_tokens = self._tokens(doc_text)
            score = len(query_tokens & doc_tokens) / len(query_tokens)
            if score > 0:
                scored.append((score, insight))
        scored.sort(key=lambda result: result[0], reverse=True)
        return [
            {**insight, "similarity_score": score}
            for score, insight in scored[:top_k]
        ]
