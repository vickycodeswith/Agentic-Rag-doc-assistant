import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.config import settings
from app.core.logging import logger


class FeedbackService:
    """Manages persistent storage and analytics for user feedback on RAG answers."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.FEEDBACK_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    query_id TEXT,
                    query TEXT,
                    answer TEXT,
                    rating TEXT NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_feedback(
        self,
        rating: str,
        query_id: Optional[str] = None,
        query: Optional[str] = None,
        answer: Optional[str] = None,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Saves user rating and comment into SQLite."""
        feedback_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        clean_rating = "up" if rating.lower() in ["up", "thumbs_up", "positive", "1", "+1"] else "down"

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, query_id, query, answer, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (feedback_id, query_id, query, answer, clean_rating, comment, created_at)
            )
            conn.commit()

        logger.info(f"Recorded feedback [{clean_rating}] ID: {feedback_id}")
        return {
            "id": feedback_id,
            "query_id": query_id,
            "rating": clean_rating,
            "comment": comment,
            "created_at": created_at,
            "status": "success"
        }

    def get_all_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent feedback items."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary(self) -> Dict[str, Any]:
        """Returns aggregate feedback statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN rating = 'up' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN rating = 'down' THEN 1 ELSE 0 END) as negative
                FROM feedback
            """)
            row = cursor.fetchone()
            total = row["total"] or 0
            pos = row["positive"] or 0
            neg = row["negative"] or 0
            satisfaction_rate = (pos / total * 100.0) if total > 0 else 0.0

            return {
                "total_feedback": total,
                "positive_count": pos,
                "negative_count": neg,
                "satisfaction_rate_percent": round(satisfaction_rate, 2)
            }


feedback_service = FeedbackService()
