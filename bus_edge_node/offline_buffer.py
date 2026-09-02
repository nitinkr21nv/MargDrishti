"""Production-grade local database queue for bus edge nodes.

Provides offline telemetry persistence using SQLite with Write-Ahead Logging (WAL)
mode to queue road telemetry data when network connectivity is degraded.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "edge_offline_queue.db")


def get_db_connection(db_path: str, timeout: float = 30.0) -> sqlite3.Connection:
    """Establishes SQLite connection configured with WAL mode for edge storage."""
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


@contextmanager
def buffer_session(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """Context manager providing automated transaction commit/rollback for edge queue operations."""
    target_path = db_path if db_path is not None else DEFAULT_DB_PATH
    conn = get_db_connection(target_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_buffer(db_path: Optional[str] = None) -> None:
    """Initializes the offline buffer schema with WAL mode enabled."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS telemetry_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        retry_count INTEGER DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_telemetry_queue_id ON telemetry_queue(id);
    """
    with buffer_session(db_path) as conn:
        conn.executescript(schema_sql)


def enqueue_payload(payload: Dict[str, Any], db_path: Optional[str] = None) -> int:
    """Serializes a telemetry payload to JSON and inserts it into the queue.
    
    Args:
        payload: Telemetry data dictionary to buffer.
        db_path: Optional path override for SQLite database.

    Returns:
        int: Inserted row ID.
    """
    payload_json = json.dumps(payload)
    created_at = datetime.now(timezone.utc).isoformat()
    query = """
    INSERT INTO telemetry_queue (payload_json, created_at, retry_count)
    VALUES (?, ?, 0)
    """
    with buffer_session(db_path) as conn:
        cursor = conn.execute(query, (payload_json, created_at))
        return cursor.lastrowid


def get_pending_payloads(limit: int = 50, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves pending queue items ordered by insertion ID ascending.
    
    Args:
        limit: Maximum number of records to retrieve.
        db_path: Optional path override for SQLite database.

    Returns:
        List[Dict[str, Any]]: List of deserialized queue item records.
    """
    query = """
    SELECT id, payload_json, created_at, retry_count
    FROM telemetry_queue
    ORDER BY id ASC
    LIMIT ?
    """
    with buffer_session(db_path) as conn:
        cursor = conn.execute(query, (limit,))
        rows = cursor.fetchall()

        pending_items: List[Dict[str, Any]] = []
        for row in rows:
            pending_items.append({
                "id": row["id"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
                "retry_count": row["retry_count"],
            })
        return pending_items


def dequeue_payload(record_id: int, db_path: Optional[str] = None) -> bool:
    """Deletes a successfully transmitted record from the queue.
    
    Args:
        record_id: Queue record ID to delete.
        db_path: Optional path override for SQLite database.

    Returns:
        bool: True if record was deleted, False otherwise.
    """
    query = "DELETE FROM telemetry_queue WHERE id = ?"
    with buffer_session(db_path) as conn:
        cursor = conn.execute(query, (record_id,))
        return cursor.rowcount > 0


def increment_retry(record_id: int, db_path: Optional[str] = None) -> None:
    """Increments retry counter for a queue item following a transmission failure.
    
    Args:
        record_id: Queue record ID to update.
        db_path: Optional path override for SQLite database.
    """
    query = "UPDATE telemetry_queue SET retry_count = retry_count + 1 WHERE id = ?"
    with buffer_session(db_path) as conn:
        conn.execute(query, (record_id,))
