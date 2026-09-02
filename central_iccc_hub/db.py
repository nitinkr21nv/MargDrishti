"""Database persistence layer for the Central ICCC Hub.

Manages SQLite storage with Write-Ahead Logging (WAL) enabled for high-concurrency
reads and writes. Exposes schema initializers, connection providers, context managers,
and parameterized CRUD interfaces for defects and ANPR alerts.
"""

from contextlib import contextmanager
import os
import sqlite3
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_PATH = os.getenv("ICCC_DB_PATH", "iccc_hub.db")


def get_db_connection(db_path: str = DEFAULT_DB_PATH, timeout: float = 30.0) -> sqlite3.Connection:
    """Establishes SQLite connection with WAL mode and dictionary row access enabled.
    
    Args:
        db_path: Filepath to the SQLite database file.
        timeout: Lock acquisition timeout in seconds for concurrent transactions.

    Returns:
        Configured sqlite3.Connection instance.
    """
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def db_session(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager providing auto-committing / auto-rolling-back DB connection.
    
    Yields:
        sqlite3.Connection: Active database connection.
    """
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def db_cursor(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Cursor]:
    """Context manager providing database cursor within auto-managed transaction session.
    
    Yields:
        sqlite3.Cursor: Active cursor bound to the transaction session.
    """
    with db_session(db_path) as conn:
        cursor = conn.cursor()
        yield cursor


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes database schema tables and indexes if they do not exist."""
    schema_sql = """
    CREATE TABLE IF NOT EXISTS defects (
        id TEXT PRIMARY KEY,
        bus_id TEXT NOT NULL,
        lat REAL NOT NULL,
        lng REAL NOT NULL,
        defect_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        z_axis_g REAL NOT NULL,
        severity_score REAL NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('OPEN', 'AUDIT_PENDING', 'VERIFIED_FIXED', 'SLA_BREACH')),
        detection_count INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS anpr_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT UNIQUE NOT NULL,
        incident_type TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('SEARCHING', 'INTERCEPTED')),
        last_lat REAL,
        last_lng REAL,
        last_seen_by TEXT,
        last_seen_time TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_defects_status ON defects(status);
    CREATE INDEX IF NOT EXISTS idx_defects_bus_id ON defects(bus_id);
    CREATE INDEX IF NOT EXISTS idx_anpr_alerts_plate ON anpr_alerts(plate_number);
    CREATE INDEX IF NOT EXISTS idx_anpr_alerts_status ON anpr_alerts(status);
    """
    with db_session(db_path) as conn:
        conn.executescript(schema_sql)


# --- Defects CRUD Interface ---

def upsert_defect(conn: sqlite3.Connection, defect_data: Dict[str, Any]) -> None:
    """Inserts a new defect or updates existing record detection count and timestamp."""
    query = """
    INSERT INTO defects (
        id, bus_id, lat, lng, defect_type, confidence, z_axis_g,
        severity_score, status, detection_count, created_at, last_seen_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        confidence = excluded.confidence,
        z_axis_g = excluded.z_axis_g,
        severity_score = excluded.severity_score,
        detection_count = defects.detection_count + 1,
        last_seen_at = excluded.last_seen_at;
    """
    conn.execute(query, (
        defect_data["id"],
        defect_data["bus_id"],
        defect_data["lat"],
        defect_data["lng"],
        defect_data["defect_type"],
        defect_data["confidence"],
        defect_data["z_axis_g"],
        defect_data["severity_score"],
        defect_data.get("status", "OPEN"),
        defect_data.get("detection_count", 1),
        defect_data["created_at"],
        defect_data["last_seen_at"],
    ))


def get_defect_by_id(conn: sqlite3.Connection, defect_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single defect record by its unique ID."""
    query = "SELECT * FROM defects WHERE id = ?"
    cursor = conn.execute(query, (defect_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def list_defects(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    bus_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Fetches defect records with optional filtering by status and bus ID."""
    conditions = []
    params: List[Any] = []

    if status:
        conditions.append("status = ?")
        params.append(status)
    if bus_id:
        conditions.append("bus_id = ?")
        params.append(bus_id)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM defects {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def update_defect_status(conn: sqlite3.Connection, defect_id: str, new_status: str) -> bool:
    """Updates the processing lifecycle status of a defect."""
    query = "UPDATE defects SET status = ? WHERE id = ?"
    cursor = conn.execute(query, (new_status, defect_id))
    return cursor.rowcount > 0


# --- ANPR Alerts CRUD Interface ---

def upsert_anpr_alert(conn: sqlite3.Connection, alert_data: Dict[str, Any]) -> int:
    """Inserts or updates ANPR watchlist alert location and sighting details."""
    query = """
    INSERT INTO anpr_alerts (
        plate_number, incident_type, status, last_lat, last_lng, last_seen_by, last_seen_time
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(plate_number) DO UPDATE SET
        incident_type = excluded.incident_type,
        status = excluded.status,
        last_lat = excluded.last_lat,
        last_lng = excluded.last_lng,
        last_seen_by = excluded.last_seen_by,
        last_seen_time = excluded.last_seen_time;
    """
    cursor = conn.execute(query, (
        alert_data["plate_number"],
        alert_data["incident_type"],
        alert_data.get("status", "SEARCHING"),
        alert_data.get("last_lat"),
        alert_data.get("last_lng"),
        alert_data.get("last_seen_by"),
        alert_data.get("last_seen_time"),
    ))
    return cursor.lastrowid


def get_anpr_alert_by_plate(conn: sqlite3.Connection, plate_number: str) -> Optional[Dict[str, Any]]:
    """Retrieves ANPR alert record by vehicle plate number."""
    query = "SELECT * FROM anpr_alerts WHERE plate_number = ?"
    cursor = conn.execute(query, (plate_number,))
    row = cursor.fetchone()
    return dict(row) if row else None


def list_anpr_alerts(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Lists ANPR alerts matching optional status filter."""
    if status:
        query = "SELECT * FROM anpr_alerts WHERE status = ? ORDER BY id DESC LIMIT ? OFFSET ?"
        params = [status, limit, offset]
    else:
        query = "SELECT * FROM anpr_alerts ORDER BY id DESC LIMIT ? OFFSET ?"
        params = [limit, offset]

    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def update_anpr_alert_status(conn: sqlite3.Connection, plate_number: str, new_status: str) -> bool:
    """Updates status for a target vehicle plate number."""
    query = "UPDATE anpr_alerts SET status = ? WHERE plate_number = ?"
    cursor = conn.execute(query, (new_status, plate_number))
    return cursor.rowcount > 0
