"""
Event History Storage for OMI
SQLite-based event audit log for memory operations.

This module provides event storage and query capabilities:
- Store all memory operation events
- Query events by type, date range, and other filters
- Audit trail for memory operations
"""

import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EventRecord:
    """A stored event record."""
    id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


class EventHistory:
    """
    Event History Storage - SQLite-based audit log for events

    Pattern: Append-only log, queryable
    Lifetime: Indefinite (with optional pruning)

    Features:
    - Store all event types (memory.stored, belief.updated, etc.)
    - Query by event_type, timestamp range
    - WAL mode for concurrent writes
    - Full-text search of event payloads
    """

    def __init__(self, db_path: Path, enable_wal: bool = True) -> None:
        """
        Initialize Event History.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for concurrent writes (default: True)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal = enable_wal
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema with indexes."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrent writes
            if self._enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")

            # Create events table
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,  -- JSON
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT  -- JSON
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp ON events(event_type, timestamp);
            """)

            conn.commit()

    def store_event(self,
                   event_type: str,
                   payload: Dict[str, Any],
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store an event in the history.

        Args:
            event_type: The type of event (e.g., "memory.stored", "belief.updated")
            payload: Event data as dictionary
            metadata: Optional additional metadata

        Returns:
            event_id: UUID of the created event record
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO events (id, event_type, payload, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                event_id,
                event_type,
                json.dumps(payload),
                timestamp,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()

        return event_id

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        """
        Retrieve an event by ID.

        Args:
            event_id: UUID of the event

        Returns:
            EventRecord or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, event_type, payload, timestamp, metadata
                FROM events WHERE id = ?
            """, (event_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return EventRecord(
                id=row[0],
                event_type=row[1],
                payload=json.loads(row[2]) if row[2] else {},
                timestamp=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                metadata=json.loads(row[4]) if row[4] else None
            )

    def query_events(self,
                    event_type: Optional[str] = None,
                    since: Optional[datetime] = None,
                    until: Optional[datetime] = None,
                    limit: int = 100) -> List[EventRecord]:
        """
        Query events with filters.

        Args:
            event_type: Filter by event type (e.g., "memory.stored")
            since: Filter events after this timestamp
            until: Filter events before this timestamp
            limit: Maximum number of results (default: 100)

        Returns:
            List of EventRecord objects
        """
        events: List[EventRecord] = []

        # Build query dynamically based on filters
        query = "SELECT id, event_type, payload, timestamp, metadata FROM events WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)

            for row in cursor:
                events.append(EventRecord(
                    id=row[0],
                    event_type=row[1],
                    payload=json.loads(row[2]) if row[2] else {},
                    timestamp=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                    metadata=json.loads(row[4]) if row[4] else None
                ))

        return events

    def get_event_types(self) -> List[str]:
        """
        Get all distinct event types in the history.

        Returns:
            List of event type strings
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT event_type FROM events ORDER BY event_type
            """)
            return [row[0] for row in cursor]

    def count_events(self,
                    event_type: Optional[str] = None,
                    since: Optional[datetime] = None,
                    until: Optional[datetime] = None) -> int:
        """
        Count events matching filters.

        Args:
            event_type: Filter by event type
            since: Filter events after this timestamp
            until: Filter events before this timestamp

        Returns:
            Count of matching events
        """
        query = "SELECT COUNT(*) FROM events WHERE 1=1"
        params: List[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()
            return int(result[0]) if result else 0

    def delete_events_before(self, timestamp: datetime) -> int:
        """
        Delete events older than the specified timestamp.
        Used for pruning old history.

        Args:
            timestamp: Delete events before this time

        Returns:
            Number of events deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM events WHERE timestamp < ?
            """, (timestamp.isoformat(),))
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with event_count, type_distribution, oldest/newest timestamps
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            event_count = cursor.fetchone()[0]

            # Type distribution
            cursor = conn.execute("""
                SELECT event_type, COUNT(*) FROM events GROUP BY event_type
            """)
            type_distribution = {row[0]: row[1] for row in cursor}

            # Timestamp range
            cursor = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM events
            """)
            row = cursor.fetchone()
            oldest = row[0]
            newest = row[1]

            return {
                "event_count": event_count,
                "type_distribution": type_distribution,
                "oldest_event": oldest,
                "newest_event": newest
            }

    def vacuum(self) -> None:
        """Optimize database (reclaim space)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

    def close(self) -> None:
        """Close connection and cleanup."""
        pass

    def __enter__(self) -> "EventHistory":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
