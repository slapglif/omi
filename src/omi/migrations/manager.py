"""
Migration Manager for OMI
Tracks schema versions and migration history.

This module provides migration tracking capabilities:
- Track current schema version via PRAGMA user_version
- Record all applied migrations in _migrations table
- Query migration history
- Detect pending migrations
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MigrationRecord:
    """A stored migration record."""
    id: int
    version: int
    description: str
    applied_at: datetime
    duration_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "version": self.version,
            "description": self.description,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata or {}
        }


class MigrationManager:
    """
    Migration Manager - Tracks schema versions and migration history

    Pattern: Schema version in PRAGMA user_version, history in _migrations table
    Lifetime: Persistent across application restarts

    Features:
    - Get/set schema version via PRAGMA user_version
    - Track all applied migrations in _migrations table
    - Query migration history
    - WAL mode for concurrent writes
    - Detect pending migrations
    """

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize Migration Manager.

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

            # Create migrations history table
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS _migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration_ms INTEGER,
                    metadata TEXT  -- JSON
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_migrations_version ON _migrations(version);
                CREATE INDEX IF NOT EXISTS idx_migrations_applied_at ON _migrations(applied_at);
            """)

            conn.commit()

    def get_schema_version(self) -> int:
        """
        Get current schema version from PRAGMA user_version.

        Returns:
            Current schema version (0 if uninitialized)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            return version

    def set_schema_version(self, version: int) -> None:
        """
        Set schema version using PRAGMA user_version.

        Args:
            version: New schema version (must be >= 0)

        Raises:
            ValueError: If version is negative
        """
        if version < 0:
            raise ValueError(f"Schema version must be >= 0, got {version}")

        with sqlite3.connect(self.db_path) as conn:
            # PRAGMA user_version doesn't support parameterized queries
            conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()

    def record_migration(self,
                        version: int,
                        description: str,
                        duration_ms: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Record a successfully applied migration.

        Args:
            version: Migration version number
            description: Human-readable description
            duration_ms: Optional execution time in milliseconds
            metadata: Optional additional metadata

        Returns:
            record_id: ID of the created migration record

        Raises:
            sqlite3.IntegrityError: If migration version already recorded
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO _migrations (version, description, applied_at, duration_ms, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                version,
                description,
                datetime.now().isoformat(),
                duration_ms,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
            return cursor.lastrowid

    def get_migration_record(self, version: int) -> Optional[MigrationRecord]:
        """
        Retrieve a migration record by version.

        Args:
            version: Migration version number

        Returns:
            MigrationRecord or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, version, description, applied_at, duration_ms, metadata
                FROM _migrations WHERE version = ?
            """, (version,))

            row = cursor.fetchone()
            if not row:
                return None

            return MigrationRecord(
                id=row[0],
                version=row[1],
                description=row[2],
                applied_at=datetime.fromisoformat(row[3]) if row[3] else None,
                duration_ms=row[4],
                metadata=json.loads(row[5]) if row[5] else None
            )

    def get_applied_migrations(self) -> List[MigrationRecord]:
        """
        Get all applied migrations ordered by version.

        Returns:
            List of MigrationRecord objects, ordered by version ascending
        """
        migrations = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, version, description, applied_at, duration_ms, metadata
                FROM _migrations
                ORDER BY version ASC
            """)

            for row in cursor:
                migrations.append(MigrationRecord(
                    id=row[0],
                    version=row[1],
                    description=row[2],
                    applied_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    duration_ms=row[4],
                    metadata=json.loads(row[5]) if row[5] else None
                ))

        return migrations

    def get_last_migration(self) -> Optional[MigrationRecord]:
        """
        Get the most recently applied migration.

        Returns:
            MigrationRecord or None if no migrations applied
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, version, description, applied_at, duration_ms, metadata
                FROM _migrations
                ORDER BY version DESC
                LIMIT 1
            """)

            row = cursor.fetchone()
            if not row:
                return None

            return MigrationRecord(
                id=row[0],
                version=row[1],
                description=row[2],
                applied_at=datetime.fromisoformat(row[3]) if row[3] else None,
                duration_ms=row[4],
                metadata=json.loads(row[5]) if row[5] else None
            )

    def is_migration_applied(self, version: int) -> bool:
        """
        Check if a specific migration has been applied.

        Args:
            version: Migration version number

        Returns:
            True if migration is recorded, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM _migrations WHERE version = ?
            """, (version,))
            count = cursor.fetchone()[0]
            return count > 0

    def get_migration_count(self) -> int:
        """
        Get total number of applied migrations.

        Returns:
            Count of applied migrations
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM _migrations")
            return cursor.fetchone()[0]

    def clear_history(self) -> None:
        """
        Clear all migration history (use with caution).

        This does NOT affect the schema version or actual database schema.
        Only clears the _migrations table records.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM _migrations")
            conn.commit()
