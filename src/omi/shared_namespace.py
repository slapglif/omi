"""
Shared namespace management for multi-agent memory coordination

Pattern: Database-backed shared namespaces with permissions and metadata
"""

import sqlite3
import json
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from .namespaces import Namespace, validate_namespace


@dataclass
class SharedNamespaceInfo:
    """Information about a shared namespace"""
    namespace: str
    created_at: datetime
    created_by: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "namespace": self.namespace,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "metadata": self.metadata or {}
        }


class SharedNamespace:
    """
    Shared namespace manager for multi-agent coordination

    Features:
    - Create/delete shared namespaces
    - Store namespace metadata
    - Query namespace information
    - Thread-safe database operations
    - WAL mode for concurrent access

    Pattern follows GraphPalace architecture:
    - Persistent connection with thread lock
    - WAL mode for concurrent writes
    - Foreign key constraints
    - Proper error handling
    """

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize SharedNamespace manager.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for concurrent writes (default: True)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal = enable_wal

        # Create persistent connection
        # check_same_thread=False allows multi-threaded access (safe with WAL mode)
        # isolation_level=None enables autocommit mode for better concurrency
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0
        )
        # Thread lock for serializing database operations
        self._db_lock = threading.Lock()

        # Enable WAL mode and foreign keys
        if self._enable_wal:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def create(
        self,
        namespace: str,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SharedNamespaceInfo:
        """
        Create a new shared namespace.

        Args:
            namespace: Namespace string (must be valid format)
            created_by: Agent ID that created the namespace
            metadata: Optional metadata dictionary

        Returns:
            SharedNamespaceInfo object

        Raises:
            ValueError: If namespace is invalid or already exists
        """
        # Validate namespace format
        if not validate_namespace(namespace):
            raise ValueError(f"Invalid namespace format: {namespace}")

        # Serialize metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None

        with self._db_lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO shared_namespaces (namespace, created_by, metadata)
                    VALUES (?, ?, ?)
                    """,
                    (namespace, created_by, metadata_json)
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"Shared namespace already exists: {namespace}")

            # Retrieve the created namespace
            cursor = self._conn.execute(
                """
                SELECT namespace, created_at, created_by, metadata
                FROM shared_namespaces
                WHERE namespace = ?
                """,
                (namespace,)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to create shared namespace")

        return self._row_to_info(row)

    def get(self, namespace: str) -> Optional[SharedNamespaceInfo]:
        """
        Get information about a shared namespace.

        Args:
            namespace: Namespace string

        Returns:
            SharedNamespaceInfo if exists, None otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT namespace, created_at, created_by, metadata
                FROM shared_namespaces
                WHERE namespace = ?
                """,
                (namespace,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_info(row)

    def exists(self, namespace: str) -> bool:
        """
        Check if a shared namespace exists.

        Args:
            namespace: Namespace string

        Returns:
            True if exists, False otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT 1 FROM shared_namespaces WHERE namespace = ?
                """,
                (namespace,)
            )
            return cursor.fetchone() is not None

    def delete(self, namespace: str) -> bool:
        """
        Delete a shared namespace.

        Note: This will cascade delete all permissions, subscriptions,
        and audit log entries for this namespace (via foreign keys).

        Args:
            namespace: Namespace string

        Returns:
            True if deleted, False if not found
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                DELETE FROM shared_namespaces WHERE namespace = ?
                """,
                (namespace,)
            )
            return cursor.rowcount > 0

    def list_all(self) -> List[SharedNamespaceInfo]:
        """
        List all shared namespaces.

        Returns:
            List of SharedNamespaceInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT namespace, created_at, created_by, metadata
                FROM shared_namespaces
                ORDER BY created_at DESC
                """
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def list_by_creator(self, agent_id: str) -> List[SharedNamespaceInfo]:
        """
        List shared namespaces created by a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of SharedNamespaceInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT namespace, created_at, created_by, metadata
                FROM shared_namespaces
                WHERE created_by = ?
                ORDER BY created_at DESC
                """,
                (agent_id,)
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def update_metadata(
        self,
        namespace: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a shared namespace.

        Args:
            namespace: Namespace string
            metadata: New metadata dictionary

        Returns:
            True if updated, False if not found
        """
        metadata_json = json.dumps(metadata)

        with self._db_lock:
            cursor = self._conn.execute(
                """
                UPDATE shared_namespaces
                SET metadata = ?
                WHERE namespace = ?
                """,
                (metadata_json, namespace)
            )
            return cursor.rowcount > 0

    def _row_to_info(self, row: tuple) -> SharedNamespaceInfo:
        """
        Convert database row to SharedNamespaceInfo.

        Args:
            row: Database row tuple

        Returns:
            SharedNamespaceInfo object
        """
        namespace, created_at, created_by, metadata_json = row

        # Parse created_at from ISO format
        created_at_dt = datetime.fromisoformat(created_at) if created_at else datetime.now()

        # Parse metadata JSON
        metadata = json.loads(metadata_json) if metadata_json else None

        return SharedNamespaceInfo(
            namespace=namespace,
            created_at=created_at_dt,
            created_by=created_by,
            metadata=metadata
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
