"""
Audit logging for tracking cross-agent operations

Pattern: Database-backed audit log for multi-agent coordination and debugging
"""

import sqlite3
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class AuditEntry:
    """An audit log entry recording an operation"""
    id: str
    agent_id: str
    action_type: str  # read, write, delete, share, subscribe, etc.
    resource_type: str  # memory, namespace, subscription, etc.
    resource_id: Optional[str]
    namespace: Optional[str]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "namespace": self.namespace,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class AuditLogger:
    """
    Audit logger for tracking cross-agent operations

    Features:
    - Log all operations (read, write, delete, share, subscribe, etc.)
    - Query logs by agent, namespace, action, resource, time range
    - Store metadata for detailed context
    - Thread-safe database operations
    - WAL mode for concurrent access

    Pattern follows SharedNamespace architecture:
    - Persistent connection with thread lock
    - WAL mode for concurrent writes
    - Foreign key constraints
    - Proper error handling

    Usage:
        logger = AuditLogger(db_path)

        # Log an operation
        logger.log(
            agent_id="agent-1",
            action_type="write",
            resource_type="memory",
            resource_id="mem-123",
            namespace="acme/research",
            metadata={"content_hash": "abc123"}
        )

        # Query logs
        entries = logger.get_by_agent("agent-1", limit=10)
        entries = logger.get_by_namespace("acme/research")
        entries = logger.get_by_action("write", limit=20)
    """

    # Common action types for consistency
    ACTION_READ = "read"
    ACTION_WRITE = "write"
    ACTION_DELETE = "delete"
    ACTION_SHARE = "share"
    ACTION_SUBSCRIBE = "subscribe"
    ACTION_UNSUBSCRIBE = "unsubscribe"
    ACTION_GRANT_PERMISSION = "grant_permission"
    ACTION_REVOKE_PERMISSION = "revoke_permission"
    ACTION_CREATE_NAMESPACE = "create_namespace"
    ACTION_DELETE_NAMESPACE = "delete_namespace"

    # Common resource types for consistency
    RESOURCE_MEMORY = "memory"
    RESOURCE_NAMESPACE = "namespace"
    RESOURCE_SUBSCRIPTION = "subscription"
    RESOURCE_PERMISSION = "permission"
    RESOURCE_BELIEF = "belief"

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize AuditLogger.

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

    def log(
        self,
        agent_id: str,
        action_type: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Log an operation to the audit log.

        Args:
            agent_id: Agent ID performing the operation
            action_type: Type of action (read, write, delete, etc.)
            resource_type: Type of resource (memory, namespace, etc.)
            resource_id: Optional specific resource ID
            namespace: Optional namespace the operation occurred in
            metadata: Optional metadata dictionary for additional context

        Returns:
            AuditEntry object

        Examples:
            # Log memory write
            logger.log(
                agent_id="agent-1",
                action_type="write",
                resource_type="memory",
                resource_id="mem-123",
                namespace="acme/research"
            )

            # Log namespace creation
            logger.log(
                agent_id="agent-1",
                action_type="create_namespace",
                resource_type="namespace",
                resource_id="acme/research",
                metadata={"created_by": "agent-1"}
            )
        """
        # Generate unique ID
        entry_id = str(uuid.uuid4())

        # Serialize metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None

        with self._db_lock:
            self._conn.execute(
                """
                INSERT INTO audit_log (id, agent_id, action_type, resource_type,
                                      resource_id, namespace, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (entry_id, agent_id, action_type, resource_type,
                 resource_id, namespace, metadata_json)
            )

            # Retrieve the created entry
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                WHERE id = ?
                """,
                (entry_id,)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to create audit log entry")

        return self._row_to_entry(row)

    def get_by_id(self, entry_id: str) -> Optional[AuditEntry]:
        """
        Get a specific audit entry by ID.

        Args:
            entry_id: Audit entry ID

        Returns:
            AuditEntry if exists, None otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                WHERE id = ?
                """,
                (entry_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_entry(row)

    def get_by_agent(
        self,
        agent_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Get audit entries for a specific agent.

        Args:
            agent_id: Agent ID
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip (default: 0)

        Returns:
            List of AuditEntry objects, ordered by timestamp (newest first)
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (agent_id, limit, offset)
            )
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_by_namespace(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Get audit entries for a specific namespace.

        Args:
            namespace: Namespace string
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip (default: 0)

        Returns:
            List of AuditEntry objects, ordered by timestamp (newest first)
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                WHERE namespace = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (namespace, limit, offset)
            )
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_by_action(
        self,
        action_type: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Get audit entries for a specific action type.

        Args:
            action_type: Action type string (e.g., "write", "read")
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip (default: 0)

        Returns:
            List of AuditEntry objects, ordered by timestamp (newest first)
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                WHERE action_type = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (action_type, limit, offset)
            )
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_by_resource(
        self,
        resource_type: str,
        resource_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Get audit entries for a specific resource type and optionally resource ID.

        Args:
            resource_type: Resource type string (e.g., "memory", "namespace")
            resource_id: Optional specific resource ID
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip (default: 0)

        Returns:
            List of AuditEntry objects, ordered by timestamp (newest first)
        """
        with self._db_lock:
            if resource_id:
                cursor = self._conn.execute(
                    """
                    SELECT id, agent_id, action_type, resource_type, resource_id,
                           namespace, timestamp, metadata
                    FROM audit_log
                    WHERE resource_type = ? AND resource_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (resource_type, resource_id, limit, offset)
                )
            else:
                cursor = self._conn.execute(
                    """
                    SELECT id, agent_id, action_type, resource_type, resource_id,
                           namespace, timestamp, metadata
                    FROM audit_log
                    WHERE resource_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (resource_type, limit, offset)
                )
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_by_time_range(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Get audit entries within a time range.

        Args:
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive). If None, uses current time
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip (default: 0)

        Returns:
            List of AuditEntry objects, ordered by timestamp (newest first)
        """
        if end_time is None:
            end_time = datetime.now()

        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (start_time.isoformat(), end_time.isoformat(), limit, offset)
            )
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_recent(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Get recent audit entries.

        Args:
            limit: Maximum number of entries to return (default: 100)
            offset: Number of entries to skip (default: 0)

        Returns:
            List of AuditEntry objects, ordered by timestamp (newest first)
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, action_type, resource_type, resource_id,
                       namespace, timestamp, metadata
                FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def count_by_agent(self, agent_id: str) -> int:
        """
        Count audit entries for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of audit entries
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT COUNT(*) FROM audit_log WHERE agent_id = ?
                """,
                (agent_id,)
            )
            row = cursor.fetchone()

        return row[0] if row else 0

    def count_by_namespace(self, namespace: str) -> int:
        """
        Count audit entries for a specific namespace.

        Args:
            namespace: Namespace string

        Returns:
            Number of audit entries
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT COUNT(*) FROM audit_log WHERE namespace = ?
                """,
                (namespace,)
            )
            row = cursor.fetchone()

        return row[0] if row else 0

    def count_by_action(self, action_type: str) -> int:
        """
        Count audit entries for a specific action type.

        Args:
            action_type: Action type string

        Returns:
            Number of audit entries
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT COUNT(*) FROM audit_log WHERE action_type = ?
                """,
                (action_type,)
            )
            row = cursor.fetchone()

        return row[0] if row else 0

    def _row_to_entry(self, row: tuple) -> AuditEntry:
        """
        Convert database row to AuditEntry.

        Args:
            row: Database row tuple

        Returns:
            AuditEntry object
        """
        entry_id, agent_id, action_type, resource_type, resource_id, namespace, timestamp, metadata_json = row

        # Parse timestamp from ISO format
        timestamp_dt = datetime.fromisoformat(timestamp) if timestamp else datetime.now()

        # Parse metadata JSON
        metadata = json.loads(metadata_json) if metadata_json else None

        return AuditEntry(
            id=entry_id,
            agent_id=agent_id,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            namespace=namespace,
            timestamp=timestamp_dt,
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
