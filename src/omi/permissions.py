"""
Permission management for shared namespace access control

Pattern: Database-backed permission checks with role-based access levels
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Set
from dataclasses import dataclass
from enum import Enum


class PermissionLevel(str, Enum):
    """
    Permission levels for shared namespace access

    Hierarchy (from least to most permissive):
    - READ: Can read memories from the namespace
    - WRITE: Can read and write memories to the namespace
    - ADMIN: Can read, write, and manage permissions
    """
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

    @classmethod
    def from_string(cls, level: str) -> "PermissionLevel":
        """
        Convert string to PermissionLevel

        Args:
            level: Permission level string (case-insensitive)

        Returns:
            PermissionLevel enum value

        Raises:
            ValueError: If level is invalid
        """
        level_lower = level.lower()
        if level_lower == "read":
            return cls.READ
        elif level_lower == "write":
            return cls.WRITE
        elif level_lower == "admin":
            return cls.ADMIN
        else:
            raise ValueError(
                f"Invalid permission level '{level}'. "
                "Must be one of: read, write, admin"
            )

    def can_read(self) -> bool:
        """Check if this level allows reading"""
        return True  # All levels can read

    def can_write(self) -> bool:
        """Check if this level allows writing"""
        return self in (PermissionLevel.WRITE, PermissionLevel.ADMIN)

    def can_admin(self) -> bool:
        """Check if this level allows administration"""
        return self == PermissionLevel.ADMIN

    def __str__(self) -> str:
        """String representation"""
        return self.value


@dataclass
class PermissionInfo:
    """Information about an agent's permission in a namespace"""
    namespace: str
    agent_id: str
    permission_level: PermissionLevel
    created_at: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "namespace": self.namespace,
            "agent_id": self.agent_id,
            "permission_level": self.permission_level.value,
            "created_at": self.created_at.isoformat()
        }


class PermissionManager:
    """
    Permission manager for shared namespace access control

    Features:
    - Grant/revoke permissions for agents in namespaces
    - Check if agent has specific permission level
    - List permissions for agent or namespace
    - Thread-safe database operations
    - WAL mode for concurrent access

    Permission hierarchy:
    - READ: Can read memories
    - WRITE: Can read and write memories (includes READ)
    - ADMIN: Can manage permissions (includes READ and WRITE)

    Pattern follows SharedNamespace architecture:
    - Persistent connection with thread lock
    - WAL mode for concurrent writes
    - Foreign key constraints
    - Proper error handling
    """

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize PermissionManager.

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

    def grant(
        self,
        namespace: str,
        agent_id: str,
        permission_level: PermissionLevel
    ) -> PermissionInfo:
        """
        Grant permission to an agent for a namespace.

        If permission already exists, it will be updated to the new level.

        Args:
            namespace: Namespace string
            agent_id: Agent ID to grant permission to
            permission_level: Permission level to grant

        Returns:
            PermissionInfo object

        Raises:
            ValueError: If namespace doesn't exist (foreign key constraint)
        """
        with self._db_lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO namespace_permissions (namespace, agent_id, permission_level)
                    VALUES (?, ?, ?)
                    ON CONFLICT(namespace, agent_id)
                    DO UPDATE SET permission_level = excluded.permission_level
                    """,
                    (namespace, agent_id, permission_level.value)
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(
                    f"Failed to grant permission: {e}. "
                    f"Namespace '{namespace}' may not exist."
                )

            # Retrieve the permission
            cursor = self._conn.execute(
                """
                SELECT namespace, agent_id, permission_level, created_at
                FROM namespace_permissions
                WHERE namespace = ? AND agent_id = ?
                """,
                (namespace, agent_id)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to grant permission")

        return self._row_to_info(row)

    def revoke(self, namespace: str, agent_id: str) -> bool:
        """
        Revoke permission from an agent for a namespace.

        Args:
            namespace: Namespace string
            agent_id: Agent ID to revoke permission from

        Returns:
            True if permission was revoked, False if it didn't exist
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                DELETE FROM namespace_permissions
                WHERE namespace = ? AND agent_id = ?
                """,
                (namespace, agent_id)
            )
            return cursor.rowcount > 0

    def get(
        self,
        namespace: str,
        agent_id: str
    ) -> Optional[PermissionLevel]:
        """
        Get permission level for an agent in a namespace.

        Args:
            namespace: Namespace string
            agent_id: Agent ID

        Returns:
            PermissionLevel if exists, None otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT permission_level
                FROM namespace_permissions
                WHERE namespace = ? AND agent_id = ?
                """,
                (namespace, agent_id)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return PermissionLevel.from_string(row[0])

    def has_permission(
        self,
        namespace: str,
        agent_id: str,
        required_level: PermissionLevel
    ) -> bool:
        """
        Check if agent has at least the required permission level.

        Permission hierarchy:
        - ADMIN >= WRITE >= READ
        - If agent has ADMIN, they satisfy WRITE and READ requirements
        - If agent has WRITE, they satisfy READ requirements

        Args:
            namespace: Namespace string
            agent_id: Agent ID
            required_level: Minimum required permission level

        Returns:
            True if agent has sufficient permission, False otherwise
        """
        current_level = self.get(namespace, agent_id)

        if current_level is None:
            return False

        # Check hierarchy
        if required_level == PermissionLevel.READ:
            return True  # All levels satisfy READ
        elif required_level == PermissionLevel.WRITE:
            return current_level in (PermissionLevel.WRITE, PermissionLevel.ADMIN)
        elif required_level == PermissionLevel.ADMIN:
            return current_level == PermissionLevel.ADMIN

        return False

    def can_read(self, namespace: str, agent_id: str) -> bool:
        """Check if agent can read from namespace"""
        return self.has_permission(namespace, agent_id, PermissionLevel.READ)

    def can_write(self, namespace: str, agent_id: str) -> bool:
        """Check if agent can write to namespace"""
        return self.has_permission(namespace, agent_id, PermissionLevel.WRITE)

    def can_admin(self, namespace: str, agent_id: str) -> bool:
        """Check if agent can administer namespace"""
        return self.has_permission(namespace, agent_id, PermissionLevel.ADMIN)

    def list_for_agent(self, agent_id: str) -> List[PermissionInfo]:
        """
        List all permissions for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of PermissionInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT namespace, agent_id, permission_level, created_at
                FROM namespace_permissions
                WHERE agent_id = ?
                ORDER BY created_at DESC
                """,
                (agent_id,)
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def list_for_namespace(self, namespace: str) -> List[PermissionInfo]:
        """
        List all permissions for a specific namespace.

        Args:
            namespace: Namespace string

        Returns:
            List of PermissionInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT namespace, agent_id, permission_level, created_at
                FROM namespace_permissions
                WHERE namespace = ?
                ORDER BY created_at DESC
                """,
                (namespace,)
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def get_agents_with_level(
        self,
        namespace: str,
        permission_level: PermissionLevel
    ) -> Set[str]:
        """
        Get all agent IDs with a specific permission level in a namespace.

        Args:
            namespace: Namespace string
            permission_level: Permission level to filter by

        Returns:
            Set of agent IDs
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT agent_id
                FROM namespace_permissions
                WHERE namespace = ? AND permission_level = ?
                """,
                (namespace, permission_level.value)
            )
            rows = cursor.fetchall()

        return {row[0] for row in rows}

    def _row_to_info(self, row: tuple) -> PermissionInfo:
        """
        Convert database row to PermissionInfo.

        Args:
            row: Database row tuple

        Returns:
            PermissionInfo object
        """
        namespace, agent_id, permission_level, created_at = row

        # Parse created_at from ISO format
        created_at_dt = datetime.fromisoformat(created_at) if created_at else datetime.now()

        # Parse permission level
        level = PermissionLevel.from_string(permission_level)

        return PermissionInfo(
            namespace=namespace,
            agent_id=agent_id,
            permission_level=level,
            created_at=created_at_dt
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


def validate_permission_level(level: str) -> bool:
    """
    Validate permission level string without creating PermissionLevel object.

    Args:
        level: Permission level string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        PermissionLevel.from_string(level)
        return True
    except ValueError:
        return False
