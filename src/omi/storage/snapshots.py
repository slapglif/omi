"""
Snapshot Manager for OMI Memory Time-Travel

Handles point-in-time snapshot creation, diff, and rollback operations.

Features:
- Point-in-time memory state capture
- Delta encoding for efficient storage
- Snapshot diff and rollback
- MoltVault integration for cloud backup
- Timeline browsing support
"""

import sqlite3
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, asdict

from .models import Memory
from .schema import init_database


@dataclass
class SnapshotInfo:
    """Information about a memory snapshot."""
    snapshot_id: str
    created_at: datetime
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    moltvault_backup_id: Optional[str] = None
    memory_count: int = 0
    is_delta: bool = False
    base_snapshot_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "description": self.description,
            "metadata": self.metadata or {},
            "moltvault_backup_id": self.moltvault_backup_id,
            "memory_count": self.memory_count,
            "is_delta": self.is_delta,
            "base_snapshot_id": self.base_snapshot_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnapshotInfo':
        """Create SnapshotInfo from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            description=data.get("description"),
            metadata=data.get("metadata"),
            moltvault_backup_id=data.get("moltvault_backup_id"),
            memory_count=data.get("memory_count", 0),
            is_delta=data.get("is_delta", False),
            base_snapshot_id=data.get("base_snapshot_id")
        )


@dataclass
class SnapshotDiff:
    """Differences between two snapshots."""
    snapshot1_id: str
    snapshot2_id: str
    added: List[str]  # Memory IDs added in snapshot2
    modified: List[str]  # Memory IDs modified between snapshots
    deleted: List[str]  # Memory IDs deleted in snapshot2
    total_changes: int = 0

    def __post_init__(self):
        """Calculate total changes."""
        self.total_changes = len(self.added) + len(self.modified) + len(self.deleted)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "snapshot1_id": self.snapshot1_id,
            "snapshot2_id": self.snapshot2_id,
            "added": self.added,
            "modified": self.modified,
            "deleted": self.deleted,
            "total_changes": self.total_changes
        }


class SnapshotManager:
    """
    Snapshot Manager - Point-in-time memory state capture and restore

    Pattern: Delta-encoded snapshots with timeline support
    Lifetime: Snapshots persist until manually deleted

    Features:
    - Create point-in-time memory snapshots
    - Delta encoding (only store changes since last snapshot)
    - Diff between snapshots
    - Rollback to previous snapshot state
    - MoltVault integration for cloud backup
    - Timeline browsing

    Example:
        manager = SnapshotManager(db_path)

        # Create snapshot
        snapshot = manager.create_snapshot("Before major changes")

        # Make changes to memories...

        # Create another snapshot (delta-encoded)
        snapshot2 = manager.create_snapshot("After changes")

        # View differences
        diff = manager.diff_snapshots(snapshot.snapshot_id, snapshot2.snapshot_id)

        # Rollback if needed
        manager.rollback_to_snapshot(snapshot.snapshot_id)
    """

    def __init__(self, db_path: Path):
        """
        Initialize Snapshot Manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

    def create_snapshot(
        self,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        moltvault_backup_id: Optional[str] = None
    ) -> SnapshotInfo:
        """
        Create a point-in-time snapshot of current memory state.

        This uses delta encoding - only memories that changed since the last
        snapshot are recorded. The first snapshot is always a full snapshot.

        Args:
            description: Optional description for the snapshot
            metadata: Optional metadata dictionary
            moltvault_backup_id: Optional MoltVault backup reference

        Returns:
            SnapshotInfo object with snapshot details

        Raises:
            sqlite3.Error: If snapshot creation fails
        """
        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        created_at = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")

            try:
                # Get the most recent snapshot for delta encoding
                cursor = conn.execute("""
                    SELECT snapshot_id, created_at
                    FROM snapshots
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                last_snapshot = cursor.fetchone()

                # Determine if this is a delta or full snapshot
                is_delta = last_snapshot is not None
                base_snapshot_id = last_snapshot[0] if is_delta else None

                # Create snapshot record
                metadata_json = json.dumps(metadata) if metadata else None
                conn.execute("""
                    INSERT INTO snapshots (snapshot_id, created_at, description, metadata_json, moltvault_backup_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (snapshot_id, created_at.isoformat(), description, metadata_json, moltvault_backup_id))

                if is_delta:
                    # Delta snapshot: only store changed memories
                    memory_count = self._create_delta_snapshot(conn, snapshot_id, base_snapshot_id)
                else:
                    # Full snapshot: store all current memories
                    memory_count = self._create_full_snapshot(conn, snapshot_id)

                conn.commit()

                return SnapshotInfo(
                    snapshot_id=snapshot_id,
                    created_at=created_at,
                    description=description,
                    metadata=metadata,
                    moltvault_backup_id=moltvault_backup_id,
                    memory_count=memory_count,
                    is_delta=is_delta,
                    base_snapshot_id=base_snapshot_id
                )

            except Exception:
                conn.rollback()
                raise

    def _create_full_snapshot(self, conn: sqlite3.Connection, snapshot_id: str) -> int:
        """
        Create a full snapshot (all current memories).

        Args:
            conn: SQLite connection
            snapshot_id: Snapshot identifier

        Returns:
            Number of memories captured
        """
        # Get all current memories with their latest versions
        cursor = conn.execute("""
            SELECT m.id, mv.version_id
            FROM memories m
            LEFT JOIN memory_versions mv ON m.id = mv.memory_id
            WHERE mv.version_id = (
                SELECT version_id
                FROM memory_versions mv2
                WHERE mv2.memory_id = m.id
                ORDER BY mv2.version_number DESC
                LIMIT 1
            )
        """)

        memories = cursor.fetchall()

        # Insert into snapshot_memories with ADDED operation
        for memory_id, version_id in memories:
            conn.execute("""
                INSERT INTO snapshot_memories (snapshot_id, memory_id, version_id, operation_type)
                VALUES (?, ?, ?, 'ADDED')
            """, (snapshot_id, memory_id, version_id))

        return len(memories)

    def _create_delta_snapshot(
        self,
        conn: sqlite3.Connection,
        snapshot_id: str,
        base_snapshot_id: str
    ) -> int:
        """
        Create a delta snapshot (only changed memories since base).

        Args:
            conn: SQLite connection
            snapshot_id: New snapshot identifier
            base_snapshot_id: Base snapshot to compare against

        Returns:
            Number of changed memories captured
        """
        # Get base snapshot creation time
        cursor = conn.execute("""
            SELECT created_at
            FROM snapshots
            WHERE snapshot_id = ?
        """, (base_snapshot_id,))
        base_time_str = cursor.fetchone()[0]
        base_time = datetime.fromisoformat(base_time_str)

        # Get memories that existed in base snapshot
        cursor = conn.execute("""
            SELECT memory_id
            FROM snapshot_memories
            WHERE snapshot_id = ?
        """, (base_snapshot_id,))
        base_memory_ids = {row[0] for row in cursor.fetchall()}

        # Get current memories
        cursor = conn.execute("""
            SELECT m.id, mv.version_id
            FROM memories m
            LEFT JOIN memory_versions mv ON m.id = mv.memory_id
            WHERE mv.version_id = (
                SELECT version_id
                FROM memory_versions mv2
                WHERE mv2.memory_id = m.id
                ORDER BY mv2.version_number DESC
                LIMIT 1
            )
        """)
        current_memories = {row[0]: row[1] for row in cursor.fetchall()}

        change_count = 0

        # Find added memories (in current but not in base)
        added = set(current_memories.keys()) - base_memory_ids
        for memory_id in added:
            conn.execute("""
                INSERT INTO snapshot_memories (snapshot_id, memory_id, version_id, operation_type)
                VALUES (?, ?, ?, 'ADDED')
            """, (snapshot_id, memory_id, current_memories[memory_id]))
            change_count += 1

        # Find deleted memories (in base but not in current)
        deleted = base_memory_ids - set(current_memories.keys())
        for memory_id in deleted:
            conn.execute("""
                INSERT INTO snapshot_memories (snapshot_id, memory_id, version_id, operation_type)
                VALUES (?, ?, NULL, 'DELETED')
            """, (snapshot_id, memory_id))
            change_count += 1

        # Find modified memories (in both, but version changed since base)
        common = base_memory_ids & set(current_memories.keys())
        for memory_id in common:
            # Check if memory has versions created after base snapshot
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM memory_versions
                WHERE memory_id = ? AND created_at > ?
            """, (memory_id, base_time.isoformat()))
            version_count = cursor.fetchone()[0]

            if version_count > 0:
                conn.execute("""
                    INSERT INTO snapshot_memories (snapshot_id, memory_id, version_id, operation_type)
                    VALUES (?, ?, ?, 'MODIFIED')
                """, (snapshot_id, memory_id, current_memories[memory_id]))
                change_count += 1

        return change_count

    def get_snapshot(self, snapshot_id: str) -> Optional[SnapshotInfo]:
        """
        Retrieve snapshot information.

        Args:
            snapshot_id: Snapshot identifier

        Returns:
            SnapshotInfo object or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT snapshot_id, created_at, description, metadata_json, moltvault_backup_id
                FROM snapshots
                WHERE snapshot_id = ?
            """, (snapshot_id,))

            row = cursor.fetchone()
            if not row:
                return None

            # Count memories in snapshot
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM snapshot_memories
                WHERE snapshot_id = ?
            """, (snapshot_id,))
            memory_count = cursor.fetchone()[0]

            # Determine if it's a delta snapshot
            cursor = conn.execute("""
                SELECT COUNT(*)
                FROM snapshots
                WHERE created_at < ?
            """, (row[1],))
            has_previous = cursor.fetchone()[0] > 0

            metadata = json.loads(row[3]) if row[3] else None

            return SnapshotInfo(
                snapshot_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                description=row[2],
                metadata=metadata,
                moltvault_backup_id=row[4],
                memory_count=memory_count,
                is_delta=has_previous,
                base_snapshot_id=None  # Not stored, could be computed if needed
            )

    def list_snapshots(self, limit: Optional[int] = None) -> List[SnapshotInfo]:
        """
        List all snapshots ordered by creation time (newest first).

        Args:
            limit: Optional maximum number of snapshots to return

        Returns:
            List of SnapshotInfo objects
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT snapshot_id, created_at, description, metadata_json, moltvault_backup_id
                FROM snapshots
                ORDER BY created_at DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query)
            rows = cursor.fetchall()

            snapshots = []
            for row in rows:
                # Count memories in snapshot
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM snapshot_memories
                    WHERE snapshot_id = ?
                """, (row[0],))
                memory_count = cursor.fetchone()[0]

                # Determine if it's a delta snapshot
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM snapshots
                    WHERE created_at < ?
                """, (row[1],))
                has_previous = cursor.fetchone()[0] > 0

                metadata = json.loads(row[3]) if row[3] else None

                snapshots.append(SnapshotInfo(
                    snapshot_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    description=row[2],
                    metadata=metadata,
                    moltvault_backup_id=row[4],
                    memory_count=memory_count,
                    is_delta=has_previous,
                    base_snapshot_id=None
                ))

            return snapshots

    def diff_snapshots(self, snapshot1_id: str, snapshot2_id: str) -> SnapshotDiff:
        """
        Compare two snapshots and show differences.

        Args:
            snapshot1_id: First snapshot ID (older)
            snapshot2_id: Second snapshot ID (newer)

        Returns:
            SnapshotDiff object with added/modified/deleted memories

        Raises:
            ValueError: If either snapshot doesn't exist
        """
        with sqlite3.connect(self.db_path) as conn:
            # Verify both snapshots exist
            for sid in [snapshot1_id, snapshot2_id]:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM snapshots WHERE snapshot_id = ?",
                    (sid,)
                )
                if cursor.fetchone()[0] == 0:
                    raise ValueError(f"Snapshot not found: {sid}")

            # Get memory state at snapshot1
            snapshot1_state = self._get_snapshot_memory_state(conn, snapshot1_id)

            # Get memory state at snapshot2
            snapshot2_state = self._get_snapshot_memory_state(conn, snapshot2_id)

            # Calculate differences
            added = list(set(snapshot2_state.keys()) - set(snapshot1_state.keys()))
            deleted = list(set(snapshot1_state.keys()) - set(snapshot2_state.keys()))

            # Find modified (same memory_id but different version_id)
            common = set(snapshot1_state.keys()) & set(snapshot2_state.keys())
            modified = [
                mid for mid in common
                if snapshot1_state[mid] != snapshot2_state[mid]
            ]

            return SnapshotDiff(
                snapshot1_id=snapshot1_id,
                snapshot2_id=snapshot2_id,
                added=added,
                modified=modified,
                deleted=deleted
            )

    def _get_snapshot_memory_state(
        self,
        conn: sqlite3.Connection,
        snapshot_id: str
    ) -> Dict[str, str]:
        """
        Reconstruct memory state at a snapshot by applying deltas.

        Args:
            conn: SQLite connection
            snapshot_id: Snapshot identifier

        Returns:
            Dictionary mapping memory_id to version_id
        """
        # Get all snapshots up to and including this one
        cursor = conn.execute("""
            SELECT snapshot_id
            FROM snapshots
            WHERE created_at <= (
                SELECT created_at
                FROM snapshots
                WHERE snapshot_id = ?
            )
            ORDER BY created_at ASC
        """, (snapshot_id,))

        snapshot_ids = [row[0] for row in cursor.fetchall()]

        # Build state by applying snapshots in order
        state: Dict[str, str] = {}

        for sid in snapshot_ids:
            cursor = conn.execute("""
                SELECT memory_id, version_id, operation_type
                FROM snapshot_memories
                WHERE snapshot_id = ?
            """, (sid,))

            for memory_id, version_id, operation_type in cursor.fetchall():
                if operation_type == 'DELETED':
                    state.pop(memory_id, None)
                else:  # ADDED or MODIFIED
                    state[memory_id] = version_id

        return state

    def rollback_to_snapshot(self, snapshot_id: str) -> int:
        """
        Rollback memory state to a specific snapshot.

        WARNING: This is a destructive operation that modifies current memory state.
        It's recommended to create a backup snapshot before rolling back.

        Args:
            snapshot_id: Snapshot to rollback to

        Returns:
            Number of memories affected

        Raises:
            ValueError: If snapshot doesn't exist
            sqlite3.Error: If rollback fails
        """
        with sqlite3.connect(self.db_path) as conn:
            # Verify snapshot exists
            cursor = conn.execute(
                "SELECT COUNT(*) FROM snapshots WHERE snapshot_id = ?",
                (snapshot_id,)
            )
            if cursor.fetchone()[0] == 0:
                raise ValueError(f"Snapshot not found: {snapshot_id}")

            conn.execute("BEGIN IMMEDIATE")

            try:
                # Get target state
                target_state = self._get_snapshot_memory_state(conn, snapshot_id)

                # Get current state
                cursor = conn.execute("""
                    SELECT m.id, mv.version_id
                    FROM memories m
                    LEFT JOIN memory_versions mv ON m.id = mv.memory_id
                    WHERE mv.version_id = (
                        SELECT version_id
                        FROM memory_versions mv2
                        WHERE mv2.memory_id = m.id
                        ORDER BY mv2.version_number DESC
                        LIMIT 1
                    )
                """)
                current_state = {row[0]: row[1] for row in cursor.fetchall()}

                changes = 0

                # Restore memories to target versions
                for memory_id, version_id in target_state.items():
                    if memory_id not in current_state or current_state[memory_id] != version_id:
                        # Get content from version
                        cursor = conn.execute("""
                            SELECT content
                            FROM memory_versions
                            WHERE version_id = ?
                        """, (version_id,))
                        row = cursor.fetchone()
                        if row:
                            # Update memory to historical content
                            conn.execute("""
                                UPDATE memories
                                SET content = ?
                                WHERE id = ?
                            """, (row[0], memory_id))
                            changes += 1

                # Delete memories not in target state
                for memory_id in current_state:
                    if memory_id not in target_state:
                        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                        changes += 1

                # Add back memories that were deleted
                for memory_id in target_state:
                    if memory_id not in current_state:
                        # Restore from version
                        cursor = conn.execute("""
                            SELECT content, version_number
                            FROM memory_versions
                            WHERE version_id = ?
                        """, (target_state[memory_id],))
                        row = cursor.fetchone()
                        if row:
                            conn.execute("""
                                INSERT INTO memories (id, content, created_at)
                                VALUES (?, ?, CURRENT_TIMESTAMP)
                            """, (memory_id, row[0]))
                            changes += 1

                conn.commit()
                return changes

            except Exception:
                conn.rollback()
                raise

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.

        Args:
            snapshot_id: Snapshot to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM snapshots
                WHERE snapshot_id = ?
            """, (snapshot_id,))

            return cursor.rowcount > 0
