"""
Conflict detection and resolution for concurrent writes in shared namespaces

Pattern: Database-backed conflict detection with configurable resolution strategies
"""

import sqlite3
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class ConflictResolutionStrategy(str, Enum):
    """
    Strategies for resolving concurrent write conflicts

    - LAST_WRITER_WINS: Accept the most recent write, discard others
    - MERGE: Attempt to merge conflicting changes (content-dependent)
    - REJECT: Reject the conflicting write, return error to agent
    """
    LAST_WRITER_WINS = "last_writer_wins"
    MERGE = "merge"
    REJECT = "reject"

    @classmethod
    def from_string(cls, strategy: str) -> "ConflictResolutionStrategy":
        """
        Convert string to ConflictResolutionStrategy

        Args:
            strategy: Strategy string (case-insensitive)

        Returns:
            ConflictResolutionStrategy enum value

        Raises:
            ValueError: If strategy is invalid
        """
        strategy_lower = strategy.lower()
        if strategy_lower == "last_writer_wins":
            return cls.LAST_WRITER_WINS
        elif strategy_lower == "merge":
            return cls.MERGE
        elif strategy_lower == "reject":
            return cls.REJECT
        else:
            raise ValueError(
                f"Invalid conflict resolution strategy '{strategy}'. "
                "Must be one of: last_writer_wins, merge, reject"
            )

    def __str__(self) -> str:
        """String representation"""
        return self.value


@dataclass
class WriteIntent:
    """
    Represents an intention to write to a memory

    Used for detecting concurrent writes before they are committed
    """
    id: str
    memory_id: str
    agent_id: str
    namespace: Optional[str]
    content_hash: str  # SHA-256 of intended content
    base_version: Optional[int]  # Version the write is based on
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "content_hash": self.content_hash,
            "base_version": self.base_version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata or {}
        }


@dataclass
class Conflict:
    """
    Represents a detected write conflict between agents
    """
    id: str
    memory_id: str
    namespace: Optional[str]
    conflicting_agents: List[str]
    conflicting_intents: List[str]  # WriteIntent IDs
    detected_at: datetime
    resolved: bool = False
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    winner_intent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "namespace": self.namespace,
            "conflicting_agents": self.conflicting_agents,
            "conflicting_intents": self.conflicting_intents,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy.value if self.resolution_strategy else None,
            "winner_intent_id": self.winner_intent_id,
            "metadata": self.metadata or {}
        }


class ConflictDetector:
    """
    Conflict detector for detecting and tracking concurrent writes

    Features:
    - Register write intents before committing
    - Detect concurrent writes to same memory
    - Track conflict resolution outcomes
    - Thread-safe database operations
    - WAL mode for concurrent access

    Pattern follows SharedNamespace/PermissionManager architecture:
    - Persistent connection with thread lock
    - WAL mode for concurrent writes
    - Foreign key constraints
    - Proper error handling

    Usage:
        detector = ConflictDetector(db_path)

        # Register write intent
        intent = detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            namespace="acme/research",
            content_hash="abc123...",
            base_version=5
        )

        # Check for conflicts
        conflict = detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        if conflict:
            # Resolve conflict
            detector.resolve_conflict(
                conflict_id=conflict.id,
                strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
                winner_intent_id=intent.id
            )

        # Commit the write
        detector.commit_intent(intent.id)
    """

    # Conflict detection window (seconds)
    # Writes within this window are considered potentially conflicting
    CONFLICT_WINDOW_SECONDS = 5

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize ConflictDetector.

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

        # Initialize schema
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema for conflict detection"""
        # Enable WAL mode and foreign keys
        if self._enable_wal:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Create tables for write intents and conflicts
        with self._db_lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS write_intents (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    namespace TEXT,
                    content_hash TEXT NOT NULL,
                    base_version INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    committed BOOLEAN DEFAULT 0,
                    metadata TEXT  -- JSON
                );

                CREATE TABLE IF NOT EXISTS conflicts (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    namespace TEXT,
                    conflicting_agents TEXT NOT NULL,  -- JSON array
                    conflicting_intents TEXT NOT NULL,  -- JSON array
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    resolution_strategy TEXT,
                    winner_intent_id TEXT,
                    metadata TEXT  -- JSON
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_write_intents_memory_id ON write_intents(memory_id);
                CREATE INDEX IF NOT EXISTS idx_write_intents_agent_id ON write_intents(agent_id);
                CREATE INDEX IF NOT EXISTS idx_write_intents_namespace ON write_intents(namespace);
                CREATE INDEX IF NOT EXISTS idx_write_intents_created_at ON write_intents(created_at);
                CREATE INDEX IF NOT EXISTS idx_write_intents_committed ON write_intents(committed);
                CREATE INDEX IF NOT EXISTS idx_conflicts_memory_id ON conflicts(memory_id);
                CREATE INDEX IF NOT EXISTS idx_conflicts_namespace ON conflicts(namespace);
                CREATE INDEX IF NOT EXISTS idx_conflicts_detected_at ON conflicts(detected_at);
                CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts(resolved);
            """)

    def register_intent(
        self,
        memory_id: str,
        agent_id: str,
        content_hash: str,
        namespace: Optional[str] = None,
        base_version: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WriteIntent:
        """
        Register an intent to write to a memory.

        This should be called BEFORE actually writing to the memory,
        to allow conflict detection.

        Args:
            memory_id: Memory ID to write to
            agent_id: Agent ID performing the write
            content_hash: SHA-256 hash of the content to write
            namespace: Optional namespace the memory belongs to
            base_version: Optional version the write is based on
            metadata: Optional metadata dictionary

        Returns:
            WriteIntent object
        """
        intent_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata) if metadata else None

        with self._db_lock:
            self._conn.execute(
                """
                INSERT INTO write_intents
                (id, memory_id, agent_id, namespace, content_hash, base_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (intent_id, memory_id, agent_id, namespace, content_hash, base_version, metadata_json)
            )

            # Retrieve the created intent
            cursor = self._conn.execute(
                """
                SELECT id, memory_id, agent_id, namespace, content_hash, base_version, created_at, metadata
                FROM write_intents
                WHERE id = ?
                """,
                (intent_id,)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to register write intent")

        return self._row_to_intent(row)

    def detect_conflict(
        self,
        memory_id: str,
        agent_id: str,
        time_window_seconds: Optional[int] = None
    ) -> Optional[Conflict]:
        """
        Detect if there are concurrent write intents for a memory.

        A conflict exists if multiple agents have uncommitted write intents
        for the same memory within the time window.

        Args:
            memory_id: Memory ID to check
            agent_id: Agent ID to check conflicts for
            time_window_seconds: Time window for conflict detection (default: CONFLICT_WINDOW_SECONDS)

        Returns:
            Conflict object if conflict detected, None otherwise
        """
        if time_window_seconds is None:
            time_window_seconds = self.CONFLICT_WINDOW_SECONDS

        with self._db_lock:
            # Find all uncommitted intents for this memory within the time window
            cursor = self._conn.execute(
                """
                SELECT id, memory_id, agent_id, namespace, content_hash, base_version, created_at, metadata
                FROM write_intents
                WHERE memory_id = ?
                  AND committed = 0
                  AND datetime(created_at) >= datetime('now', '-' || ? || ' seconds')
                ORDER BY created_at ASC
                """,
                (memory_id, time_window_seconds)
            )
            rows = cursor.fetchall()

        if len(rows) <= 1:
            # No conflict if only one or zero intents
            return None

        # Extract unique agents
        agents = list(set(row[2] for row in rows))  # row[2] is agent_id

        if len(agents) <= 1:
            # No conflict if all intents are from the same agent
            return None

        # Conflict detected - create conflict record
        conflict_id = str(uuid.uuid4())
        intent_ids = [row[0] for row in rows]
        namespace = rows[0][3]  # Assume same namespace for all intents

        with self._db_lock:
            self._conn.execute(
                """
                INSERT INTO conflicts
                (id, memory_id, namespace, conflicting_agents, conflicting_intents)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conflict_id, memory_id, namespace, json.dumps(agents), json.dumps(intent_ids))
            )

            # Retrieve the created conflict
            cursor = self._conn.execute(
                """
                SELECT id, memory_id, namespace, conflicting_agents, conflicting_intents,
                       detected_at, resolved, resolution_strategy, winner_intent_id, metadata
                FROM conflicts
                WHERE id = ?
                """,
                (conflict_id,)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to create conflict record")

        return self._row_to_conflict(row)

    def get_conflict(self, conflict_id: str) -> Optional[Conflict]:
        """
        Get a conflict by ID.

        Args:
            conflict_id: Conflict ID

        Returns:
            Conflict object if exists, None otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, memory_id, namespace, conflicting_agents, conflicting_intents,
                       detected_at, resolved, resolution_strategy, winner_intent_id, metadata
                FROM conflicts
                WHERE id = ?
                """,
                (conflict_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_conflict(row)

    def resolve_conflict(
        self,
        conflict_id: str,
        strategy: ConflictResolutionStrategy,
        winner_intent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conflict:
        """
        Resolve a conflict using the specified strategy.

        Args:
            conflict_id: Conflict ID to resolve
            strategy: Resolution strategy to use
            winner_intent_id: Optional winning intent ID (required for LAST_WRITER_WINS)
            metadata: Optional metadata about the resolution

        Returns:
            Updated Conflict object

        Raises:
            ValueError: If conflict doesn't exist or is already resolved
        """
        conflict = self.get_conflict(conflict_id)
        if not conflict:
            raise ValueError(f"Conflict not found: {conflict_id}")

        if conflict.resolved:
            raise ValueError(f"Conflict already resolved: {conflict_id}")

        metadata_json = json.dumps(metadata) if metadata else None

        with self._db_lock:
            self._conn.execute(
                """
                UPDATE conflicts
                SET resolved = 1,
                    resolution_strategy = ?,
                    winner_intent_id = ?,
                    metadata = ?
                WHERE id = ?
                """,
                (strategy.value, winner_intent_id, metadata_json, conflict_id)
            )

            # Retrieve the updated conflict
            cursor = self._conn.execute(
                """
                SELECT id, memory_id, namespace, conflicting_agents, conflicting_intents,
                       detected_at, resolved, resolution_strategy, winner_intent_id, metadata
                FROM conflicts
                WHERE id = ?
                """,
                (conflict_id,)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to resolve conflict")

        return self._row_to_conflict(row)

    def commit_intent(self, intent_id: str) -> bool:
        """
        Mark a write intent as committed.

        This should be called after the write has been successfully applied.

        Args:
            intent_id: WriteIntent ID to commit

        Returns:
            True if committed, False if intent not found
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                UPDATE write_intents
                SET committed = 1
                WHERE id = ?
                """,
                (intent_id,)
            )
            return cursor.rowcount > 0

    def get_intent(self, intent_id: str) -> Optional[WriteIntent]:
        """
        Get a write intent by ID.

        Args:
            intent_id: WriteIntent ID

        Returns:
            WriteIntent object if exists, None otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, memory_id, agent_id, namespace, content_hash, base_version, created_at, metadata
                FROM write_intents
                WHERE id = ?
                """,
                (intent_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_intent(row)

    def get_pending_intents(
        self,
        memory_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> List[WriteIntent]:
        """
        Get pending (uncommitted) write intents.

        Args:
            memory_id: Optional memory ID to filter by
            agent_id: Optional agent ID to filter by
            namespace: Optional namespace to filter by

        Returns:
            List of WriteIntent objects
        """
        query = """
            SELECT id, memory_id, agent_id, namespace, content_hash, base_version, created_at, metadata
            FROM write_intents
            WHERE committed = 0
        """
        params = []

        if memory_id:
            query += " AND memory_id = ?"
            params.append(memory_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)

        query += " ORDER BY created_at ASC"

        with self._db_lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_intent(row) for row in rows]

    def get_conflicts(
        self,
        memory_id: Optional[str] = None,
        namespace: Optional[str] = None,
        resolved: Optional[bool] = None
    ) -> List[Conflict]:
        """
        Get conflicts with optional filters.

        Args:
            memory_id: Optional memory ID to filter by
            namespace: Optional namespace to filter by
            resolved: Optional resolved status to filter by

        Returns:
            List of Conflict objects
        """
        query = """
            SELECT id, memory_id, namespace, conflicting_agents, conflicting_intents,
                   detected_at, resolved, resolution_strategy, winner_intent_id, metadata
            FROM conflicts
            WHERE 1=1
        """
        params = []

        if memory_id:
            query += " AND memory_id = ?"
            params.append(memory_id)
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(1 if resolved else 0)

        query += " ORDER BY detected_at DESC"

        with self._db_lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_conflict(row) for row in rows]

    def cleanup_old_intents(self, age_seconds: int = 3600) -> int:
        """
        Clean up old committed write intents.

        Args:
            age_seconds: Delete intents older than this (default: 1 hour)

        Returns:
            Number of intents deleted
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                DELETE FROM write_intents
                WHERE committed = 1
                  AND datetime(created_at) < datetime('now', '-' || ? || ' seconds')
                """,
                (age_seconds,)
            )
            return cursor.rowcount

    def _row_to_intent(self, row: tuple) -> WriteIntent:
        """
        Convert database row to WriteIntent.

        Args:
            row: Database row tuple

        Returns:
            WriteIntent object
        """
        intent_id, memory_id, agent_id, namespace, content_hash, base_version, created_at, metadata_json = row

        # Parse created_at from ISO format
        created_at_dt = datetime.fromisoformat(created_at) if created_at else datetime.now()

        # Parse metadata JSON
        metadata = json.loads(metadata_json) if metadata_json else None

        return WriteIntent(
            id=intent_id,
            memory_id=memory_id,
            agent_id=agent_id,
            namespace=namespace,
            content_hash=content_hash,
            base_version=base_version,
            created_at=created_at_dt,
            metadata=metadata
        )

    def _row_to_conflict(self, row: tuple) -> Conflict:
        """
        Convert database row to Conflict.

        Args:
            row: Database row tuple

        Returns:
            Conflict object
        """
        (conflict_id, memory_id, namespace, conflicting_agents_json, conflicting_intents_json,
         detected_at, resolved, resolution_strategy, winner_intent_id, metadata_json) = row

        # Parse detected_at from ISO format
        detected_at_dt = datetime.fromisoformat(detected_at) if detected_at else datetime.now()

        # Parse JSON fields
        conflicting_agents = json.loads(conflicting_agents_json) if conflicting_agents_json else []
        conflicting_intents = json.loads(conflicting_intents_json) if conflicting_intents_json else []
        metadata = json.loads(metadata_json) if metadata_json else None

        # Parse resolution strategy
        strategy = None
        if resolution_strategy:
            strategy = ConflictResolutionStrategy.from_string(resolution_strategy)

        return Conflict(
            id=conflict_id,
            memory_id=memory_id,
            namespace=namespace,
            conflicting_agents=conflicting_agents,
            conflicting_intents=conflicting_intents,
            detected_at=detected_at_dt,
            resolved=bool(resolved),
            resolution_strategy=strategy,
            winner_intent_id=winner_intent_id,
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


@dataclass
class ResolutionResult:
    """
    Result of a conflict resolution operation
    """
    winner_intent_id: Optional[str]
    strategy: ConflictResolutionStrategy
    merged_content: Optional[str] = None  # For merge strategy
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "winner_intent_id": self.winner_intent_id,
            "strategy": self.strategy.value,
            "merged_content": self.merged_content,
            "reason": self.reason,
            "metadata": self.metadata or {}
        }


class LastWriterWinsResolver:
    """
    Last-writer-wins conflict resolution strategy

    Resolves conflicts by accepting the most recent write intent
    and discarding all others.

    Pattern: Simple timestamp-based resolution
    - Find the intent with the most recent created_at timestamp
    - Mark it as the winner
    - Discard all other conflicting intents

    Usage:
        resolver = LastWriterWinsResolver()
        result = resolver.resolve(conflict, intents)

        if result.winner_intent_id:
            # Apply the winning write
            apply_write(result.winner_intent_id)
    """

    def resolve(
        self,
        conflict: Conflict,
        intents: List[WriteIntent]
    ) -> ResolutionResult:
        """
        Resolve conflict using last-writer-wins strategy.

        Args:
            conflict: Conflict to resolve
            intents: List of conflicting WriteIntent objects

        Returns:
            ResolutionResult with winner_intent_id set to most recent intent

        Raises:
            ValueError: If no intents provided
        """
        if not intents:
            raise ValueError("No write intents provided for resolution")

        # Sort by timestamp, most recent first
        sorted_intents = sorted(intents, key=lambda i: i.created_at, reverse=True)
        winner = sorted_intents[0]

        return ResolutionResult(
            winner_intent_id=winner.id,
            strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
            reason=f"Most recent write at {winner.created_at.isoformat()}",
            metadata={
                "winner_agent": winner.agent_id,
                "winner_timestamp": winner.created_at.isoformat(),
                "discarded_intents": [i.id for i in sorted_intents[1:]]
            }
        )


class MergeResolver:
    """
    Merge conflict resolution strategy

    Attempts to merge conflicting changes intelligently.

    Pattern: Content-aware merging
    - If content_hash is identical, accept any write (they're the same)
    - If base_version differs, attempt three-way merge
    - If merge fails, fall back to last-writer-wins

    Merge strategies:
    1. Identical content: Accept any (no conflict)
    2. Non-overlapping changes: Combine both changes
    3. Overlapping changes: Fall back to last-writer-wins

    Usage:
        resolver = MergeResolver()
        result = resolver.resolve(conflict, intents, get_content_func)

        if result.merged_content:
            # Apply the merged content
            apply_merged_write(result.merged_content)
        elif result.winner_intent_id:
            # Merge failed, use winner
            apply_write(result.winner_intent_id)
    """

    def resolve(
        self,
        conflict: Conflict,
        intents: List[WriteIntent]
    ) -> ResolutionResult:
        """
        Resolve conflict using merge strategy.

        Args:
            conflict: Conflict to resolve
            intents: List of conflicting WriteIntent objects

        Returns:
            ResolutionResult with merged_content or winner_intent_id

        Raises:
            ValueError: If no intents provided
        """
        if not intents:
            raise ValueError("No write intents provided for resolution")

        # Check if all content hashes are identical
        unique_hashes = set(i.content_hash for i in intents)

        if len(unique_hashes) == 1:
            # All writes are identical - accept any
            winner = intents[0]
            return ResolutionResult(
                winner_intent_id=winner.id,
                strategy=ConflictResolutionStrategy.MERGE,
                reason="All write intents have identical content",
                metadata={
                    "merge_type": "identical",
                    "content_hash": winner.content_hash
                }
            )

        # Check if all intents have same base_version
        # If base_versions match, changes are based on same state
        base_versions = [i.base_version for i in intents if i.base_version is not None]

        if len(base_versions) == len(intents) and len(set(base_versions)) == 1:
            # Same base version - attempt merge
            # For now, fall back to last-writer-wins
            # TODO: Implement content-aware merging
            sorted_intents = sorted(intents, key=lambda i: i.created_at, reverse=True)
            winner = sorted_intents[0]

            return ResolutionResult(
                winner_intent_id=winner.id,
                strategy=ConflictResolutionStrategy.MERGE,
                reason="Content-aware merge not yet implemented, using last-writer-wins",
                metadata={
                    "merge_type": "fallback_lww",
                    "base_version": base_versions[0],
                    "conflicting_agents": [i.agent_id for i in intents]
                }
            )

        # Different base versions or missing base_version - cannot merge safely
        # Fall back to last-writer-wins
        sorted_intents = sorted(intents, key=lambda i: i.created_at, reverse=True)
        winner = sorted_intents[0]

        return ResolutionResult(
            winner_intent_id=winner.id,
            strategy=ConflictResolutionStrategy.MERGE,
            reason="Cannot merge writes with different base versions, using last-writer-wins",
            metadata={
                "merge_type": "fallback_lww",
                "reason": "incompatible_base_versions"
            }
        )


class RejectResolver:
    """
    Reject conflict resolution strategy

    Rejects conflicting writes and returns an error.

    Pattern: Conservative conflict handling
    - Detect conflict
    - Reject all writes
    - Return error to agents
    - Require manual resolution

    This strategy ensures no data loss but requires human/agent intervention.

    Usage:
        resolver = RejectResolver()
        result = resolver.resolve(conflict, intents)

        if not result.winner_intent_id:
            # All writes rejected - notify agents
            notify_conflict(conflict, result.reason)
    """

    def resolve(
        self,
        conflict: Conflict,
        intents: List[WriteIntent]
    ) -> ResolutionResult:
        """
        Resolve conflict by rejecting all writes.

        Args:
            conflict: Conflict to resolve
            intents: List of conflicting WriteIntent objects

        Returns:
            ResolutionResult with winner_intent_id=None (rejected)

        Raises:
            ValueError: If no intents provided
        """
        if not intents:
            raise ValueError("No write intents provided for resolution")

        agent_ids = list(set(i.agent_id for i in intents))

        return ResolutionResult(
            winner_intent_id=None,
            strategy=ConflictResolutionStrategy.REJECT,
            reason=f"Concurrent writes rejected - manual resolution required",
            metadata={
                "conflicting_agents": agent_ids,
                "conflicting_intents": [i.id for i in intents],
                "conflict_id": conflict.id,
                "memory_id": conflict.memory_id
            }
        )
