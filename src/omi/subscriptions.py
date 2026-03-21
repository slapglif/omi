"""
Subscription management for topic-based memory notifications

Pattern: Database-backed subscription system for multi-agent coordination
"""

import sqlite3
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Set, Dict, Any
from dataclasses import dataclass


@dataclass
class SubscriptionInfo:
    """Information about a subscription"""
    id: str
    agent_id: str
    namespace: Optional[str]
    memory_id: Optional[str]
    event_types: List[str]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "namespace": self.namespace,
            "memory_id": self.memory_id,
            "event_types": self.event_types,
            "created_at": self.created_at.isoformat()
        }


class SubscriptionManager:
    """
    Subscription manager for topic-based memory notifications

    Features:
    - Subscribe to specific namespaces, memories, or event types
    - Filter subscriptions by agent, namespace, or memory
    - Match subscriptions against events for notification routing
    - Thread-safe database operations
    - WAL mode for concurrent access

    Subscription types:
    - Global: namespace=None, memory_id=None (all events)
    - Namespace: namespace=X, memory_id=None (all events in namespace)
    - Memory-specific: memory_id=X (events for specific memory)
    - Event-filtered: event_types=[...] (only specific event types)

    Pattern follows SharedNamespace architecture:
    - Persistent connection with thread lock
    - WAL mode for concurrent writes
    - Foreign key constraints
    - Proper error handling

    Usage:
        manager = SubscriptionManager(db_path)

        # Subscribe to all events in a namespace
        manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["memory.stored", "belief.updated"]
        )

        # Subscribe to specific memory
        manager.subscribe(
            agent_id="agent-2",
            memory_id="mem-123",
            event_types=["belief.updated"]
        )

        # Find matching subscriptions for an event
        subscriptions = manager.match_subscriptions(
            event_type="memory.stored",
            namespace="acme/research",
            memory_id="mem-456"
        )
    """

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize SubscriptionManager.

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

    def subscribe(
        self,
        agent_id: str,
        event_types: List[str],
        namespace: Optional[str] = None,
        memory_id: Optional[str] = None
    ) -> SubscriptionInfo:
        """
        Create a subscription for an agent.

        Args:
            agent_id: Agent ID creating the subscription
            event_types: List of event types to subscribe to (e.g., ["memory.stored"])
                        Use ["*"] to subscribe to all event types
            namespace: Optional namespace to filter by
            memory_id: Optional memory ID to filter by

        Returns:
            SubscriptionInfo object

        Raises:
            ValueError: If namespace or memory_id doesn't exist (foreign key constraint)
            ValueError: If event_types is empty

        Examples:
            # Subscribe to all events in namespace
            subscribe(agent_id="agent-1", namespace="acme/research", event_types=["*"])

            # Subscribe to memory stored events only
            subscribe(agent_id="agent-1", event_types=["memory.stored"])

            # Subscribe to specific memory updates
            subscribe(agent_id="agent-1", memory_id="mem-123", event_types=["belief.updated"])
        """
        if not event_types:
            raise ValueError("event_types cannot be empty")

        # Generate unique ID
        subscription_id = str(uuid.uuid4())

        # Serialize event_types to JSON
        event_types_json = json.dumps(event_types)

        with self._db_lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO subscriptions (id, agent_id, namespace, memory_id, event_types)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (subscription_id, agent_id, namespace, memory_id, event_types_json)
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(
                    f"Failed to create subscription: {e}. "
                    f"Namespace or memory may not exist."
                )

            # Retrieve the created subscription
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, namespace, memory_id, event_types, created_at
                FROM subscriptions
                WHERE id = ?
                """,
                (subscription_id,)
            )
            row = cursor.fetchone()

        if not row:
            raise RuntimeError("Failed to create subscription")

        return self._row_to_info(row)

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove a subscription.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if subscription was removed, False if it didn't exist
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                DELETE FROM subscriptions WHERE id = ?
                """,
                (subscription_id,)
            )
            return cursor.rowcount > 0

    def unsubscribe_agent_from_namespace(
        self,
        agent_id: str,
        namespace: str
    ) -> int:
        """
        Remove all subscriptions for an agent in a specific namespace.

        Args:
            agent_id: Agent ID
            namespace: Namespace string

        Returns:
            Number of subscriptions removed
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                DELETE FROM subscriptions
                WHERE agent_id = ? AND namespace = ?
                """,
                (agent_id, namespace)
            )
            return cursor.rowcount

    def get(self, subscription_id: str) -> Optional[SubscriptionInfo]:
        """
        Get a specific subscription by ID.

        Args:
            subscription_id: Subscription ID

        Returns:
            SubscriptionInfo if exists, None otherwise
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, namespace, memory_id, event_types, created_at
                FROM subscriptions
                WHERE id = ?
                """,
                (subscription_id,)
            )
            row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_info(row)

    def list_for_agent(self, agent_id: str) -> List[SubscriptionInfo]:
        """
        List all subscriptions for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of SubscriptionInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, namespace, memory_id, event_types, created_at
                FROM subscriptions
                WHERE agent_id = ?
                ORDER BY created_at DESC
                """,
                (agent_id,)
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def list_for_namespace(self, namespace: str) -> List[SubscriptionInfo]:
        """
        List all subscriptions for a specific namespace.

        Args:
            namespace: Namespace string

        Returns:
            List of SubscriptionInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, namespace, memory_id, event_types, created_at
                FROM subscriptions
                WHERE namespace = ?
                ORDER BY created_at DESC
                """,
                (namespace,)
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def list_for_memory(self, memory_id: str) -> List[SubscriptionInfo]:
        """
        List all subscriptions for a specific memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of SubscriptionInfo objects
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, namespace, memory_id, event_types, created_at
                FROM subscriptions
                WHERE memory_id = ?
                ORDER BY created_at DESC
                """,
                (memory_id,)
            )
            rows = cursor.fetchall()

        return [self._row_to_info(row) for row in rows]

    def match_subscriptions(
        self,
        event_type: str,
        namespace: Optional[str] = None,
        memory_id: Optional[str] = None
    ) -> List[SubscriptionInfo]:
        """
        Find subscriptions that match an event.

        Matching logic:
        1. Event type must match (or subscription has wildcard "*")
        2. If memory_id provided, match memory-specific subscriptions
        3. If namespace provided, match namespace subscriptions
        4. Always include global subscriptions (namespace=None, memory_id=None)

        Args:
            event_type: Event type string (e.g., "memory.stored")
            namespace: Optional namespace the event occurred in
            memory_id: Optional memory ID the event relates to

        Returns:
            List of matching SubscriptionInfo objects

        Example:
            # Event: memory.stored in namespace "acme/research" for memory "mem-123"
            match_subscriptions(
                event_type="memory.stored",
                namespace="acme/research",
                memory_id="mem-123"
            )
            # Returns subscriptions for:
            # - agent-1: global subscription to "memory.stored"
            # - agent-2: namespace subscription to "acme/research"
            # - agent-3: memory subscription to "mem-123"
        """
        matching_subscriptions = []

        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT id, agent_id, namespace, memory_id, event_types, created_at
                FROM subscriptions
                """
            )
            rows = cursor.fetchall()

        for row in rows:
            subscription = self._row_to_info(row)

            # Check event type match
            if not self._event_type_matches(event_type, subscription.event_types):
                continue

            # Check scope match
            if subscription.memory_id:
                # Memory-specific subscription
                if subscription.memory_id == memory_id:
                    matching_subscriptions.append(subscription)
            elif subscription.namespace:
                # Namespace subscription
                if subscription.namespace == namespace:
                    matching_subscriptions.append(subscription)
            else:
                # Global subscription
                matching_subscriptions.append(subscription)

        return matching_subscriptions

    def get_subscribed_agents(
        self,
        event_type: str,
        namespace: Optional[str] = None,
        memory_id: Optional[str] = None
    ) -> Set[str]:
        """
        Get all agent IDs subscribed to an event.

        Args:
            event_type: Event type string
            namespace: Optional namespace
            memory_id: Optional memory ID

        Returns:
            Set of agent IDs
        """
        subscriptions = self.match_subscriptions(event_type, namespace, memory_id)
        return {sub.agent_id for sub in subscriptions}

    def count_for_agent(self, agent_id: str) -> int:
        """
        Count subscriptions for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Number of active subscriptions
        """
        with self._db_lock:
            cursor = self._conn.execute(
                """
                SELECT COUNT(*) FROM subscriptions WHERE agent_id = ?
                """,
                (agent_id,)
            )
            row = cursor.fetchone()

        return row[0] if row else 0

    def _event_type_matches(
        self,
        event_type: str,
        subscription_event_types: List[str]
    ) -> bool:
        """
        Check if event type matches subscription event types.

        Args:
            event_type: Actual event type (e.g., "memory.stored")
            subscription_event_types: List of event types from subscription

        Returns:
            True if matches, False otherwise
        """
        # Wildcard match
        if "*" in subscription_event_types:
            return True

        # Exact match
        if event_type in subscription_event_types:
            return True

        return False

    def _row_to_info(self, row: tuple) -> SubscriptionInfo:
        """
        Convert database row to SubscriptionInfo.

        Args:
            row: Database row tuple

        Returns:
            SubscriptionInfo object
        """
        sub_id, agent_id, namespace, memory_id, event_types_json, created_at = row

        # Parse created_at from ISO format
        created_at_dt = datetime.fromisoformat(created_at) if created_at else datetime.now()

        # Parse event_types JSON
        event_types = json.loads(event_types_json) if event_types_json else []

        return SubscriptionInfo(
            id=sub_id,
            agent_id=agent_id,
            namespace=namespace,
            memory_id=memory_id,
            event_types=event_types,
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
