"""
Unit tests for Conflict Detection and Resolution

Tests cover:
- ConflictResolutionStrategy enum
- WriteIntent registration and tracking
- Conflict detection (concurrent writes)
- Conflict resolution strategies:
  - Last-writer-wins
  - Merge (identical content, fallback)
  - Reject
- Intent commit/rollback
- Cleanup operations
- Thread-safe database operations
- Edge cases and error handling
"""

import unittest
import tempfile
import time
import threading
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.conflict_resolution import (
    ConflictResolutionStrategy,
    ConflictDetector,
    WriteIntent,
    Conflict,
    LastWriterWinsResolver,
    MergeResolver,
    RejectResolver,
    ResolutionResult
)


class TestConflictResolutionStrategy(unittest.TestCase):
    """Test suite for ConflictResolutionStrategy enum."""

    def test_from_string_last_writer_wins(self):
        """Test converting string to LAST_WRITER_WINS strategy."""
        strategy = ConflictResolutionStrategy.from_string("last_writer_wins")
        self.assertEqual(strategy, ConflictResolutionStrategy.LAST_WRITER_WINS)

    def test_from_string_merge(self):
        """Test converting string to MERGE strategy."""
        strategy = ConflictResolutionStrategy.from_string("merge")
        self.assertEqual(strategy, ConflictResolutionStrategy.MERGE)

    def test_from_string_reject(self):
        """Test converting string to REJECT strategy."""
        strategy = ConflictResolutionStrategy.from_string("reject")
        self.assertEqual(strategy, ConflictResolutionStrategy.REJECT)

    def test_from_string_case_insensitive(self):
        """Test from_string is case-insensitive."""
        strategy = ConflictResolutionStrategy.from_string("LAST_WRITER_WINS")
        self.assertEqual(strategy, ConflictResolutionStrategy.LAST_WRITER_WINS)

    def test_from_string_invalid(self):
        """Test invalid strategy string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            ConflictResolutionStrategy.from_string("invalid")

        self.assertIn("Invalid conflict resolution strategy", str(context.exception))

    def test_strategy_to_string(self):
        """Test strategy string representation."""
        self.assertEqual(str(ConflictResolutionStrategy.LAST_WRITER_WINS), "last_writer_wins")
        self.assertEqual(str(ConflictResolutionStrategy.MERGE), "merge")
        self.assertEqual(str(ConflictResolutionStrategy.REJECT), "reject")


class TestConflictDetector(unittest.TestCase):
    """Test suite for ConflictDetector."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_conflicts.sqlite"
        self.detector = ConflictDetector(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        self.detector.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _hash_content(self, content: str) -> str:
        """Generate SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    # ==================== Write Intent Registration ====================

    def test_register_intent(self):
        """Test registering a write intent."""
        content_hash = self._hash_content("Test content")

        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=content_hash,
            namespace="acme/research",
            base_version=5
        )

        self.assertIsNotNone(intent.id)
        self.assertEqual(intent.memory_id, "mem-123")
        self.assertEqual(intent.agent_id, "agent-1")
        self.assertEqual(intent.content_hash, content_hash)
        self.assertEqual(intent.namespace, "acme/research")
        self.assertEqual(intent.base_version, 5)
        self.assertIsInstance(intent.created_at, datetime)

    def test_register_intent_with_metadata(self):
        """Test registering intent with metadata."""
        content_hash = self._hash_content("Test content")
        metadata = {"source": "api", "priority": "high"}

        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=content_hash,
            metadata=metadata
        )

        self.assertEqual(intent.metadata, metadata)

    def test_get_intent(self):
        """Test retrieving a write intent by ID."""
        content_hash = self._hash_content("Test content")

        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=content_hash
        )

        retrieved = self.detector.get_intent(intent.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, intent.id)
        self.assertEqual(retrieved.memory_id, "mem-123")
        self.assertEqual(retrieved.agent_id, "agent-1")

    def test_get_intent_nonexistent(self):
        """Test retrieving nonexistent intent returns None."""
        retrieved = self.detector.get_intent("nonexistent-id")
        self.assertIsNone(retrieved)

    def test_commit_intent(self):
        """Test committing a write intent."""
        content_hash = self._hash_content("Test content")

        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=content_hash
        )

        # Commit the intent
        result = self.detector.commit_intent(intent.id)
        self.assertTrue(result)

        # Intent should no longer appear in pending
        pending = self.detector.get_pending_intents(memory_id="mem-123")
        self.assertEqual(len(pending), 0)

    def test_commit_intent_nonexistent(self):
        """Test committing nonexistent intent returns False."""
        result = self.detector.commit_intent("nonexistent-id")
        self.assertFalse(result)

    # ==================== Pending Intents ====================

    def test_get_pending_intents_all(self):
        """Test getting all pending intents."""
        # Register multiple intents
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        pending = self.detector.get_pending_intents()
        self.assertEqual(len(pending), 2)

    def test_get_pending_intents_by_memory(self):
        """Test getting pending intents filtered by memory ID."""
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        pending = self.detector.get_pending_intents(memory_id="mem-123")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].memory_id, "mem-123")

    def test_get_pending_intents_by_agent(self):
        """Test getting pending intents filtered by agent ID."""
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 2")
        )
        self.detector.register_intent(
            memory_id="mem-789",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 3")
        )

        pending = self.detector.get_pending_intents(agent_id="agent-1")
        self.assertEqual(len(pending), 2)
        for intent in pending:
            self.assertEqual(intent.agent_id, "agent-1")

    def test_get_pending_intents_by_namespace(self):
        """Test getting pending intents filtered by namespace."""
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            namespace="acme/research",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-2",
            namespace="other/namespace",
            content_hash=self._hash_content("Content 2")
        )

        pending = self.detector.get_pending_intents(namespace="acme/research")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].namespace, "acme/research")

    def test_pending_intents_excludes_committed(self):
        """Test that committed intents are excluded from pending."""
        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content")
        )

        # Should be pending
        pending = self.detector.get_pending_intents(memory_id="mem-123")
        self.assertEqual(len(pending), 1)

        # Commit it
        self.detector.commit_intent(intent.id)

        # Should no longer be pending
        pending = self.detector.get_pending_intents(memory_id="mem-123")
        self.assertEqual(len(pending), 0)

    # ==================== Conflict Detection ====================

    def test_detect_conflict_concurrent_writes(self):
        """Test detecting conflict from concurrent writes by different agents."""
        # Register two intents for same memory
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        # Detect conflict
        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.memory_id, "mem-123")
        self.assertEqual(len(conflict.conflicting_agents), 2)
        self.assertIn("agent-1", conflict.conflicting_agents)
        self.assertIn("agent-2", conflict.conflicting_agents)
        self.assertEqual(len(conflict.conflicting_intents), 2)
        self.assertFalse(conflict.resolved)

    def test_detect_conflict_no_conflict_single_agent(self):
        """Test no conflict when only one agent has pending writes."""
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        self.assertIsNone(conflict)

    def test_detect_conflict_no_conflict_same_agent(self):
        """Test no conflict when multiple writes from same agent."""
        # Multiple writes from same agent
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 2")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        # Same agent writing multiple times is not a conflict
        self.assertIsNone(conflict)

    def test_detect_conflict_time_window(self):
        """Test conflict detection respects time window."""
        # Register first intent
        intent1 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )

        # Wait longer to ensure first intent is outside short window
        time.sleep(1.1)

        # Register second intent from different agent
        intent2 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        # Should detect conflict with default window (5 seconds)
        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )
        self.assertIsNotNone(conflict)

        # Commit the first intent so it's no longer pending
        self.detector.commit_intent(intent1.id)

        # With 1-second window, should only see the second uncommitted intent (single agent = no conflict)
        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-2",
            time_window_seconds=1
        )
        self.assertIsNone(conflict)

    def test_get_conflict(self):
        """Test retrieving a conflict by ID."""
        # Create conflict
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        # Retrieve it
        retrieved = self.detector.get_conflict(conflict.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, conflict.id)
        self.assertEqual(retrieved.memory_id, "mem-123")

    def test_get_conflict_nonexistent(self):
        """Test retrieving nonexistent conflict returns None."""
        retrieved = self.detector.get_conflict("nonexistent-id")
        self.assertIsNone(retrieved)

    # ==================== Conflict Resolution ====================

    def test_resolve_conflict(self):
        """Test resolving a conflict."""
        # Create conflict
        intent1 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        # Resolve it
        resolved = self.detector.resolve_conflict(
            conflict_id=conflict.id,
            strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
            winner_intent_id=intent1.id
        )

        self.assertTrue(resolved.resolved)
        self.assertEqual(resolved.resolution_strategy, ConflictResolutionStrategy.LAST_WRITER_WINS)
        self.assertEqual(resolved.winner_intent_id, intent1.id)

    def test_resolve_conflict_with_metadata(self):
        """Test resolving conflict with metadata."""
        intent1 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        metadata = {"resolved_by": "admin", "reason": "priority"}
        resolved = self.detector.resolve_conflict(
            conflict_id=conflict.id,
            strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
            winner_intent_id=intent1.id,
            metadata=metadata
        )

        self.assertEqual(resolved.metadata, metadata)

    def test_resolve_conflict_nonexistent(self):
        """Test resolving nonexistent conflict raises error."""
        with self.assertRaises(ValueError) as context:
            self.detector.resolve_conflict(
                conflict_id="nonexistent-id",
                strategy=ConflictResolutionStrategy.LAST_WRITER_WINS
            )

        self.assertIn("Conflict not found", str(context.exception))

    def test_resolve_conflict_already_resolved(self):
        """Test resolving already resolved conflict raises error."""
        intent1 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        # Resolve once
        self.detector.resolve_conflict(
            conflict_id=conflict.id,
            strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
            winner_intent_id=intent1.id
        )

        # Try to resolve again
        with self.assertRaises(ValueError) as context:
            self.detector.resolve_conflict(
                conflict_id=conflict.id,
                strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
                winner_intent_id=intent1.id
            )

        self.assertIn("already resolved", str(context.exception))

    # ==================== Conflict Listing ====================

    def test_get_conflicts_all(self):
        """Test getting all conflicts."""
        # Create multiple conflicts
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )
        self.detector.detect_conflict(memory_id="mem-123", agent_id="agent-1")

        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-3",
            content_hash=self._hash_content("Content 3")
        )
        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-4",
            content_hash=self._hash_content("Content 4")
        )
        self.detector.detect_conflict(memory_id="mem-456", agent_id="agent-3")

        conflicts = self.detector.get_conflicts()
        self.assertEqual(len(conflicts), 2)

    def test_get_conflicts_by_memory(self):
        """Test getting conflicts filtered by memory ID."""
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )
        self.detector.detect_conflict(memory_id="mem-123", agent_id="agent-1")

        conflicts = self.detector.get_conflicts(memory_id="mem-123")
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].memory_id, "mem-123")

    def test_get_conflicts_by_resolved_status(self):
        """Test getting conflicts filtered by resolved status."""
        intent1 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )
        conflict1 = self.detector.detect_conflict(memory_id="mem-123", agent_id="agent-1")

        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-3",
            content_hash=self._hash_content("Content 3")
        )
        self.detector.register_intent(
            memory_id="mem-456",
            agent_id="agent-4",
            content_hash=self._hash_content("Content 4")
        )
        conflict2 = self.detector.detect_conflict(memory_id="mem-456", agent_id="agent-3")

        # Resolve one conflict
        self.detector.resolve_conflict(
            conflict_id=conflict1.id,
            strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
            winner_intent_id=intent1.id
        )

        # Get unresolved conflicts
        unresolved = self.detector.get_conflicts(resolved=False)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].id, conflict2.id)

        # Get resolved conflicts
        resolved = self.detector.get_conflicts(resolved=True)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].id, conflict1.id)

    # ==================== Cleanup Operations ====================

    def test_cleanup_old_intents(self):
        """Test cleaning up old committed intents."""
        # Register and commit an intent
        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content")
        )
        self.detector.commit_intent(intent.id)

        # Sleep briefly to ensure intent is old enough
        time.sleep(0.1)

        # Cleanup with very short age (should delete)
        count = self.detector.cleanup_old_intents(age_seconds=0)
        self.assertEqual(count, 1)

    def test_cleanup_preserves_uncommitted(self):
        """Test cleanup preserves uncommitted intents."""
        # Register but don't commit
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content")
        )

        # Cleanup should not delete uncommitted
        count = self.detector.cleanup_old_intents(age_seconds=0)
        self.assertEqual(count, 0)

        # Verify it's still there
        pending = self.detector.get_pending_intents(memory_id="mem-123")
        self.assertEqual(len(pending), 1)

    # ==================== Serialization ====================

    def test_write_intent_to_dict(self):
        """Test WriteIntent serialization to dict."""
        intent = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content"),
            namespace="acme/research",
            base_version=5,
            metadata={"key": "value"}
        )

        data = intent.to_dict()

        self.assertEqual(data["id"], intent.id)
        self.assertEqual(data["memory_id"], "mem-123")
        self.assertEqual(data["agent_id"], "agent-1")
        self.assertEqual(data["namespace"], "acme/research")
        self.assertIsInstance(data["content_hash"], str)
        self.assertEqual(data["base_version"], 5)
        self.assertIsInstance(data["created_at"], str)
        self.assertEqual(data["metadata"], {"key": "value"})

    def test_conflict_to_dict(self):
        """Test Conflict serialization to dict."""
        intent1 = self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-1",
            content_hash=self._hash_content("Content 1")
        )
        self.detector.register_intent(
            memory_id="mem-123",
            agent_id="agent-2",
            content_hash=self._hash_content("Content 2")
        )

        conflict = self.detector.detect_conflict(
            memory_id="mem-123",
            agent_id="agent-1"
        )

        data = conflict.to_dict()

        self.assertEqual(data["id"], conflict.id)
        self.assertEqual(data["memory_id"], "mem-123")
        self.assertIsInstance(data["conflicting_agents"], list)
        self.assertIsInstance(data["conflicting_intents"], list)
        self.assertIsInstance(data["detected_at"], str)
        self.assertFalse(data["resolved"])


class TestLastWriterWinsResolver(unittest.TestCase):
    """Test suite for LastWriterWinsResolver."""

    def _create_intent(self, agent_id: str, created_at: datetime) -> WriteIntent:
        """Helper to create a WriteIntent for testing."""
        return WriteIntent(
            id=f"intent-{agent_id}",
            memory_id="mem-123",
            agent_id=agent_id,
            namespace=None,
            content_hash="abc123",
            base_version=None,
            created_at=created_at
        )

    def test_resolve_most_recent_wins(self):
        """Test that most recent write wins."""
        resolver = LastWriterWinsResolver()

        now = datetime.now()
        intents = [
            self._create_intent("agent-1", now - timedelta(seconds=10)),
            self._create_intent("agent-2", now - timedelta(seconds=5)),
            self._create_intent("agent-3", now),  # Most recent
        ]

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=["agent-1", "agent-2", "agent-3"],
            conflicting_intents=["intent-agent-1", "intent-agent-2", "intent-agent-3"],
            detected_at=now
        )

        result = resolver.resolve(conflict, intents)

        self.assertEqual(result.winner_intent_id, "intent-agent-3")
        self.assertEqual(result.strategy, ConflictResolutionStrategy.LAST_WRITER_WINS)
        self.assertIn("Most recent", result.reason)
        self.assertEqual(result.metadata["winner_agent"], "agent-3")

    def test_resolve_no_intents(self):
        """Test resolving with no intents raises error."""
        resolver = LastWriterWinsResolver()

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=[],
            conflicting_intents=[],
            detected_at=datetime.now()
        )

        with self.assertRaises(ValueError) as context:
            resolver.resolve(conflict, [])

        self.assertIn("No write intents", str(context.exception))


class TestMergeResolver(unittest.TestCase):
    """Test suite for MergeResolver."""

    def _create_intent(
        self,
        agent_id: str,
        content_hash: str,
        base_version: int = None,
        created_at: datetime = None
    ) -> WriteIntent:
        """Helper to create a WriteIntent for testing."""
        return WriteIntent(
            id=f"intent-{agent_id}",
            memory_id="mem-123",
            agent_id=agent_id,
            namespace=None,
            content_hash=content_hash,
            base_version=base_version,
            created_at=created_at or datetime.now()
        )

    def test_resolve_identical_content(self):
        """Test resolving when all writes have identical content."""
        resolver = MergeResolver()

        # Same content hash for all intents
        intents = [
            self._create_intent("agent-1", "abc123"),
            self._create_intent("agent-2", "abc123"),
            self._create_intent("agent-3", "abc123"),
        ]

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=["agent-1", "agent-2", "agent-3"],
            conflicting_intents=["intent-agent-1", "intent-agent-2", "intent-agent-3"],
            detected_at=datetime.now()
        )

        result = resolver.resolve(conflict, intents)

        self.assertIsNotNone(result.winner_intent_id)
        self.assertEqual(result.strategy, ConflictResolutionStrategy.MERGE)
        self.assertIn("identical", result.reason)
        self.assertEqual(result.metadata["merge_type"], "identical")

    def test_resolve_different_content_fallback(self):
        """Test fallback to last-writer-wins for different content."""
        resolver = MergeResolver()

        now = datetime.now()
        intents = [
            self._create_intent("agent-1", "abc123", created_at=now - timedelta(seconds=10)),
            self._create_intent("agent-2", "def456", created_at=now),
        ]

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=["agent-1", "agent-2"],
            conflicting_intents=["intent-agent-1", "intent-agent-2"],
            detected_at=now
        )

        result = resolver.resolve(conflict, intents)

        # Should fall back to last-writer-wins
        self.assertEqual(result.winner_intent_id, "intent-agent-2")
        self.assertEqual(result.strategy, ConflictResolutionStrategy.MERGE)
        self.assertIn("fallback", result.metadata["merge_type"])

    def test_resolve_same_base_version(self):
        """Test resolving writes with same base version."""
        resolver = MergeResolver()

        now = datetime.now()
        intents = [
            self._create_intent("agent-1", "abc123", base_version=5, created_at=now - timedelta(seconds=5)),
            self._create_intent("agent-2", "def456", base_version=5, created_at=now),
        ]

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=["agent-1", "agent-2"],
            conflicting_intents=["intent-agent-1", "intent-agent-2"],
            detected_at=now
        )

        result = resolver.resolve(conflict, intents)

        # Should attempt merge but fall back to last-writer-wins
        self.assertIsNotNone(result.winner_intent_id)
        self.assertEqual(result.strategy, ConflictResolutionStrategy.MERGE)

    def test_resolve_no_intents(self):
        """Test resolving with no intents raises error."""
        resolver = MergeResolver()

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=[],
            conflicting_intents=[],
            detected_at=datetime.now()
        )

        with self.assertRaises(ValueError) as context:
            resolver.resolve(conflict, [])

        self.assertIn("No write intents", str(context.exception))


class TestRejectResolver(unittest.TestCase):
    """Test suite for RejectResolver."""

    def _create_intent(self, agent_id: str) -> WriteIntent:
        """Helper to create a WriteIntent for testing."""
        return WriteIntent(
            id=f"intent-{agent_id}",
            memory_id="mem-123",
            agent_id=agent_id,
            namespace=None,
            content_hash="abc123",
            base_version=None,
            created_at=datetime.now()
        )

    def test_resolve_rejects_all(self):
        """Test that resolver rejects all writes."""
        resolver = RejectResolver()

        intents = [
            self._create_intent("agent-1"),
            self._create_intent("agent-2"),
        ]

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=["agent-1", "agent-2"],
            conflicting_intents=["intent-agent-1", "intent-agent-2"],
            detected_at=datetime.now()
        )

        result = resolver.resolve(conflict, intents)

        self.assertIsNone(result.winner_intent_id)
        self.assertEqual(result.strategy, ConflictResolutionStrategy.REJECT)
        self.assertIn("rejected", result.reason)
        self.assertIn("manual resolution", result.reason)
        self.assertEqual(result.metadata["conflict_id"], "conflict-1")

    def test_resolve_no_intents(self):
        """Test resolving with no intents raises error."""
        resolver = RejectResolver()

        conflict = Conflict(
            id="conflict-1",
            memory_id="mem-123",
            namespace=None,
            conflicting_agents=[],
            conflicting_intents=[],
            detected_at=datetime.now()
        )

        with self.assertRaises(ValueError) as context:
            resolver.resolve(conflict, [])

        self.assertIn("No write intents", str(context.exception))


class TestConflictDetectorThreadSafety(unittest.TestCase):
    """Test suite for ConflictDetector thread safety."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_conflicts.sqlite"
        self.detector = ConflictDetector(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        self.detector.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_intent_registration(self):
        """Test thread-safe concurrent intent registration."""
        num_threads = 10
        intents_per_thread = 5
        results = [[] for _ in range(num_threads)]

        def register_intents(thread_id):
            for i in range(intents_per_thread):
                intent = self.detector.register_intent(
                    memory_id=f"mem-{thread_id}",
                    agent_id=f"agent-{thread_id}",
                    content_hash=hashlib.sha256(f"Content-{i}".encode()).hexdigest()
                )
                results[thread_id].append(intent)

        # Register intents concurrently
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=register_intents, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all intents were created
        for thread_results in results:
            self.assertEqual(len(thread_results), intents_per_thread)

        # Verify total count
        all_pending = self.detector.get_pending_intents()
        self.assertEqual(len(all_pending), num_threads * intents_per_thread)


if __name__ == "__main__":
    unittest.main()
