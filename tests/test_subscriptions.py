"""
Unit tests for SubscriptionManager - Topic-based memory notifications

Tests cover:
- Subscribe/unsubscribe operations
- Subscription filtering (agent, namespace, memory)
- Event matching logic (global, namespace, memory-specific)
- Wildcard event types
- Thread-safe database operations
- Edge cases and error handling
"""

import unittest
import tempfile
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.subscriptions import SubscriptionManager, SubscriptionInfo


class TestSubscriptionManager(unittest.TestCase):
    """Test suite for SubscriptionManager."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_subscriptions.sqlite"

        # Create the subscriptions table that SubscriptionManager expects
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                namespace TEXT,
                memory_id TEXT,
                event_types TEXT NOT NULL,  -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.close()

        self.manager = SubscriptionManager(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        self.manager.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ==================== Subscribe/Unsubscribe Operations ====================

    def test_subscribe_global(self):
        """Test subscribing to all events (global subscription)."""
        sub = self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored", "belief.updated"]
        )

        self.assertIsNotNone(sub.id)
        self.assertEqual(sub.agent_id, "agent-1")
        self.assertIsNone(sub.namespace)
        self.assertIsNone(sub.memory_id)
        self.assertEqual(sub.event_types, ["memory.stored", "belief.updated"])
        self.assertIsInstance(sub.created_at, datetime)

    def test_subscribe_namespace(self):
        """Test subscribing to namespace events."""
        sub = self.manager.subscribe(
            agent_id="agent-2",
            namespace="acme/research",
            event_types=["*"]
        )

        self.assertEqual(sub.agent_id, "agent-2")
        self.assertEqual(sub.namespace, "acme/research")
        self.assertIsNone(sub.memory_id)
        self.assertEqual(sub.event_types, ["*"])

    def test_subscribe_memory_specific(self):
        """Test subscribing to specific memory events."""
        sub = self.manager.subscribe(
            agent_id="agent-3",
            memory_id="mem-123",
            event_types=["belief.updated"]
        )

        self.assertEqual(sub.agent_id, "agent-3")
        self.assertIsNone(sub.namespace)
        self.assertEqual(sub.memory_id, "mem-123")
        self.assertEqual(sub.event_types, ["belief.updated"])

    def test_subscribe_empty_event_types(self):
        """Test subscribing with empty event_types raises error."""
        with self.assertRaises(ValueError) as context:
            self.manager.subscribe(
                agent_id="agent-1",
                event_types=[]
            )

        self.assertIn("event_types cannot be empty", str(context.exception))

    def test_unsubscribe(self):
        """Test unsubscribing by subscription ID."""
        sub = self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored"]
        )

        # Unsubscribe
        result = self.manager.unsubscribe(sub.id)
        self.assertTrue(result)

        # Verify subscription is gone
        retrieved = self.manager.get(sub.id)
        self.assertIsNone(retrieved)

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing nonexistent subscription returns False."""
        result = self.manager.unsubscribe("nonexistent-id")
        self.assertFalse(result)

    def test_unsubscribe_agent_from_namespace(self):
        """Test unsubscribing agent from all subscriptions in namespace."""
        # Create multiple subscriptions
        self.manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["memory.stored"]
        )
        self.manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["belief.updated"]
        )
        self.manager.subscribe(
            agent_id="agent-1",
            namespace="other/namespace",
            event_types=["*"]
        )

        # Unsubscribe from acme/research
        count = self.manager.unsubscribe_agent_from_namespace("agent-1", "acme/research")
        self.assertEqual(count, 2)

        # Verify remaining subscriptions
        subs = self.manager.list_for_agent("agent-1")
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].namespace, "other/namespace")

    # ==================== Subscription Retrieval ====================

    def test_get_subscription(self):
        """Test retrieving a subscription by ID."""
        sub = self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored"]
        )

        retrieved = self.manager.get(sub.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, sub.id)
        self.assertEqual(retrieved.agent_id, "agent-1")
        self.assertEqual(retrieved.event_types, ["memory.stored"])

    def test_get_nonexistent_subscription(self):
        """Test retrieving nonexistent subscription returns None."""
        retrieved = self.manager.get("nonexistent-id")
        self.assertIsNone(retrieved)

    def test_list_for_agent(self):
        """Test listing all subscriptions for an agent."""
        # Create subscriptions for different agents
        self.manager.subscribe(agent_id="agent-1", event_types=["memory.stored"])
        self.manager.subscribe(agent_id="agent-1", event_types=["belief.updated"])
        self.manager.subscribe(agent_id="agent-2", event_types=["*"])

        # List for agent-1
        subs = self.manager.list_for_agent("agent-1")
        self.assertEqual(len(subs), 2)

        # All should belong to agent-1
        for sub in subs:
            self.assertEqual(sub.agent_id, "agent-1")

        # List for agent-2
        subs = self.manager.list_for_agent("agent-2")
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0].agent_id, "agent-2")

    def test_list_for_namespace(self):
        """Test listing all subscriptions for a namespace."""
        # Create subscriptions in different namespaces
        self.manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["memory.stored"]
        )
        self.manager.subscribe(
            agent_id="agent-2",
            namespace="acme/research",
            event_types=["belief.updated"]
        )
        self.manager.subscribe(
            agent_id="agent-3",
            namespace="other/namespace",
            event_types=["*"]
        )

        # List for acme/research
        subs = self.manager.list_for_namespace("acme/research")
        self.assertEqual(len(subs), 2)

        # All should belong to acme/research
        for sub in subs:
            self.assertEqual(sub.namespace, "acme/research")

    def test_list_for_memory(self):
        """Test listing all subscriptions for a memory."""
        # Create subscriptions for different memories
        self.manager.subscribe(
            agent_id="agent-1",
            memory_id="mem-123",
            event_types=["belief.updated"]
        )
        self.manager.subscribe(
            agent_id="agent-2",
            memory_id="mem-123",
            event_types=["*"]
        )
        self.manager.subscribe(
            agent_id="agent-3",
            memory_id="mem-456",
            event_types=["memory.stored"]
        )

        # List for mem-123
        subs = self.manager.list_for_memory("mem-123")
        self.assertEqual(len(subs), 2)

        # All should belong to mem-123
        for sub in subs:
            self.assertEqual(sub.memory_id, "mem-123")

    def test_count_for_agent(self):
        """Test counting subscriptions for an agent."""
        # Initially zero
        count = self.manager.count_for_agent("agent-1")
        self.assertEqual(count, 0)

        # Add subscriptions
        self.manager.subscribe(agent_id="agent-1", event_types=["memory.stored"])
        self.manager.subscribe(agent_id="agent-1", event_types=["belief.updated"])
        self.manager.subscribe(agent_id="agent-2", event_types=["*"])

        # Count should be 2 for agent-1
        count = self.manager.count_for_agent("agent-1")
        self.assertEqual(count, 2)

        # Count should be 1 for agent-2
        count = self.manager.count_for_agent("agent-2")
        self.assertEqual(count, 1)

    # ==================== Event Matching ====================

    def test_match_subscriptions_global(self):
        """Test matching global subscriptions."""
        # Create global subscription
        sub = self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored"]
        )

        # Should match any event of type memory.stored
        matches = self.manager.match_subscriptions(
            event_type="memory.stored",
            namespace="acme/research",
            memory_id="mem-123"
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].id, sub.id)

    def test_match_subscriptions_namespace(self):
        """Test matching namespace-specific subscriptions."""
        # Create namespace subscription
        sub = self.manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["memory.stored"]
        )

        # Should match events in acme/research
        matches = self.manager.match_subscriptions(
            event_type="memory.stored",
            namespace="acme/research",
            memory_id="mem-123"
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].id, sub.id)

        # Should NOT match events in other namespaces
        matches = self.manager.match_subscriptions(
            event_type="memory.stored",
            namespace="other/namespace",
            memory_id="mem-123"
        )
        self.assertEqual(len(matches), 0)

    def test_match_subscriptions_memory_specific(self):
        """Test matching memory-specific subscriptions."""
        # Create memory-specific subscription
        sub = self.manager.subscribe(
            agent_id="agent-1",
            memory_id="mem-123",
            event_types=["belief.updated"]
        )

        # Should match events for mem-123
        matches = self.manager.match_subscriptions(
            event_type="belief.updated",
            namespace="acme/research",
            memory_id="mem-123"
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].id, sub.id)

        # Should NOT match events for other memories
        matches = self.manager.match_subscriptions(
            event_type="belief.updated",
            namespace="acme/research",
            memory_id="mem-456"
        )
        self.assertEqual(len(matches), 0)

    def test_match_subscriptions_wildcard(self):
        """Test matching wildcard event types."""
        # Create wildcard subscription
        sub = self.manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["*"]
        )

        # Should match any event type
        matches = self.manager.match_subscriptions(
            event_type="memory.stored",
            namespace="acme/research"
        )
        self.assertEqual(len(matches), 1)

        matches = self.manager.match_subscriptions(
            event_type="belief.updated",
            namespace="acme/research"
        )
        self.assertEqual(len(matches), 1)

        matches = self.manager.match_subscriptions(
            event_type="custom.event",
            namespace="acme/research"
        )
        self.assertEqual(len(matches), 1)

    def test_match_subscriptions_event_type_filter(self):
        """Test event type filtering in matching."""
        # Create subscription for specific event types
        self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored", "belief.updated"]
        )

        # Should match memory.stored
        matches = self.manager.match_subscriptions(event_type="memory.stored")
        self.assertEqual(len(matches), 1)

        # Should match belief.updated
        matches = self.manager.match_subscriptions(event_type="belief.updated")
        self.assertEqual(len(matches), 1)

        # Should NOT match other event types
        matches = self.manager.match_subscriptions(event_type="memory.deleted")
        self.assertEqual(len(matches), 0)

    def test_match_subscriptions_multiple_matches(self):
        """Test matching multiple subscriptions for same event."""
        # Create overlapping subscriptions
        sub1 = self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored"]  # Global
        )
        sub2 = self.manager.subscribe(
            agent_id="agent-2",
            namespace="acme/research",
            event_types=["memory.stored"]  # Namespace
        )
        sub3 = self.manager.subscribe(
            agent_id="agent-3",
            memory_id="mem-123",
            event_types=["*"]  # Memory-specific with wildcard
        )

        # Should match all three
        matches = self.manager.match_subscriptions(
            event_type="memory.stored",
            namespace="acme/research",
            memory_id="mem-123"
        )

        self.assertEqual(len(matches), 3)
        match_ids = {m.id for m in matches}
        self.assertEqual(match_ids, {sub1.id, sub2.id, sub3.id})

    def test_get_subscribed_agents(self):
        """Test getting agent IDs subscribed to an event."""
        # Create subscriptions
        self.manager.subscribe(agent_id="agent-1", event_types=["memory.stored"])
        self.manager.subscribe(agent_id="agent-2", event_types=["memory.stored"])
        self.manager.subscribe(agent_id="agent-3", event_types=["belief.updated"])

        # Get agents subscribed to memory.stored
        agents = self.manager.get_subscribed_agents(event_type="memory.stored")
        self.assertEqual(len(agents), 2)
        self.assertEqual(agents, {"agent-1", "agent-2"})

        # Get agents subscribed to belief.updated
        agents = self.manager.get_subscribed_agents(event_type="belief.updated")
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents, {"agent-3"})

    # ==================== Serialization ====================

    def test_subscription_info_to_dict(self):
        """Test SubscriptionInfo serialization to dict."""
        sub = self.manager.subscribe(
            agent_id="agent-1",
            namespace="acme/research",
            event_types=["memory.stored"]
        )

        data = sub.to_dict()

        self.assertEqual(data["id"], sub.id)
        self.assertEqual(data["agent_id"], "agent-1")
        self.assertEqual(data["namespace"], "acme/research")
        self.assertIsNone(data["memory_id"])
        self.assertEqual(data["event_types"], ["memory.stored"])
        self.assertIsInstance(data["created_at"], str)

    # ==================== Thread Safety ====================

    def test_concurrent_subscriptions(self):
        """Test thread-safe concurrent subscription creation."""
        num_threads = 10
        subscriptions_per_thread = 5
        results = [[] for _ in range(num_threads)]

        def create_subscriptions(thread_id):
            for i in range(subscriptions_per_thread):
                sub = self.manager.subscribe(
                    agent_id=f"agent-{thread_id}",
                    event_types=["memory.stored"]
                )
                results[thread_id].append(sub)

        # Create subscriptions concurrently
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=create_subscriptions, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Verify all subscriptions were created
        for thread_results in results:
            self.assertEqual(len(thread_results), subscriptions_per_thread)

        # Verify total count
        total_count = sum(
            self.manager.count_for_agent(f"agent-{i}")
            for i in range(num_threads)
        )
        self.assertEqual(total_count, num_threads * subscriptions_per_thread)

    # ==================== Edge Cases ====================

    def test_list_for_agent_empty(self):
        """Test listing subscriptions for agent with no subscriptions."""
        subs = self.manager.list_for_agent("nonexistent-agent")
        self.assertEqual(len(subs), 0)

    def test_list_for_namespace_empty(self):
        """Test listing subscriptions for namespace with no subscriptions."""
        subs = self.manager.list_for_namespace("nonexistent/namespace")
        self.assertEqual(len(subs), 0)

    def test_list_for_memory_empty(self):
        """Test listing subscriptions for memory with no subscriptions."""
        subs = self.manager.list_for_memory("nonexistent-memory")
        self.assertEqual(len(subs), 0)

    def test_match_subscriptions_no_matches(self):
        """Test event matching with no matching subscriptions."""
        self.manager.subscribe(
            agent_id="agent-1",
            event_types=["memory.stored"]
        )

        # Different event type
        matches = self.manager.match_subscriptions(event_type="belief.updated")
        self.assertEqual(len(matches), 0)

    def test_context_manager(self):
        """Test using SubscriptionManager as context manager."""
        db_path = Path(self.temp_dir) / "context_manager.sqlite"

        # Create the subscriptions table first
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                namespace TEXT,
                memory_id TEXT,
                event_types TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.close()

        with SubscriptionManager(db_path) as manager:
            sub = manager.subscribe(
                agent_id="agent-1",
                event_types=["memory.stored"]
            )
            self.assertIsNotNone(sub.id)


if __name__ == "__main__":
    unittest.main()
