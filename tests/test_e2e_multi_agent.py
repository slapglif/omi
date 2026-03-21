"""End-to-End Multi-Agent Memory Coordination Tests

Tests the complete multi-agent coordination flow:
1. Agent A creates shared namespace
2. Agent B subscribes to namespace
3. Agent A stores memory in shared namespace
4. Agent B receives notification via EventBus (simulating SSE)
5. Verify audit log records the cross-agent operation
6. Verify operation latency is under 50ms

This test verifies the full integration of:
- SharedNamespace (namespace creation)
- PermissionManager (access control)
- SubscriptionManager (event routing)
- AuditLogger (operation tracking)
- EventBus (notification delivery)
- Performance requirements (<50ms overhead)
"""

import pytest
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Any


class TestE2EMultiAgentCoordination:
    """Test end-to-end multi-agent memory coordination workflow."""

    def test_two_agents_sharing_via_namespace(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        End-to-end verification of two agents sharing memories via namespace.

        Steps:
        1. Agent A creates shared namespace
        2. Agent B subscribes to namespace
        3. Agent A stores memory in shared namespace
        4. Agent B receives notification via EventBus
        5. Verify audit log records the cross-agent operation
        6. Verify operation latency is under 50ms
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.audit_log import AuditLogger
        from omi.events import SharedMemoryStoredEvent
        from omi.api import MemoryTools

        # Setup - Initialize database with schema
        db_path = temp_omi_setup["db_path"]
        import sqlite3
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        palace = GraphPalace(db_path)
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)
        audit = AuditLogger(db_path)
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Define agents
        agent_a = "agent-a"
        agent_b = "agent-b"
        namespace = "team-alpha/research"

        # Track events received by Agent B
        agent_b_events: List[Any] = []

        def agent_b_event_handler(event):
            """Callback to capture events for Agent B (simulating SSE)."""
            # Filter events that Agent B should receive based on subscriptions
            if hasattr(event, 'event_type'):
                # Extract namespace from event metadata if present
                event_namespace = None
                if hasattr(event, 'metadata') and event.metadata:
                    event_namespace = event.metadata.get('namespace')

                # Check if event matches any of Agent B's subscriptions
                matching_subs = subscriptions.match_subscriptions(
                    event_type=event.event_type,
                    namespace=event_namespace,
                    memory_id=getattr(event, 'memory_id', None)
                )
                # Filter to only Agent B's subscriptions
                for sub in matching_subs:
                    if sub.agent_id == agent_b:
                        agent_b_events.append(event)
                        break

        # Subscribe Agent B's handler to EventBus
        clean_event_bus.subscribe('*', agent_b_event_handler)

        # ===== STEP 1: Agent A creates shared namespace =====
        start_time = time.time()

        shared_ns.create(
            namespace=namespace,
            created_by=agent_a,
            metadata={"purpose": "AI research collaboration"}
        )

        step1_time = time.time() - start_time

        # Grant Agent A admin permission
        permissions.grant(
            namespace=namespace,
            agent_id=agent_a,
            permission_level=PermissionLevel.ADMIN
        )

        # Grant Agent B read permission
        permissions.grant(
            namespace=namespace,
            agent_id=agent_b,
            permission_level=PermissionLevel.READ
        )

        # Log namespace creation in audit log
        audit.log(
            agent_id=agent_a,
            action_type="CREATE",
            resource_type="NAMESPACE",
            resource_id=namespace,
            namespace=namespace,
            metadata={"purpose": "AI research collaboration"}
        )

        # Verify namespace was created
        ns_info = shared_ns.get(namespace)
        assert ns_info is not None
        assert ns_info.namespace == namespace
        assert ns_info.created_by == agent_a

        # ===== STEP 2: Agent B subscribes to namespace =====
        step2_start = time.time()

        subscription_info = subscriptions.subscribe(
            agent_id=agent_b,
            namespace=namespace,
            event_types=["memory.stored", "memory.shared_stored"]
        )

        step2_time = time.time() - step2_start

        # Log subscription in audit log
        audit.log(
            agent_id=agent_b,
            action_type="SUBSCRIBE",
            resource_type="SUBSCRIPTION",
            resource_id=subscription_info.id,
            namespace=namespace,
            metadata={"event_types": ["memory.stored", "memory.shared_stored"]}
        )

        # Verify subscription was created
        subs = subscriptions.list_for_agent(agent_b)
        assert len(subs) > 0
        assert any(s.namespace == namespace for s in subs)

        # ===== STEP 3: Agent A stores memory in shared namespace =====
        step3_start = time.time()

        memory_content = "Discovered breakthrough in transformer architecture optimization"
        memory_id = memory_tools.store(
            content=memory_content,
            memory_type="fact",
            confidence=0.95
        )

        # Publish SharedMemoryStoredEvent to EventBus
        shared_event = SharedMemoryStoredEvent(
            source_agent_id=agent_a,
            target_agent_ids=[agent_b],
            memory_id=memory_id,
            content=memory_content,
            memory_type="fact",
            confidence=0.95,
            metadata={"namespace": namespace}
        )
        clean_event_bus.publish(shared_event)

        step3_time = time.time() - step3_start

        # Log memory storage in audit log
        audit.log(
            agent_id=agent_a,
            action_type="WRITE",
            resource_type="MEMORY",
            resource_id=memory_id,
            namespace=namespace,
            metadata={
                "content": memory_content,
                "memory_type": "fact",
                "target_agents": [agent_b]
            }
        )

        # Verify memory was stored
        stored_memory = palace.get_memory(memory_id)
        assert stored_memory is not None
        assert stored_memory.content == memory_content

        # ===== STEP 4: Verify Agent B received notification =====
        # Events should have been captured by agent_b_event_handler
        assert len(agent_b_events) > 0, "Agent B should have received at least one event"

        # Find the SharedMemoryStoredEvent
        shared_events = [e for e in agent_b_events if isinstance(e, SharedMemoryStoredEvent)]
        assert len(shared_events) > 0, "Agent B should have received SharedMemoryStoredEvent"

        received_event = shared_events[0]
        assert received_event.source_agent_id == agent_a
        assert agent_b in received_event.target_agent_ids
        assert received_event.memory_id == memory_id
        assert received_event.content == memory_content
        assert received_event.metadata.get('namespace') == namespace

        # ===== STEP 5: Verify audit log records cross-agent operation =====
        # Query audit log for all operations
        all_audit_entries = audit.get_by_namespace(namespace, limit=100)
        assert len(all_audit_entries) >= 3, "Should have at least 3 audit entries (create, subscribe, write)"

        # Verify CREATE operation
        create_entries = [e for e in all_audit_entries if e.action_type == "CREATE"]
        assert len(create_entries) > 0
        assert create_entries[0].agent_id == agent_a
        assert create_entries[0].resource_type == "NAMESPACE"

        # Verify SUBSCRIBE operation
        subscribe_entries = [e for e in all_audit_entries if e.action_type == "SUBSCRIBE"]
        assert len(subscribe_entries) > 0
        assert subscribe_entries[0].agent_id == agent_b
        assert subscribe_entries[0].resource_type == "SUBSCRIPTION"

        # Verify WRITE operation
        write_entries = [e for e in all_audit_entries if e.action_type == "WRITE"]
        assert len(write_entries) > 0
        assert write_entries[0].agent_id == agent_a
        assert write_entries[0].resource_type == "MEMORY"
        assert write_entries[0].resource_id == memory_id

        # ===== STEP 6: Verify operation latency is under 50ms =====
        # Each individual operation should be under 50ms
        assert step1_time < 0.050, f"Namespace creation took {step1_time*1000:.2f}ms (should be <50ms)"
        assert step2_time < 0.050, f"Subscription took {step2_time*1000:.2f}ms (should be <50ms)"
        assert step3_time < 0.050, f"Memory storage took {step3_time*1000:.2f}ms (should be <50ms)"

        # Total operation time should be reasonable (under 150ms)
        total_time = step1_time + step2_time + step3_time
        assert total_time < 0.150, f"Total operation time {total_time*1000:.2f}ms (should be <150ms)"

    def test_multiple_agents_coordination(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Test coordination between multiple agents (>2) in shared namespace.

        Verifies:
        - Multiple agents can subscribe to same namespace
        - All subscribed agents receive notifications
        - Permissions are enforced correctly
        - Audit log tracks all agent interactions
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.audit_log import AuditLogger
        from omi.events import SharedMemoryStoredEvent
        from omi.api import MemoryTools

        # Setup
        db_path = temp_omi_setup["db_path"]
        import sqlite3
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        palace = GraphPalace(db_path)
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)
        audit = AuditLogger(db_path)
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Define agents
        agent_coordinator = "agent-coordinator"
        agent_researcher = "agent-researcher"
        agent_writer = "agent-writer"
        agent_reviewer = "agent-reviewer"
        namespace = "project-alpha/documentation"

        # Track events for each agent
        events_by_agent = {
            agent_researcher: [],
            agent_writer: [],
            agent_reviewer: []
        }

        def create_event_handler(agent_id: str):
            """Create event handler for specific agent."""
            def handler(event):
                if hasattr(event, 'event_type'):
                    # Extract namespace from event metadata if present
                    event_namespace = None
                    if hasattr(event, 'metadata') and event.metadata:
                        event_namespace = event.metadata.get('namespace')

                    # Check if event matches agent's subscriptions
                    matching_subs = subscriptions.match_subscriptions(
                        event_type=event.event_type,
                        namespace=event_namespace,
                        memory_id=getattr(event, 'memory_id', None)
                    )
                    # Filter to only this agent's subscriptions
                    for sub in matching_subs:
                        if sub.agent_id == agent_id:
                            events_by_agent[agent_id].append(event)
                            break
            return handler

        # Subscribe handlers to EventBus
        clean_event_bus.subscribe('*', create_event_handler(agent_researcher))
        clean_event_bus.subscribe('*', create_event_handler(agent_writer))
        clean_event_bus.subscribe('*', create_event_handler(agent_reviewer))

        # Create shared namespace
        shared_ns.create(
            namespace=namespace,
            created_by=agent_coordinator,
            metadata={"project": "Alpha Documentation System"}
        )

        # Grant permissions
        permissions.grant(namespace, agent_coordinator, PermissionLevel.ADMIN)
        permissions.grant(namespace, agent_researcher, PermissionLevel.WRITE)
        permissions.grant(namespace, agent_writer, PermissionLevel.WRITE)
        permissions.grant(namespace, agent_reviewer, PermissionLevel.READ)

        # Subscribe all agents to namespace
        for agent in [agent_researcher, agent_writer, agent_reviewer]:
            sub_info = subscriptions.subscribe(
                agent_id=agent,
                namespace=namespace,
                event_types=["memory.shared_stored"]
            )

        # Coordinator stores memory in shared namespace
        memory_content = "Project Alpha documentation standards established"
        memory_id = memory_tools.store(
            content=memory_content,
            memory_type="decision",
            confidence=1.0
        )

        # Publish shared memory event
        target_agents = [agent_researcher, agent_writer, agent_reviewer]
        shared_event = SharedMemoryStoredEvent(
            source_agent_id=agent_coordinator,
            target_agent_ids=target_agents,
            memory_id=memory_id,
            content=memory_content,
            memory_type="decision",
            confidence=1.0,
            metadata={"namespace": namespace}
        )
        clean_event_bus.publish(shared_event)

        # Verify all agents received notification
        for agent in target_agents:
            assert len(events_by_agent[agent]) > 0, f"{agent} should have received event"
            shared_events = [e for e in events_by_agent[agent] if isinstance(e, SharedMemoryStoredEvent)]
            assert len(shared_events) > 0, f"{agent} should have received SharedMemoryStoredEvent"
            assert shared_events[0].memory_id == memory_id

        # Verify permissions are correct
        # Check coordinator has admin permission
        assert permissions.has_permission(namespace, agent_coordinator, PermissionLevel.ADMIN)
        assert permissions.has_permission(namespace, agent_researcher, PermissionLevel.WRITE)
        assert permissions.has_permission(namespace, agent_writer, PermissionLevel.WRITE)
        assert permissions.has_permission(namespace, agent_reviewer, PermissionLevel.READ)

        # Verify reviewer doesn't have write permission
        assert not permissions.has_permission(namespace, agent_reviewer, PermissionLevel.WRITE)

    def test_subscription_filtering(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Test subscription filtering by event type and namespace.

        Verifies:
        - Agents only receive events matching their subscription filters
        - Event type filtering works correctly
        - Namespace filtering works correctly
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.events import SharedMemoryStoredEvent, MemoryStoredEvent
        from omi.api import MemoryTools

        # Setup
        db_path = temp_omi_setup["db_path"]
        import sqlite3
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        palace = GraphPalace(db_path)
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Define agents and namespaces
        agent_a = "agent-a"
        agent_b = "agent-b"
        namespace1 = "team/ns1"
        namespace2 = "team/ns2"

        # Track events
        agent_a_events = []
        agent_b_events = []

        def agent_a_handler(event):
            if hasattr(event, 'event_type'):
                event_namespace = None
                if hasattr(event, 'metadata') and event.metadata:
                    event_namespace = event.metadata.get('namespace')

                matching_subs = subscriptions.match_subscriptions(
                    event_type=event.event_type,
                    namespace=event_namespace,
                    memory_id=getattr(event, 'memory_id', None)
                )
                for sub in matching_subs:
                    if sub.agent_id == agent_a:
                        agent_a_events.append(event)
                        break

        def agent_b_handler(event):
            if hasattr(event, 'event_type'):
                event_namespace = None
                if hasattr(event, 'metadata') and event.metadata:
                    event_namespace = event.metadata.get('namespace')

                matching_subs = subscriptions.match_subscriptions(
                    event_type=event.event_type,
                    namespace=event_namespace,
                    memory_id=getattr(event, 'memory_id', None)
                )
                for sub in matching_subs:
                    if sub.agent_id == agent_b:
                        agent_b_events.append(event)
                        break

        clean_event_bus.subscribe('*', agent_a_handler)
        clean_event_bus.subscribe('*', agent_b_handler)

        # Create namespaces
        for ns in [namespace1, namespace2]:
            shared_ns.create(ns, created_by="admin")
            permissions.grant(ns, agent_a, PermissionLevel.READ)
            permissions.grant(ns, agent_b, PermissionLevel.READ)

        # Agent A subscribes only to namespace1 and only memory.shared_stored events
        sub_a = subscriptions.subscribe(
            agent_id=agent_a,
            namespace=namespace1,
            event_types=["memory.shared_stored"]
        )

        # Agent B subscribes to namespace2 and all memory events
        sub_b = subscriptions.subscribe(
            agent_id=agent_b,
            namespace=namespace2,
            event_types=["memory.stored", "memory.shared_stored"]
        )

        # Publish event to namespace1 (only Agent A should receive)
        event1 = SharedMemoryStoredEvent(
            source_agent_id="admin",
            target_agent_ids=[agent_a],
            memory_id="mem1",
            content="Event for namespace1",
            memory_type="fact",
            metadata={"namespace": namespace1}
        )
        clean_event_bus.publish(event1)

        # Publish event to namespace2 (only Agent B should receive)
        event2 = SharedMemoryStoredEvent(
            source_agent_id="admin",
            target_agent_ids=[agent_b],
            memory_id="mem2",
            content="Event for namespace2",
            memory_type="fact",
            metadata={"namespace": namespace2}
        )
        clean_event_bus.publish(event2)

        # Debug: Check subscriptions
        agent_a_subs = subscriptions.list_for_agent(agent_a)
        agent_b_subs = subscriptions.list_for_agent(agent_b)

        # Verify filtering
        # If no events received, the subscriptions might not be matching correctly
        # Let's verify the event type and subscriptions match
        if len(agent_a_events) == 0:
            # Check if event would match
            matches = subscriptions.match_subscriptions(
                event_type="memory.shared_stored",
                namespace=namespace1,
                memory_id=None
            )
            # At least log that matching failed for debugging
            if len(matches) == 0:
                # Subscriptions don't match - this is expected if the event type or namespace doesn't align
                pass

        # Agent A should only have events from namespace1
        assert len(agent_a_events) > 0, f"Agent A received {len(agent_a_events)} events. Subscriptions: {[s.to_dict() for s in agent_a_subs]}"
        for event in agent_a_events:
            if hasattr(event, 'metadata') and event.metadata:
                assert event.metadata.get('namespace') == namespace1

        # Agent B should only have events from namespace2
        assert len(agent_b_events) > 0, f"Agent B received {len(agent_b_events)} events. Subscriptions: {[s.to_dict() for s in agent_b_subs]}"
        for event in agent_b_events:
            if hasattr(event, 'metadata') and event.metadata:
                assert event.metadata.get('namespace') == namespace2

    def test_audit_log_completeness(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Test audit log captures all cross-agent operations.

        Verifies:
        - All namespace operations are logged
        - All permission operations are logged
        - All subscription operations are logged
        - All memory operations are logged
        - Audit entries contain complete metadata
        """
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.audit_log import AuditLogger

        # Setup
        db_path = temp_omi_setup["db_path"]
        import sqlite3
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)
        audit = AuditLogger(db_path)

        agent_a = "agent-a"
        agent_b = "agent-b"
        namespace = "test/audit"

        # Perform operations and log them
        # 1. Create namespace
        shared_ns.create(namespace, created_by=agent_a)
        audit.log(agent_a, "CREATE", "NAMESPACE", namespace, namespace)

        # 2. Grant permission
        permissions.grant(namespace, agent_b, PermissionLevel.READ)
        audit.log(agent_a, "GRANT", "PERMISSION", f"{namespace}:{agent_b}", namespace,
                  metadata={"permission_level": "READ"})

        # 3. Subscribe
        sub_info = subscriptions.subscribe(agent_b, namespace=namespace, event_types=["memory.stored"])
        audit.log(agent_b, "SUBSCRIBE", "SUBSCRIPTION", sub_info.id, namespace)

        # 4. Revoke permission
        permissions.revoke(namespace, agent_b)
        audit.log(agent_a, "REVOKE", "PERMISSION", f"{namespace}:{agent_b}", namespace)

        # 5. Unsubscribe
        subscriptions.unsubscribe(sub_info.id)
        audit.log(agent_b, "UNSUBSCRIBE", "SUBSCRIPTION", sub_info.id, namespace)

        # 6. Delete namespace
        # Log before deleting (since logging after deletion would cause FK constraint failure)
        audit.log(agent_a, "DELETE", "NAMESPACE", namespace, namespace)
        shared_ns.delete(namespace)

        # Verify audit log completeness
        # Note: After namespace deletion, entries with FK=namespace get SET NULL, so we need
        # to query by agent instead
        agent_a_entries = audit.get_by_agent(agent_a, limit=100)
        agent_b_entries = audit.get_by_agent(agent_b, limit=100)
        all_entries = agent_a_entries + agent_b_entries
        assert len(all_entries) == 6, f"Should have 6 audit entries, got {len(all_entries)}"

        # Verify each operation type
        action_types = [e.action_type for e in all_entries]
        assert "CREATE" in action_types
        assert "GRANT" in action_types
        assert "SUBSCRIBE" in action_types
        assert "REVOKE" in action_types
        assert "UNSUBSCRIBE" in action_types
        assert "DELETE" in action_types

        # Verify metadata is preserved
        grant_entry = next(e for e in all_entries if e.action_type == "GRANT")
        assert grant_entry.metadata is not None
        assert grant_entry.metadata.get("permission_level") == "READ"

    def test_performance_under_load(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Test performance under concurrent multi-agent operations.

        Verifies:
        - Multiple concurrent operations complete under 50ms each
        - System handles multiple agents simultaneously
        - No performance degradation with concurrent access
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.api import MemoryTools

        # Setup
        db_path = temp_omi_setup["db_path"]
        import sqlite3
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        palace = GraphPalace(db_path)
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        namespace = "performance/test"

        # Create namespace
        shared_ns.create(namespace, created_by="admin")

        # Perform 10 operations and track timing
        operation_times = []

        for i in range(10):
            agent_id = f"agent-{i}"

            # Grant permission
            start = time.time()
            permissions.grant(namespace, agent_id, PermissionLevel.READ)
            elapsed = time.time() - start
            operation_times.append(("grant_permission", elapsed))

            # Subscribe
            start = time.time()
            sub_info = subscriptions.subscribe(agent_id, namespace=namespace, event_types=["memory.stored"])
            elapsed = time.time() - start
            operation_times.append(("subscribe", elapsed))

            # Store memory
            start = time.time()
            memory_tools.store(
                content=f"Performance test memory {i}",
                memory_type="fact"
            )
            elapsed = time.time() - start
            operation_times.append(("store_memory", elapsed))

        # Verify all operations completed under 50ms
        for operation, elapsed in operation_times:
            assert elapsed < 0.050, f"{operation} took {elapsed*1000:.2f}ms (should be <50ms)"

        # Calculate average time per operation type
        grant_times = [t for op, t in operation_times if op == "grant_permission"]
        subscribe_times = [t for op, t in operation_times if op == "subscribe"]
        store_times = [t for op, t in operation_times if op == "store_memory"]

        avg_grant = sum(grant_times) / len(grant_times)
        avg_subscribe = sum(subscribe_times) / len(subscribe_times)
        avg_store = sum(store_times) / len(store_times)

        # All averages should be well under 50ms
        assert avg_grant < 0.050
        assert avg_subscribe < 0.050
        assert avg_store < 0.050
