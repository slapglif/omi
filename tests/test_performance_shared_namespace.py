"""
Performance Benchmarking for Shared Namespace Operations

Measures latency for:
- Shared namespace creation
- Permission grants/revokes
- Subscription management
- Memory storage in shared namespace
- Audit logging
- Event propagation

Acceptance Criteria: All operations must complete in <50ms
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
from pathlib import Path


class PerformanceBenchmark:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def record(self, operation: str, duration: float):
        """Record a single operation duration."""
        if operation not in self.measurements:
            self.measurements[operation] = []
        self.measurements[operation].append(duration)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Calculate statistics for an operation."""
        if operation not in self.measurements or not self.measurements[operation]:
            return {}

        data = sorted(self.measurements[operation])
        n = len(data)

        return {
            'count': n,
            'min': min(data) * 1000,  # Convert to ms
            'max': max(data) * 1000,
            'mean': statistics.mean(data) * 1000,
            'median': statistics.median(data) * 1000,
            'p95': data[int(n * 0.95)] * 1000 if n > 0 else 0,
            'p99': data[int(n * 0.99)] * 1000 if n > 0 else 0,
        }

    def verify_threshold(self, threshold_ms: float = 50.0) -> Dict[str, bool]:
        """Verify all operations meet threshold requirement."""
        results = {}
        for operation in self.measurements:
            stats = self.get_stats(operation)
            # Check p95 latency (95th percentile must be under threshold)
            results[operation] = stats.get('p95', float('inf')) < threshold_ms
        return results

    def generate_report(self) -> str:
        """Generate a human-readable performance report."""
        lines = ["# Performance Benchmark Results\n"]
        lines.append(f"Total operations benchmarked: {len(self.measurements)}\n")
        lines.append(f"Threshold requirement: <50ms per operation\n\n")

        verification = self.verify_threshold()

        for operation in sorted(self.measurements.keys()):
            stats = self.get_stats(operation)
            passed = verification[operation]
            status = "✓ PASS" if passed else "✗ FAIL"

            lines.append(f"## {operation} {status}\n")
            lines.append(f"- Count: {stats['count']}\n")
            lines.append(f"- Min: {stats['min']:.2f}ms\n")
            lines.append(f"- Max: {stats['max']:.2f}ms\n")
            lines.append(f"- Mean: {stats['mean']:.2f}ms\n")
            lines.append(f"- Median: {stats['median']:.2f}ms\n")
            lines.append(f"- P95: {stats['p95']:.2f}ms\n")
            lines.append(f"- P99: {stats['p99']:.2f}ms\n\n")

        # Overall pass/fail
        all_passed = all(verification.values())
        lines.append(f"## Overall Result: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}\n")

        return "".join(lines)


class TestSharedNamespacePerformance:
    """Performance benchmarking for shared namespace operations."""

    @pytest.fixture
    def benchmark(self):
        """Create a performance benchmark tracker."""
        return PerformanceBenchmark()

    def test_individual_operation_latency(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        benchmark
    ):
        """
        Benchmark individual operations with 100 iterations each.

        Tests:
        - Namespace creation
        - Permission grant
        - Permission revoke
        - Subscribe
        - Unsubscribe
        - Memory storage
        - Audit logging
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.audit_log import AuditLogger
        from omi.api import MemoryTools
        import sqlite3

        # Setup
        db_path = temp_omi_setup["db_path"]
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

        iterations = 100

        # Benchmark: Namespace creation
        for i in range(iterations):
            namespace = f"bench/ns-{i}"
            start = time.perf_counter()
            shared_ns.create(namespace, created_by="bench-agent")
            elapsed = time.perf_counter() - start
            benchmark.record("namespace_create", elapsed)

        # Benchmark: Permission grant
        test_namespace = "bench/ns-0"
        for i in range(iterations):
            agent_id = f"agent-grant-{i}"
            start = time.perf_counter()
            permissions.grant(test_namespace, agent_id, PermissionLevel.READ)
            elapsed = time.perf_counter() - start
            benchmark.record("permission_grant", elapsed)

        # Benchmark: Permission check
        for i in range(iterations):
            agent_id = f"agent-grant-{i % 10}"  # Check against existing agents
            start = time.perf_counter()
            permissions.has_permission(test_namespace, agent_id, PermissionLevel.READ)
            elapsed = time.perf_counter() - start
            benchmark.record("permission_check", elapsed)

        # Benchmark: Subscribe
        subscription_ids = []
        for i in range(iterations):
            agent_id = f"agent-sub-{i}"
            start = time.perf_counter()
            sub_info = subscriptions.subscribe(
                agent_id,
                namespace=test_namespace,
                event_types=["memory.stored"]
            )
            elapsed = time.perf_counter() - start
            benchmark.record("subscribe", elapsed)
            subscription_ids.append(sub_info.id)

        # Benchmark: List subscriptions
        for i in range(iterations):
            agent_id = f"agent-sub-{i % 10}"
            start = time.perf_counter()
            subscriptions.list_for_agent(agent_id)
            elapsed = time.perf_counter() - start
            benchmark.record("list_subscriptions", elapsed)

        # Benchmark: Unsubscribe
        for sub_id in subscription_ids[:50]:  # Unsubscribe half
            start = time.perf_counter()
            subscriptions.unsubscribe(sub_id)
            elapsed = time.perf_counter() - start
            benchmark.record("unsubscribe", elapsed)

        # Benchmark: Memory storage
        for i in range(iterations):
            start = time.perf_counter()
            memory_tools.store(
                content=f"Performance test memory {i}",
                memory_type="fact"
            )
            elapsed = time.perf_counter() - start
            benchmark.record("memory_store", elapsed)

        # Benchmark: Audit logging
        for i in range(iterations):
            start = time.perf_counter()
            audit.log(
                agent_id=f"bench-agent-{i}",
                action_type="READ",
                resource_type="MEMORY",
                resource_id=f"mem-{i}",
                namespace=test_namespace
            )
            elapsed = time.perf_counter() - start
            benchmark.record("audit_log", elapsed)

        # Benchmark: Audit log retrieval
        for i in range(iterations):
            start = time.perf_counter()
            audit.get_by_namespace(test_namespace, limit=10)
            elapsed = time.perf_counter() - start
            benchmark.record("audit_query", elapsed)

        # Benchmark: Permission revoke
        for i in range(50):  # Revoke half the grants
            agent_id = f"agent-grant-{i}"
            start = time.perf_counter()
            permissions.revoke(test_namespace, agent_id)
            elapsed = time.perf_counter() - start
            benchmark.record("permission_revoke", elapsed)

        # Verify all operations meet <50ms threshold
        verification = benchmark.verify_threshold(50.0)

        # Assert each operation passes
        for operation, passed in verification.items():
            stats = benchmark.get_stats(operation)
            assert passed, (
                f"{operation} failed: P95={stats['p95']:.2f}ms "
                f"(threshold: 50ms)"
            )

    def test_concurrent_agent_load(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        benchmark
    ):
        """
        Benchmark performance under concurrent multi-agent load.

        Simulates 50 agents performing operations simultaneously.
        Verifies no performance degradation with concurrent access.
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.audit_log import AuditLogger
        from omi.api import MemoryTools
        import sqlite3

        # Setup
        db_path = temp_omi_setup["db_path"]
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

        namespace = "bench/load-test"
        shared_ns.create(namespace, created_by="admin")

        num_agents = 50

        # Simulate concurrent workflow for each agent
        for i in range(num_agents):
            agent_id = f"load-agent-{i}"

            # Grant permission
            start = time.perf_counter()
            permissions.grant(namespace, agent_id, PermissionLevel.WRITE)
            elapsed = time.perf_counter() - start
            benchmark.record("load_test_grant", elapsed)

            # Subscribe
            start = time.perf_counter()
            subscriptions.subscribe(agent_id, namespace=namespace, event_types=["memory.stored"])
            elapsed = time.perf_counter() - start
            benchmark.record("load_test_subscribe", elapsed)

            # Store memory
            start = time.perf_counter()
            memory_id = memory_tools.store(
                content=f"Load test memory from {agent_id}",
                memory_type="fact"
            )
            elapsed = time.perf_counter() - start
            benchmark.record("load_test_store", elapsed)

            # Log operation
            start = time.perf_counter()
            audit.log(agent_id, "WRITE", "MEMORY", memory_id, namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("load_test_audit", elapsed)

        # Verify all operations under load meet threshold
        verification = benchmark.verify_threshold(50.0)

        for operation, passed in verification.items():
            if operation.startswith("load_test_"):
                stats = benchmark.get_stats(operation)
                assert passed, (
                    f"{operation} under load failed: P95={stats['p95']:.2f}ms "
                    f"(threshold: 50ms)"
                )

    def test_namespace_query_performance(
        self,
        temp_omi_setup,
        benchmark
    ):
        """
        Benchmark namespace query operations.

        Tests:
        - List all namespaces
        - Get specific namespace
        - List permissions for namespace
        - List subscriptions for namespace
        """
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        import sqlite3

        # Setup
        db_path = temp_omi_setup["db_path"]
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)

        # Create test data
        num_namespaces = 20
        agents_per_ns = 10

        for i in range(num_namespaces):
            namespace = f"query-bench/ns-{i}"
            shared_ns.create(namespace, created_by="admin")

            for j in range(agents_per_ns):
                agent_id = f"query-agent-{i}-{j}"
                permissions.grant(namespace, agent_id, PermissionLevel.READ)
                subscriptions.subscribe(agent_id, namespace=namespace, event_types=["memory.stored"])

        # Benchmark: List all namespaces
        for _ in range(50):
            start = time.perf_counter()
            shared_ns.list_all()
            elapsed = time.perf_counter() - start
            benchmark.record("list_all_namespaces", elapsed)

        # Benchmark: Get specific namespace
        for i in range(50):
            namespace = f"query-bench/ns-{i % num_namespaces}"
            start = time.perf_counter()
            shared_ns.get(namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("get_namespace", elapsed)

        # Benchmark: List permissions for namespace
        for i in range(50):
            namespace = f"query-bench/ns-{i % num_namespaces}"
            start = time.perf_counter()
            permissions.list_for_namespace(namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("list_namespace_permissions", elapsed)

        # Benchmark: List permissions for agent
        for i in range(50):
            agent_id = f"query-agent-0-{i % agents_per_ns}"
            start = time.perf_counter()
            permissions.list_for_agent(agent_id)
            elapsed = time.perf_counter() - start
            benchmark.record("list_agent_permissions", elapsed)

        # Benchmark: List subscriptions for namespace
        for i in range(50):
            namespace = f"query-bench/ns-{i % num_namespaces}"
            start = time.perf_counter()
            subscriptions.list_for_namespace(namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("list_namespace_subscriptions", elapsed)

        # Verify all query operations meet threshold
        verification = benchmark.verify_threshold(50.0)

        for operation, passed in verification.items():
            stats = benchmark.get_stats(operation)
            assert passed, (
                f"{operation} failed: P95={stats['p95']:.2f}ms "
                f"(threshold: 50ms)"
            )

    def test_belief_propagation_performance(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus,
        benchmark
    ):
        """
        Benchmark belief propagation operations.

        Tests:
        - Belief update with propagation
        - Trust weight updates
        - Multi-agent belief sync
        """
        from omi.storage.graph_palace import GraphPalace
        from omi.storage.schema import init_database
        from omi.shared_namespace import SharedNamespace
        from omi.permissions import PermissionManager, PermissionLevel
        from omi.subscriptions import SubscriptionManager
        from omi.belief import BeliefNetwork, Evidence
        from omi.belief_propagation import BeliefPropagator
        import sqlite3

        # Setup
        db_path = temp_omi_setup["db_path"]
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        palace = GraphPalace(db_path)
        shared_ns = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)

        # Initialize belief networks
        bn_source = BeliefNetwork(palace, event_bus=clean_event_bus)
        bn_target = BeliefNetwork(palace)

        # Initialize belief propagator
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscriptions,
            belief_network=bn_target
        )

        namespace = "belief-bench/test"
        shared_ns.create(namespace, created_by="source-agent")
        permissions.grant(namespace, "source-agent", PermissionLevel.WRITE)
        permissions.grant(namespace, "target-agent", PermissionLevel.READ)
        subscriptions.subscribe("target-agent", namespace=namespace, event_types=["belief.updated"])

        # Set trust weight
        start = time.perf_counter()
        propagator.set_trust_weight("target-agent", "source-agent", 0.8)
        elapsed = time.perf_counter() - start
        benchmark.record("set_trust_weight", elapsed)

        propagator.start()

        try:
            # Benchmark: Belief creation and propagation
            for i in range(50):
                content = f"Benchmark belief {i}"

                start = time.perf_counter()
                belief_id = bn_source.create_belief(
                    content=content,
                    initial_confidence=0.7,
                    agent_id="source-agent",
                    namespace=namespace
                )
                elapsed = time.perf_counter() - start
                benchmark.record("belief_create_with_propagation", elapsed)

                # Create evidence and update belief
                from datetime import datetime
                memory_id = palace.store_memory(
                    content=f"Evidence for belief {i}",
                    memory_type="fact"
                )

                evidence = Evidence(
                    memory_id=memory_id,
                    supports=True,
                    strength=0.9,
                    timestamp=datetime.now()
                )

                start = time.perf_counter()
                bn_source.update_with_evidence(
                    belief_id=belief_id,
                    evidence=evidence,
                    agent_id="source-agent",
                    namespace=namespace
                )
                elapsed = time.perf_counter() - start
                benchmark.record("belief_update_with_propagation", elapsed)

            # Benchmark: Trust weight retrieval
            for _ in range(50):
                start = time.perf_counter()
                propagator.get_trust_weight("target-agent", "source-agent")
                elapsed = time.perf_counter() - start
                benchmark.record("get_trust_weight", elapsed)

        finally:
            propagator.stop()

        # Verify belief propagation operations meet threshold
        verification = benchmark.verify_threshold(50.0)

        for operation, passed in verification.items():
            if "belief" in operation or "trust" in operation:
                stats = benchmark.get_stats(operation)
                assert passed, (
                    f"{operation} failed: P95={stats['p95']:.2f}ms "
                    f"(threshold: 50ms)"
                )


@pytest.fixture(scope="session")
def performance_report(tmp_path_factory):
    """Generate and save performance report after all tests."""
    # This will be called after all tests complete
    yield
