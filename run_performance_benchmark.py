#!/usr/bin/env python3
"""
Performance Benchmarking Script for Shared Namespace Operations

Runs comprehensive performance tests and generates a detailed report.
"""

import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List
import tempfile
import sqlite3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omi.storage.graph_palace import GraphPalace
from omi.storage.schema import init_database
from omi.shared_namespace import SharedNamespace
from omi.permissions import PermissionManager, PermissionLevel
from omi.subscriptions import SubscriptionManager
from omi.audit_log import AuditLogger
from omi.belief import BeliefNetwork, Evidence
from omi.belief_propagation import BeliefPropagator
from omi.event_bus import EventBus
from datetime import datetime


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
        lines = ["# Performance Benchmark Results\n\n"]
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lines.append(f"**Total operations benchmarked:** {len(self.measurements)}\n\n")
        lines.append(f"**Threshold requirement:** <50ms per operation (P95 latency)\n\n")

        verification = self.verify_threshold()
        all_passed = all(verification.values())

        lines.append("## Summary\n\n")
        passed_count = sum(1 for v in verification.values() if v)
        failed_count = len(verification) - passed_count
        lines.append(f"- ✓ Passed: {passed_count}/{len(verification)}\n")
        lines.append(f"- ✗ Failed: {failed_count}/{len(verification)}\n\n")

        lines.append("---\n\n")
        lines.append("## Detailed Results\n\n")

        for operation in sorted(self.measurements.keys()):
            stats = self.get_stats(operation)
            passed = verification[operation]
            status = "✓ PASS" if passed else "✗ FAIL"

            lines.append(f"### {operation.replace('_', ' ').title()}\n\n")
            lines.append(f"**Status:** {status}\n\n")
            lines.append(f"| Metric | Value |\n")
            lines.append(f"|--------|-------|\n")
            lines.append(f"| Count | {stats['count']} |\n")
            lines.append(f"| Min | {stats['min']:.3f} ms |\n")
            lines.append(f"| Max | {stats['max']:.3f} ms |\n")
            lines.append(f"| Mean | {stats['mean']:.3f} ms |\n")
            lines.append(f"| Median | {stats['median']:.3f} ms |\n")
            lines.append(f"| P95 | {stats['p95']:.3f} ms |\n")
            lines.append(f"| P99 | {stats['p99']:.3f} ms |\n\n")

        lines.append("---\n\n")
        lines.append(f"## Overall Result\n\n")
        if all_passed:
            lines.append("### ✓ ALL TESTS PASSED\n\n")
            lines.append("All shared namespace operations completed within the 50ms latency requirement.\n")
        else:
            lines.append("### ✗ SOME TESTS FAILED\n\n")
            lines.append("The following operations exceeded the 50ms threshold:\n\n")
            for operation, passed in verification.items():
                if not passed:
                    stats = self.get_stats(operation)
                    lines.append(f"- {operation}: P95={stats['p95']:.3f}ms\n")

        return "".join(lines)


class MockEmbedder:
    """Mock embedder for testing."""
    def embed(self, text: str) -> List[float]:
        return [0.1] * 768


class MockEmbeddingCache:
    """Mock embedding cache for testing."""
    def get(self, text: str):
        return None
    def set(self, text: str, embedding):
        pass


def run_benchmarks():
    """Run all performance benchmarks."""
    print("Starting Performance Benchmarks for Shared Namespace Operations")
    print("=" * 70)

    benchmark = PerformanceBenchmark()

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        # Initialize database
        conn = sqlite3.connect(db_path)
        init_database(conn)
        conn.close()

        # Initialize components
        palace = GraphPalace(str(db_path))
        shared_ns = SharedNamespace(str(db_path))
        permissions = PermissionManager(str(db_path))
        subscriptions = SubscriptionManager(str(db_path))
        audit = AuditLogger(str(db_path))
        event_bus = EventBus()

        print("\n1. Benchmarking Individual Operations (100 iterations each)...")

        # Benchmark: Namespace creation
        print("   - Namespace creation...")
        for i in range(100):
            namespace = f"bench/ns-{i}"
            start = time.perf_counter()
            shared_ns.create(namespace, created_by="bench-agent")
            elapsed = time.perf_counter() - start
            benchmark.record("namespace_create", elapsed)

        # Benchmark: Permission operations
        print("   - Permission grant...")
        test_namespace = "bench/ns-0"
        for i in range(100):
            agent_id = f"agent-grant-{i}"
            start = time.perf_counter()
            permissions.grant(test_namespace, agent_id, PermissionLevel.READ)
            elapsed = time.perf_counter() - start
            benchmark.record("permission_grant", elapsed)

        print("   - Permission check...")
        for i in range(100):
            agent_id = f"agent-grant-{i % 10}"
            start = time.perf_counter()
            permissions.has_permission(test_namespace, agent_id, PermissionLevel.READ)
            elapsed = time.perf_counter() - start
            benchmark.record("permission_check", elapsed)

        # Benchmark: Subscription operations
        print("   - Subscribe...")
        subscription_ids = []
        for i in range(100):
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

        print("   - List subscriptions...")
        for i in range(100):
            agent_id = f"agent-sub-{i % 10}"
            start = time.perf_counter()
            subscriptions.list_for_agent(agent_id)
            elapsed = time.perf_counter() - start
            benchmark.record("list_subscriptions", elapsed)

        print("   - Unsubscribe...")
        for sub_id in subscription_ids[:50]:
            start = time.perf_counter()
            subscriptions.unsubscribe(sub_id)
            elapsed = time.perf_counter() - start
            benchmark.record("unsubscribe", elapsed)

        # Benchmark: Memory storage
        print("   - Memory storage...")
        for i in range(100):
            start = time.perf_counter()
            palace.store_memory(
                content=f"Performance test memory {i}",
                memory_type="fact"
            )
            elapsed = time.perf_counter() - start
            benchmark.record("memory_store", elapsed)

        # Benchmark: Audit logging
        print("   - Audit logging...")
        for i in range(100):
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

        print("   - Audit query...")
        for i in range(100):
            start = time.perf_counter()
            audit.get_by_namespace(test_namespace, limit=10)
            elapsed = time.perf_counter() - start
            benchmark.record("audit_query", elapsed)

        print("\n2. Benchmarking Under Load (50 concurrent agents)...")
        load_namespace = "bench/load-test"
        shared_ns.create(load_namespace, created_by="admin")

        for i in range(50):
            agent_id = f"load-agent-{i}"

            start = time.perf_counter()
            permissions.grant(load_namespace, agent_id, PermissionLevel.WRITE)
            elapsed = time.perf_counter() - start
            benchmark.record("load_grant", elapsed)

            start = time.perf_counter()
            subscriptions.subscribe(agent_id, namespace=load_namespace, event_types=["memory.stored"])
            elapsed = time.perf_counter() - start
            benchmark.record("load_subscribe", elapsed)

            start = time.perf_counter()
            memory_id = palace.store_memory(
                content=f"Load test memory from {agent_id}",
                memory_type="fact"
            )
            elapsed = time.perf_counter() - start
            benchmark.record("load_memory_store", elapsed)

            start = time.perf_counter()
            audit.log(agent_id, "WRITE", "MEMORY", memory_id, load_namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("load_audit", elapsed)

        print("\n3. Benchmarking Query Operations...")

        # Create test data
        for i in range(20):
            ns = f"query-bench/ns-{i}"
            shared_ns.create(ns, created_by="admin")
            for j in range(10):
                agent_id = f"query-agent-{i}-{j}"
                permissions.grant(ns, agent_id, PermissionLevel.READ)
                subscriptions.subscribe(agent_id, namespace=ns, event_types=["memory.stored"])

        print("   - List all namespaces...")
        for _ in range(50):
            start = time.perf_counter()
            shared_ns.list_all()
            elapsed = time.perf_counter() - start
            benchmark.record("query_list_all_namespaces", elapsed)

        print("   - Get specific namespace...")
        for i in range(50):
            namespace = f"query-bench/ns-{i % 20}"
            start = time.perf_counter()
            shared_ns.get(namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("query_get_namespace", elapsed)

        print("   - List namespace permissions...")
        for i in range(50):
            namespace = f"query-bench/ns-{i % 20}"
            start = time.perf_counter()
            permissions.list_for_namespace(namespace)
            elapsed = time.perf_counter() - start
            benchmark.record("query_namespace_permissions", elapsed)

        print("\n4. Benchmarking Belief Propagation...")

        bn_source = BeliefNetwork(palace, event_bus=event_bus)
        bn_target = BeliefNetwork(palace)

        propagator = BeliefPropagator(
            event_bus=event_bus,
            subscription_manager=subscriptions,
            belief_network=bn_target
        )

        belief_ns = "belief-bench/test"
        shared_ns.create(belief_ns, created_by="source-agent")
        permissions.grant(belief_ns, "source-agent", PermissionLevel.WRITE)
        permissions.grant(belief_ns, "target-agent", PermissionLevel.READ)
        subscriptions.subscribe("target-agent", namespace=belief_ns, event_types=["belief.updated"])

        print("   - Set trust weight...")
        start = time.perf_counter()
        propagator.set_trust_weight("target-agent", "source-agent", 0.8)
        elapsed = time.perf_counter() - start
        benchmark.record("trust_set_weight", elapsed)

        propagator.start()

        try:
            print("   - Belief creation with propagation...")
            for i in range(50):
                content = f"Benchmark belief {i}"

                start = time.perf_counter()
                belief_id = bn_source.create_belief(
                    content=content,
                    initial_confidence=0.7,
                    agent_id="source-agent",
                    namespace=belief_ns
                )
                elapsed = time.perf_counter() - start
                benchmark.record("belief_create", elapsed)

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
                    namespace=belief_ns
                )
                elapsed = time.perf_counter() - start
                benchmark.record("belief_update", elapsed)

            print("   - Trust weight retrieval...")
            for _ in range(50):
                start = time.perf_counter()
                propagator.get_trust_weight("target-agent", "source-agent")
                elapsed = time.perf_counter() - start
                benchmark.record("trust_get_weight", elapsed)

        finally:
            propagator.stop()

    print("\n" + "=" * 70)
    print("Benchmarks Complete!")
    print("=" * 70 + "\n")

    # Generate and save report
    report = benchmark.generate_report()

    report_path = Path(__file__).parent / "performance_results.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Performance report saved to: {report_path}\n")

    # Print summary
    verification = benchmark.verify_threshold()
    all_passed = all(verification.values())

    if all_passed:
        print("✓ ALL OPERATIONS PASSED: All operations completed within 50ms threshold")
        return 0
    else:
        print("✗ SOME OPERATIONS FAILED: Review performance_results.md for details")
        failed = [op for op, passed in verification.items() if not passed]
        for op in failed:
            stats = benchmark.get_stats(op)
            print(f"  - {op}: P95={stats['p95']:.3f}ms")
        return 1


if __name__ == "__main__":
    sys.exit(run_benchmarks())
