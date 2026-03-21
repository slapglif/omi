"""Performance Benchmark for Cursor-Based Pagination

This test verifies that cursor-based pagination adds less than 5ms overhead
compared to unpaginated queries, as specified in the acceptance criteria.

Acceptance Criteria (from spec):
- Performance: Paginated queries add less than 5ms overhead vs. unpaginated

Issue: https://github.com/slapglif/omi/issues/46
"""
import pytest
import tempfile
import os
import time
from typing import List, Dict, Any
from statistics import mean, stdev

from omi.storage.graph_palace import GraphPalace


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def palace_with_data(temp_db):
    """Create a GraphPalace instance with 1000 test memories."""
    palace = GraphPalace(temp_db)

    # Create 1000 memories for realistic performance testing
    for i in range(1000):
        palace.store_memory(
            content=f"Test memory {i:04d} - this is some content for performance testing",
            memory_type="fact"
        )

    return palace


def benchmark_query(func, iterations: int = 20) -> Dict[str, float]:
    """
    Benchmark a query function multiple times and return statistics.

    Args:
        func: Function to benchmark (no arguments)
        iterations: Number of iterations to run

    Returns:
        Dictionary with mean, stdev, min, max times in milliseconds
    """
    times = []

    # Warmup run (not counted)
    func()

    # Actual benchmark runs
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        "mean_ms": mean(times),
        "stdev_ms": stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times)
    }


class TestPaginationPerformance:
    """Test that pagination overhead is less than 5ms."""

    def test_list_memories_pagination_overhead(self, palace_with_data):
        """
        Benchmark list_memories with pagination vs raw SQL query.

        Compares:
        1. Baseline: Raw SQL SELECT with LIMIT (unpaginated)
        2. Paginated: list_memories() with cursor-based pagination

        Assert: Overhead < 5ms
        """
        palace = palace_with_data
        limit = 50

        # Baseline: Raw SQL query with LIMIT (no pagination logic)
        def baseline_query():
            cursor = palace._conn.execute("""
                SELECT id, content, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            results = cursor.fetchall()
            return results

        # Paginated query using list_memories
        def paginated_query():
            result = palace.list_memories(limit=limit, order_by="created_at", order_dir="desc")
            return result["memories"]

        # Run benchmarks
        baseline_stats = benchmark_query(baseline_query, iterations=30)
        paginated_stats = benchmark_query(paginated_query, iterations=30)

        # Calculate overhead
        overhead_ms = paginated_stats["mean_ms"] - baseline_stats["mean_ms"]

        # Print results for visibility
        print(f"\n=== list_memories Pagination Performance ===")
        print(f"Baseline (raw SQL):   {baseline_stats['mean_ms']:.3f}ms (±{baseline_stats['stdev_ms']:.3f}ms)")
        print(f"Paginated:            {paginated_stats['mean_ms']:.3f}ms (±{paginated_stats['stdev_ms']:.3f}ms)")
        print(f"Overhead:             {overhead_ms:.3f}ms")
        print(f"Target:               <5.000ms")

        # Verify overhead is less than 5ms
        assert overhead_ms < 5.0, (
            f"Pagination overhead {overhead_ms:.3f}ms exceeds 5ms limit. "
            f"Baseline: {baseline_stats['mean_ms']:.3f}ms, "
            f"Paginated: {paginated_stats['mean_ms']:.3f}ms"
        )

    def test_list_beliefs_pagination_overhead(self, palace_with_data):
        """
        Benchmark list_beliefs with pagination vs raw SQL query.

        Assert: Overhead < 5ms
        """
        palace = palace_with_data

        # Create some beliefs for testing
        for i in range(100):
            palace.store_memory(
                content=f"Belief {i}",
                memory_type="belief",
                confidence=0.5 + (i * 0.005)
            )

        limit = 20

        # Baseline: Raw SQL query
        def baseline_query():
            cursor = palace._conn.execute("""
                SELECT id, content, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                WHERE memory_type = 'belief'
                ORDER BY confidence DESC
                LIMIT ?
            """, (limit,))
            results = cursor.fetchall()
            return results

        # Paginated query
        def paginated_query():
            result = palace.list_beliefs(limit=limit, order_by="confidence", order_dir="desc")
            return result["beliefs"]

        # Run benchmarks
        baseline_stats = benchmark_query(baseline_query, iterations=30)
        paginated_stats = benchmark_query(paginated_query, iterations=30)

        # Calculate overhead
        overhead_ms = paginated_stats["mean_ms"] - baseline_stats["mean_ms"]

        # Print results
        print(f"\n=== list_beliefs Pagination Performance ===")
        print(f"Baseline (raw SQL):   {baseline_stats['mean_ms']:.3f}ms (±{baseline_stats['stdev_ms']:.3f}ms)")
        print(f"Paginated:            {paginated_stats['mean_ms']:.3f}ms (±{paginated_stats['stdev_ms']:.3f}ms)")
        print(f"Overhead:             {overhead_ms:.3f}ms")
        print(f"Target:               <5.000ms")

        # Verify overhead is less than 5ms
        assert overhead_ms < 5.0, (
            f"Pagination overhead {overhead_ms:.3f}ms exceeds 5ms limit"
        )

    def test_list_edges_pagination_overhead(self, palace_with_data):
        """
        Benchmark list_edges with pagination vs raw SQL query.

        Assert: Overhead < 5ms
        """
        palace = palace_with_data

        # Create some memories and edges
        memory_ids = []
        for i in range(20):
            memory_id = palace.store_memory(
                content=f"Memory for edges {i}",
                memory_type="fact"
            )
            memory_ids.append(memory_id)

        # Create 200 edges
        for i in range(200):
            source_idx = i % 19
            target_idx = (i + 1) % 20
            palace.create_edge(
                source_id=memory_ids[source_idx],
                target_id=memory_ids[target_idx],
                edge_type="RELATED_TO",
                strength=0.5 + (i * 0.002)
            )

        limit = 50

        # Baseline: Raw SQL query
        def baseline_query():
            cursor = palace._conn.execute("""
                SELECT id, source_id, target_id, edge_type, strength, created_at
                FROM edges
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            results = cursor.fetchall()
            return results

        # Paginated query
        def paginated_query():
            result = palace.list_edges(limit=limit, order_by="created_at", order_dir="desc")
            return result["edges"]

        # Run benchmarks
        baseline_stats = benchmark_query(baseline_query, iterations=30)
        paginated_stats = benchmark_query(paginated_query, iterations=30)

        # Calculate overhead
        overhead_ms = paginated_stats["mean_ms"] - baseline_stats["mean_ms"]

        # Print results
        print(f"\n=== list_edges Pagination Performance ===")
        print(f"Baseline (raw SQL):   {baseline_stats['mean_ms']:.3f}ms (±{baseline_stats['stdev_ms']:.3f}ms)")
        print(f"Paginated:            {paginated_stats['mean_ms']:.3f}ms (±{paginated_stats['stdev_ms']:.3f}ms)")
        print(f"Overhead:             {overhead_ms:.3f}ms")
        print(f"Target:               <5.000ms")

        # Verify overhead is less than 5ms
        assert overhead_ms < 5.0, (
            f"Pagination overhead {overhead_ms:.3f}ms exceeds 5ms limit"
        )

    def test_cursor_pagination_overhead(self, palace_with_data):
        """
        Benchmark cursor-based pagination (2nd page) vs offset-based query.

        This tests the cursor decoding and filtering overhead.

        Assert: Overhead < 5ms
        """
        palace = palace_with_data
        limit = 50

        # Get first page to obtain cursor
        page1 = palace.list_memories(limit=limit, order_by="created_at", order_dir="desc")
        cursor = page1["next_cursor"]

        # Get the last_id from the cursor for baseline comparison
        import base64
        import json
        cursor_data = json.loads(base64.b64decode(cursor).decode('utf-8'))
        last_id = cursor_data["last_id"]

        # Baseline: Offset-based pagination (traditional approach)
        def baseline_query():
            # Offset-based: skip first 50, get next 50
            cursor = palace._conn.execute("""
                SELECT id, content, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, limit))
            results = cursor.fetchall()
            return results

        # Cursor-based pagination
        def paginated_query():
            result = palace.list_memories(limit=limit, cursor=cursor, order_by="created_at", order_dir="desc")
            return result["memories"]

        # Run benchmarks
        baseline_stats = benchmark_query(baseline_query, iterations=30)
        paginated_stats = benchmark_query(paginated_query, iterations=30)

        # Calculate overhead
        overhead_ms = paginated_stats["mean_ms"] - baseline_stats["mean_ms"]

        # Print results
        print(f"\n=== Cursor-based vs Offset-based Pagination ===")
        print(f"Baseline (OFFSET):    {baseline_stats['mean_ms']:.3f}ms (±{baseline_stats['stdev_ms']:.3f}ms)")
        print(f"Cursor-based:         {paginated_stats['mean_ms']:.3f}ms (±{paginated_stats['stdev_ms']:.3f}ms)")
        print(f"Overhead:             {overhead_ms:.3f}ms")
        print(f"Target:               <5.000ms")

        # Verify overhead is less than 5ms
        assert overhead_ms < 5.0, (
            f"Cursor pagination overhead {overhead_ms:.3f}ms exceeds 5ms limit"
        )

    def test_pagination_with_filters_overhead(self, palace_with_data):
        """
        Benchmark paginated queries with filters vs raw SQL with same filters.

        Assert: Overhead < 5ms even with filtering
        """
        palace = palace_with_data

        # Add some beliefs and decisions for filtering
        for i in range(100):
            palace.store_memory(
                content=f"Belief {i}",
                memory_type="belief",
                confidence=0.7
            )
            palace.store_memory(
                content=f"Decision {i}",
                memory_type="decision"
            )

        limit = 30

        # Baseline: Raw SQL with filter
        def baseline_query():
            cursor = palace._conn.execute("""
                SELECT id, content, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                WHERE memory_type = 'fact'
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            results = cursor.fetchall()
            return results

        # Paginated query with filter
        def paginated_query():
            result = palace.list_memories(
                limit=limit,
                memory_type="fact",
                order_by="created_at",
                order_dir="desc"
            )
            return result["memories"]

        # Run benchmarks
        baseline_stats = benchmark_query(baseline_query, iterations=30)
        paginated_stats = benchmark_query(paginated_query, iterations=30)

        # Calculate overhead
        overhead_ms = paginated_stats["mean_ms"] - baseline_stats["mean_ms"]

        # Print results
        print(f"\n=== Filtered Pagination Performance ===")
        print(f"Baseline (raw SQL):   {baseline_stats['mean_ms']:.3f}ms (±{baseline_stats['stdev_ms']:.3f}ms)")
        print(f"Paginated:            {paginated_stats['mean_ms']:.3f}ms (±{paginated_stats['stdev_ms']:.3f}ms)")
        print(f"Overhead:             {overhead_ms:.3f}ms")
        print(f"Target:               <5.000ms")

        # Verify overhead is less than 5ms
        assert overhead_ms < 5.0, (
            f"Filtered pagination overhead {overhead_ms:.3f}ms exceeds 5ms limit"
        )
