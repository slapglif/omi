"""
test_ann_performance.py - Performance Benchmark for ANN Vector Index

Benchmarks ANN (HNSW) vs brute-force vector search performance:
- 100k synthetic memories with 1024-dim embeddings
- Top-10 recall queries
- Target: <50ms for ANN search (vs. much slower brute-force)

This validates the core performance requirement from:
Issue: https://github.com/slapglif/omi/issues/[ANN-index]
Spec: Acceptance Criteria - Search performance: <50ms for top-10 recall on 100k memories
"""
import time
import tempfile
from pathlib import Path
from typing import List, Tuple

import pytest
import numpy as np

from omi.storage.ann_index import ANNIndex
from omi.storage.search import MemorySearch
from omi.storage.graph_palace import GraphPalace


def generate_synthetic_embeddings(count: int, dim: int = 1024) -> List[Tuple[str, List[float]]]:
    """
    Generate synthetic embeddings for benchmarking.

    Creates random normalized vectors that simulate real embeddings.

    Args:
        count: Number of embeddings to generate
        dim: Embedding dimension (default: 1024 for NIM)

    Returns:
        List of (memory_id, embedding) tuples
    """
    embeddings = []
    for i in range(count):
        # Generate random unit vector (normalized)
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize to unit length
        memory_id = f"mem_{i:06d}"
        embeddings.append((memory_id, vec.tolist()))

    return embeddings


def benchmark_ann_search(embeddings: List[Tuple[str, List[float]]], num_queries: int = 100) -> dict:
    """
    Benchmark ANN index search performance.

    Args:
        embeddings: List of (memory_id, embedding) tuples
        num_queries: Number of search queries to benchmark

    Returns:
        Dictionary with timing results
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "ann_bench.db"

        # Create ANN index and build it
        ann_index = ANNIndex(str(db_path), dim=1024, enable_persistence=False)

        # Benchmark index build time
        build_start = time.time()
        ann_index.rebuild_from_embeddings(embeddings)
        build_time = time.time() - build_start

        # Generate random query embeddings
        query_embeddings = []
        for _ in range(num_queries):
            vec = np.random.randn(1024).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            query_embeddings.append(vec.tolist())

        # Benchmark search time (warm-up first query)
        _ = ann_index.search(query_embeddings[0], k=10)

        # Benchmark actual queries
        search_start = time.time()
        for query_emb in query_embeddings:
            results = ann_index.search(query_emb, k=10)
        search_time = time.time() - search_start

        avg_query_time_ms = (search_time / num_queries) * 1000

        return {
            'build_time': build_time,
            'total_search_time': search_time,
            'num_queries': num_queries,
            'avg_query_time_ms': avg_query_time_ms,
            'index_size': ann_index.get_size()
        }


def benchmark_bruteforce_search(embeddings: List[Tuple[str, List[float]]], num_queries: int = 100) -> dict:
    """
    Benchmark brute-force cosine similarity search performance.

    Simulates the old brute-force approach for comparison.

    Args:
        embeddings: List of (memory_id, embedding) tuples
        num_queries: Number of search queries to benchmark

    Returns:
        Dictionary with timing results
    """
    # Convert to numpy arrays for vectorized operations
    embedding_matrix = np.array([emb for _, emb in embeddings], dtype=np.float32)
    ids = [mid for mid, _ in embeddings]

    # Generate random query embeddings
    query_embeddings = []
    for _ in range(num_queries):
        vec = np.random.randn(1024).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        query_embeddings.append(vec)

    # Benchmark brute-force search
    search_start = time.time()
    for query_emb in query_embeddings:
        # Compute cosine similarity with all vectors
        similarities = np.dot(embedding_matrix, query_emb)
        # Get top-10 indices
        top_k_indices = np.argsort(similarities)[-10:][::-1]
        # Get results (would normally return these)
        results = [(ids[idx], float(similarities[idx])) for idx in top_k_indices]
    search_time = time.time() - search_start

    avg_query_time_ms = (search_time / num_queries) * 1000

    return {
        'total_search_time': search_time,
        'num_queries': num_queries,
        'avg_query_time_ms': avg_query_time_ms,
        'index_size': len(embeddings)
    }


def print_benchmark_results(ann_results: dict, brute_results: dict, num_memories: int):
    """
    Print benchmark results in a readable format.

    Args:
        ann_results: ANN benchmark results
        brute_results: Brute-force benchmark results
        num_memories: Number of memories in the benchmark
    """
    print("\n" + "="*70)
    print("ANN vs BRUTE-FORCE VECTOR SEARCH BENCHMARK")
    print("="*70)

    print(f"\nDataset: {num_memories:,} memories with 1024-dim embeddings")
    print(f"Queries: {ann_results['num_queries']} random queries (top-10 recall)")

    print("\n" + "-"*70)
    print("ANN (HNSW) Results:")
    print("-"*70)
    print(f"  Index build time:     {ann_results['build_time']:.2f}s")
    print(f"  Total search time:    {ann_results['total_search_time']:.4f}s")
    print(f"  Avg query time:       {ann_results['avg_query_time_ms']:.2f}ms")
    print(f"  Index size:           {ann_results['index_size']:,} vectors")

    print("\n" + "-"*70)
    print("Brute-Force (Cosine) Results:")
    print("-"*70)
    print(f"  Total search time:    {brute_results['total_search_time']:.4f}s")
    print(f"  Avg query time:       {brute_results['avg_query_time_ms']:.2f}ms")

    print("\n" + "-"*70)
    print("Performance Comparison:")
    print("-"*70)

    speedup = brute_results['avg_query_time_ms'] / ann_results['avg_query_time_ms']
    print(f"  Speedup:              {speedup:.1f}x faster")
    print(f"  ANN target:           <50ms per query")

    target_met = ann_results['avg_query_time_ms'] < 50.0
    status = "✓ PASS" if target_met else "✗ FAIL"
    print(f"  Target met:           {status}")

    print("\n" + "="*70)


@pytest.mark.slow
class TestANNPerformance:
    """
    Performance benchmark tests for ANN vector index.

    These tests are marked as 'slow' because they:
    - Create 100k synthetic memories
    - Run extensive benchmarks
    - Take several seconds to complete

    Run with: pytest tests/test_ann_performance.py -v -m slow
    """

    def test_ann_performance_100k_memories(self):
        """
        Benchmark ANN search performance on 100k memories.

        Validates acceptance criteria:
        - Search performance: <50ms for top-10 recall on 100k memories

        This is the primary performance requirement for the ANN index feature.
        """
        num_memories = 100_000
        num_queries = 100

        print(f"\nGenerating {num_memories:,} synthetic embeddings...")
        embeddings = generate_synthetic_embeddings(num_memories, dim=1024)

        print(f"Benchmarking ANN search...")
        ann_results = benchmark_ann_search(embeddings, num_queries)

        print(f"Benchmarking brute-force search...")
        brute_results = benchmark_bruteforce_search(embeddings, num_queries)

        # Print results
        print_benchmark_results(ann_results, brute_results, num_memories)

        # Assert performance requirement
        assert ann_results['avg_query_time_ms'] < 50.0, (
            f"ANN search took {ann_results['avg_query_time_ms']:.2f}ms, "
            f"exceeding 50ms target"
        )

        # Assert that ANN is actually faster than brute-force
        assert ann_results['avg_query_time_ms'] < brute_results['avg_query_time_ms'], (
            "ANN search should be faster than brute-force"
        )

    def test_ann_performance_10k_memories(self):
        """
        Benchmark ANN search performance on 10k memories.

        Smaller benchmark for faster iteration during development.
        Should be even faster than the 100k target.
        """
        num_memories = 10_000
        num_queries = 100

        print(f"\nGenerating {num_memories:,} synthetic embeddings...")
        embeddings = generate_synthetic_embeddings(num_memories, dim=1024)

        print(f"Benchmarking ANN search...")
        ann_results = benchmark_ann_search(embeddings, num_queries)

        print(f"\nResults for {num_memories:,} memories:")
        print(f"  Index build time:   {ann_results['build_time']:.2f}s")
        print(f"  Avg query time:     {ann_results['avg_query_time_ms']:.2f}ms")
        print(f"  Target:             <50ms per query")

        # Should be well under 50ms for 10k memories
        assert ann_results['avg_query_time_ms'] < 50.0, (
            f"ANN search took {ann_results['avg_query_time_ms']:.2f}ms, "
            f"exceeding 50ms target"
        )

    def test_ann_performance_with_different_dimensions(self):
        """
        Benchmark ANN search with different embedding dimensions.

        Tests both common dimensions:
        - 768-dim (Ollama nomic-embed-text)
        - 1024-dim (NVIDIA NIM baai/bge-m3)
        """
        num_memories = 10_000
        num_queries = 100

        results = {}

        for dim in [768, 1024]:
            print(f"\nBenchmarking {dim}-dim embeddings...")
            embeddings = generate_synthetic_embeddings(num_memories, dim=dim)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                db_path = tmp_path / f"ann_bench_{dim}.db"

                # Create ANN index and build it
                ann_index = ANNIndex(str(db_path), dim=dim, enable_persistence=False)

                build_start = time.time()
                ann_index.rebuild_from_embeddings(embeddings)
                build_time = time.time() - build_start

                # Generate random query embeddings
                query_embeddings = []
                for _ in range(num_queries):
                    vec = np.random.randn(dim).astype(np.float32)
                    vec = vec / np.linalg.norm(vec)
                    query_embeddings.append(vec.tolist())

                # Warm-up
                _ = ann_index.search(query_embeddings[0], k=10)

                # Benchmark
                search_start = time.time()
                for query_emb in query_embeddings:
                    _ = ann_index.search(query_emb, k=10)
                search_time = time.time() - search_start

                avg_query_time_ms = (search_time / num_queries) * 1000

                results[dim] = {
                    'build_time': build_time,
                    'avg_query_time_ms': avg_query_time_ms
                }

                print(f"  Build time:     {build_time:.2f}s")
                print(f"  Avg query time: {avg_query_time_ms:.2f}ms")

        # Both dimensions should meet performance target
        for dim, result in results.items():
            assert result['avg_query_time_ms'] < 50.0, (
                f"ANN search for {dim}-dim took {result['avg_query_time_ms']:.2f}ms, "
                f"exceeding 50ms target"
            )


if __name__ == "__main__":
    """
    Run benchmark as standalone script for development/profiling.

    Usage: python tests/test_ann_performance.py
    """
    print("Running ANN Performance Benchmark...")
    print("This will take a few seconds...\n")

    num_memories = 100_000
    num_queries = 100

    print(f"Generating {num_memories:,} synthetic embeddings...")
    embeddings = generate_synthetic_embeddings(num_memories, dim=1024)

    print(f"Benchmarking ANN search...")
    ann_results = benchmark_ann_search(embeddings, num_queries)

    print(f"Benchmarking brute-force search...")
    brute_results = benchmark_bruteforce_search(embeddings, num_queries)

    print_benchmark_results(ann_results, brute_results, num_memories)
