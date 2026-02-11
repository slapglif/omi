"""
benchmark_async.py - Performance Benchmark for Async vs Sync OMI Operations

Compares sync vs async performance for batch memory operations:
- 10 concurrent store operations
- 10 concurrent recall operations

Expected: Async should be at least 2x faster than sync for batch operations

Issue: https://github.com/slapglif/omi/issues/4
"""
import asyncio
import time
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Sync imports
from omi.api import MemoryTools
from omi.storage.graph_palace import GraphPalace
from omi.embeddings import OllamaEmbedder, EmbeddingCache

# Async imports
from omi.async_api import AsyncMemoryTools
from omi.storage.async_graph_palace import AsyncGraphPalace
from omi.async_embeddings import AsyncNIMEmbedder, AsyncEmbeddingCache


def create_mock_sync_embedder():
    """Create mock sync embedder that returns consistent embeddings."""
    mock = MagicMock()

    def embed_with_delay(text):
        # Simulate realistic embedding generation time (API call ~10ms)
        time.sleep(0.01)
        return [0.1] * 1024

    mock.embed.side_effect = embed_with_delay
    mock.embed_batch.return_value = [[0.1] * 1024]

    def mock_similarity(e1, e2):
        if e1 == e2:
            return 1.0
        return 0.85

    mock.similarity = mock_similarity
    return mock


def create_mock_async_embedder():
    """Create mock async embedder that returns consistent embeddings."""
    mock = AsyncMock()

    async def embed_with_delay(text):
        # Simulate realistic async embedding generation (API call ~10ms)
        await asyncio.sleep(0.01)
        return [0.1] * 1024

    mock.embed.side_effect = embed_with_delay
    mock.embed_batch.return_value = [[0.1] * 1024]

    def mock_similarity(e1, e2):
        if e1 == e2:
            return 1.0
        return 0.85

    mock.similarity = mock_similarity
    return mock


def benchmark_sync_batch_operations(num_operations=10):
    """Benchmark sync batch operations (sequential)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        base_path = tmp_path / "omi_sync"
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / "palace.sqlite"
        cache_dir = base_path / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create sync components
        embedder = create_mock_sync_embedder()
        cache = EmbeddingCache(cache_dir, embedder)
        palace = GraphPalace(db_path)
        memory_tools = MemoryTools(palace, embedder, cache)

        # Benchmark store operations
        store_start = time.time()
        stored_ids = []
        for i in range(num_operations):
            content = f"Sync memory {i}: Important information about topic {i}"
            memory_id = memory_tools.store(content, memory_type="experience")
            stored_ids.append(memory_id)
        store_time = time.time() - store_start

        # Benchmark recall operations
        recall_start = time.time()
        for i in range(num_operations):
            query = f"topic {i}"
            # Embed query first, then pass to palace recall
            query_embedding = cache.get_or_compute(query)
            results = palace.recall(query_embedding, limit=5)
        recall_time = time.time() - recall_start

        total_time = store_time + recall_time

        # Cleanup
        palace.close()

        return {
            'store_time': store_time,
            'recall_time': recall_time,
            'total_time': total_time,
            'operations': num_operations
        }


async def benchmark_async_batch_operations(num_operations=10):
    """Benchmark async batch operations (concurrent)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        base_path = tmp_path / "omi_async"
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / "palace.sqlite"
        cache_dir = base_path / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create async components
        embedder = create_mock_async_embedder()
        cache = AsyncEmbeddingCache(cache_dir, embedder)

        async with AsyncGraphPalace(db_path) as palace:
            memory_tools = AsyncMemoryTools(palace, embedder, cache)

            # Benchmark concurrent store operations
            store_start = time.time()

            async def store_memory(i):
                content = f"Async memory {i}: Important information about topic {i}"
                return await memory_tools.store(content, memory_type="experience")

            stored_ids = await asyncio.gather(*[store_memory(i) for i in range(num_operations)])
            store_time = time.time() - store_start

            # Benchmark concurrent recall operations
            recall_start = time.time()

            async def recall_memory(i):
                query = f"topic {i}"
                return await memory_tools.recall(query, limit=5)

            results = await asyncio.gather(*[recall_memory(i) for i in range(num_operations)])
            recall_time = time.time() - recall_start

            total_time = store_time + recall_time

            return {
                'store_time': store_time,
                'recall_time': recall_time,
                'total_time': total_time,
                'operations': num_operations
            }


def print_results(sync_results, async_results):
    """Print benchmark results in a readable format."""
    print("\n" + "="*70)
    print("ASYNC vs SYNC BATCH OPERATIONS BENCHMARK")
    print("="*70)

    print(f"\nNumber of operations: {sync_results['operations']}")

    print("\n" + "-"*70)
    print("SYNC (Sequential) Results:")
    print("-"*70)
    print(f"  Store operations:  {sync_results['store_time']:.4f}s")
    print(f"  Recall operations: {sync_results['recall_time']:.4f}s")
    print(f"  Total time:        {sync_results['total_time']:.4f}s")

    print("\n" + "-"*70)
    print("ASYNC (Concurrent) Results:")
    print("-"*70)
    print(f"  Store operations:  {async_results['store_time']:.4f}s")
    print(f"  Recall operations: {async_results['recall_time']:.4f}s")
    print(f"  Total time:        {async_results['total_time']:.4f}s")

    print("\n" + "-"*70)
    print("Performance Comparison:")
    print("-"*70)

    store_speedup = sync_results['store_time'] / async_results['store_time']
    recall_speedup = sync_results['recall_time'] / async_results['recall_time']
    total_speedup = sync_results['total_time'] / async_results['total_time']

    print(f"  Store speedup:     {store_speedup:.2f}x faster")
    print(f"  Recall speedup:    {recall_speedup:.2f}x faster")
    print(f"  Total speedup:     {total_speedup:.2f}x faster")

    print("\n" + "="*70)

    # Acceptance criteria check
    if total_speedup >= 2.0:
        print("✓ PASS: Async is at least 2x faster than sync")
        print("="*70)
        return True
    else:
        print(f"✗ FAIL: Async is only {total_speedup:.2f}x faster (expected >= 2.0x)")
        print("="*70)
        return False


def main():
    """Run the benchmark."""
    print("\nRunning sync benchmark...")
    sync_results = benchmark_sync_batch_operations(num_operations=10)

    print("Running async benchmark...")
    async_results = asyncio.run(benchmark_async_batch_operations(num_operations=10))

    # Print and verify results
    passed = print_results(sync_results, async_results)

    # Exit with appropriate code
    import sys
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
