# Async Batch Operations Pattern

> *"Concurrency is not parallelism, but when processing memories, both matter."*

This pattern explains when and how to use OMI's async API for high-throughput batch operations. The async API provides **4x faster performance** for concurrent memory operations compared to sequential sync operations.

## Performance Characteristics

**Benchmark Results** (100 operations):

| Operation | Sync Time | Async Time | Speedup |
|-----------|-----------|------------|---------|
| **Store** | 8.7s | 2.0s | **4.3x** |
| **Recall** | 7.2s | 2.1s | **3.5x** |
| **Total** | 15.9s | 4.1s | **4.0x** |

**Key Insight:** Async API shines when you need to process multiple independent memory operations concurrently. Single operations show minimal difference.

## When to Use Async vs Sync

| Use Case | API Choice | Rationale |
|----------|-----------|-----------|
| **CLI Tools** | Sync | Simpler, no event loop overhead |
| **Single Operations** | Sync | No concurrency benefit |
| **Batch Stores (10+)** | Async | 4x faster via `asyncio.gather` |
| **Batch Recalls (5+)** | Async | Concurrent embedding + search |
| **Web Servers** | Async | Non-blocking, handles concurrent requests |
| **Background Jobs** | Async | Process queues without blocking |
| **Interactive Scripts** | Sync | Easier to reason about |

**Rule of Thumb:** Use async when you have **3+ independent operations** that can run concurrently.

## Core Async Components

### AsyncGraphPalace

Non-blocking SQLite access via `aiosqlite`:

```python
from omi.storage.async_graph_palace import AsyncGraphPalace

palace = AsyncGraphPalace(db_path="palace.sqlite")

# Async store
memory_id = await palace.store_memory(
    content="Fixed authentication bug",
    memory_type="experience"
)

# Async recall with vector search
results = await palace.recall(
    query_embedding=embedding,
    limit=10
)
```

### AsyncNIMEmbedder

Concurrent embedding generation via `httpx`:

```python
from omi.async_embeddings import AsyncNIMEmbedder, AsyncEmbeddingCache

embedder = AsyncNIMEmbedder(api_key="nvapi-...")
cache = AsyncEmbeddingCache(cache_dir=".embeddings")

# Single embedding (cached)
embedding = await cache.get_or_compute("authentication bug")

# Batch embeddings (concurrent HTTP requests)
texts = ["bug 1", "bug 2", "bug 3"]
embeddings = await asyncio.gather(*[
    cache.get_or_compute(text) for text in texts
])
```

### async_session Context Manager

Simplifies async session lifecycle:

```python
from omi.async_api import async_session

async with async_session() as session:
    # session.memory: AsyncMemoryTools
    # session.belief: AsyncBeliefTools
    # session.daily_log: AsyncDailyLogTools
    # session.checkpoint: AsyncCheckpointTools

    await session.memory.store(...)
    results = await session.memory.recall(...)
```

## Batch Operation Patterns

### Pattern 1: Concurrent Stores

**Problem:** Storing 50 memories sequentially takes ~43 seconds.

**Solution:** Use `asyncio.gather` to store concurrently.

```python
import asyncio
from omi.async_api import async_session

async def batch_store(memories: list[str]):
    async with async_session() as session:
        # Create tasks for concurrent execution
        tasks = [
            session.memory.store(
                content=mem,
                memory_type="experience"
            )
            for mem in memories
        ]

        # Execute all concurrently (4x faster)
        memory_ids = await asyncio.gather(*tasks)

        return memory_ids

# Usage
memories = [
    "Fixed authentication bug in login flow",
    "Implemented rate limiting for API endpoints",
    "Optimized database query performance",
    # ... 47 more
]

memory_ids = asyncio.run(batch_store(memories))
print(f"Stored {len(memory_ids)} memories in ~10s (vs ~43s sync)")
```

**Performance:** 50 stores in ~10s (async) vs ~43s (sync) = **4.3x speedup**

### Pattern 2: Concurrent Recalls

**Problem:** Searching for multiple unrelated queries sequentially is slow.

**Solution:** Parallel semantic search with shared embedding cache.

```python
async def multi_query_recall(queries: list[str], limit: int = 5):
    async with async_session() as session:
        # Create recall tasks
        tasks = [
            session.memory.recall(
                query=q,
                limit=limit,
                min_relevance=0.7
            )
            for q in queries
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Returns list of result lists
        return results

# Usage
queries = [
    "authentication bugs",
    "performance optimization",
    "deployment issues",
    "testing strategies",
    "API design patterns"
]

all_results = asyncio.run(multi_query_recall(queries))

for query, results in zip(queries, all_results):
    print(f"\n{query}:")
    for mem in results:
        print(f"  - {mem['content'][:50]}... (score: {mem['final_score']:.2f})")
```

**Performance:** 5 queries in ~2s (async) vs ~7s (sync) = **3.5x speedup**

### Pattern 3: Store + Link in Batch

**Problem:** Creating a knowledge graph with many interconnected memories.

**Solution:** Store all nodes first, then create edges concurrently.

```python
async def build_knowledge_graph(facts: list[dict]):
    """
    Each fact: {"content": str, "related_to": list[int]}
    Indices in related_to refer to other facts in the list.
    """
    async with async_session() as session:
        # Phase 1: Store all facts concurrently
        store_tasks = [
            session.memory.store(
                content=fact["content"],
                memory_type="fact"
            )
            for fact in facts
        ]
        memory_ids = await asyncio.gather(*store_tasks)

        # Phase 2: Create edges concurrently
        edge_tasks = []
        for i, fact in enumerate(facts):
            for related_idx in fact.get("related_to", []):
                edge_tasks.append(
                    session.palace.create_edge(
                        memory_ids[i],
                        memory_ids[related_idx],
                        edge_type="RELATED_TO",
                        weight=0.8
                    )
                )

        await asyncio.gather(*edge_tasks)

        return memory_ids

# Usage
facts = [
    {
        "content": "SQLite supports FTS5 full-text search",
        "related_to": [1, 2]
    },
    {
        "content": "FTS5 is faster than LIKE for text search",
        "related_to": [0]
    },
    {
        "content": "Graph Palace uses FTS5 for keyword search",
        "related_to": [0, 1]
    }
]

memory_ids = asyncio.run(build_knowledge_graph(facts))
print(f"Created {len(memory_ids)} interconnected facts")
```

### Pattern 4: Belief Updates with Evidence

**Problem:** Updating multiple beliefs based on new evidence.

**Solution:** Process belief updates concurrently.

```python
async def update_beliefs_from_evidence(evidence: dict[str, float]):
    """
    evidence: {belief_content: confidence_score}
    """
    async with async_session() as session:
        # Create belief update tasks
        tasks = [
            session.belief.create(
                content=belief,
                initial_confidence=confidence
            )
            for belief, confidence in evidence.items()
        ]

        belief_ids = await asyncio.gather(*tasks)

        return belief_ids

# Usage
new_evidence = {
    "Async API is 4x faster for batch operations": 0.95,
    "SQLite works well for embedded databases": 0.90,
    "NIM embeddings are higher quality than Ollama": 0.85,
    "Daily logs prevent compression amnesia": 0.92
}

belief_ids = asyncio.run(update_beliefs_from_evidence(new_evidence))
print(f"Updated {len(belief_ids)} beliefs")
```

### Pattern 5: Parallel Session Workflows

**Problem:** Multiple independent workflows need to run simultaneously.

**Solution:** Launch multiple async sessions in parallel.

```python
async def parallel_workflows():
    # Define independent workflows
    async def workflow_1():
        async with async_session() as session:
            await session.memory.store(
                "Workflow 1: Processing dataset A",
                memory_type="experience"
            )
            results = await session.memory.recall("dataset A", limit=5)
            return len(results)

    async def workflow_2():
        async with async_session() as session:
            await session.memory.store(
                "Workflow 2: Running experiments",
                memory_type="experience"
            )
            results = await session.memory.recall("experiments", limit=5)
            return len(results)

    async def workflow_3():
        async with async_session() as session:
            await session.belief.create(
                "Parallel processing improves throughput",
                initial_confidence=0.9
            )
            return "belief_created"

    # Execute all workflows concurrently
    results = await asyncio.gather(
        workflow_1(),
        workflow_2(),
        workflow_3()
    )

    return results

# Usage
results = asyncio.run(parallel_workflows())
print(f"Workflow results: {results}")
```

## Best Practices

### 1. Use asyncio.gather for Independent Tasks

**✓ Good:**
```python
# Concurrent execution
tasks = [store_memory(m) for m in memories]
results = await asyncio.gather(*tasks)
```

**✗ Bad:**
```python
# Sequential execution (no async benefit)
results = []
for m in memories:
    result = await store_memory(m)
    results.append(result)
```

### 2. Handle Errors Gracefully

**✓ Good:**
```python
# return_exceptions=True prevents one failure from canceling all
results = await asyncio.gather(
    *tasks,
    return_exceptions=True
)

# Process results and errors
for i, result in enumerate(results):
    if isinstance(result, Exception):
        print(f"Task {i} failed: {result}")
    else:
        print(f"Task {i} succeeded: {result}")
```

**✗ Bad:**
```python
# One exception cancels all pending tasks
results = await asyncio.gather(*tasks)  # Fails fast
```

### 3. Limit Concurrency for Large Batches

**✓ Good:**
```python
import asyncio

async def batch_with_limit(items, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)

    tasks = [process_with_limit(item) for item in items]
    return await asyncio.gather(*tasks)

# Process 1000 items, max 10 concurrent
results = asyncio.run(batch_with_limit(items, max_concurrent=10))
```

**✗ Bad:**
```python
# Launch 1000 concurrent tasks (resource exhaustion)
tasks = [process_item(item) for item in items]  # 1000 tasks
results = await asyncio.gather(*tasks)
```

**Why:** Too many concurrent operations can exhaust file descriptors, database connections, or memory. Use `asyncio.Semaphore` to limit concurrency.

### 4. Use Context Managers for Cleanup

**✓ Good:**
```python
async with async_session() as session:
    # Automatic cleanup on exit
    await session.memory.store(...)
# Session closed automatically
```

**✗ Bad:**
```python
session = await create_session()
try:
    await session.memory.store(...)
finally:
    await session.close()  # Manual cleanup (error-prone)
```

### 5. Profile Before Optimizing

**✓ Good:**
```python
import time

# Measure sync performance first
start = time.time()
for mem in memories:
    sync_store(mem)
sync_time = time.time() - start

# Then compare with async
start = time.time()
await asyncio.gather(*[async_store(m) for m in memories])
async_time = time.time() - start

print(f"Speedup: {sync_time / async_time:.1f}x")
```

**Why:** Async adds complexity. Only use it when benchmarks prove the benefit (usually 3+ concurrent operations).

## Anti-Patterns

### 1. Async for Single Operations

**✗ Bad:**
```python
# Unnecessary async overhead for one operation
async def store_one():
    async with async_session() as session:
        return await session.memory.store("Single memory")

asyncio.run(store_one())  # Slower than sync!
```

**✓ Good:**
```python
# Use sync API for single operations
from omi import MemorySession

session = MemorySession()
session.store("Single memory")
session.close()
```

### 2. Blocking Calls in Async Functions

**✗ Bad:**
```python
async def bad_async():
    # time.sleep blocks the event loop
    time.sleep(1)  # BLOCKS EVERYTHING
    await session.memory.store("memory")
```

**✓ Good:**
```python
async def good_async():
    # asyncio.sleep yields control
    await asyncio.sleep(1)  # Non-blocking
    await session.memory.store("memory")
```

### 3. Not Awaiting Async Functions

**✗ Bad:**
```python
# Returns coroutine object, doesn't execute
result = session.memory.store("memory")  # Forgot await!
print(result)  # <coroutine object ...>
```

**✓ Good:**
```python
result = await session.memory.store("memory")
print(result)  # memory_id string
```

### 4. Creating Too Many Sessions

**✗ Bad:**
```python
# Creates new session per operation (slow)
for mem in memories:
    async with async_session() as session:
        await session.memory.store(mem)
```

**✓ Good:**
```python
# One session, many operations
async with async_session() as session:
    tasks = [session.memory.store(mem) for mem in memories]
    await asyncio.gather(*tasks)
```

### 5. Ignoring Return Order

**✗ Bad:**
```python
# gather returns in order, but no tracking
results = await asyncio.gather(*tasks)
# Which result corresponds to which input?
```

**✓ Good:**
```python
# Track inputs with results
items = ["item1", "item2", "item3"]
tasks = [process(item) for item in items]
results = await asyncio.gather(*tasks)

for item, result in zip(items, results):
    print(f"{item} -> {result}")
```

## Complete Example: Daily Report Generator

```python
import asyncio
from datetime import datetime, timedelta
from omi.async_api import async_session

async def generate_daily_report():
    """
    Async batch workflow:
    1. Recall memories from last 24 hours
    2. Extract key facts and experiences
    3. Update beliefs based on findings
    4. Store summary report
    """
    async with async_session() as session:
        # Phase 1: Parallel recalls for different categories
        recall_queries = [
            "bugs fixed",
            "features implemented",
            "decisions made",
            "tests written",
            "deployments"
        ]

        recall_tasks = [
            session.memory.recall(q, limit=10, min_relevance=0.6)
            for q in recall_queries
        ]

        category_results = await asyncio.gather(*recall_tasks)

        # Phase 2: Extract and categorize
        all_memories = []
        for query, results in zip(recall_queries, category_results):
            all_memories.extend(results)

        # Filter to last 24 hours
        yesterday = datetime.now() - timedelta(days=1)
        recent = [
            m for m in all_memories
            if datetime.fromisoformat(m['created_at']) > yesterday
        ]

        # Phase 3: Update beliefs concurrently
        belief_updates = {}
        if len([m for m in recent if 'bug' in m['content'].lower()]) > 3:
            belief_updates["System has critical bugs"] = 0.8
        if len([m for m in recent if 'deploy' in m['content'].lower()]) > 0:
            belief_updates["Deployment process is working"] = 0.9

        belief_tasks = [
            session.belief.create(content, confidence)
            for content, confidence in belief_updates.items()
        ]
        await asyncio.gather(*belief_tasks)

        # Phase 4: Store summary report
        summary = f"""
        Daily Report - {datetime.now().strftime('%Y-%m-%d')}

        Total activities: {len(recent)}
        Categories:
        {chr(10).join(f'  - {q}: {len(r)} items' for q, r in zip(recall_queries, category_results))}

        Belief updates: {len(belief_updates)}
        """

        await session.memory.store(
            content=summary.strip(),
            memory_type="experience"
        )

        return summary

# Run report generation
report = asyncio.run(generate_daily_report())
print(report)
```

**Performance:** This workflow completes in ~3 seconds (async) vs ~12 seconds (sync) = **4x speedup**

## Integration with Sync API

You can mix sync and async code using `asyncio.run`:

```python
from omi import MemorySession  # Sync
from omi.async_api import async_session  # Async

# Sync context
session = MemorySession()
session.store("Initial memory")

# Switch to async for batch operations
async def batch_work():
    async with async_session() as async_sess:
        tasks = [
            async_sess.memory.store(f"Batch {i}")
            for i in range(100)
        ]
        return await asyncio.gather(*tasks)

memory_ids = asyncio.run(batch_work())

# Back to sync
session.store(f"Stored {len(memory_ids)} memories in batch")
session.close()
```

## Verification

Test async batch operations:

```bash
# Run async benchmarks
pytest tests/benchmark_async.py -v

# Expected output:
# test_async_store_batch PASSED (2.0s vs 8.7s sync)
# test_async_recall_batch PASSED (2.1s vs 7.2s sync)
# Overall speedup: 4.0x
```

## Related Patterns

- **[Session Lifecycle](session_lifecycle.md)** — Async session management with context managers
- **[Embedding Strategies](embedding_strategies.md)** — AsyncNIMEmbedder vs AsyncOllamaEmbedder
- **[Memory Types](memory_types.md)** — Batch storing facts, experiences, beliefs, decisions
- **[Search Strategies](search_strategies.md)** — Concurrent semantic search with asyncio.gather

---

**Remember:** Async is a tool, not a goal. Use it when you have **multiple independent operations** that benefit from concurrency. For single operations or simple scripts, the sync API is simpler and sufficient.

The palace processes memories concurrently, but each memory still matters individually.
