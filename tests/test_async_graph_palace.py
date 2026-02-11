"""
Unit tests for AsyncGraphPalace - Tier 3 Storage for OMI

Tests cover:
- CRUD operations (memories, edges)
- Semantic search with cosine similarity
- Recency decay scoring
- Graph traversal (BFS)
- Centrality calculation
- Full-text search (FTS5)
- Edge relationships
- Concurrent writes (WAL mode)
- Performance benchmarks
- Async/await patterns
"""

import unittest
import tempfile
import time
import math
import hashlib
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.storage.async_graph_palace import AsyncGraphPalace, Memory, Edge


class TestAsyncGraphPalace(unittest.IsolatedAsyncioTestCase):
    """Test suite for AsyncGraphPalace."""

    async def asyncSetUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_palace.sqlite"
        self.palace = AsyncGraphPalace(self.db_path)

    async def asyncTearDown(self):
        """Clean up test database."""
        await self.palace.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_embedding(self, dim: int = 1024) -> List[float]:
        """Generate a random normalized embedding vector."""
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _generate_similar_embedding(self, base: List[float], similarity: float = 0.9) -> List[float]:
        """Generate an embedding similar to base vector."""
        noise = np.random.randn(len(base))
        noise = noise / np.linalg.norm(noise)
        similar = np.array(base) * similarity + noise * (1 - similarity)
        similar = similar / np.linalg.norm(similar)
        return similar.tolist()

    # ==================== CRUD Operations ====================

    async def test_store_and_get_memory(self):
        """Test storing and retrieving a memory."""
        embedding = self._generate_embedding()

        memory_id = await self.palace.store_memory(
            content="Test memory content",
            embedding=embedding,
            memory_type="fact",
            confidence=0.95
        )

        self.assertIsNotNone(memory_id)
        self.assertTrue(len(memory_id) > 0)

        # Retrieve and verify
        memory = await self.palace.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, "Test memory content")
        self.assertEqual(memory.memory_type, "fact")
        self.assertAlmostEqual(memory.confidence, 0.95, places=4)
        self.assertEqual(memory.access_count, 1)  # Retrieved once

    async def test_store_memory_no_embedding(self):
        """Test storing a memory without embedding."""
        memory_id = await self.palace.store_memory(
            content="Memory without embedding",
            memory_type="experience"
        )

        memory = await self.palace.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertIsNone(memory.embedding)
        self.assertEqual(memory.memory_type, "experience")

    async def test_invalid_memory_type(self):
        """Test invalid memory type raises error."""
        with self.assertRaises(ValueError):
            await self.palace.store_memory(
                content="Invalid type",
                memory_type="invalid_type"
            )

    async def test_invalid_confidence(self):
        """Test invalid confidence value raises error."""
        with self.assertRaises(ValueError):
            await self.palace.store_memory(
                content="Invalid confidence",
                memory_type="belief",
                confidence=1.5
            )

        with self.assertRaises(ValueError):
            await self.palace.store_memory(
                content="Invalid confidence",
                memory_type="belief",
                confidence=-0.1
            )

    async def test_update_embedding(self):
        """Test updating memory embedding."""
        old_embedding = self._generate_embedding()
        new_embedding = self._generate_embedding()

        memory_id = await self.palace.store_memory(
            content="Test memory",
            embedding=old_embedding,
            memory_type="fact"
        )

        result = await self.palace.update_embedding(memory_id, new_embedding)
        self.assertTrue(result)

        memory = await self.palace.get_memory(memory_id)
        self.assertTrue(np.allclose(memory.embedding, new_embedding, rtol=1e-5))

    async def test_delete_memory(self):
        """Test deleting a memory."""
        memory_id = await self.palace.store_memory(
            content="To be deleted",
            memory_type="experience"
        )

        result = await self.palace.delete_memory(memory_id)
        self.assertTrue(result)

        memory = await self.palace.get_memory(memory_id)
        self.assertIsNone(memory)

    async def test_get_nonexistent_memory(self):
        """Test retrieving non-existent memory."""
        memory = await self.palace.get_memory("nonexistent-id")
        self.assertIsNone(memory)

    # ==================== Edge Relationships ====================

    async def test_create_edge(self):
        """Test creating an edge between memories."""
        memory1_id = await self.palace.store_memory(
            content="Memory 1",
            memory_type="fact"
        )
        memory2_id = await self.palace.store_memory(
            content="Memory 2",
            memory_type="fact"
        )

        edge_id = await self.palace.create_edge(
            source_id=memory1_id,
            target_id=memory2_id,
            edge_type="RELATED_TO",
            strength=0.8
        )

        self.assertIsNotNone(edge_id)

        # Get edges and verify
        edges = await self.palace.get_edges(memory1_id)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_id, memory1_id)
        self.assertEqual(edges[0].target_id, memory2_id)
        self.assertEqual(edges[0].edge_type, "RELATED_TO")
        self.assertAlmostEqual(edges[0].strength, 0.8, places=4)

    async def test_invalid_edge_type(self):
        """Test invalid edge type raises error."""
        with self.assertRaises(ValueError):
            await self.palace.create_edge("id1", "id2", "INVALID_TYPE")

    async def test_get_neighbors(self):
        """Test getting neighboring memories."""
        memory1_id = await self.palace.store_memory(content="Central memory", memory_type="fact")
        memory2_id = await self.palace.store_memory(content="Neighbor 1", memory_type="fact")
        memory3_id = await self.palace.store_memory(content="Neighbor 2", memory_type="fact")

        await self.palace.create_edge(memory1_id, memory2_id, "RELATED_TO")
        await self.palace.create_edge(memory3_id, memory1_id, "SUPPORTS")

        neighbors = await self.palace.get_neighbors(memory1_id)
        self.assertEqual(len(neighbors), 2)

        neighbor_ids = {n.id for n in neighbors}
        self.assertEqual(neighbor_ids, {memory2_id, memory3_id})

    async def test_delete_edge(self):
        """Test deleting an edge."""
        memory1_id = await self.palace.store_memory(content="Source", memory_type="fact")
        memory2_id = await self.palace.store_memory(content="Target", memory_type="fact")

        edge_id = await self.palace.create_edge(memory1_id, memory2_id, "RELATED_TO")
        result = await self.palace.delete_edge(edge_id)

        self.assertTrue(result)

        edges = await self.palace.get_edges(memory1_id)
        self.assertEqual(len(edges), 0)

    # ==================== Semantic Search ====================

    async def test_recall_basic(self):
        """Test basic semantic search/recall."""
        base_embedding = self._generate_embedding()
        similar_embedding = self._generate_similar_embedding(base_embedding, similarity=0.95)
        different_embedding = self._generate_embedding()

        # Store memories
        await self.palace.store_memory(
            content="Similar memory",
            embedding=similar_embedding,
            memory_type="fact"
        )
        await self.palace.store_memory(
            content="Different memory",
            embedding=different_embedding,
            memory_type="fact"
        )

        # Search with base query
        results = await self.palace.recall(query_embedding=base_embedding, limit=10, min_relevance=0.5)

        self.assertGreater(len(results), 0)
        # First result should be the similar one
        contents = [r[0].content for r in results]
        self.assertIn("Similar memory", contents)

    async def test_recall_min_relevance(self):
        """Test recall with minimum relevance threshold."""
        base = self._generate_embedding()
        v1 = self._generate_similar_embedding(base, 0.95)
        v2 = self._generate_similar_embedding(base, 0.6)
        v3 = self._generate_embedding()  # Very different

        await self.palace.store_memory(content="High relevance", embedding=v1, memory_type="fact")
        await self.palace.store_memory(content="Medium relevance", embedding=v2, memory_type="fact")
        await self.palace.store_memory(content="Low relevance", embedding=v3, memory_type="fact")

        # High threshold
        results = await self.palace.recall(base, limit=10, min_relevance=0.8)
        contents = [r[0].content for r in results]
        self.assertIn("High relevance", contents)
        self.assertNotIn("Low relevance", contents)

    async def test_recal_relevance_score_range(self):
        """Test that relevance scores are in valid range."""
        embedding = self._generate_embedding()
        await self.palace.store_memory(content="Test", embedding=embedding, memory_type="fact")

        results = await self.palace.recall(query_embedding=embedding, limit=10, min_relevance=0.0)

        for memory, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    # ==================== Full-Text Search ====================

    async def test_full_text_search(self):
        """Test FTS5 full-text search."""
        await self.palace.store_memory(content="Python is great for machine learning", memory_type="fact")
        await self.palace.store_memory(content="SQLite works well for embedded systems", memory_type="fact")
        await self.palace.store_memory(content="Cats sleep 15 hours a day", memory_type="fact")

        results = await self.palace.full_text_search("machine learning", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Python is great for machine learning")

    async def test_full_text_search_multiple_words(self):
        """Test FTS5 with multi-word query."""
        await self.palace.store_memory(content="Neural networks for natural language processing", memory_type="fact")
        await self.palace.store_memory(content="Graph databases store relationships", memory_type="fact")

        results = await self.palace.full_text_search("neural language", limit=5)
        self.assertEqual(len(results), 1)

    # ==================== Centrality ====================

    async def test_centrality_basic(self):
        """Test basic centrality calculation."""
        memory_id = await self.palace.store_memory(
            content="Central node",
            memory_type="fact"
        )

        # Add neighbors
        for i in range(3):
            neighbor_id = await self.palace.store_memory(content=f"Neighbor {i}", memory_type="fact")
            await self.palace.create_edge(memory_id, neighbor_id, "RELATED_TO")

        centrality = await self.palace.get_centrality(memory_id)
        self.assertGreater(centrality, 0.0)
        self.assertLessEqual(centrality, 1.0)

    async def test_centrality_with_access_count(self):
        """Test centrality increases with access count."""
        memory_id = await self.palace.store_memory(content="Popular memory", memory_type="fact")

        # Access multiple times
        for _ in range(5):
            await self.palace.get_memory(memory_id)

        centrality1 = await self.palace.get_centrality(memory_id)

        # More accesses
        for _ in range(10):
            await self.palace.get_memory(memory_id)

        centrality2 = await self.palace.get_centrality(memory_id)

        self.assertGreater(centrality2, centrality1)

    async def test_top_central(self):
        """Test getting top central memories."""
        # Create memories with different centralities
        hub_id = await self.palace.store_memory(content="Hub memory", memory_type="fact")

        for i in range(5):
            leaf_id = await self.palace.store_memory(content=f"Leaf {i}", memory_type="fact")
            await self.palace.create_edge(hub_id, leaf_id, "RELATED_TO")

        await self.palace.get_memory(hub_id)  # Access it

        top = await self.palace.get_top_central(limit=5)
        self.assertGreater(len(top), 0)

    # ==================== Graph Traversal ====================

    async def test_get_connected_depth_1(self):
        """Test BFS traversal depth 1."""
        node_a = await self.palace.store_memory(content="Node A", memory_type="fact")
        node_b = await self.palace.store_memory(content="Node B", memory_type="fact")
        node_c = await self.palace.store_memory(content="Node C", memory_type="fact")

        await self.palace.create_edge(node_a, node_b, "RELATED_TO")
        await self.palace.create_edge(node_a, node_c, "RELATED_TO")

        connected = await self.palace.get_connected(node_a, depth=1)
        self.assertEqual(len(connected), 2)

    async def test_get_connected_depth_2(self):
        """Test BFS traversal depth 2."""
        # A -> B -> C (depth 2 from A)
        node_a = await self.palace.store_memory(content="Node A", memory_type="fact")
        node_b = await self.palace.store_memory(content="Node B", memory_type="fact")
        node_c = await self.palace.store_memory(content="Node C", memory_type="fact")

        await self.palace.create_edge(node_a, node_b, "RELATED_TO")
        await self.palace.create_edge(node_b, node_c, "RELATED_TO")

        connected_depth1 = await self.palace.get_connected(node_a, depth=1)
        self.assertEqual(len(connected_depth1), 1)  # Only B

        connected_depth2 = await self.palace.get_connected(node_a, depth=2)
        self.assertEqual(len(connected_depth2), 2)  # B and C

    async def test_get_connected_depth_0(self):
        """Test BFS with depth 0."""
        node_a = await self.palace.store_memory(content="Node A", memory_type="fact")
        _ = await self.palace.store_memory(content="Node B", memory_type="fact")

        await self.palace.create_edge(node_a, _, "RELATED_TO")

        connected = await self.palace.get_connected(node_a, depth=0)
        self.assertEqual(len(connected), 0)

    # ==================== Special Relationships ====================

    async def test_find_contradictions(self):
        """Test finding contradicting memories."""
        belief_id = await self.palace.store_memory(content="I should always work", memory_type="belief")
        contradiction_id = await self.palace.store_memory(content="I should rest", memory_type="belief")

        await self.palace.create_edge(belief_id, contradiction_id, "CONTRADICTS", strength=0.9)

        contradictions = await self.palace.find_contradictions(belief_id)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].id, contradiction_id)

    async def test_get_supporting_evidence(self):
        """Test getting supporting evidence."""
        belief_id = await self.palace.store_memory(content="AI is transformative", memory_type="belief")
        evidence_id = await self.palace.store_memory(content="AI adoption rose 50%", memory_type="fact")

        await self.palace.create_edge(belief_id, evidence_id, "SUPPORTS", strength=0.85)

        evidence = await self.palace.get_supporting_evidence(belief_id)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].id, evidence_id)

    # ==================== Statistics ====================

    async def test_get_stats(self):
        """Test database statistics."""
        # Populate with sample data
        for i in range(5):
            await self.palace.store_memory(content=f"Memory {i}", memory_type="fact")

        stats = await self.palace.get_stats()

        self.assertEqual(stats["memory_count"], 5)
        self.assertIn("fact", stats["type_distribution"])
        self.assertEqual(stats["type_distribution"]["fact"], 5)

    async def test_get_stats_with_edges(self):
        """Test stats with edges."""
        m1 = await self.palace.store_memory(content="Source", memory_type="fact")
        m2 = await self.palace.store_memory(content="Target", memory_type="fact")
        await self.palace.create_edge(m1, m2, "RELATED_TO")

        stats = await self.palace.get_stats()
        self.assertEqual(stats["edge_count"], 1)

    # ==================== Performance Benchmarks ====================

    async def test_performance_recall_1000_memories(self):
        """Benchmark: 1000 memories searchable in <500ms."""
        # Create 1000 memories with random embeddings
        import random
        random.seed(42)

        print("\n  Creating 1000 memories...")
        for i in range(1000):
            embedding = [random.random() for _ in range(1024)]
            # Normalize
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]

            await self.palace.store_memory(
                content=f"Memory {i}: This is a test memory for performance benchmarking",
                embedding=embedding,
                memory_type="fact"
            )

        print("  Benchmarking recall...")
        query_embedding = [random.random() for _ in range(1024)]
        norm = sum(x*x for x in query_embedding) ** 0.5
        query_embedding = [x/norm for x in query_embedding]

        start = time.time()
        results = await self.palace.recall(query_embedding, limit=10, min_relevance=0.0)
        elapsed = time.time() - start

        print(f"  Recall 1000 memories: {elapsed*1000:.2f}ms, found {len(results)} results")

        # Target: <500ms
        self.assertLess(elapsed, 0.5, f"Query too slow: {elapsed*1000:.2f}ms")

    async def test_performance_graph_traversal_10k_edges(self):
        """Benchmark: Graph queries scale to 10k edges."""
        import random
        random.seed(42)

        print("\n  Creating 500 nodes...")
        node_ids = []
        for i in range(500):
            node_id = await self.palace.store_memory(content=f"Node {i}", memory_type="fact")
            node_ids.append(node_id)

        print("  Creating ~10k edges...")
        edge_count = 0
        for i in range(len(node_ids)):
            # Each node connects to ~20 random other nodes
            targets = random.sample(node_ids, min(20, len(node_ids)))
            for target in targets:
                if target != node_ids[i]:
                    await self.palace.create_edge(node_ids[i], target, "RELATED_TO")
                    edge_count += 1

        print(f"  Created {edge_count} edges")

        # Test centrality calculation
        print("  Benchmarking centrality for 100 nodes...")
        start = time.time()
        sample_nodes = random.sample(node_ids, min(100, len(node_ids)))
        for node_id in sample_nodes:
            _ = await self.palace.get_centrality(node_id)
        elapsed = time.time() - start

        print(f"  Centrality for 100 nodes: {elapsed*1000:.2f}ms")

        # Should complete in reasonable time (let's say <5s for 100 nodes)
        self.assertLess(elapsed, 5.0)

        # Test graph traversal
        print("  Benchmarking BFS traversal...")
        start = time.time()
        connected = await self.palace.get_connected(node_ids[0], depth=2)
        elapsed = time.time() - start

        print(f"  BFS depth 2 from central node: {elapsed*1000:.2f}ms, found {len(connected)} nodes")
        self.assertGreater(len(connected), 0)

    async def test_performance_wal_concurrent_writes(self):
        """Test that WAL mode allows concurrent writes."""
        import threading
        import time

        errors = []
        success_count = [0]
        db_path = self.db_path  # Capture for use in threads

        def writer_thread(thread_id, count):
            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                async def do_writes():
                    # Each thread needs its own AsyncGraphPalace instance
                    async with AsyncGraphPalace(db_path) as palace:
                        for i in range(count):
                            # Retry logic for occasional database locks
                            max_retries = 3
                            for retry in range(max_retries):
                                try:
                                    memory_id = await palace.store_memory(
                                        content=f"Thread {thread_id} Memory {i}",
                                        memory_type="experience"
                                    )
                                    success_count[0] += 1
                                    break
                                except Exception as e:
                                    if "locked" in str(e) and retry < max_retries - 1:
                                        await asyncio.sleep(0.01 * (retry + 1))  # Exponential backoff
                                    else:
                                        raise

                loop.run_until_complete(do_writes())
                loop.close()
            except Exception as e:
                errors.append((thread_id, str(e)))

        print("\n  Testing concurrent writes with WAL mode...")
        threads = []
        num_threads = 4
        writes_per_thread = 25

        start = time.time()
        for i in range(num_threads):
            t = threading.Thread(target=writer_thread, args=(i, writes_per_thread))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        elapsed = time.time() - start

        print(f"  {num_threads} threads x {writes_per_thread} writes = {success_count[0]} in {elapsed*1000:.2f}ms")

        if errors:
            print(f"  Errors: {errors}")

        # WAL mode should enable most concurrent writes to succeed
        # Allow for a small failure rate due to timing
        self.assertEqual(len(errors), 0, f"Concurrent writes failed: {errors}")
        self.assertEqual(success_count[0], num_threads * writes_per_thread)

    # ==================== Integrity ====================

    async def test_content_hash(self):
        """Test content hash is computed correctly."""
        memory_id = await self.palace.store_memory(content="Test content", memory_type="fact")
        memory = await self.palace.get_memory(memory_id)

        expected_hash = hashlib.sha256("Test content".encode()).hexdigest()
        self.assertEqual(memory.content_hash, expected_hash)

    async def test_access_count_increment(self):
        """Test access count increments on get."""
        memory_id = await self.palace.store_memory(content="Popular", memory_type="fact")

        memory = await self.palace.get_memory(memory_id)
        self.assertEqual(memory.access_count, 1)

        memory = await self.palace.get_memory(memory_id)
        self.assertEqual(memory.access_count, 2)

    async def test_last_accessed_update(self):
        """Test last_accessed timestamp updates."""
        memory_id = await self.palace.store_memory(content="Test", memory_type="fact")
        first_access = (await self.palace.get_memory(memory_id)).last_accessed

        await asyncio.sleep(0.01)  # Small delay
        second_access = (await self.palace.get_memory(memory_id)).last_accessed

        self.assertGreater(second_access, first_access)

    async def test_context_manager(self):
        """Test async context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"

            async with AsyncGraphPalace(db_path) as palace:
                memory_id = await palace.store_memory(content="In context", memory_type="fact")
                memory = await palace.get_memory(memory_id)
                self.assertIsNotNone(memory)

    async def test_vacuum(self):
        """Test vacuum operation."""
        # Should not raise error
        await self.palace.vacuum()

    async def test_update_memory_content(self):
        """Test updating memory content."""
        memory_id = await self.palace.store_memory(content="Original content", memory_type="fact")

        result = await self.palace.update_memory_content(memory_id, "Updated content")
        self.assertTrue(result)

        memory = await self.palace.get_memory(memory_id)
        self.assertEqual(memory.content, "Updated content")

        # Check hash was updated
        expected_hash = hashlib.sha256("Updated content".encode()).hexdigest()
        self.assertEqual(memory.content_hash, expected_hash)

    async def test_get_compression_stats(self):
        """Test compression statistics calculation."""
        # Create some test memories
        for i in range(10):
            await self.palace.store_memory(
                content=f"Test memory {i}" * 10,
                memory_type="fact"
            )

        stats = await self.palace.get_compression_stats()

        self.assertEqual(stats["total_memories"], 10)
        self.assertGreater(stats["total_chars"], 0)
        self.assertGreater(stats["estimated_tokens"], 0)
        self.assertIn("fact", stats["memories_by_type"])

    async def test_get_memories_before(self):
        """Test getting memories before a threshold."""
        # Create memories with timestamps
        past_time = datetime.now() - timedelta(days=10)

        memory_id = await self.palace.store_memory(content="Old memory", memory_type="fact")

        # Query memories before now
        memories = await self.palace.get_memories_before(datetime.now() + timedelta(days=1))

        self.assertGreater(len(memories), 0)
        self.assertTrue(any(m.id == memory_id for m in memories))


class TestMemoryDataclass(unittest.TestCase):
    """Test Memory dataclass."""

    def test_memory_to_dict(self):
        """Test Memory.to_dict method."""
        memory = Memory(
            id="test-id",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            memory_type="fact",
            confidence=0.9
        )

        d = memory.to_dict()
        self.assertEqual(d["id"], "test-id")
        self.assertEqual(d["content"], "Test content")
        self.assertEqual(d["memory_type"], "fact")


class TestEdgeTypes(unittest.IsolatedAsyncioTestCase):
    """Test all valid edge types."""

    async def asyncSetUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "edges.sqlite"
        self.palace = AsyncGraphPalace(self.db_path)

    async def asyncTearDown(self):
        await self.palace.close()
        import shutil
        if self.db_path.parent.exists():
            shutil.rmtree(str(self.db_path.parent), ignore_errors=True)

    async def test_all_edge_types(self):
        """Test all valid edge types can be created."""
        source = await self.palace.store_memory(content="Source", memory_type="fact")

        valid_types = ["SUPPORTS", "CONTRADICTS", "RELATED_TO", "DEPENDS_ON", "POSTED", "DISCUSSED"]

        for edge_type in valid_types:
            with self.subTest(edge_type=edge_type):
                target = await self.palace.store_memory(content=f"Target {edge_type}", memory_type="fact")
                edge_id = await self.palace.create_edge(source, target, edge_type, strength=0.5)
                self.assertIsNotNone(edge_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
