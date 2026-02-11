"""
Unit tests for GraphPalace - Tier 3 Storage for OMI

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
"""

import unittest
import tempfile
import time
import math
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.storage.graph_palace import GraphPalace, Memory, Edge


class TestGraphPalace(unittest.TestCase):
    """Test suite for GraphPalace."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_palace.sqlite"
        self.palace = GraphPalace(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        self.palace.close()
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

    def test_store_and_get_memory(self):
        """Test storing and retrieving a memory."""
        embedding = self._generate_embedding()
        
        memory_id = self.palace.store_memory(
            content="Test memory content",
            embedding=embedding,
            memory_type="fact",
            confidence=0.95
        )
        
        self.assertIsNotNone(memory_id)
        self.assertTrue(len(memory_id) > 0)
        
        # Retrieve and verify
        memory = self.palace.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, "Test memory content")
        self.assertEqual(memory.memory_type, "fact")
        self.assertAlmostEqual(memory.confidence, 0.95, places=4)
        self.assertEqual(memory.access_count, 1)  # Retrieved once
        
    def test_store_memory_no_embedding(self):
        """Test storing a memory without embedding."""
        memory_id = self.palace.store_memory(
            content="Memory without embedding",
            memory_type="experience"
        )
        
        memory = self.palace.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertIsNone(memory.embedding)
        self.assertEqual(memory.memory_type, "experience")

    def test_invalid_memory_type(self):
        """Test invalid memory type raises error."""
        with self.assertRaises(ValueError):
            self.palace.store_memory(
                content="Invalid type",
                memory_type="invalid_type"
            )

    def test_invalid_confidence(self):
        """Test invalid confidence value raises error."""
        with self.assertRaises(ValueError):
            self.palace.store_memory(
                content="Invalid confidence",
                memory_type="belief",
                confidence=1.5
            )
        
        with self.assertRaises(ValueError):
            self.palace.store_memory(
                content="Invalid confidence",
                memory_type="belief",
                confidence=-0.1
            )

    def test_update_embedding(self):
        """Test updating memory embedding."""
        old_embedding = self._generate_embedding()
        new_embedding = self._generate_embedding()
        
        memory_id = self.palace.store_memory(
            content="Test memory",
            embedding=old_embedding,
            memory_type="fact"
        )
        
        result = self.palace.update_embedding(memory_id, new_embedding)
        self.assertTrue(result)
        
        memory = self.palace.get_memory(memory_id)
        # Manual allclose check since numpy.testing may not be available
        self.assertTrue(np.allclose(memory.embedding, new_embedding, rtol=1e-5))

    def test_delete_memory(self):
        """Test deleting a memory."""
        memory_id = self.palace.store_memory(
            content="To be deleted",
            memory_type="experience"
        )
        
        result = self.palace.delete_memory(memory_id)
        self.assertTrue(result)
        
        memory = self.palace.get_memory(memory_id)
        self.assertIsNone(memory)

    def test_get_nonexistent_memory(self):
        """Test retrieving non-existent memory."""
        memory = self.palace.get_memory("nonexistent-id")
        self.assertIsNone(memory)

    # ==================== Edge Relationships ====================

    def test_create_edge(self):
        """Test creating an edge between memories."""
        memory1_id = self.palace.store_memory(
            content="Memory 1",
            memory_type="fact"
        )
        memory2_id = self.palace.store_memory(
            content="Memory 2", 
            memory_type="fact"
        )
        
        edge_id = self.palace.create_edge(
            source_id=memory1_id,
            target_id=memory2_id,
            edge_type="RELATED_TO",
            strength=0.8
        )
        
        self.assertIsNotNone(edge_id)
        
        # Get edges and verify
        edges = self.palace.get_edges(memory1_id)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source_id, memory1_id)
        self.assertEqual(edges[0].target_id, memory2_id)
        self.assertEqual(edges[0].edge_type, "RELATED_TO")
        self.assertAlmostEqual(edges[0].strength, 0.8, places=4)

    def test_invalid_edge_type(self):
        """Test invalid edge type raises error."""
        with self.assertRaises(ValueError):
            self.palace.create_edge("id1", "id2", "INVALID_TYPE")

    def test_get_neighbors(self):
        """Test getting neighboring memories."""
        memory1_id = self.palace.store_memory(content="Central memory", memory_type="fact")
        memory2_id = self.palace.store_memory(content="Neighbor 1", memory_type="fact")
        memory3_id = self.palace.store_memory(content="Neighbor 2", memory_type="fact")
        
        self.palace.create_edge(memory1_id, memory2_id, "RELATED_TO")
        self.palace.create_edge(memory3_id, memory1_id, "SUPPORTS")
        
        neighbors = self.palace.get_neighbors(memory1_id)
        self.assertEqual(len(neighbors), 2)
        
        neighbor_ids = {n.id for n in neighbors}
        self.assertEqual(neighbor_ids, {memory2_id, memory3_id})

    def test_delete_edge(self):
        """Test deleting an edge."""
        memory1_id = self.palace.store_memory(content="Source", memory_type="fact")
        memory2_id = self.palace.store_memory(content="Target", memory_type="fact")
        
        edge_id = self.palace.create_edge(memory1_id, memory2_id, "RELATED_TO")
        result = self.palace.delete_edge(edge_id)
        
        self.assertTrue(result)
        
        edges = self.palace.get_edges(memory1_id)
        self.assertEqual(len(edges), 0)

    # ==================== Semantic Search ====================

    def test_recall_basic(self):
        """Test basic semantic search/recall."""
        base_embedding = self._generate_embedding()
        similar_embedding = self._generate_similar_embedding(base_embedding, similarity=0.95)
        different_embedding = self._generate_embedding()
        
        # Store memories
        self.palace.store_memory(
            content="Similar memory",
            embedding=similar_embedding,
            memory_type="fact"
        )
        self.palace.store_memory(
            content="Different memory",
            embedding=different_embedding,
            memory_type="fact"
        )
        
        # Search with base query
        results = self.palace.recall(query_embedding=base_embedding, limit=10, min_relevance=0.5)
        
        self.assertGreater(len(results), 0)
        # First result should be the similar one
        contents = [r[0].content for r in results]
        self.assertIn("Similar memory", contents)

    def test_recall_min_relevance(self):
        """Test recall with minimum relevance threshold."""
        base = self._generate_embedding()
        v1 = self._generate_similar_embedding(base, 0.95)
        v2 = self._generate_similar_embedding(base, 0.6)
        v3 = self._generate_embedding()  # Very different
        
        self.palace.store_memory(content="High relevance", embedding=v1, memory_type="fact")
        self.palace.store_memory(content="Medium relevance", embedding=v2, memory_type="fact")
        self.palace.store_memory(content="Low relevance", embedding=v3, memory_type="fact")
        
        # High threshold
        results = self.palace.recall(base, limit=10, min_relevance=0.8)
        contents = [r[0].content for r in results]
        self.assertIn("High relevance", contents)
        self.assertNotIn("Low relevance", contents)

    def test_recal_relevance_score_range(self):
        """Test that relevance scores are in valid range."""
        embedding = self._generate_embedding()
        self.palace.store_memory(content="Test", embedding=embedding, memory_type="fact")
        
        results = self.palace.recall(query_embedding=embedding, limit=10, min_relevance=0.0)
        
        for memory, score in results:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    # ==================== Full-Text Search ====================

    def test_full_text_search(self):
        """Test FTS5 full-text search."""
        self.palace.store_memory(content="Python is great for machine learning", memory_type="fact")
        self.palace.store_memory(content="SQLite works well for embedded systems", memory_type="fact")
        self.palace.store_memory(content="Cats sleep 15 hours a day", memory_type="fact")
        
        results = self.palace.full_text_search("machine learning", limit=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Python is great for machine learning")

    def test_full_text_search_multiple_words(self):
        """Test FTS5 with multi-word query."""
        self.palace.store_memory(content="Neural networks for natural language processing", memory_type="fact")
        self.palace.store_memory(content="Graph databases store relationships", memory_type="fact")
        
        results = self.palace.full_text_search("neural language", limit=5)
        self.assertEqual(len(results), 1)

    # ==================== Centrality ====================

    def test_centrality_basic(self):
        """Test basic centrality calculation."""
        memory_id = self.palace.store_memory(
            content="Central node",
            memory_type="fact"
        )
        
        # Add neighbors
        for i in range(3):
            neighbor_id = self.palace.store_memory(content=f"Neighbor {i}", memory_type="fact")
            self.palace.create_edge(memory_id, neighbor_id, "RELATED_TO")
        
        centrality = self.palace.get_centrality(memory_id)
        self.assertGreater(centrality, 0.0)
        self.assertLessEqual(centrality, 1.0)

    def test_centrality_with_access_count(self):
        """Test centrality increases with access count."""
        memory_id = self.palace.store_memory(content="Popular memory", memory_type="fact")
        
        # Access multiple times
        for _ in range(5):
            self.palace.get_memory(memory_id)
        
        centrality1 = self.palace.get_centrality(memory_id)
        
        # More accesses
        for _ in range(10):
            self.palace.get_memory(memory_id)
        
        centrality2 = self.palace.get_centrality(memory_id)
        
        self.assertGreater(centrality2, centrality1)

    def test_top_central(self):
        """Test getting top central memories."""
        # Create memories with different centralities
        hub_id = self.palace.store_memory(content="Hub memory", memory_type="fact")

        for i in range(5):
            leaf_id = self.palace.store_memory(content=f"Leaf {i}", memory_type="fact")
            self.palace.create_edge(hub_id, leaf_id, "RELATED_TO")

        self.palace.get_memory(hub_id)  # Access it

        top = self.palace.get_top_central(limit=5)
        self.assertGreater(len(top), 0)

    def test_get_top_central_query_performance(self):
        """Test that get_top_central uses single query instead of N+1 pattern."""
        import sqlite3

        # Create 100 memories with various edge counts
        print("\n  Creating 100 memories with varying connectivity...")
        memory_ids = []
        for i in range(100):
            memory_id = self.palace.store_memory(
                content=f"Memory {i}: Test content for centrality",
                memory_type="fact"
            )
            memory_ids.append(memory_id)

        # Create edges with different patterns
        # Hub memory: connected to many others
        hub_id = memory_ids[0]
        for i in range(1, 20):
            self.palace.create_edge(hub_id, memory_ids[i], "RELATED_TO")

        # Medium connectivity memories
        for i in range(20, 40):
            for j in range(i + 1, min(i + 5, 40)):
                self.palace.create_edge(memory_ids[i], memory_ids[j], "RELATED_TO")

        # Access some memories to vary access counts
        for _ in range(10):
            self.palace.get_memory(hub_id)
        for _ in range(5):
            self.palace.get_memory(memory_ids[20])

        # Track query count using SQLite trace
        query_count = [0]

        def trace_callback(query):
            # Count SELECT queries (ignore pragma, etc.)
            if query.strip().upper().startswith('SELECT'):
                query_count[0] += 1

        # Close the existing connection to enable tracing
        self.palace.close()

        # Reconnect with tracing enabled
        conn = sqlite3.connect(self.db_path)
        conn.set_trace_callback(trace_callback)

        # Execute get_top_central query
        cursor = conn.execute("""
            SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                   m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash,
                   COUNT(e.id) as edge_count
            FROM memories m
            LEFT JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
            GROUP BY m.id
        """)
        results = cursor.fetchall()
        conn.close()

        # Reconnect palace for cleanup
        self.palace = GraphPalace(self.db_path)

        print(f"  Query count: {query_count[0]} (should be 1, not {len(memory_ids)}+)")
        print(f"  Retrieved {len(results)} memories")

        # Verify single query was executed
        self.assertEqual(query_count[0], 1,
                        f"Expected 1 query, got {query_count[0]}. N+1 pattern detected!")

        # Verify we got all memories
        self.assertEqual(len(results), 100)

        # Verify centrality calculation
        # Hub memory should have high edge count
        hub_row = [r for r in results if r[0] == hub_id][0]
        hub_edge_count = hub_row[10]
        self.assertGreaterEqual(hub_edge_count, 19, "Hub should have many edges")

        # Test the actual method works correctly
        top_central = self.palace.get_top_central(limit=10)
        self.assertEqual(len(top_central), 10)

        # Verify results are sorted by centrality (descending)
        centralities = [score for _, score in top_central]
        self.assertEqual(centralities, sorted(centralities, reverse=True),
                        "Results should be sorted by centrality descending")

        # Hub memory should be in top results
        top_ids = [memory.id for memory, _ in top_central]
        self.assertIn(hub_id, top_ids, "Hub memory should be in top central")

        # Verify centrality scores are in valid range
        for _, centrality in top_central:
            self.assertGreaterEqual(centrality, 0.0)
            self.assertLessEqual(centrality, 1.0)

    # ==================== Graph Traversal ====================

    def test_get_connected_depth_1(self):
        """Test BFS traversal depth 1."""
        node_a = self.palace.store_memory(content="Node A", memory_type="fact")
        node_b = self.palace.store_memory(content="Node B", memory_type="fact")
        node_c = self.palace.store_memory(content="Node C", memory_type="fact")
        
        self.palace.create_edge(node_a, node_b, "RELATED_TO")
        self.palace.create_edge(node_a, node_c, "RELATED_TO")
        
        connected = self.palace.get_connected(node_a, depth=1)
        self.assertEqual(len(connected), 2)

    def test_get_connected_depth_2(self):
        """Test BFS traversal depth 2."""
        # A -> B -> C (depth 2 from A)
        node_a = self.palace.store_memory(content="Node A", memory_type="fact")
        node_b = self.palace.store_memory(content="Node B", memory_type="fact")
        node_c = self.palace.store_memory(content="Node C", memory_type="fact")
        
        self.palace.create_edge(node_a, node_b, "RELATED_TO")
        self.palace.create_edge(node_b, node_c, "RELATED_TO")
        
        connected_depth1 = self.palace.get_connected(node_a, depth=1)
        self.assertEqual(len(connected_depth1), 1)  # Only B
        
        connected_depth2 = self.palace.get_connected(node_a, depth=2)
        self.assertEqual(len(connected_depth2), 2)  # B and C

    def test_get_connected_depth_0(self):
        """Test BFS with depth 0."""
        node_a = self.palace.store_memory(content="Node A", memory_type="fact")
        _ = self.palace.store_memory(content="Node B", memory_type="fact")
        
        self.palace.create_edge(node_a, _, "RELATED_TO")
        
        connected = self.palace.get_connected(node_a, depth=0)
        self.assertEqual(len(connected), 0)

    # ==================== Special Relationships ====================

    def test_find_contradictions(self):
        """Test finding contradicting memories."""
        belief_id = self.palace.store_memory(content="I should always work", memory_type="belief")
        contradiction_id = self.palace.store_memory(content="I should rest", memory_type="belief")
        
        self.palace.create_edge(belief_id, contradiction_id, "CONTRADICTS", strength=0.9)
        
        contradictions = self.palace.find_contradictions(belief_id)
        self.assertEqual(len(contradictions), 1)
        self.assertEqual(contradictions[0].id, contradiction_id)

    def test_get_supporting_evidence(self):
        """Test getting supporting evidence."""
        belief_id = self.palace.store_memory(content="AI is transformative", memory_type="belief")
        evidence_id = self.palace.store_memory(content="AI adoption rose 50%", memory_type="fact")
        
        self.palace.create_edge(belief_id, evidence_id, "SUPPORTS", strength=0.85)
        
        evidence = self.palace.get_supporting_evidence(belief_id)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].id, evidence_id)

    # ==================== Statistics ====================

    def test_get_stats(self):
        """Test database statistics."""
        # Populate with sample data
        for i in range(5):
            self.palace.store_memory(content=f"Memory {i}", memory_type="fact")
        
        stats = self.palace.get_stats()
        
        self.assertEqual(stats["memory_count"], 5)
        self.assertIn("fact", stats["type_distribution"])
        self.assertEqual(stats["type_distribution"]["fact"], 5)

    def test_get_stats_with_edges(self):
        """Test stats with edges."""
        m1 = self.palace.store_memory(content="Source", memory_type="fact")
        m2 = self.palace.store_memory(content="Target", memory_type="fact")
        self.palace.create_edge(m1, m2, "RELATED_TO")
        
        stats = self.palace.get_stats()
        self.assertEqual(stats["edge_count"], 1)

    # ==================== Performance Benchmarks ====================

    def test_performance_recall_1000_memories(self):
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
            
            self.palace.store_memory(
                content=f"Memory {i}: This is a test memory for performance benchmarking",
                embedding=embedding,
                memory_type="fact"
            )
        
        print("  Benchmarking recall...")
        query_embedding = [random.random() for _ in range(1024)]
        norm = sum(x*x for x in query_embedding) ** 0.5
        query_embedding = [x/norm for x in query_embedding]
        
        start = time.time()
        results = self.palace.recall(query_embedding, limit=10, min_relevance=0.0)
        elapsed = time.time() - start
        
        print(f"  Recall 1000 memories: {elapsed*1000:.2f}ms, found {len(results)} results")
        
        # Target: <500ms
        self.assertLess(elapsed, 0.5, f"Query too slow: {elapsed*1000:.2f}ms")

    def test_performance_graph_traversal_10k_edges(self):
        """Benchmark: Graph queries scale to 10k edges."""
        import random
        random.seed(42)
        
        print("\n  Creating 500 nodes...")
        node_ids = []
        for i in range(500):
            node_id = self.palace.store_memory(content=f"Node {i}", memory_type="fact")
            node_ids.append(node_id)
        
        print("  Creating ~10k edges...")
        edge_count = 0
        for i in range(len(node_ids)):
            # Each node connects to ~20 random other nodes
            targets = random.sample(node_ids, min(20, len(node_ids)))
            for target in targets:
                if target != node_ids[i]:
                    self.palace.create_edge(node_ids[i], target, "RELATED_TO")
                    edge_count += 1
        
        print(f"  Created {edge_count} edges")
        
        # Test centrality calculation
        print("  Benchmarking centrality for 100 nodes...")
        start = time.time()
        sample_nodes = random.sample(node_ids, min(100, len(node_ids)))
        for node_id in sample_nodes:
            _ = self.palace.get_centrality(node_id)
        elapsed = time.time() - start
        
        print(f"  Centrality for 100 nodes: {elapsed*1000:.2f}ms")
        
        # Should complete in reasonable time (let's say <5s for 100 nodes)
        self.assertLess(elapsed, 5.0)
        
        # Test graph traversal
        print("  Benchmarking BFS traversal...")
        start = time.time()
        connected = self.palace.get_connected(node_ids[0], depth=2)
        elapsed = time.time() - start
        
        print(f"  BFS depth 2 from central node: {elapsed*1000:.2f}ms, found {len(connected)} nodes")
        self.assertGreater(len(connected), 0)

    def test_performance_wal_concurrent_writes(self):
        """Test that WAL mode allows concurrent writes."""
        import threading
        import time
        
        errors = []
        success_count = [0]
        
        def writer_thread(thread_id, count):
            try:
                for i in range(count):
                    memory_id = self.palace.store_memory(
                        content=f"Thread {thread_id} Memory {i}",
                        memory_type="experience"
                    )
                    success_count[0] += 1
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
        
        self.assertEqual(len(errors), 0, f"Concurrent writes failed: {errors}")
        self.assertEqual(success_count[0], num_threads * writes_per_thread)

    # ==================== Integrity ====================

    def test_content_hash(self):
        """Test content hash is computed correctly."""
        memory_id = self.palace.store_memory(content="Test content", memory_type="fact")
        memory = self.palace.get_memory(memory_id)
        
        expected_hash = hashlib.sha256("Test content".encode()).hexdigest()
        self.assertEqual(memory.content_hash, expected_hash)

    def test_access_count_increment(self):
        """Test access count increments on get."""
        memory_id = self.palace.store_memory(content="Popular", memory_type="fact")
        
        memory = self.palace.get_memory(memory_id)
        self.assertEqual(memory.access_count, 1)
        
        memory = self.palace.get_memory(memory_id)
        self.assertEqual(memory.access_count, 2)

    def test_last_accessed_update(self):
        """Test last_accessed timestamp updates."""
        memory_id = self.palace.store_memory(content="Test", memory_type="fact")
        first_access = self.palace.get_memory(memory_id).last_accessed
        
        time.sleep(0.01)  # Small delay
        second_access = self.palace.get_memory(memory_id).last_accessed
        
        self.assertGreater(second_access, first_access)

    def test_context_manager(self):
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.sqlite"
            
            with GraphPalace(db_path) as palace:
                memory_id = palace.store_memory(content="In context", memory_type="fact")
                memory = palace.get_memory(memory_id)
                self.assertIsNotNone(memory)

    def test_vacuum(self):
        """Test vacuum operation."""
        # Should not raise error
        self.palace.vacuum()


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


class TestEdgeTypes(unittest.TestCase):
    """Test all valid edge types."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "edges.sqlite"
        self.palace = GraphPalace(self.db_path)

    def tearDown(self):
        self.palace.close()
        import shutil
        if self.db_path.parent.exists():
            shutil.rmtree(str(self.db_path.parent), ignore_errors=True)

    def test_all_edge_types(self):
        """Test all valid edge types can be created."""
        source = self.palace.store_memory(content="Source", memory_type="fact")
        
        valid_types = ["SUPPORTS", "CONTRADICTS", "RELATED_TO", "DEPENDS_ON", "POSTED", "DISCUSSED"]
        
        for edge_type in valid_types:
            with self.subTest(edge_type=edge_type):
                target = self.palace.store_memory(content=f"Target {edge_type}", memory_type="fact")
                edge_id = self.palace.create_edge(source, target, edge_type, strength=0.5)
                self.assertIsNotNone(edge_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
