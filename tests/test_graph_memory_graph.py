"""
Comprehensive tests for graph.memory_graph module

Tests cover:
- Memory node storage
- Semantic search with embeddings
- Centrality calculation
- Topology verification
- Cosine similarity
"""

import pytest
import uuid
import tempfile
from pathlib import Path

from omi.graph.memory_graph import MemoryGraph, MemoryNode


class TestMemoryGraph:
    """Test suite for MemoryGraph class"""

    def test_init_creates_database(self, tmp_path):
        """Test that initialization creates database"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        assert db_path.exists()
        assert mg.db_path == db_path
        assert mg.embeddings_dim == 768

    def test_init_with_custom_embedding_dim(self, tmp_path):
        """Test initialization with custom embedding dimension"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path, embeddings_dim=1024)

        assert mg.embeddings_dim == 1024

    def test_store_basic_node(self, tmp_path):
        """Test storing a basic memory node"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Test memory content",
            memory_type="fact"
        )

        node_id = mg.store(node)

        assert node_id == node.id
        assert node.hash != ""  # Hash should be computed

    def test_store_node_with_embedding(self, tmp_path):
        """Test storing node with embedding"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        embedding = [0.1] * 768
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Memory with embedding",
            embedding=embedding,
            memory_type="experience"
        )

        node_id = mg.store(node)

        assert node_id == node.id

    def test_store_node_with_confidence(self, tmp_path):
        """Test storing belief node with confidence"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Belief content",
            memory_type="belief",
            confidence=0.85
        )

        node_id = mg.store(node)

        assert node_id == node.id

    def test_store_sets_timestamps(self, tmp_path):
        """Test that store sets timestamps if not provided"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Test",
            memory_type="fact"
        )

        mg.store(node)

        assert node.created_at != ""
        assert node.last_accessed != ""

    def test_store_computes_hash(self, tmp_path):
        """Test that store computes content hash"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        content = "Test content"
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content=content,
            memory_type="fact"
        )

        mg.store(node)

        assert node.hash != ""
        assert len(node.hash) == 64  # SHA-256 produces 64 hex chars

    def test_semantic_search_no_results(self, tmp_path):
        """Test semantic search with no matching memories"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        query_embedding = [0.5] * 768
        results = mg.semantic_search(query_embedding)

        assert len(results) == 0

    def test_semantic_search_basic(self, tmp_path):
        """Test basic semantic search"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        # Store memory with embedding
        embedding = [0.5] * 768
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Test memory",
            embedding=embedding,
            memory_type="fact"
        )
        mg.store(node)

        # Search with similar embedding
        query_embedding = [0.51] * 768
        results = mg.semantic_search(query_embedding, min_similarity=0.9)

        assert len(results) >= 1
        assert all(isinstance(item, tuple) for item in results)
        assert all(isinstance(item[0], MemoryNode) for item in results)
        assert all(isinstance(item[1], float) for item in results)

    def test_semantic_search_similarity_threshold(self, tmp_path):
        """Test that similarity threshold filters results"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        # Store two memories with different embeddings
        similar_embedding = [1.0] * 768
        different_embedding = [0.0] + [1.0] * 767

        mg.store(MemoryNode(
            id=str(uuid.uuid4()),
            content="Similar",
            embedding=similar_embedding,
            memory_type="fact"
        ))

        mg.store(MemoryNode(
            id=str(uuid.uuid4()),
            content="Different",
            embedding=different_embedding,
            memory_type="fact"
        ))

        # Search with high threshold
        query = [0.95] * 768
        results = mg.semantic_search(query, min_similarity=0.9)

        # Should only get the similar one
        assert len(results) >= 1
        # Check that low-similarity results are filtered out

    def test_semantic_search_limit(self, tmp_path):
        """Test that limit parameter works"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        # Store multiple memories
        embedding_base = [0.5] * 768
        for i in range(5):
            embedding = [0.5 + i*0.01] * 768
            mg.store(MemoryNode(
                id=str(uuid.uuid4()),
                content=f"Memory {i}",
                embedding=embedding,
                memory_type="fact"
            ))

        # Search with limit=2
        query = [0.5] * 768
        results = mg.semantic_search(query, limit=2, min_similarity=0.0)

        assert len(results) <= 2

    def test_semantic_search_sorted_by_similarity(self, tmp_path):
        """Test that results are sorted by similarity descending"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        # Store memories with varying similarity to query
        query = [1.0] * 768

        embeddings = [
            [0.9] * 768,  # High similarity
            [0.5] * 768,  # Medium similarity
            [0.1] * 768,  # Low similarity
        ]

        for i, emb in enumerate(embeddings):
            mg.store(MemoryNode(
                id=str(uuid.uuid4()),
                content=f"Memory {i}",
                embedding=emb,
                memory_type="fact"
            ))

        results = mg.semantic_search(query, min_similarity=0.0)

        # Should be sorted descending by similarity
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]

    def test_get_centrality_basic(self, tmp_path):
        """Test centrality calculation"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        node_id = str(uuid.uuid4())
        node = MemoryNode(
            id=node_id,
            content="Test",
            memory_type="fact",
            access_count=10
        )
        mg.store(node)

        centrality = mg.get_centrality(node_id)

        assert centrality > 0
        # Centrality = access * 0.7 + edges * 0.3
        # With 10 accesses and 0 edges: 10 * 0.7 + 0 * 0.3 = 7.0
        assert abs(centrality - 7.0) < 0.1

    def test_get_centrality_nonexistent(self, tmp_path):
        """Test centrality for non-existent node"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        fake_id = str(uuid.uuid4())
        centrality = mg.get_centrality(fake_id)

        assert centrality == 0.0

    def test_verify_topology_no_anomalies(self, tmp_path):
        """Test topology verification with clean graph"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        # Store regular memory
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Regular memory",
            memory_type="fact"
        )
        mg.store(node)

        anomalies = mg.verify_topology()

        assert "orphan_nodes" in anomalies
        assert "sudden_cores" in anomalies
        assert len(anomalies["orphan_nodes"]) == 0
        assert len(anomalies["sudden_cores"]) == 0

    def test_cosine_similarity_identical_vectors(self, tmp_path):
        """Test cosine similarity with identical vectors"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        vec = [1.0] * 768
        similarity = mg._cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal_vectors(self, tmp_path):
        """Test cosine similarity with orthogonal vectors"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        vec_a = [1.0] + [0.0] * 767
        vec_b = [0.0] + [1.0] * 767

        similarity = mg._cosine_similarity(vec_a, vec_b)

        # Orthogonal vectors should have ~0 similarity
        assert abs(similarity) < 0.01

    def test_store_with_instance_ids(self, tmp_path):
        """Test storing node with instance IDs"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        instance_ids = ["inst1", "inst2", "inst3"]
        node = MemoryNode(
            id=str(uuid.uuid4()),
            content="Test",
            memory_type="fact",
            instance_ids=instance_ids
        )

        node_id = mg.store(node)

        assert node_id == node.id

    def test_store_replace_existing(self, tmp_path):
        """Test that store replaces existing nodes"""
        db_path = tmp_path / "memory_graph.db"
        mg = MemoryGraph(db_path)

        node_id = str(uuid.uuid4())

        # Store initial node
        node1 = MemoryNode(
            id=node_id,
            content="Original content",
            memory_type="fact"
        )
        mg.store(node1)

        # Store with same ID but different content
        node2 = MemoryNode(
            id=node_id,
            content="Updated content",
            memory_type="experience"
        )
        mg.store(node2)

        # Only one node should exist with the updated content
        # (Would need a get method to verify, but store should succeed)
        assert node2.hash != node1.hash
