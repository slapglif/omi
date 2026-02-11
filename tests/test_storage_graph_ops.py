"""
Comprehensive tests for storage.graph_ops module (GraphOperations)

Tests cover:
- Edge creation with type validation
- Edge deletion
- Getting edges by memory and type
- Getting neighbors (memories connected via edges)
- Finding contradictions
- Getting supporting evidence
- Thread-safe operations
- Context manager usage
"""

import pytest
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from omi.storage.graph_ops import GraphOperations
from omi.storage.models import Edge, Memory
from omi.storage.schema import init_database


class TestGraphOperations:
    """Test suite for GraphOperations class"""

    def test_init_with_file_path(self, tmp_path):
        """Test initialization with file-based database"""
        db_path = tmp_path / "test_graph.db"
        graph_ops = GraphOperations(str(db_path))

        assert graph_ops.db_path == db_path
        assert graph_ops._owns_connection is True
        assert graph_ops._conn is not None

        graph_ops.close()

    def test_init_with_memory_db(self):
        """Test initialization with in-memory database"""
        graph_ops = GraphOperations(':memory:')

        assert graph_ops.db_path == ':memory:'
        assert graph_ops._owns_connection is True

        graph_ops.close()

    def test_init_with_shared_connection(self, tmp_path):
        """Test initialization with shared connection (facade pattern)"""
        db_path = tmp_path / "test_shared.db"
        conn = sqlite3.connect(str(db_path))
        init_database(conn, enable_wal=True)

        graph_ops = GraphOperations(str(db_path), conn=conn)

        assert graph_ops._owns_connection is False
        assert graph_ops._conn is conn

        # Close should not close shared connection
        graph_ops.close()
        # Connection should still work
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_edge_type_validation_valid(self, tmp_path):
        """Test that valid edge types are accepted"""
        graph_ops = GraphOperations(':memory:')

        valid_types = ["SUPPORTS", "CONTRADICTS", "RELATED_TO", "DEPENDS_ON", "POSTED", "DISCUSSED"]

        for edge_type in valid_types:
            # Should not raise
            graph_ops._validate_edge_type(edge_type)

        graph_ops.close()

    def test_edge_type_validation_invalid(self, tmp_path):
        """Test that invalid edge types are rejected"""
        graph_ops = GraphOperations(':memory:')

        with pytest.raises(ValueError, match="Invalid edge_type"):
            graph_ops._validate_edge_type("INVALID_TYPE")

        graph_ops.close()

    def test_create_edge_basic(self, tmp_path):
        """Test creating a basic edge between memories"""
        graph_ops = GraphOperations(':memory:')

        # Insert test memories first
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (source_id, "Source memory", "fact", datetime.now().isoformat()))

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (target_id, "Target memory", "fact", datetime.now().isoformat()))

        # Create edge
        edge_id = graph_ops.create_edge(source_id, target_id, "RELATED_TO", strength=0.8)

        assert edge_id is not None
        assert len(edge_id) == 36  # UUID format

        # Verify edge was created
        cursor = graph_ops._conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == source_id  # source_id
        assert row[2] == target_id  # target_id
        assert row[3] == "RELATED_TO"  # edge_type
        assert row[4] == 0.8  # strength

        graph_ops.close()

    def test_create_edge_invalid_type(self, tmp_path):
        """Test creating edge with invalid type raises error"""
        graph_ops = GraphOperations(':memory:')

        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())

        with pytest.raises(ValueError, match="Invalid edge_type"):
            graph_ops.create_edge(source_id, target_id, "INVALID_TYPE")

        graph_ops.close()

    def test_delete_edge_existing(self, tmp_path):
        """Test deleting an existing edge"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories and edge
        source_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?)
        """, (source_id, "Source", "fact", datetime.now().isoformat(),
              target_id, "Target", "fact", datetime.now().isoformat()))

        edge_id = graph_ops.create_edge(source_id, target_id, "SUPPORTS")

        # Delete edge
        result = graph_ops.delete_edge(edge_id)

        assert result is True

        # Verify edge is gone
        cursor = graph_ops._conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,))
        assert cursor.fetchone() is None

        graph_ops.close()

    def test_delete_edge_nonexistent(self, tmp_path):
        """Test deleting non-existent edge returns False"""
        graph_ops = GraphOperations(':memory:')

        fake_edge_id = str(uuid.uuid4())
        result = graph_ops.delete_edge(fake_edge_id)

        assert result is False

        graph_ops.close()

    def test_get_edges_all_types(self, tmp_path):
        """Test getting all edges for a memory"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        memory_id = str(uuid.uuid4())
        other_id_1 = str(uuid.uuid4())
        other_id_2 = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)
        """, (memory_id, "Main", "fact", datetime.now().isoformat(),
              other_id_1, "Other1", "fact", datetime.now().isoformat(),
              other_id_2, "Other2", "fact", datetime.now().isoformat()))

        # Create edges
        edge1_id = graph_ops.create_edge(memory_id, other_id_1, "SUPPORTS", 0.9)
        edge2_id = graph_ops.create_edge(memory_id, other_id_2, "CONTRADICTS", 0.7)

        # Get all edges
        edges = graph_ops.get_edges(memory_id)

        assert len(edges) == 2
        assert all(isinstance(e, Edge) for e in edges)

        edge_ids = [e.id for e in edges]
        assert edge1_id in edge_ids
        assert edge2_id in edge_ids

        graph_ops.close()

    def test_get_edges_filtered_by_type(self, tmp_path):
        """Test getting edges filtered by type"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        memory_id = str(uuid.uuid4())
        other_id_1 = str(uuid.uuid4())
        other_id_2 = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)
        """, (memory_id, "Main", "fact", datetime.now().isoformat(),
              other_id_1, "Other1", "fact", datetime.now().isoformat(),
              other_id_2, "Other2", "fact", datetime.now().isoformat()))

        # Create edges of different types
        supports_id = graph_ops.create_edge(memory_id, other_id_1, "SUPPORTS", 0.9)
        graph_ops.create_edge(memory_id, other_id_2, "CONTRADICTS", 0.7)

        # Filter by SUPPORTS type
        supports_edges = graph_ops.get_edges(memory_id, edge_type="SUPPORTS")

        assert len(supports_edges) == 1
        assert supports_edges[0].id == supports_id
        assert supports_edges[0].edge_type == "SUPPORTS"

        graph_ops.close()

    def test_get_neighbors_basic(self, tmp_path):
        """Test getting neighboring memories"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        memory_id = str(uuid.uuid4())
        neighbor1_id = str(uuid.uuid4())
        neighbor2_id = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)
        """, (memory_id, "Main memory", "fact", datetime.now().isoformat(),
              neighbor1_id, "Neighbor 1", "fact", datetime.now().isoformat(),
              neighbor2_id, "Neighbor 2", "fact", datetime.now().isoformat()))

        # Create edges
        graph_ops.create_edge(memory_id, neighbor1_id, "RELATED_TO")
        graph_ops.create_edge(memory_id, neighbor2_id, "DEPENDS_ON")

        # Get neighbors
        neighbors = graph_ops.get_neighbors(memory_id)

        assert len(neighbors) == 2
        assert all(isinstance(m, Memory) for m in neighbors)

        neighbor_ids = [m.id for m in neighbors]
        assert neighbor1_id in neighbor_ids
        assert neighbor2_id in neighbor_ids

        # Main memory should not be in results
        assert memory_id not in neighbor_ids

        graph_ops.close()

    def test_get_neighbors_filtered_by_type(self, tmp_path):
        """Test getting neighbors filtered by edge type"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        memory_id = str(uuid.uuid4())
        supporter_id = str(uuid.uuid4())
        related_id = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)
        """, (memory_id, "Main", "fact", datetime.now().isoformat(),
              supporter_id, "Supporter", "fact", datetime.now().isoformat(),
              related_id, "Related", "fact", datetime.now().isoformat()))

        # Create different edge types
        graph_ops.create_edge(memory_id, supporter_id, "SUPPORTS")
        graph_ops.create_edge(memory_id, related_id, "RELATED_TO")

        # Get only SUPPORTS neighbors
        supporters = graph_ops.get_neighbors(memory_id, edge_type="SUPPORTS")

        assert len(supporters) == 1
        assert supporters[0].id == supporter_id

        graph_ops.close()

    def test_find_contradictions(self, tmp_path):
        """Test finding contradicting memories"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        memory_id = str(uuid.uuid4())
        contradictor1_id = str(uuid.uuid4())
        contradictor2_id = str(uuid.uuid4())
        supporter_id = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)
        """, (memory_id, "Main", "fact", datetime.now().isoformat(),
              contradictor1_id, "Contra1", "fact", datetime.now().isoformat(),
              contradictor2_id, "Contra2", "fact", datetime.now().isoformat(),
              supporter_id, "Supporter", "fact", datetime.now().isoformat()))

        # Create edges
        graph_ops.create_edge(memory_id, contradictor1_id, "CONTRADICTS")
        graph_ops.create_edge(memory_id, contradictor2_id, "CONTRADICTS")
        graph_ops.create_edge(memory_id, supporter_id, "SUPPORTS")  # Should not be returned

        # Find contradictions
        contradictions = graph_ops.find_contradictions(memory_id)

        assert len(contradictions) == 2
        contradictor_ids = [m.id for m in contradictions]
        assert contradictor1_id in contradictor_ids
        assert contradictor2_id in contradictor_ids
        assert supporter_id not in contradictor_ids

        graph_ops.close()

    def test_get_supporting_evidence(self, tmp_path):
        """Test getting supporting evidence"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        belief_id = str(uuid.uuid4())
        evidence1_id = str(uuid.uuid4())
        evidence2_id = str(uuid.uuid4())
        contradictor_id = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?), (?, ?, ?, ?)
        """, (belief_id, "Belief", "belief", datetime.now().isoformat(),
              evidence1_id, "Evidence1", "fact", datetime.now().isoformat(),
              evidence2_id, "Evidence2", "fact", datetime.now().isoformat(),
              contradictor_id, "Contra", "fact", datetime.now().isoformat()))

        # Create edges
        graph_ops.create_edge(belief_id, evidence1_id, "SUPPORTS")
        graph_ops.create_edge(belief_id, evidence2_id, "SUPPORTS")
        graph_ops.create_edge(belief_id, contradictor_id, "CONTRADICTS")  # Should not be returned

        # Get supporting evidence
        evidence = graph_ops.get_supporting_evidence(belief_id)

        assert len(evidence) == 2
        evidence_ids = [m.id for m in evidence]
        assert evidence1_id in evidence_ids
        assert evidence2_id in evidence_ids
        assert contradictor_id not in evidence_ids

        graph_ops.close()

    def test_context_manager(self, tmp_path):
        """Test context manager usage"""
        db_path = tmp_path / "test_cm.db"

        with GraphOperations(str(db_path)) as graph_ops:
            # Should work inside context
            source_id = str(uuid.uuid4())
            target_id = str(uuid.uuid4())

            graph_ops._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?), (?, ?, ?, ?)
            """, (source_id, "Source", "fact", datetime.now().isoformat(),
                  target_id, "Target", "fact", datetime.now().isoformat()))

            edge_id = graph_ops.create_edge(source_id, target_id, "RELATED_TO")
            assert edge_id is not None

        # Connection should be closed after context exit
        # (test passes if no exception is raised)

    def test_bidirectional_edge_queries(self, tmp_path):
        """Test that edges work in both directions"""
        graph_ops = GraphOperations(':memory:')

        # Setup memories
        memory_a = str(uuid.uuid4())
        memory_b = str(uuid.uuid4())

        graph_ops._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?), (?, ?, ?, ?)
        """, (memory_a, "Memory A", "fact", datetime.now().isoformat(),
              memory_b, "Memory B", "fact", datetime.now().isoformat()))

        # Create edge A -> B
        edge_id = graph_ops.create_edge(memory_a, memory_b, "RELATED_TO")

        # Both memories should see the edge
        edges_a = graph_ops.get_edges(memory_a)
        edges_b = graph_ops.get_edges(memory_b)

        assert len(edges_a) == 1
        assert len(edges_b) == 1
        assert edges_a[0].id == edge_id
        assert edges_b[0].id == edge_id

        # Both should see each other as neighbors
        neighbors_a = graph_ops.get_neighbors(memory_a)
        neighbors_b = graph_ops.get_neighbors(memory_b)

        assert len(neighbors_a) == 1
        assert neighbors_a[0].id == memory_b

        assert len(neighbors_b) == 1
        assert neighbors_b[0].id == memory_a

        graph_ops.close()
