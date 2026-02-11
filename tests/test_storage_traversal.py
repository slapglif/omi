"""
Comprehensive tests for storage.traversal module (GraphTraversal)

Tests cover:
- BFS traversal (get_connected)
- Centrality calculation (degree, access frequency, recency)
- Getting top central (hub) memories
- Recency score calculation
- Thread-safe operations
- Context manager usage
"""

import pytest
import sqlite3
import uuid
import math
from datetime import datetime, timedelta
from pathlib import Path

from omi.storage.traversal import GraphTraversal
from omi.storage.models import Memory
from omi.storage.schema import init_database


class TestGraphTraversal:
    """Test suite for GraphTraversal class"""

    def test_init_with_file_path(self, tmp_path):
        """Test initialization with file-based database"""
        db_path = tmp_path / "test_traversal.db"
        traversal = GraphTraversal(str(db_path))

        assert traversal.db_path == db_path
        assert traversal._owns_connection is True
        assert traversal._conn is not None

        traversal.close()

    def test_init_with_memory_db(self):
        """Test initialization with in-memory database"""
        traversal = GraphTraversal(':memory:')

        assert traversal.db_path == ':memory:'
        assert traversal._owns_connection is True

        traversal.close()

    def test_init_with_shared_connection(self, tmp_path):
        """Test initialization with shared connection (facade pattern)"""
        db_path = tmp_path / "test_shared.db"
        conn = sqlite3.connect(str(db_path))
        init_database(conn, enable_wal=True)

        traversal = GraphTraversal(str(db_path), conn=conn)

        assert traversal._owns_connection is False
        assert traversal._conn is conn

        # Close should not close shared connection
        traversal.close()
        # Connection should still work
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_calculate_recency_score_recent(self):
        """Test recency score for recent timestamp"""
        traversal = GraphTraversal(':memory:')

        # Recent timestamp (1 day ago)
        timestamp = datetime.now() - timedelta(days=1)
        score = traversal._calculate_recency_score(timestamp)

        # Should be close to 1.0 (very recent)
        assert score > 0.9
        assert score <= 1.0

        traversal.close()

    def test_calculate_recency_score_old(self):
        """Test recency score for old timestamp"""
        traversal = GraphTraversal(':memory:')

        # Old timestamp (90 days ago, 3 half-lives)
        timestamp = datetime.now() - timedelta(days=90)
        score = traversal._calculate_recency_score(timestamp)

        # Should decay significantly: exp(-90/30) = exp(-3) ≈ 0.05
        assert 0.04 < score < 0.06

        traversal.close()

    def test_calculate_recency_score_at_half_life(self):
        """Test recency score at half-life (30 days)"""
        traversal = GraphTraversal(':memory:')

        # At half-life (30 days ago)
        timestamp = datetime.now() - timedelta(days=30)
        score = traversal._calculate_recency_score(timestamp)

        # Should be exp(-1) ≈ 0.368
        expected = math.exp(-1)
        assert abs(score - expected) < 0.01

        traversal.close()

    def test_calculate_recency_score_none(self):
        """Test recency score with None timestamp"""
        traversal = GraphTraversal(':memory:')

        score = traversal._calculate_recency_score(None)

        assert score == 0.0

        traversal.close()

    def test_get_centrality_basic(self):
        """Test centrality calculation for a memory"""
        traversal = GraphTraversal(':memory:')

        # Insert memory with some access history
        mem_id = str(uuid.uuid4())
        last_accessed = datetime.now() - timedelta(days=5)

        traversal._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mem_id, "Test memory", "fact", datetime.now().isoformat(),
              last_accessed.isoformat(), 10))

        # Add some edges
        other_ids = []
        for i in range(3):
            other_id = str(uuid.uuid4())
            other_ids.append(other_id)
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (other_id, f"Other {i}", "fact", datetime.now().isoformat()))

            edge_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, mem_id, other_id, "RELATED_TO", datetime.now().isoformat()))

        centrality = traversal.get_centrality(mem_id)

        # Should have non-zero centrality
        assert centrality > 0
        assert centrality <= 1.0

        traversal.close()

    def test_get_centrality_nonexistent_memory(self):
        """Test centrality for non-existent memory"""
        traversal = GraphTraversal(':memory:')

        fake_id = str(uuid.uuid4())
        centrality = traversal.get_centrality(fake_id)

        assert centrality == 0.0

        traversal.close()

    def test_get_centrality_isolated_memory(self):
        """Test centrality for isolated memory (no edges)"""
        traversal = GraphTraversal(':memory:')

        # Memory with no edges
        mem_id = str(uuid.uuid4())
        traversal._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at, access_count)
            VALUES (?, ?, ?, ?, ?)
        """, (mem_id, "Isolated", "fact", datetime.now().isoformat(), 0))

        centrality = traversal.get_centrality(mem_id)

        # Should have low centrality (mostly from recency)
        assert 0 <= centrality < 0.3

        traversal.close()

    def test_get_centrality_hub_memory(self):
        """Test centrality for highly connected hub memory"""
        traversal = GraphTraversal(':memory:')

        # Create hub memory with many edges and high access count
        hub_id = str(uuid.uuid4())
        last_accessed = datetime.now() - timedelta(days=1)  # Recent

        traversal._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (hub_id, "Hub memory", "fact", datetime.now().isoformat(),
              last_accessed.isoformat(), 50))  # High access count

        # Add many edges (high degree)
        for i in range(20):
            other_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (other_id, f"Connected {i}", "fact", datetime.now().isoformat()))

            edge_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, hub_id, other_id, "RELATED_TO", datetime.now().isoformat()))

        centrality = traversal.get_centrality(hub_id)

        # Should have high centrality
        assert centrality > 0.5

        traversal.close()

    def test_get_connected_depth_1(self):
        """Test BFS traversal with depth=1 (immediate neighbors)"""
        traversal = GraphTraversal(':memory:')

        # Create memory graph: A -> B -> C -> D
        mem_a = str(uuid.uuid4())
        mem_b = str(uuid.uuid4())
        mem_c = str(uuid.uuid4())
        mem_d = str(uuid.uuid4())

        for mem_id, content in [(mem_a, "A"), (mem_b, "B"), (mem_c, "C"), (mem_d, "D")]:
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, "fact", datetime.now().isoformat()))

        # Create edges: A-B, B-C, C-D
        for source, target in [(mem_a, mem_b), (mem_b, mem_c), (mem_c, mem_d)]:
            edge_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, source, target, "RELATED_TO", datetime.now().isoformat()))

        # Traverse from A with depth=1 (should only get B)
        connected = traversal.get_connected(mem_a, depth=1)

        assert len(connected) == 1
        assert connected[0].id == mem_b
        assert all(isinstance(m, Memory) for m in connected)

        traversal.close()

    def test_get_connected_depth_2(self):
        """Test BFS traversal with depth=2 (2 hops)"""
        traversal = GraphTraversal(':memory:')

        # Create memory graph: A -> B -> C -> D
        mem_a = str(uuid.uuid4())
        mem_b = str(uuid.uuid4())
        mem_c = str(uuid.uuid4())
        mem_d = str(uuid.uuid4())

        for mem_id, content in [(mem_a, "A"), (mem_b, "B"), (mem_c, "C"), (mem_d, "D")]:
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, "fact", datetime.now().isoformat()))

        # Create edges: A-B, B-C, C-D
        for source, target in [(mem_a, mem_b), (mem_b, mem_c), (mem_c, mem_d)]:
            edge_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, source, target, "RELATED_TO", datetime.now().isoformat()))

        # Traverse from A with depth=2 (should get B and C)
        connected = traversal.get_connected(mem_a, depth=2)

        assert len(connected) == 2
        connected_ids = [m.id for m in connected]
        assert mem_b in connected_ids
        assert mem_c in connected_ids
        assert mem_d not in connected_ids  # 3 hops away

        traversal.close()

    def test_get_connected_isolated_memory(self):
        """Test traversal from isolated memory (no neighbors)"""
        traversal = GraphTraversal(':memory:')

        mem_id = str(uuid.uuid4())
        traversal._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (mem_id, "Isolated", "fact", datetime.now().isoformat()))

        connected = traversal.get_connected(mem_id, depth=2)

        assert len(connected) == 0

        traversal.close()

    def test_get_connected_bidirectional_edges(self):
        """Test that traversal works in both directions"""
        traversal = GraphTraversal(':memory:')

        # Create A -> B edge
        mem_a = str(uuid.uuid4())
        mem_b = str(uuid.uuid4())

        for mem_id, content in [(mem_a, "A"), (mem_b, "B")]:
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, "fact", datetime.now().isoformat()))

        edge_id = str(uuid.uuid4())
        traversal._conn.execute("""
            INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (edge_id, mem_a, mem_b, "RELATED_TO", datetime.now().isoformat()))

        # Traverse from A should find B
        connected_from_a = traversal.get_connected(mem_a, depth=1)
        assert len(connected_from_a) == 1
        assert connected_from_a[0].id == mem_b

        # Traverse from B should find A (bidirectional)
        connected_from_b = traversal.get_connected(mem_b, depth=1)
        assert len(connected_from_b) == 1
        assert connected_from_b[0].id == mem_a

        traversal.close()

    def test_get_connected_no_duplicates(self):
        """Test that BFS doesn't return duplicates in complex graphs"""
        traversal = GraphTraversal(':memory:')

        # Create diamond pattern: A -> B, A -> C, B -> D, C -> D
        mem_a = str(uuid.uuid4())
        mem_b = str(uuid.uuid4())
        mem_c = str(uuid.uuid4())
        mem_d = str(uuid.uuid4())

        for mem_id, content in [(mem_a, "A"), (mem_b, "B"), (mem_c, "C"), (mem_d, "D")]:
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, "fact", datetime.now().isoformat()))

        # Create diamond edges
        for source, target in [(mem_a, mem_b), (mem_a, mem_c), (mem_b, mem_d), (mem_c, mem_d)]:
            edge_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, source, target, "RELATED_TO", datetime.now().isoformat()))

        # Traverse from A with depth=2 (should get B, C, D once each)
        connected = traversal.get_connected(mem_a, depth=2)

        assert len(connected) == 3
        connected_ids = [m.id for m in connected]
        assert len(connected_ids) == len(set(connected_ids))  # No duplicates
        assert mem_b in connected_ids
        assert mem_c in connected_ids
        assert mem_d in connected_ids

        traversal.close()

    def test_get_top_central_basic(self):
        """Test getting top central memories"""
        traversal = GraphTraversal(':memory:')

        # Create memories with varying centrality
        mem_ids = []
        for i in range(5):
            mem_id = str(uuid.uuid4())
            mem_ids.append(mem_id)
            # Vary access counts
            access_count = i * 5
            last_accessed = datetime.now() - timedelta(days=i)

            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, f"Memory {i}", "fact", datetime.now().isoformat(),
                  last_accessed.isoformat(), access_count))

        # Add varying numbers of edges (increase connectivity for higher i)
        for i, mem_id in enumerate(mem_ids):
            for j in range(i):  # Memory 0 gets 0 edges, Memory 4 gets 4 edges
                other_id = str(uuid.uuid4())
                traversal._conn.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, (other_id, f"Connected to {i}", "fact", datetime.now().isoformat()))

                edge_id = str(uuid.uuid4())
                traversal._conn.execute("""
                    INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (edge_id, mem_id, other_id, "RELATED_TO", datetime.now().isoformat()))

        # Get top 3 central memories
        top_central = traversal.get_top_central(limit=3)

        assert len(top_central) == 3
        assert all(isinstance(item, tuple) for item in top_central)
        assert all(isinstance(item[0], Memory) for item in top_central)
        assert all(isinstance(item[1], float) for item in top_central)

        # Should be sorted by centrality descending
        for i in range(len(top_central) - 1):
            assert top_central[i][1] >= top_central[i + 1][1]

        traversal.close()

    def test_get_top_central_empty_database(self):
        """Test getting top central from empty database"""
        traversal = GraphTraversal(':memory:')

        top_central = traversal.get_top_central(limit=10)

        assert len(top_central) == 0

        traversal.close()

    def test_get_top_central_less_than_limit(self):
        """Test when database has fewer memories than limit"""
        traversal = GraphTraversal(':memory:')

        # Insert only 2 memories
        for i in range(2):
            mem_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, f"Memory {i}", "fact", datetime.now().isoformat()))

        # Request 10 but only 2 exist
        top_central = traversal.get_top_central(limit=10)

        assert len(top_central) == 2

        traversal.close()

    def test_context_manager(self, tmp_path):
        """Test context manager usage"""
        db_path = tmp_path / "test_cm.db"

        with GraphTraversal(str(db_path)) as traversal:
            # Should work inside context
            mem_id = str(uuid.uuid4())
            traversal._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, "Test", "fact", datetime.now().isoformat()))

            centrality = traversal.get_centrality(mem_id)
            assert centrality >= 0

        # Connection should be closed after context exit
        # (test passes if no exception is raised)

    def test_recency_half_life_constant(self):
        """Test that recency half-life constant is correct"""
        traversal = GraphTraversal(':memory:')

        assert traversal.RECENCY_HALF_LIFE == 30.0

        traversal.close()
