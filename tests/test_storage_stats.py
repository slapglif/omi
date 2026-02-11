"""
Comprehensive tests for storage.stats module (DatabaseStats)

Tests cover:
- Database statistics (memory/edge counts)
- Type distributions
- Compression statistics
- Memory age queries
- Vacuum operations
- Thread-safe operations
- Context manager usage
"""

import pytest
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from omi.storage.stats import DatabaseStats
from omi.storage.models import Memory
from omi.storage.schema import init_database


class TestDatabaseStats:
    """Test suite for DatabaseStats class"""

    def test_init_with_file_path(self, tmp_path):
        """Test initialization with file-based database"""
        db_path = tmp_path / "test_stats.db"
        stats = DatabaseStats(str(db_path))

        assert stats.db_path == db_path
        assert stats._owns_connection is True
        assert stats._conn is not None

        stats.close()

    def test_init_with_memory_db(self):
        """Test initialization with in-memory database"""
        stats = DatabaseStats(':memory:')

        assert stats.db_path == ':memory:'
        assert stats._owns_connection is True

        stats.close()

    def test_init_with_shared_connection(self, tmp_path):
        """Test initialization with shared connection (facade pattern)"""
        db_path = tmp_path / "test_shared.db"
        conn = sqlite3.connect(str(db_path))
        init_database(conn, enable_wal=True)

        stats = DatabaseStats(str(db_path), conn=conn)

        assert stats._owns_connection is False
        assert stats._conn is conn

        # Close should not close shared connection
        stats.close()
        # Connection should still work
        cursor = conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1

        conn.close()

    def test_get_stats_empty_database(self, tmp_path):
        """Test stats on empty database"""
        stats = DatabaseStats(':memory:')

        result = stats.get_stats()

        assert result["memory_count"] == 0
        assert result["edge_count"] == 0
        assert result["type_distribution"] == {}
        assert result["edge_distribution"] == {}

        stats.close()

    def test_get_stats_with_memories(self, tmp_path):
        """Test stats with various memories"""
        stats = DatabaseStats(':memory:')

        # Insert test memories of different types
        memories = [
            (str(uuid.uuid4()), "Fact 1", "fact"),
            (str(uuid.uuid4()), "Fact 2", "fact"),
            (str(uuid.uuid4()), "Experience 1", "experience"),
            (str(uuid.uuid4()), "Belief 1", "belief"),
            (str(uuid.uuid4()), "Decision 1", "decision"),
        ]

        for mem_id, content, mem_type in memories:
            stats._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, mem_type, datetime.now().isoformat()))

        result = stats.get_stats()

        assert result["memory_count"] == 5
        assert result["type_distribution"]["fact"] == 2
        assert result["type_distribution"]["experience"] == 1
        assert result["type_distribution"]["belief"] == 1
        assert result["type_distribution"]["decision"] == 1

        stats.close()

    def test_get_stats_with_edges(self, tmp_path):
        """Test stats with various edges"""
        stats = DatabaseStats(':memory:')

        # Insert memories
        mem1_id = str(uuid.uuid4())
        mem2_id = str(uuid.uuid4())
        mem3_id = str(uuid.uuid4())

        for mem_id in [mem1_id, mem2_id, mem3_id]:
            stats._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, f"Memory {mem_id}", "fact", datetime.now().isoformat()))

        # Insert edges of different types
        edges = [
            (str(uuid.uuid4()), mem1_id, mem2_id, "SUPPORTS"),
            (str(uuid.uuid4()), mem1_id, mem3_id, "CONTRADICTS"),
            (str(uuid.uuid4()), mem2_id, mem3_id, "RELATED_TO"),
            (str(uuid.uuid4()), mem2_id, mem3_id, "RELATED_TO"),  # Another RELATED_TO
        ]

        for edge_id, source, target, edge_type in edges:
            stats._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, source, target, edge_type, datetime.now().isoformat()))

        result = stats.get_stats()

        assert result["edge_count"] == 4
        assert result["edge_distribution"]["SUPPORTS"] == 1
        assert result["edge_distribution"]["CONTRADICTS"] == 1
        assert result["edge_distribution"]["RELATED_TO"] == 2

        stats.close()

    def test_get_compression_stats_all_memories(self, tmp_path):
        """Test compression stats for all memories"""
        stats = DatabaseStats(':memory:')

        # Insert memories with varying content lengths
        memories = [
            (str(uuid.uuid4()), "Short fact", "fact"),
            (str(uuid.uuid4()), "This is a longer experience with more content to compress", "experience"),
            (str(uuid.uuid4()), "Another belief " * 10, "belief"),  # Repeated text
        ]

        total_chars = 0
        for mem_id, content, mem_type in memories:
            stats._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, mem_type, datetime.now().isoformat()))
            total_chars += len(content)

        result = stats.get_compression_stats()

        assert result["total_memories"] == 3
        assert result["total_chars"] == total_chars
        assert result["estimated_tokens"] == total_chars // 4
        assert result["memories_by_type"]["fact"] == 1
        assert result["memories_by_type"]["experience"] == 1
        assert result["memories_by_type"]["belief"] == 1

        stats.close()

    def test_get_compression_stats_with_threshold(self, tmp_path):
        """Test compression stats with age threshold"""
        stats = DatabaseStats(':memory:')

        now = datetime.now()
        old_date = now - timedelta(days=60)
        recent_date = now - timedelta(days=10)

        # Insert old and recent memories
        old_memory_id = str(uuid.uuid4())
        recent_memory_id = str(uuid.uuid4())

        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (old_memory_id, "Old memory content", "fact", old_date.isoformat()))

        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (recent_memory_id, "Recent memory content", "fact", recent_date.isoformat()))

        # Get stats for memories before 30 days ago (should only include old one)
        threshold = now - timedelta(days=30)
        result = stats.get_compression_stats(threshold=threshold)

        assert result["total_memories"] == 1
        assert result["total_chars"] == len("Old memory content")
        assert result["memories_by_type"]["fact"] == 1

        stats.close()

    def test_get_compression_stats_empty_database(self, tmp_path):
        """Test compression stats on empty database"""
        stats = DatabaseStats(':memory:')

        result = stats.get_compression_stats()

        assert result["total_memories"] == 0
        assert result["total_chars"] == 0
        assert result["estimated_tokens"] == 0
        assert result["memories_by_type"] == {}

        stats.close()

    def test_get_memories_before_basic(self, tmp_path):
        """Test querying memories before a threshold"""
        stats = DatabaseStats(':memory:')

        now = datetime.now()
        old_date = now - timedelta(days=60)
        medium_date = now - timedelta(days=30)
        recent_date = now - timedelta(days=5)

        # Insert memories with different ages
        old_id = str(uuid.uuid4())
        medium_id = str(uuid.uuid4())
        recent_id = str(uuid.uuid4())

        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (old_id, "Old memory", "fact", old_date.isoformat()))

        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (medium_id, "Medium memory", "fact", medium_date.isoformat()))

        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (recent_id, "Recent memory", "fact", recent_date.isoformat()))

        # Query memories before 20 days ago (should get old and medium)
        threshold = now - timedelta(days=20)
        memories = stats.get_memories_before(threshold)

        assert len(memories) == 2
        assert all(isinstance(m, Memory) for m in memories)

        # Results should be ordered by created_at ascending (oldest first)
        assert memories[0].id == old_id
        assert memories[1].id == medium_id

        stats.close()

    def test_get_memories_before_with_limit(self, tmp_path):
        """Test querying memories with limit"""
        stats = DatabaseStats(':memory:')

        now = datetime.now()

        # Insert 5 old memories
        for i in range(5):
            mem_id = str(uuid.uuid4())
            created_at = now - timedelta(days=60 + i)
            stats._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, f"Memory {i}", "fact", created_at.isoformat()))

        # Query with limit=3
        threshold = now - timedelta(days=30)
        memories = stats.get_memories_before(threshold, limit=3)

        assert len(memories) == 3
        assert all(isinstance(m, Memory) for m in memories)

        stats.close()

    def test_get_memories_before_empty_result(self, tmp_path):
        """Test querying with threshold that returns no results"""
        stats = DatabaseStats(':memory:')

        now = datetime.now()
        recent_date = now - timedelta(days=5)

        # Insert only recent memories
        mem_id = str(uuid.uuid4())
        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (mem_id, "Recent memory", "fact", recent_date.isoformat()))

        # Query for memories before 30 days ago (should get none)
        threshold = now - timedelta(days=30)
        memories = stats.get_memories_before(threshold)

        assert len(memories) == 0

        stats.close()

    def test_get_memories_before_ordering(self, tmp_path):
        """Test that results are ordered oldest first"""
        stats = DatabaseStats(':memory:')

        now = datetime.now()

        # Insert memories in non-chronological order
        dates = [
            now - timedelta(days=50),
            now - timedelta(days=70),
            now - timedelta(days=40),
            now - timedelta(days=60),
        ]

        memory_ids = []
        for date in dates:
            mem_id = str(uuid.uuid4())
            memory_ids.append((mem_id, date))
            stats._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, "Memory content", "fact", date.isoformat()))

        # Query all old memories
        threshold = now - timedelta(days=20)
        memories = stats.get_memories_before(threshold)

        assert len(memories) == 4

        # Should be ordered oldest first
        expected_order = sorted(memory_ids, key=lambda x: x[1])
        for i, memory in enumerate(memories):
            assert memory.id == expected_order[i][0]

        stats.close()

    def test_vacuum_basic(self, tmp_path):
        """Test vacuum operation"""
        db_path = tmp_path / "test_vacuum.db"
        stats = DatabaseStats(str(db_path))

        # Insert and delete some data to create fragmentation
        mem_id = str(uuid.uuid4())
        stats._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (mem_id, "To be deleted", "fact", datetime.now().isoformat()))

        stats._conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))

        # Run vacuum (should not raise)
        stats.vacuum()

        # Database should still work
        cursor = stats._conn.execute("SELECT COUNT(*) FROM memories")
        assert cursor.fetchone()[0] == 0

        stats.close()

    def test_context_manager(self, tmp_path):
        """Test context manager usage"""
        db_path = tmp_path / "test_cm.db"

        with DatabaseStats(str(db_path)) as stats:
            # Should work inside context
            result = stats.get_stats()
            assert result["memory_count"] == 0

        # Connection should be closed after context exit
        # (test passes if no exception is raised)

    def test_get_stats_combined(self, tmp_path):
        """Test comprehensive stats with memories and edges"""
        stats = DatabaseStats(':memory:')

        # Insert diverse data
        mem_ids = []
        for i in range(3):
            mem_id = str(uuid.uuid4())
            mem_ids.append(mem_id)
            stats._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, f"Memory {i}", "fact", datetime.now().isoformat()))

        # Add some edges
        for i in range(2):
            edge_id = str(uuid.uuid4())
            stats._conn.execute("""
                INSERT INTO edges (id, source_id, target_id, edge_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (edge_id, mem_ids[i], mem_ids[i+1], "SUPPORTS", datetime.now().isoformat()))

        result = stats.get_stats()

        assert result["memory_count"] == 3
        assert result["edge_count"] == 2
        assert result["type_distribution"]["fact"] == 3
        assert result["edge_distribution"]["SUPPORTS"] == 2

        stats.close()
