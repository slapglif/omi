"""
Comprehensive tests for storage.search module (MemorySearch)

Tests cover:
- Semantic search with vector embeddings
- Cosine similarity calculation
- Recency weighting
- Full-text search via FTS5
- Thread-safe operations
"""

import pytest
import sqlite3
import uuid
import numpy as np
from datetime import datetime, timedelta

from omi.storage.search import MemorySearch
from omi.storage.schema import init_database
from omi.storage.embeddings import embed_to_blob


class TestMemorySearch:
    """Test suite for MemorySearch class"""

    def test_init_with_memory_db(self):
        """Test initialization with in-memory database"""
        search = MemorySearch(':memory:')
        assert search.db_path == ':memory:'
        assert search._owns_connection is True
        search.close()

    def test_init_with_shared_connection(self, tmp_path):
        """Test initialization with shared connection"""
        db_path = tmp_path / "test_shared.db"
        conn = sqlite3.connect(str(db_path))
        init_database(conn, enable_wal=True)

        search = MemorySearch(str(db_path), conn=conn)
        assert search._owns_connection is False
        search.close()
        conn.close()

    def test_calculate_recency_score_recent(self):
        """Test recency score for recent timestamp"""
        search = MemorySearch(':memory:')
        timestamp = datetime.now() - timedelta(days=1)
        score = search._calculate_recency_score(timestamp)
        assert score > 0.9
        search.close()

    def test_calculate_recency_score_old(self):
        """Test recency score for old timestamp"""
        search = MemorySearch(':memory:')
        timestamp = datetime.now() - timedelta(days=90)
        score = search._calculate_recency_score(timestamp)
        assert 0.04 < score < 0.06
        search.close()

    def test_calculate_recency_score_none(self):
        """Test recency score with None timestamp"""
        search = MemorySearch(':memory:')
        score = search._calculate_recency_score(None)
        assert score == 0.0
        search.close()

    def test_recall_basic(self):
        """Test basic semantic search"""
        search = MemorySearch(':memory:')

        # Insert memories with embeddings
        query_embedding = [0.1] * 768
        similar_embedding = [0.11] * 768
        different_embedding = [0.9] * 768

        for embedding, content in [
            (similar_embedding, "Similar content"),
            (different_embedding, "Different content"),
        ]:
            mem_id = str(uuid.uuid4())
            embedding_blob = embed_to_blob(embedding)
            search._conn.execute("""
                INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, content, embedding_blob, "fact",
                  datetime.now().isoformat(), datetime.now().isoformat()))

        results = search.recall(query_embedding, limit=10, min_relevance=0.7)

        assert len(results) > 0
        assert all(isinstance(item, tuple) for item in results)
        search.close()

    def test_recall_empty_query(self):
        """Test recall with empty query embedding"""
        search = MemorySearch(':memory:')
        results = search.recall([], limit=10)
        assert len(results) == 0
        search.close()

    def test_recall_zero_norm_query(self):
        """Test recall with zero-norm query vector"""
        search = MemorySearch(':memory:')
        results = search.recall([0.0] * 768, limit=10)
        assert len(results) == 0
        search.close()

    def test_recall_no_memories(self):
        """Test recall on empty database"""
        search = MemorySearch(':memory:')
        results = search.recall([0.1] * 768, limit=10)
        assert len(results) == 0
        search.close()

    def test_recall_min_relevance_filter(self):
        """Test that min_relevance filters low-similarity results"""
        search = MemorySearch(':memory:')

        # Insert two memories with known similarities
        query_embedding = [1.0] * 768  # All 1.0
        similar_embedding = [0.95] * 768  # Very similar (high cosine ~0.95)
        different_embedding = [0.0] + [1.0] * 767  # Very different (low cosine)

        for embedding, content in [
            (similar_embedding, "Similar"),
            (different_embedding, "Different"),
        ]:
            mem_id = str(uuid.uuid4())
            embedding_blob = embed_to_blob(embedding)
            search._conn.execute("""
                INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, content, embedding_blob, "fact",
                  datetime.now().isoformat(), datetime.now().isoformat()))

        # Low threshold should return more results
        results_low = search.recall(query_embedding, limit=10, min_relevance=0.1)
        assert len(results_low) >= 1

        # High threshold should filter more aggressively
        results_high = search.recall(query_embedding, limit=10, min_relevance=0.9)
        assert len(results_high) <= len(results_low)

        search.close()

    def test_recall_limit(self):
        """Test that limit parameter works"""
        search = MemorySearch(':memory:')

        # Insert 5 memories
        query_embedding = [0.5] * 768
        for i in range(5):
            mem_id = str(uuid.uuid4())
            embedding = [0.5 + i*0.01] * 768
            embedding_blob = embed_to_blob(embedding)
            search._conn.execute("""
                INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, f"Memory {i}", embedding_blob, "fact",
                  datetime.now().isoformat(), datetime.now().isoformat()))

        # Request only 2
        results = search.recall(query_embedding, limit=2, min_relevance=0.0)
        assert len(results) <= 2
        search.close()

    def test_recall_sorted_by_score(self):
        """Test that results are sorted by final score"""
        search = MemorySearch(':memory:')

        # Insert memories with different similarities
        query_embedding = [1.0] + [0.0] * 767

        embeddings = [
            ([0.9] + [0.1] * 767, "High"),  # High similarity
            ([0.5] + [0.5] * 767, "Medium"),  # Medium similarity
            ([0.1] + [0.9] * 767, "Low"),  # Low similarity
        ]

        for embedding, content in embeddings:
            mem_id = str(uuid.uuid4())
            embedding_blob = embed_to_blob(embedding)
            search._conn.execute("""
                INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, content, embedding_blob, "fact",
                  datetime.now().isoformat(), datetime.now().isoformat()))

        results = search.recall(query_embedding, limit=10, min_relevance=0.0)

        # Should be sorted descending by score
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]

        search.close()

    def test_recall_recency_weighting(self):
        """Test that recency affects final score"""
        search = MemorySearch(':memory:')

        query_embedding = [0.5] * 768
        similar_embedding = [0.51] * 768

        # Recent memory
        recent_id = str(uuid.uuid4())
        recent_time = datetime.now() - timedelta(days=1)
        embedding_blob = embed_to_blob(similar_embedding)
        search._conn.execute("""
            INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (recent_id, "Recent", embedding_blob, "fact",
              recent_time.isoformat(), recent_time.isoformat()))

        # Old memory (same embedding, different recency)
        old_id = str(uuid.uuid4())
        old_time = datetime.now() - timedelta(days=90)
        search._conn.execute("""
            INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (old_id, "Old", embedding_blob, "fact",
              old_time.isoformat(), old_time.isoformat()))

        results = search.recall(query_embedding, limit=10, min_relevance=0.0)
        assert len(results) == 2

        # Recent should have higher score due to recency weighting
        results_dict = {r[0].id: r[1] for r in results}
        assert results_dict[recent_id] > results_dict[old_id]

        search.close()

    def test_full_text_search_basic(self):
        """Test basic full-text search"""
        search = MemorySearch(':memory:')

        # Insert memories
        mem_id = str(uuid.uuid4())
        search._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (mem_id, "Python programming language", "fact", datetime.now().isoformat()))

        search._conn.execute("""
            INSERT INTO memories_fts(memory_id, content)
            VALUES (?, ?)
        """, (mem_id, "Python programming language"))

        results = search.full_text_search("Python", limit=10)

        assert len(results) >= 1
        assert any("Python" in m.content for m in results)
        search.close()

    def test_full_text_search_no_results(self):
        """Test FTS with no matching results"""
        search = MemorySearch(':memory:')

        # Insert memory
        mem_id = str(uuid.uuid4())
        search._conn.execute("""
            INSERT INTO memories (id, content, memory_type, created_at)
            VALUES (?, ?, ?, ?)
        """, (mem_id, "Python programming", "fact", datetime.now().isoformat()))

        search._conn.execute("""
            INSERT INTO memories_fts(memory_id, content)
            VALUES (?, ?)
        """, (mem_id, "Python programming"))

        results = search.full_text_search("Haskell", limit=10)
        assert len(results) == 0
        search.close()

    def test_full_text_search_limit(self):
        """Test that FTS respects limit"""
        search = MemorySearch(':memory:')

        # Insert multiple matching memories
        for i in range(5):
            mem_id = str(uuid.uuid4())
            content = f"Python programming example {i}"
            search._conn.execute("""
                INSERT INTO memories (id, content, memory_type, created_at)
                VALUES (?, ?, ?, ?)
            """, (mem_id, content, "fact", datetime.now().isoformat()))

            search._conn.execute("""
                INSERT INTO memories_fts(memory_id, content)
                VALUES (?, ?)
            """, (mem_id, content))

        results = search.full_text_search("Python", limit=2)
        assert len(results) <= 2
        search.close()

    def test_context_manager(self, tmp_path):
        """Test context manager usage"""
        db_path = tmp_path / "test_cm.db"

        with MemorySearch(str(db_path)) as search:
            results = search.full_text_search("test", limit=10)
            assert len(results) == 0

    def test_recall_mismatched_embedding_dimensions(self):
        """Test that memories with wrong dimension embeddings are skipped"""
        search = MemorySearch(':memory:')

        # Insert memory with wrong dimension
        mem_id = str(uuid.uuid4())
        wrong_dim_embedding = [0.1] * 512  # Wrong dimension
        embedding_blob = embed_to_blob(wrong_dim_embedding)
        search._conn.execute("""
            INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mem_id, "Wrong dimension", embedding_blob, "fact",
              datetime.now().isoformat(), datetime.now().isoformat()))

        # Query with 768 dimensions
        query_embedding = [0.1] * 768
        results = search.recall(query_embedding, limit=10)

        # Should not crash, should skip mismatched memory
        assert len(results) == 0
        search.close()
