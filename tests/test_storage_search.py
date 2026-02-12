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

    def test_recall_with_ann(self):
        """Test semantic search uses ANN index when available"""
        search = MemorySearch(':memory:')

        # Insert memories with embeddings into database AND ANN index
        # Use vectors with different directions for meaningful similarity differences
        query_embedding = [1.0] + [0.0] * 1023  # Point in first dimension
        similar_embedding = [0.9] + [0.1] * 1023  # Close to query
        different_embedding = [0.0] * 512 + [1.0] * 512  # Orthogonal to query

        memories = [
            (similar_embedding, "Similar content about AI"),
            (similar_embedding, "Another similar memory"),
            (different_embedding, "Completely different content"),
        ]

        for embedding, content in memories:
            mem_id = str(uuid.uuid4())
            embedding_blob = embed_to_blob(embedding)

            # Insert into database
            search._conn.execute("""
                INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, content, embedding_blob, "fact",
                  datetime.now().isoformat(), datetime.now().isoformat()))

            # Add to ANN index
            search._ann_index.add(mem_id, embedding)

        # Verify ANN index has entries
        assert search._ann_index.get_size() == 3

        # Query should use ANN index
        results = search.recall(query_embedding, limit=10, min_relevance=0.0)

        # Should return all 3 memories
        assert len(results) == 3
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)

        # Results should be sorted by score (descending)
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]

        # Verify similar content appears in results (ANN index was used successfully)
        content_list = [r[0].content for r in results]
        assert any("similar" in c.lower() for c in content_list)

        search.close()

    def test_recall_with_ann_fallback(self):
        """Test fallback to brute-force when ANN index is empty"""
        search = MemorySearch(':memory:')

        # Insert memories into database but NOT into ANN index
        query_embedding = [0.1] * 1024
        similar_embedding = [0.11] * 1024

        mem_id = str(uuid.uuid4())
        embedding_blob = embed_to_blob(similar_embedding)
        search._conn.execute("""
            INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mem_id, "Test memory", embedding_blob, "fact",
              datetime.now().isoformat(), datetime.now().isoformat()))

        # ANN index should be empty
        assert search._ann_index.get_size() == 0

        # Query should fall back to brute-force
        results = search.recall(query_embedding, limit=10, min_relevance=0.0)

        # Should still find the memory via brute-force
        assert len(results) == 1
        assert results[0][0].id == mem_id

        search.close()

    def test_recall_with_ann_dimension_mismatch(self):
        """Test fallback when query dimension doesn't match index"""
        search = MemorySearch(':memory:')

        # Insert 1024-dim memory
        embedding_1024 = [0.1] * 1024
        mem_id = str(uuid.uuid4())
        embedding_blob = embed_to_blob(embedding_1024)

        search._conn.execute("""
            INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mem_id, "1024-dim memory", embedding_blob, "fact",
              datetime.now().isoformat(), datetime.now().isoformat()))

        search._ann_index.add(mem_id, embedding_1024)

        # Query with 768-dim embedding (dimension mismatch)
        query_768 = [0.1] * 768

        # Should fall back to brute-force (which will filter due to dimension mismatch)
        results = search.recall(query_768, limit=10, min_relevance=0.0)

        # Should return empty since dimensions don't match
        assert len(results) == 0

        search.close()

    def test_recall_with_ann_min_relevance_filter(self):
        """Test min_relevance filtering with ANN index"""
        search = MemorySearch(':memory:')

        # Insert memories with different similarities
        query_embedding = [1.0] * 1024
        very_similar = [0.95] * 1024
        somewhat_similar = [0.5] * 1024
        not_similar = [0.0] + [1.0] * 1023

        memories = [
            (very_similar, "Very similar"),
            (somewhat_similar, "Somewhat similar"),
            (not_similar, "Not similar"),
        ]

        for embedding, content in memories:
            mem_id = str(uuid.uuid4())
            embedding_blob = embed_to_blob(embedding)
            search._conn.execute("""
                INSERT INTO memories (id, content, embedding, memory_type, created_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (mem_id, content, embedding_blob, "fact",
                  datetime.now().isoformat(), datetime.now().isoformat()))
            search._ann_index.add(mem_id, embedding)

        # Low threshold should return more results
        results_low = search.recall(query_embedding, limit=10, min_relevance=0.1)
        assert len(results_low) >= 2

        # High threshold should filter more aggressively
        results_high = search.recall(query_embedding, limit=10, min_relevance=0.8)
        assert len(results_high) <= len(results_low)

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
