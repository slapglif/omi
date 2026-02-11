"""
Search Operations for Graph Palace Memories

This module handles semantic and full-text search operations:
- recall: Semantic search with vector embeddings and recency weighting
- full_text_search: FTS5 full-text search
- Recency decay scoring for time-sensitive relevance
"""

import sqlite3
import json
import math
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

from .models import Memory
from .schema import init_database
from .embeddings import blob_to_embed


class MemorySearch:
    """
    Memory search operations for Graph Palace.

    Handles semantic and full-text search with:
    - Vector similarity search (cosine similarity)
    - Recency decay: score = relevance * exp(-days/30)
    - Full-text search via FTS5
    - Thread-safe database operations
    """

    # Default half-life for recency decay (30 days)
    RECENCY_HALF_LIFE = 30.0

    # Target: <500ms for 1000 memories
    QUERY_TIMEOUT_MS = 500

    def __init__(self, db_path: str, enable_wal: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize Memory Search operations.

        Args:
            db_path: Path to SQLite database file (or ':memory:' for in-memory)
            enable_wal: Enable WAL mode for concurrent writes (default: True)
            conn: Optional shared connection (for :memory: databases in facade pattern)
        """
        self.db_path = Path(db_path) if db_path != ':memory:' else db_path
        self._owns_connection = conn is None

        if conn is not None:
            # Use shared connection (facade pattern)
            self._conn = conn
            self._db_lock = threading.Lock()
        else:
            # Create parent directory if needed (skip for :memory:)
            if self.db_path != ':memory:':
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create persistent connection
            # check_same_thread=False allows multi-threaded access (safe with WAL mode)
            # isolation_level=None enables autocommit mode for better concurrency
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
                timeout=30.0
            )

            # Thread lock for serializing database operations
            self._db_lock = threading.Lock()

            # Initialize database schema
            init_database(self._conn, enable_wal)

    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate recency score.
        Formula: exp(-days_ago / half_life)
        """
        if timestamp is None:
            return 0.0
        days_ago = (datetime.now() - timestamp).days
        return math.exp(-days_ago / self.RECENCY_HALF_LIFE)

    def recall(self,
               query_embedding: List[float],
               limit: int = 10,
               min_relevance: float = 0.7) -> List[Tuple[Memory, float]]:
        """
        Semantic search with recency weighting.

        Algorithm:
        1. Calculate cosine similarity between query and all memories (vectorized)
        2. Apply recency decay: score = relevance * exp(-days/30)
        3. Sort by final score and return top results

        Target: <500ms for 1000 memories

        Args:
            query_embedding: Query vector (1024-dim)
            limit: Max results to return
            min_relevance: Minimum similarity threshold

        Returns:
            List of (Memory, final_score) tuples
        """
        if not query_embedding:
            return []

        # Convert query to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        # Fetch all memories with embeddings
        memory_data = []
        embeddings = []

        cursor = self._conn.execute("""
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories WHERE embedding IS NOT NULL
        """)

        for row in cursor:
            embedding = blob_to_embed(row[2])
            if not embedding or len(embedding) != len(query_embedding):
                continue

            memory_data.append(row)
            embeddings.append(embedding)

        if not embeddings:
            return []

        # Vectorized cosine similarity calculation
        # Convert to numpy matrix: (n_memories, embedding_dim)
        embeddings_matrix = np.array(embeddings, dtype=np.float32)

        # Compute norms for all embeddings: (n_memories,)
        norms = np.linalg.norm(embeddings_matrix, axis=1)

        # Compute dot products: (n_memories,)
        dots = np.dot(embeddings_matrix, query_vec)

        # Compute cosine similarities: (n_memories,)
        # Avoid division by zero
        similarities = np.divide(dots, norms * query_norm,
                                out=np.zeros_like(dots),
                                where=(norms * query_norm) > 0)

        # Filter by min_relevance
        valid_indices = np.where(similarities >= min_relevance)[0]

        if len(valid_indices) == 0:
            return []

        # Build results for valid memories
        results = []
        now = datetime.now()

        for idx in valid_indices:
            row = memory_data[idx]
            relevance = float(similarities[idx])

            # Calculate recency score
            last_accessed = datetime.fromisoformat(row[6]) if row[6] else now
            recency = self._calculate_recency_score(last_accessed)

            # Weight: 70% relevance, 30% recency
            final_score = min((relevance * 0.7) + (recency * 0.3), 1.0)

            memory = Memory(
                id=row[0],
                content=row[1],
                embedding=embeddings[idx],
                memory_type=row[3],
                confidence=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                last_accessed=last_accessed,
                access_count=row[7],
                instance_ids=json.loads(row[8]) if row[8] else [],
                content_hash=row[9]
            )
            results.append((memory, final_score))

        # Sort by final score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def full_text_search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Full-text search using FTS5.

        Args:
            query: Search query text
            limit: Max results

        Returns:
            List of matching memories
        """
        memories = []

        # Use FTS5 MATCH via standalone FTS table
        cursor = self._conn.execute("""
            SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                   m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.memory_id
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        for row in cursor:
            embedding = blob_to_embed(row[2]) if row[2] else None
            memory = Memory(
                id=row[0],
                content=row[1],
                embedding=embedding,
                memory_type=row[3],
                confidence=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                last_accessed=datetime.fromisoformat(row[6]) if row[6] else None,
                access_count=row[7],
                instance_ids=json.loads(row[8]) if row[8] else [],
                content_hash=row[9]
            )
            memories.append(memory)

        return memories

    def close(self) -> None:
        """Close connection and cleanup."""
        if self._owns_connection and hasattr(self, '_conn') and self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
