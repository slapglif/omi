"""
Database Statistics and Maintenance for Graph Palace

This module handles database statistics and maintenance operations:
- get_stats: Get memory and edge counts with type distributions
- get_compression_stats: Calculate compression statistics for memories
- get_memories_before: Query old memories for summarization/compression
- vacuum: Optimize database and reclaim space
"""

import sqlite3
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

from .models import Memory
from .schema import init_database
from .embeddings import blob_to_embed


class DatabaseStats:
    """
    Database statistics and maintenance operations for Graph Palace.

    Provides tools for:
    - Database statistics (memory/edge counts, type distributions)
    - Compression analysis (token estimation, content stats)
    - Memory age queries (for archival/summarization)
    - Database optimization (vacuum)
    """

    def __init__(self, db_path: str, enable_wal: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize Database Statistics operations.

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

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with memory_count, edge_count, type_distribution, edge_distribution
        """
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        memory_count = cursor.fetchone()[0]

        cursor = self._conn.execute("SELECT COUNT(*) FROM edges")
        edge_count = cursor.fetchone()[0]

        cursor = self._conn.execute("""
            SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type
        """)
        type_distribution = {row[0]: row[1] for row in cursor}

        cursor = self._conn.execute("""
            SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type
        """)
        edge_distribution = {row[0]: row[1] for row in cursor}

        return {
            "memory_count": memory_count,
            "edge_count": edge_count,
            "type_distribution": type_distribution,
            "edge_distribution": edge_distribution
        }

    def get_compression_stats(self, threshold: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate compression statistics for memories (optionally filtered by age).

        Used for dry-run reporting to estimate token savings before compression.

        Args:
            threshold: Optional datetime threshold - only include memories created before this.
                      If None, calculates stats for all memories.

        Returns:
            Dict with total_memories, total_chars, estimated_tokens, memories_by_type
        """
        if threshold is not None:
            # Query memories before threshold
            cursor = self._conn.execute("""
                SELECT content, memory_type FROM memories
                WHERE created_at < ?
            """, (threshold.isoformat(),))
        else:
            # Query all memories
            cursor = self._conn.execute("""
                SELECT content, memory_type FROM memories
            """)

        total_memories = 0
        total_chars = 0
        memories_by_type: Dict[str, int] = {}

        for row in cursor:
            content = row[0]
            memory_type = row[1]

            total_memories += 1
            total_chars += len(content)

            memories_by_type[memory_type] = memories_by_type.get(memory_type, 0) + 1

        # Estimate tokens using common approximation: 1 token ≈ 4 characters
        estimated_tokens = total_chars // 4

        return {
            "total_memories": total_memories,
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "memories_by_type": memories_by_type
        }

    def get_memories_before(self, threshold: datetime, limit: Optional[int] = None) -> List[Memory]:
        """
        Query memories older than a threshold datetime.

        Used for: finding old memories for summarization/compression

        Args:
            threshold: Datetime threshold - returns memories created before this
            limit: Optional max number of results

        Returns:
            List of Memory objects created before threshold, ordered by created_at ascending (oldest first)
        """
        memories = []

        with self._db_lock:
            if limit is not None:
                cursor = self._conn.execute("""
                    SELECT id, content, embedding, memory_type, confidence,
                           created_at, last_accessed, access_count, instance_ids, content_hash
                    FROM memories
                    WHERE created_at < ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (threshold.isoformat(), limit))
            else:
                cursor = self._conn.execute("""
                    SELECT id, content, embedding, memory_type, confidence,
                           created_at, last_accessed, access_count, instance_ids, content_hash
                    FROM memories
                    WHERE created_at < ?
                    ORDER BY created_at ASC
                """, (threshold.isoformat(),))

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

    def vacuum(self) -> None:
        """Optimize database (reclaim space)."""
        self._conn.execute("VACUUM")
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._owns_connection and hasattr(self, '_conn') and self._conn:
            self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
