"""
Graph Traversal Operations for Graph Palace

This module handles graph traversal and centrality calculations:
- get_connected: BFS traversal to find memories within N hops
- get_centrality: Calculate centrality score for a memory
- get_top_central: Retrieve the most central (hub) memories
"""

import sqlite3
import json
import math
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from collections import deque

from .models import Memory
from .schema import init_database
from .embeddings import blob_to_embed


class GraphTraversal:
    """
    Graph traversal and centrality operations for Graph Palace.

    Provides graph algorithms for navigating memory relationships:
    - BFS traversal for context loading
    - Centrality calculation for hub detection
    - Weighted scoring based on degree, access frequency, and recency
    """

    # Default half-life for recency decay (30 days)
    RECENCY_HALF_LIFE = 30.0

    def __init__(self, db_path: str, enable_wal: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize Graph Traversal operations.

        Args:
            db_path: Path to SQLite database file (or ':memory:' for in-memory)
            enable_wal: Enable WAL mode for concurrent writes (default: True)
            conn: Optional shared connection (for :memory: databases in facade pattern)
        """
        self.db_path = Path(db_path) if db_path != ':memory:' else db_path
        self._enable_wal = enable_wal
        self._owns_connection = conn is None

        if conn is not None:
            # Use shared connection (facade pattern)
            self._conn = conn
        else:
            # Initialize database
            if self.db_path != ':memory:':
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create persistent connection
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
                timeout=30.0
            )

            # Initialize schema
            init_database(self._conn, enable_wal=self._enable_wal)

    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate recency score.
        Formula: exp(-days_ago / half_life)

        Args:
            timestamp: The timestamp to calculate recency for

        Returns:
            Recency score (0.0-1.0)
        """
        if timestamp is None:
            return 0.0
        days_ago = (datetime.now() - timestamp).days
        return math.exp(-days_ago / self.RECENCY_HALF_LIFE)

    def get_centrality(self, memory_id: str) -> float:
        """
        Calculate centrality score for a memory.

        Combined score based on:
        - Degree centrality (number of edges) - weight: 40%
        - Access frequency (access_count) - weight: 35%
        - Recency (last_accessed) - weight: 25%

        Hub memories = high centrality, many connections

        Args:
            memory_id: The memory ID

        Returns:
            Centrality score (0.0-1.0)
        """
        # Get memory stats
        cursor = self._conn.execute("""
            SELECT access_count, last_accessed, created_at
            FROM memories WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        if not row:
            return 0.0

        access_count = row[0] or 0
        last_accessed = datetime.fromisoformat(row[1]) if row[1] else datetime.now()

        # Count edges (degree centrality)
        cursor = self._conn.execute("""
            SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?
        """, (memory_id, memory_id))
        edge_count = cursor.fetchone()[0]

        # Normalize metrics (0-1 scale)
        # Assume max 100 edges for normalization
        degree_score = min(edge_count / 100.0, 1.0)

        # Access frequency (log scale to reduce dominance of very high counts)
        access_score = min(math.log1p(access_count) / math.log1p(100), 1.0)

        # Recency decay
        recency_score = self._calculate_recency_score(last_accessed)

        # Weighted combination
        centrality = (degree_score * 0.40) + (access_score * 0.35) + (recency_score * 0.25)

        return round(centrality, 4)

    def get_connected(self, memory_id: str, depth: int = 2) -> List[Memory]:
        """
        BFS traversal to get all memories connected up to N hops.

        Used for: loading context around a memory

        Args:
            memory_id: Starting memory ID
            depth: Maximum traversal depth (default: 2)

        Returns:
            List of Memory objects (excluding starting memory)
        """
        visited = {memory_id}
        result = []
        queue = deque([(memory_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get neighbors
            cursor = self._conn.execute("""
                SELECT DISTINCT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                       m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
                FROM memories m
                JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND m.id != ?
            """, (current_id, current_id, current_id))

            for row in cursor:
                neighbor_id = row[0]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)

                    embedding = blob_to_embed(row[2]) if row[2] else None
                    memory = Memory(
                        id=neighbor_id,
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
                    result.append(memory)
                    queue.append((neighbor_id, current_depth + 1))

        return result

    def get_top_central(self, limit: int = 10) -> List[Tuple[Memory, float]]:
        """
        Get the most central (hub) memories.

        Args:
            limit: Number of results

        Returns:
            List of (Memory, centrality_score) tuples
        """
        memories = []

        # Single aggregated query: fetch all memory data with edge counts
        cursor = self._conn.execute("""
            SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                   m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash,
                   COUNT(e.id) as edge_count
            FROM memories m
            LEFT JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
            GROUP BY m.id
        """)

        for row in cursor:
            # Extract memory data
            memory_id = row[0]
            access_count = row[7] or 0
            last_accessed = datetime.fromisoformat(row[6]) if row[6] else datetime.now()
            edge_count = row[10]

            # Calculate centrality score (same algorithm as get_centrality)
            # Degree centrality (40% weight)
            degree_score = min(edge_count / 100.0, 1.0)

            # Access frequency (35% weight, log scale)
            access_score = min(math.log1p(access_count) / math.log1p(100), 1.0)

            # Recency decay (25% weight)
            recency_score = self._calculate_recency_score(last_accessed)

            # Weighted combination
            centrality = (degree_score * 0.40) + (access_score * 0.35) + (recency_score * 0.25)
            centrality = round(centrality, 4)

            # Build Memory object
            embedding = blob_to_embed(row[2]) if row[2] else None
            memory = Memory(
                id=memory_id,
                content=row[1],
                embedding=embedding,
                memory_type=row[3],
                confidence=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None,
                last_accessed=last_accessed,
                access_count=access_count,
                instance_ids=json.loads(row[8]) if row[8] else [],
                content_hash=row[9]
            )

            memories.append((memory, centrality))

        # Sort by centrality (descending) and limit
        memories.sort(key=lambda x: x[1], reverse=True)
        return memories[:limit]

    def close(self) -> None:
        """Close database connection."""
        if self._owns_connection and hasattr(self, '_conn') and self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
