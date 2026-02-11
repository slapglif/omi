"""
Graph Operations for Graph Palace

This module handles edge operations and graph relationships:
- create_edge: Create relationships between memories
- delete_edge: Remove edges by ID
- get_edges: Query edges connected to a memory
- get_neighbors: Get memories directly connected via edges
- find_contradictions: Find memories connected via CONTRADICTS edges
- get_supporting_evidence: Find memories connected via SUPPORTS edges
"""

import sqlite3
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from .models import Memory, Edge
from .schema import init_database
from .embeddings import blob_to_embed


class GraphOperations:
    """
    Graph edge operations for Graph Palace.

    Handles relationship management between memory nodes:
    - Edge creation with type validation
    - Edge queries (by memory, by type)
    - Neighbor discovery (BFS-style edge traversal)
    - Semantic relationship queries (contradictions, support)
    - Thread-safe database operations
    """

    # Valid edge types
    EDGE_TYPES = {"SUPPORTS", "CONTRADICTS", "RELATED_TO", "DEPENDS_ON", "POSTED", "DISCUSSED"}

    def __init__(self, db_path: str, enable_wal: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize Graph Operations.

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

    def _validate_edge_type(self, edge_type: str) -> None:
        """Validate edge type."""
        if edge_type not in self.EDGE_TYPES:
            raise ValueError(f"Invalid edge_type: {edge_type}. Must be one of: {self.EDGE_TYPES}")

    def create_edge(self,
                   source_id: str,
                   target_id: str,
                   edge_type: str,
                   strength: Optional[float] = None) -> str:
        """
        Create a relationship edge between memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            edge_type: One of (SUPPORTS, CONTRADICTS, RELATED_TO, DEPENDS_ON, POSTED, DISCUSSED)
            strength: Relationship strength 0.0-1.0

        Returns:
            edge_id: UUID of the created edge
        """
        self._validate_edge_type(edge_type)

        edge_id = str(uuid.uuid4())

        self._conn.execute("""
            INSERT INTO edges (id, source_id, target_id, edge_type, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (edge_id, source_id, target_id, edge_type, strength, datetime.now().isoformat()))
        self._conn.commit()

        return edge_id

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by ID."""
        cursor = self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_edges(self, memory_id: str, edge_type: Optional[str] = None) -> List[Edge]:
        """
        Get all edges connected to a memory.

        Args:
            memory_id: The memory ID
            edge_type: Optional filter by edge type

        Returns:
            List of Edge objects
        """
        edges = []

        if edge_type:
            cursor = self._conn.execute("""
                SELECT id, source_id, target_id, edge_type, strength, created_at
                FROM edges WHERE (source_id = ? OR target_id = ?) AND edge_type = ?
            """, (memory_id, memory_id, edge_type))
        else:
            cursor = self._conn.execute("""
                SELECT id, source_id, target_id, edge_type, strength, created_at
                FROM edges WHERE source_id = ? OR target_id = ?
            """, (memory_id, memory_id))

        for row in cursor:
            edges.append(Edge(
                id=row[0],
                source_id=row[1],
                target_id=row[2],
                edge_type=row[3],
                strength=row[4],
                created_at=datetime.fromisoformat(row[5]) if row[5] else None
            ))

        return edges

    def get_neighbors(self, memory_id: str, edge_type: Optional[str] = None) -> List[Memory]:
        """
        Get all memories directly connected to a memory.

        Args:
            memory_id: The memory ID
            edge_type: Optional filter by edge type

        Returns:
            List of Memory objects
        """
        memories = []

        if edge_type:
            cursor = self._conn.execute("""
                SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                       m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
                FROM memories m
                JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND m.id != ?
                AND e.edge_type = ?
            """, (memory_id, memory_id, memory_id, edge_type))
        else:
            cursor = self._conn.execute("""
                SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                       m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
                FROM memories m
                JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND m.id != ?
            """, (memory_id, memory_id, memory_id))

        for row in cursor:
            embedding = blob_to_embed(row[2]) if row[2] else None
            memories.append(Memory(
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
            ))

        return memories

    def find_contradictions(self, memory_id: str) -> List[Memory]:
        """
        Find memories that contradict a given memory.

        Args:
            memory_id: Memory to check

        Returns:
            List of contradicting memories
        """
        # Get memories connected via CONTRADICTS edge
        contradictions = self.get_neighbors(memory_id, edge_type="CONTRADICTS")
        return contradictions

    def get_supporting_evidence(self, memory_id: str) -> List[Memory]:
        """
        Get memories that support a given memory.

        Args:
            memory_id: Memory to get evidence for

        Returns:
            List of supporting memories
        """
        return self.get_neighbors(memory_id, edge_type="SUPPORTS")

    def close(self) -> None:
        """Close connection and cleanup."""
        if self._owns_connection and hasattr(self, '_conn') and self._conn:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
