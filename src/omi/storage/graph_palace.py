"""
Graph Palace - Tier 3 Storage for OMI
SQLite-based graph storage with vector embeddings and semantic search.

This is the CORE of OMI's intelligence - everything depends on it.
- Semantic relationships between memories
- Centrality-weighted access
- Fast vector search with recency decay
- Graph traversal (BFS)
"""

import sqlite3
import json
import hashlib
import uuid
import math
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import struct
from omi.storage.schema import init_database


@dataclass
class Memory:
    """A memory node in the graph palace."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    memory_type: str = "experience"  # fact | experience | belief | decision
    confidence: Optional[float] = None  # 0.0-1.0 for beliefs
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    instance_ids: Optional[List[str]] = None
    content_hash: Optional[str] = None  # SHA-256 for integrity

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.content_hash is None and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        if self.instance_ids is None:
            self.instance_ids = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "instance_ids": self.instance_ids,
            "content_hash": self.content_hash
        }


@dataclass
class Edge:
    """A relationship edge between memories."""
    id: str
    source_id: str
    target_id: str
    edge_type: str  # SUPPORTS | CONTRADICTS | RELATED_TO | DEPENDS_ON | POSTED | DISCUSSED
    strength: Optional[float] = None  # 0.0-1.0
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now()


class GraphPalace:
    """
    Tier 3: Graph Palace - Semantic memories, relationships, beliefs
    
    Pattern: Structured, queryable, centrality-weighted
    Lifetime: Indefinite (with decay)
    
    Features:
    - In-memory embeddings with SQLite fallback
    - Semantic search with cosine similarity
    - Recency decay: score = relevance * exp(-days/30)
    - Graph traversal (BFS)
    - Centrality calculation (hub detection)
    - WAL mode for concurrent writes
    - Full-text search via FTS5
    """

    # Embedding dimension for bge-m3
    EMBEDDING_DIM = 1024
    
    # Valid memory types
    MEMORY_TYPES = {"fact", "experience", "belief", "decision"}
    
    # Valid edge types
    EDGE_TYPES = {"SUPPORTS", "CONTRADICTS", "RELATED_TO", "DEPENDS_ON", "POSTED", "DISCUSSED"}
    
    # Default half-life for recency decay (30 days)
    RECENCY_HALF_LIFE = 30.0
    
    # Target: <500ms for 1000 memories
    QUERY_TIMEOUT_MS = 500

    def __init__(self, db_path: Path, enable_wal: bool = True, embedding_dim: int = None):
        """
        Initialize Graph Palace.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for concurrent writes (default: True)
            embedding_dim: Embedding dimension (default: 1024 for bge-m3)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal = enable_wal
        self.embedding_dim = embedding_dim if embedding_dim is not None else self.EMBEDDING_DIM

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
        self._init_db()

        # In-memory embedding cache for fast access
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_loaded = False

    def _init_db(self) -> None:
        """Initialize database schema with indexes and FTS5."""
        # Use the centralized schema initialization from schema.py
        # This includes memories, memory_versions, edges, snapshots, and all indexes
        init_database(self._conn, enable_wal=self._enable_wal)

    def _embed_to_blob(self, embedding: List[float]) -> bytes:
        """Convert embedding list to binary blob (float32)."""
        return struct.pack(f'{len(embedding)}f', *embedding)
    
    def _blob_to_embed(self, blob: bytes) -> List[float]:
        """Convert binary blob to embedding list (float32)."""
        if not blob:
            return []
        num_floats = len(blob) // 4
        return list(struct.unpack(f'{num_floats}f', blob))

    def _validate_memory_type(self, memory_type: str) -> None:
        """Validate memory type."""
        if memory_type not in self.MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type: {memory_type}. Must be one of: {self.MEMORY_TYPES}")
    
    def _validate_edge_type(self, edge_type: str) -> None:
        """Validate edge type."""
        if edge_type not in self.EDGE_TYPES:
            raise ValueError(f"Invalid edge_type: {edge_type}. Must be one of: {self.EDGE_TYPES}")

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not v1 or not v2:
            return 0.0
        arr1 = np.array(v1, dtype=np.float32)  # type: ignore[attr-defined]
        arr2 = np.array(v2, dtype=np.float32)  # type: ignore[attr-defined]
        dot = np.dot(arr1, arr2)  # type: ignore[attr-defined]
        norm = np.linalg.norm(arr1) * np.linalg.norm(arr2)
        return float(dot / norm) if norm > 0 else 0.0

    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """
        Calculate recency score.
        Formula: exp(-days_ago / half_life)
        """
        if timestamp is None:
            return 0.0
        days_ago = (datetime.now() - timestamp).days
        return math.exp(-days_ago / self.RECENCY_HALF_LIFE)

    def store_memory(self,
                   content: str,
                   embedding: Optional[List[float]] = None,
                   memory_type: str = "experience",
                   confidence: Optional[float] = None,
                   memory_id: Optional[str] = None) -> str:
        """
        Store a memory in the palace.

        If memory_id is provided and exists, creates a new version (UPDATE operation).
        Otherwise, creates a new memory (CREATE operation).

        Args:
            content: The memory content text
            embedding: Vector embedding (1024-dim for bge-m3)
            memory_type: One of (fact, experience, belief, decision)
            confidence: 0.0-1.0 for beliefs
            memory_id: Optional UUID for updating existing memory

        Returns:
            memory_id: UUID of the created/updated memory
        """
        self._validate_memory_type(memory_type)

        if confidence is not None and (confidence < 0 or confidence > 1):
            raise ValueError("confidence must be between 0.0 and 1.0")

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        now = datetime.now().isoformat()

        # Convert embedding to blob
        embedding_blob = self._embed_to_blob(embedding) if embedding else None

        # Use lock for thread-safe database access
        with self._db_lock:
            # Check if this is an update (memory_id provided and exists)
            is_update = False
            if memory_id is not None:
                cursor = self._conn.execute(
                    "SELECT id FROM memories WHERE id = ?", (memory_id,)
                )
                is_update = cursor.fetchone() is not None

            if is_update:
                # UPDATE: Create new version and update memory
                # Get the next version number
                cursor = self._conn.execute("""
                    SELECT COALESCE(MAX(version_number), 0) + 1
                    FROM memory_versions
                    WHERE memory_id = ?
                """, (memory_id,))
                next_version = cursor.fetchone()[0]

                # Get the previous version_id (most recent version)
                cursor = self._conn.execute("""
                    SELECT version_id
                    FROM memory_versions
                    WHERE memory_id = ?
                    ORDER BY version_number DESC
                    LIMIT 1
                """, (memory_id,))
                prev_row = cursor.fetchone()
                previous_version_id = prev_row[0] if prev_row else None

                # Insert new version
                version_id = str(uuid.uuid4())
                self._conn.execute("""
                    INSERT INTO memory_versions
                    (version_id, memory_id, content, version_number, operation_type, created_at, previous_version_id)
                    VALUES (?, ?, ?, ?, 'UPDATE', ?, ?)
                """, (version_id, memory_id, content, next_version, now, previous_version_id))

                # Update the memories table
                self._conn.execute("""
                    UPDATE memories
                    SET content = ?, embedding = ?, memory_type = ?, confidence = ?,
                        last_accessed = ?, content_hash = ?
                    WHERE id = ?
                """, (content, embedding_blob, memory_type, confidence, now, content_hash, memory_id))

                # Update FTS index
                self._conn.execute("""
                    UPDATE memories_fts SET content = ? WHERE memory_id = ?
                """, (content, memory_id))

            else:
                # CREATE: New memory
                if memory_id is None:
                    memory_id = str(uuid.uuid4())

                # Insert into memories table
                self._conn.execute("""
                    INSERT INTO memories
                    (id, content, embedding, memory_type, confidence, created_at,
                     last_accessed, access_count, instance_ids, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    content,
                    embedding_blob,
                    memory_type,
                    confidence,
                    now,
                    now,
                    0,
                    json.dumps([]),
                    content_hash
                ))

                # Insert initial version
                version_id = str(uuid.uuid4())
                self._conn.execute("""
                    INSERT INTO memory_versions
                    (version_id, memory_id, content, version_number, operation_type, created_at, previous_version_id)
                    VALUES (?, ?, ?, 1, 'CREATE', ?, NULL)
                """, (version_id, memory_id, content, now))

                # Insert into FTS index
                self._conn.execute("""
                    INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)
                """, (memory_id, content))

            self._conn.commit()

        # Cache the embedding for fast access
        if embedding:
            self._embedding_cache[memory_id] = embedding

        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        Also updates access_count and last_accessed.

        Args:
            memory_id: UUID of the memory

        Returns:
            Memory object or None if not found
        """
        # Update access stats
        self._conn.execute("""
            UPDATE memories
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))
        self._conn.commit()

        # Retrieve memory
        cursor = self._conn.execute("""
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        if not row:
            return None

        # Parse embedding from blob
        embedding = self._blob_to_embed(row[2]) if row[2] else None
        instance_ids = json.loads(row[8]) if row[8] else []

        memory = Memory(
            id=row[0],
            content=row[1],
            embedding=embedding,
            memory_type=row[3],
            confidence=row[4],
            created_at=datetime.fromisoformat(row[5]) if row[5] else None,
            last_accessed=datetime.fromisoformat(row[6]) if row[6] else None,
            access_count=row[7],
            instance_ids=instance_ids,
            content_hash=row[9]
        )

        # Update cache
        if embedding:
            self._embedding_cache[memory_id] = embedding

        return memory

    def get_belief(self, belief_id: str) -> Optional[dict]:
        """
        Retrieve a belief by ID (beliefs are memories with type='belief').

        Args:
            belief_id: UUID of the belief

        Returns:
            Dictionary with belief data or None if not found or not a belief
        """
        memory = self.get_memory(belief_id)
        if memory and memory.memory_type == 'belief':
            return {
                'id': memory.id,
                'content': memory.content,
                'confidence': memory.confidence,
                'memory_type': memory.memory_type,
                'created_at': memory.created_at.isoformat() if memory.created_at else None,
                'last_accessed': memory.last_accessed.isoformat() if memory.last_accessed else None,
                'access_count': memory.access_count
            }
        return None

    def update_belief_confidence(self, belief_id: str, new_confidence: float) -> None:
        """
        Update the confidence value of a belief.

        Args:
            belief_id: UUID of the belief
            new_confidence: New confidence value (0.0-1.0)
        """
        # Validate confidence range
        if not 0.0 <= new_confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {new_confidence}")

        # Update the confidence field
        self._conn.execute("""
            UPDATE memories
            SET confidence = ?
            WHERE id = ? AND memory_type = 'belief'
        """, (new_confidence, belief_id))
        self._conn.commit()

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
        query_vec = np.array(query_embedding, dtype=np.float32)  # type: ignore[attr-defined]
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
            embedding = self._blob_to_embed(row[2])
            if not embedding or len(embedding) != len(query_embedding):
                continue

            memory_data.append(row)
            embeddings.append(embedding)

        if not embeddings:
            return []

        # Vectorized cosine similarity calculation
        # Convert to numpy matrix: (n_memories, embedding_dim)
        embeddings_matrix = np.array(embeddings, dtype=np.float32)  # type: ignore[attr-defined]

        # Compute norms for all embeddings: (n_memories,)
        norms = np.linalg.norm(embeddings_matrix, axis=1)

        # Compute dot products: (n_memories,)
        dots = np.dot(embeddings_matrix, query_vec)  # type: ignore[attr-defined]

        # Compute cosine similarities: (n_memories,)
        # Avoid division by zero
        similarities = np.divide(dots, norms * query_norm,  # type: ignore[attr-defined]
                                out=np.zeros_like(dots),  # type: ignore[attr-defined]
                                where=(norms * query_norm) > 0)

        # Filter by min_relevance
        valid_indices = np.where(similarities >= min_relevance)[0]  # type: ignore[attr-defined]

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

    def recall_at(self, timestamp: datetime) -> List[Memory]:
        """
        Point-in-time query: Reconstruct memory state as it existed at a specific timestamp.

        This method queries the memory_versions table to reconstruct what memories
        existed at the given timestamp, including their content at that time.

        Algorithm:
        1. Find all unique memory_ids that had versions created at or before timestamp
        2. For each memory_id, get the most recent version at or before timestamp
        3. Exclude memories where the most recent operation was DELETE
        4. Return Memory objects with historical content

        Args:
            timestamp: Point in time to query

        Returns:
            List of Memory objects as they existed at the timestamp
        """
        memories = []

        # Query to get the most recent version for each memory_id at or before timestamp
        # Uses a subquery to find max version_number per memory_id up to timestamp
        cursor = self._conn.execute("""
            SELECT mv.memory_id, mv.content, mv.operation_type, mv.created_at
            FROM memory_versions mv
            INNER JOIN (
                SELECT memory_id, MAX(version_number) as max_version
                FROM memory_versions
                WHERE created_at <= ?
                GROUP BY memory_id
            ) latest ON mv.memory_id = latest.memory_id AND mv.version_number = latest.max_version
            WHERE mv.operation_type != 'DELETE'
        """, (timestamp.isoformat(),))

        for row in cursor:
            memory_id = row[0]
            content = row[1]
            operation_type = row[2]
            created_at_str = row[3]

            # Get additional memory metadata from memories table if it exists
            # Note: The memory might not exist in memories table if it was deleted,
            # but we're reconstructing historical state from versions
            metadata_cursor = self._conn.execute("""
                SELECT memory_type, confidence, embedding
                FROM memories
                WHERE id = ?
            """, (memory_id,))
            metadata_row = metadata_cursor.fetchone()

            if metadata_row:
                memory_type = metadata_row[0] or "experience"
                confidence = metadata_row[1]
                embedding_blob = metadata_row[2]
                embedding = self._blob_to_embed(embedding_blob) if embedding_blob else None
            else:
                # Memory was deleted from main table, use defaults
                memory_type = "experience"
                confidence = None
                embedding = None

            memory = Memory(
                id=memory_id,
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                confidence=confidence,
                created_at=datetime.fromisoformat(created_at_str) if created_at_str else None,
                last_accessed=None,
                access_count=0,
                instance_ids=[],
                content_hash=hashlib.sha256(content.encode()).hexdigest()
            )
            memories.append(memory)

        return memories

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
            embedding = self._blob_to_embed(row[2]) if row[2] else None
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
            embedding = self._blob_to_embed(row[2]) if row[2] else None
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

        return float(round(centrality, 4))

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

                    embedding = self._blob_to_embed(row[2]) if row[2] else None
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
            embedding = self._blob_to_embed(row[2]) if row[2] else None
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

    def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding vector for a memory.

        Args:
            memory_id: Memory ID
            embedding: New embedding vector

        Returns:
            True if successful
        """
        embedding_blob = self._embed_to_blob(embedding) if embedding else None

        cursor = self._conn.execute("""
            UPDATE memories SET embedding = ? WHERE id = ?
        """, (embedding_blob, memory_id))
        self._conn.commit()

        if cursor.rowcount > 0 and embedding:
            self._embedding_cache[memory_id] = embedding

        return cursor.rowcount > 0

    def update_memory_content(self, memory_id: str, new_content: str) -> bool:
        """
        Update the content of a memory and recalculate hash and timestamp.

        Args:
            memory_id: Memory ID
            new_content: New content text

        Returns:
            True if successful
        """
        new_content_hash = hashlib.sha256(new_content.encode()).hexdigest()
        now = datetime.now().isoformat()

        cursor = self._conn.execute("""
            UPDATE memories
            SET content = ?, content_hash = ?, last_accessed = ?
            WHERE id = ?
        """, (new_content, new_content_hash, now, memory_id))

        # Update FTS index
        if cursor.rowcount > 0:
            self._conn.execute("""
                UPDATE memories_fts
                SET content = ?
                WHERE memory_id = ?
            """, (new_content, memory_id))

        self._conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory and all its edges.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Remove from FTS index first
        self._conn.execute("""
            DELETE FROM memories_fts WHERE memory_id = ?
        """, (memory_id,))
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()

        # Remove from cache
        if memory_id in self._embedding_cache:
            del self._embedding_cache[memory_id]

        return cursor.rowcount > 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with memory_count, edge_count, type_distribution
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

        with sqlite3.connect(self.db_path) as conn:
            if limit is not None:
                cursor = conn.execute("""
                    SELECT id, content, embedding, memory_type, confidence,
                           created_at, last_accessed, access_count, instance_ids, content_hash
                    FROM memories
                    WHERE created_at < ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (threshold.isoformat(), limit))
            else:
                cursor = conn.execute("""
                    SELECT id, content, embedding, memory_type, confidence,
                           created_at, last_accessed, access_count, instance_ids, content_hash
                    FROM memories
                    WHERE created_at < ?
                    ORDER BY created_at ASC
                """, (threshold.isoformat(),))

            for row in cursor:
                embedding = self._blob_to_embed(row[2]) if row[2] else None
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
        """Optimize database ( reclaim space )."""
        self._conn.execute("VACUUM")
        self._conn.commit()

    def close(self) -> None:
        """Close connection and cleanup."""
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
        self._embedding_cache.clear()

    def __enter__(self) -> "GraphPalace":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
