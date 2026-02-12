"""
Async Graph Palace - Tier 3 Storage for OMI
Async/await version using aiosqlite for non-blocking database access.

This is the CORE of OMI's intelligence - everything depends on it.
- Semantic relationships between memories
- Centrality-weighted access
- Fast vector search with recency decay
- Graph traversal (BFS)
"""

import aiosqlite
import json
import hashlib
import uuid
import math
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
import struct


@dataclass
class Memory:
    """A memory node in the graph palace."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    memory_type: str = "experience"  # fact | experience | belief | decision
    confidence: Optional[float] = None  # 0.0-1.0 for beliefs
    created_at: datetime = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    instance_ids: Optional[List[str]] = None
    content_hash: Optional[str] = None  # SHA-256 for integrity

    def __post_init__(self):
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
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AsyncGraphPalace:
    """
    Tier 3: Async Graph Palace - Semantic memories, relationships, beliefs

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
    - Async/await for non-blocking I/O
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

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize Async Graph Palace.

        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for concurrent writes (default: True)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal = enable_wal
        self._conn = None

        # In-memory embedding cache for fast access
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_loaded = False

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(
                self.db_path,
                timeout=30.0
            )
            await self._init_db()
        return self._conn

    async def _init_db(self) -> None:
        """Initialize database schema with indexes and FTS5."""
        conn = await self._get_connection()

        # Enable WAL mode for concurrent writes
        if self._enable_wal:
            await conn.execute("PRAGMA journal_mode=WAL")

        # Foreign key constraints
        await conn.execute("PRAGMA foreign_keys=ON")

        # Create memories table with vector support
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB,  -- 1024-dim float32 for bge-m3
                memory_type TEXT CHECK(memory_type IN ('fact','experience','belief','decision')),
                confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                instance_ids TEXT,  -- JSON array
                content_hash TEXT  -- SHA-256 for integrity
            );

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT CHECK(edge_type IN ('SUPPORTS','CONTRADICTS','RELATED_TO','DEPENDS_ON','POSTED','DISCUSSED')),
                strength REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_bidirectional ON edges(source_id, target_id);
        """)

        # Create standalone FTS5 virtual table for full-text search
        # Note: Using standalone FTS5 (no content= sync) because memories.id
        # is TEXT (UUID), and FTS5 content_rowid requires INTEGER.
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id,
                content
            )
        """)

        await conn.commit()

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
        arr1 = np.array(v1, dtype=np.float32)
        arr2 = np.array(v2, dtype=np.float32)
        dot = np.dot(arr1, arr2)
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

    @staticmethod
    def _encode_cursor(cursor_data: Dict[str, Any]) -> str:
        """
        Encode cursor data to a base64 string for pagination.

        Args:
            cursor_data: Dictionary containing cursor state (last_id, order_by, etc.)

        Returns:
            Base64-encoded cursor string

        Example:
            >>> AsyncGraphPalace._encode_cursor({"last_id": "abc123", "order_by": "created_at"})
            'eyJsYXN0X2lkIjogImFiYzEyMyIsICJvcmRlcl9ieSI6ICJjcmVhdGVkX2F0In0='
        """
        if not cursor_data:
            return ""

        # Serialize to JSON and encode to base64
        json_bytes = json.dumps(cursor_data, sort_keys=True).encode('utf-8')
        return base64.urlsafe_b64encode(json_bytes).decode('utf-8')

    @staticmethod
    def _decode_cursor(cursor: Optional[str]) -> Dict[str, Any]:
        """
        Decode a base64 cursor string to cursor data.

        Args:
            cursor: Base64-encoded cursor string (or None/empty for first page)

        Returns:
            Dictionary containing cursor state, or empty dict if cursor is invalid/empty

        Example:
            >>> AsyncGraphPalace._decode_cursor('eyJsYXN0X2lkIjogImFiYzEyMyJ9')
            {'last_id': 'abc123'}
        """
        if not cursor:
            return {}

        try:
            # Decode from base64 and parse JSON
            json_bytes = base64.urlsafe_b64decode(cursor.encode('utf-8'))
            return json.loads(json_bytes.decode('utf-8'))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
            # Invalid cursor - return empty dict to start from beginning
            return {}

    async def store_memory(self,
                   content: str,
                   embedding: List[float] = None,
                   memory_type: str = "experience",
                   confidence: Optional[float] = None) -> str:
        """
        Store a memory in the palace.

        Args:
            content: The memory content text
            embedding: Vector embedding (1024-dim for bge-m3)
            memory_type: One of (fact, experience, belief, decision)
            confidence: 0.0-1.0 for beliefs

        Returns:
            memory_id: UUID of the created memory
        """
        self._validate_memory_type(memory_type)

        if confidence is not None and (confidence < 0 or confidence > 1):
            raise ValueError("confidence must be between 0.0 and 1.0")

        memory_id = str(uuid.uuid4())
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        now = datetime.now().isoformat()

        # Convert embedding to blob
        embedding_blob = self._embed_to_blob(embedding) if embedding else None

        conn = await self._get_connection()
        await conn.execute("""
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
        # Insert into FTS index
        await conn.execute("""
            INSERT INTO memories_fts(memory_id, content) VALUES (?, ?)
        """, (memory_id, content))
        await conn.commit()

        # Cache the embedding for fast access
        if embedding:
            self._embedding_cache[memory_id] = embedding

        return memory_id

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        Also updates access_count and last_accessed.

        Args:
            memory_id: UUID of the memory

        Returns:
            Memory object or None if not found
        """
        conn = await self._get_connection()

        # Update access stats
        await conn.execute("""
            UPDATE memories
            SET access_count = access_count + 1, last_accessed = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))
        await conn.commit()

        # Retrieve memory
        async with conn.execute("""
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories WHERE id = ?
        """, (memory_id,)) as cursor:
            row = await cursor.fetchone()
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

    async def recall(self,
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

        conn = await self._get_connection()
        async with conn.execute("""
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories WHERE embedding IS NOT NULL
        """) as cursor:
            async for row in cursor:
                embedding = self._blob_to_embed(row[2])
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

    async def full_text_search(self, query: str, limit: int = 10) -> List[Memory]:
        """
        Full-text search using FTS5.

        Args:
            query: Search query text
            limit: Max results

        Returns:
            List of matching memories
        """
        memories = []

        conn = await self._get_connection()
        # Use FTS5 MATCH via standalone FTS table
        async with conn.execute("""
            SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                   m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.memory_id
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit)) as cursor:
            async for row in cursor:
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

    async def create_edge(self,
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

        conn = await self._get_connection()
        await conn.execute("""
            INSERT INTO edges (id, source_id, target_id, edge_type, strength, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (edge_id, source_id, target_id, edge_type, strength, datetime.now().isoformat()))
        await conn.commit()

        return edge_id

    async def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge by ID."""
        conn = await self._get_connection()
        async with conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,)) as cursor:
            await conn.commit()
            return cursor.rowcount > 0

    async def get_edges(self, memory_id: str, edge_type: Optional[str] = None) -> List[Edge]:
        """
        Get all edges connected to a memory.

        Args:
            memory_id: The memory ID
            edge_type: Optional filter by edge type

        Returns:
            List of Edge objects
        """
        edges = []

        conn = await self._get_connection()
        if edge_type:
            async with conn.execute("""
                SELECT id, source_id, target_id, edge_type, strength, created_at
                FROM edges WHERE (source_id = ? OR target_id = ?) AND edge_type = ?
            """, (memory_id, memory_id, edge_type)) as cursor:
                async for row in cursor:
                    edges.append(Edge(
                        id=row[0],
                        source_id=row[1],
                        target_id=row[2],
                        edge_type=row[3],
                        strength=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else None
                    ))
        else:
            async with conn.execute("""
                SELECT id, source_id, target_id, edge_type, strength, created_at
                FROM edges WHERE source_id = ? OR target_id = ?
            """, (memory_id, memory_id)) as cursor:
                async for row in cursor:
                    edges.append(Edge(
                        id=row[0],
                        source_id=row[1],
                        target_id=row[2],
                        edge_type=row[3],
                        strength=row[4],
                        created_at=datetime.fromisoformat(row[5]) if row[5] else None
                    ))

        return edges

    async def get_neighbors(self, memory_id: str, edge_type: Optional[str] = None) -> List[Memory]:
        """
        Get all memories directly connected to a memory.

        Args:
            memory_id: The memory ID
            edge_type: Optional filter by edge type

        Returns:
            List of Memory objects
        """
        memories = []

        conn = await self._get_connection()
        if edge_type:
            async with conn.execute("""
                SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                       m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
                FROM memories m
                JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND m.id != ?
                AND e.edge_type = ?
            """, (memory_id, memory_id, memory_id, edge_type)) as cursor:
                async for row in cursor:
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
        else:
            async with conn.execute("""
                SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                       m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
                FROM memories m
                JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND m.id != ?
            """, (memory_id, memory_id, memory_id)) as cursor:
                async for row in cursor:
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

    async def get_centrality(self, memory_id: str) -> float:
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
        conn = await self._get_connection()

        # Get memory stats
        async with conn.execute("""
            SELECT access_count, last_accessed, created_at
            FROM memories WHERE id = ?
        """, (memory_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return 0.0

            access_count = row[0] or 0
            last_accessed = datetime.fromisoformat(row[1]) if row[1] else datetime.now()

        # Count edges (degree centrality)
        async with conn.execute("""
            SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?
        """, (memory_id, memory_id)) as cursor:
            edge_count = (await cursor.fetchone())[0]

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

    async def get_connected(self, memory_id: str, depth: int = 2) -> List[Memory]:
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

        conn = await self._get_connection()

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get neighbors
            async with conn.execute("""
                SELECT DISTINCT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                       m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash
                FROM memories m
                JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
                WHERE (e.source_id = ? OR e.target_id = ?)
                AND m.id != ?
            """, (current_id, current_id, current_id)) as cursor:
                async for row in cursor:
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

    async def get_top_central(self, limit: int = 10) -> List[Tuple[Memory, float]]:
        """
        Get the most central (hub) memories.

        Args:
            limit: Number of results

        Returns:
            List of (Memory, centrality_score) tuples
        """
        memories = []

        conn = await self._get_connection()
        # Single aggregated query: fetch all memory data with edge counts
        async with conn.execute("""
            SELECT m.id, m.content, m.embedding, m.memory_type, m.confidence,
                   m.created_at, m.last_accessed, m.access_count, m.instance_ids, m.content_hash,
                   COUNT(e.id) as edge_count
            FROM memories m
            LEFT JOIN edges e ON (m.id = e.source_id OR m.id = e.target_id)
            GROUP BY m.id
        """) as cursor:
            async for row in cursor:
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

    async def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding vector for a memory.

        Args:
            memory_id: Memory ID
            embedding: New embedding vector

        Returns:
            True if successful
        """
        embedding_blob = self._embed_to_blob(embedding) if embedding else None

        conn = await self._get_connection()
        async with conn.execute("""
            UPDATE memories SET embedding = ? WHERE id = ?
        """, (embedding_blob, memory_id)) as cursor:
            await conn.commit()

            if cursor.rowcount > 0 and embedding:
                self._embedding_cache[memory_id] = embedding

            return cursor.rowcount > 0

    async def update_memory_content(self, memory_id: str, new_content: str) -> bool:
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

        conn = await self._get_connection()
        async with conn.execute("""
            UPDATE memories
            SET content = ?, content_hash = ?, last_accessed = ?
            WHERE id = ?
        """, (new_content, new_content_hash, now, memory_id)) as cursor:
            # Update FTS index
            if cursor.rowcount > 0:
                await conn.execute("""
                    UPDATE memories_fts
                    SET content = ?
                    WHERE memory_id = ?
                """, (new_content, memory_id))

            await conn.commit()
            return cursor.rowcount > 0

    async def update_belief_confidence(self, belief_id: str, new_confidence: float) -> bool:
        """
        Update the confidence value of a belief.

        Args:
            belief_id: Belief ID
            new_confidence: New confidence value (0.0-1.0)

        Returns:
            True if successful
        """
        conn = await self._get_connection()
        async with conn.execute("""
            UPDATE memories
            SET confidence = ?
            WHERE id = ? AND memory_type = 'belief'
        """, (new_confidence, belief_id)) as cursor:
            await conn.commit()
            return cursor.rowcount > 0

    async def list_memories(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        memory_type: Optional[str] = None,
        order_by: str = "created_at",
        order_dir: str = "desc"
    ) -> Dict[str, Any]:
        """
        List memories with cursor-based pagination.

        Args:
            limit: Maximum number of memories to return (default 50, max 500)
            cursor: Pagination cursor from previous response (base64 encoded)
            memory_type: Filter by memory type (fact, experience, belief, decision)
            order_by: Field to order by (created_at, access_count, last_accessed)
            order_dir: Order direction (asc, desc)

        Returns:
            Dictionary containing:
                - memories: List of memory dictionaries (without embeddings)
                - total_count: Total number of memories matching filters
                - next_cursor: Cursor for next page (empty string if no more results)
                - has_more: Boolean indicating if more results exist

        Raises:
            ValueError: If invalid parameters provided
        """
        # Validate limit
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if limit > 500:
            raise ValueError(f"limit must be <= 500, got {limit}")

        # Validate memory_type if provided
        if memory_type is not None and memory_type not in self.MEMORY_TYPES:
            raise ValueError(
                f"Invalid memory_type: {memory_type}. Must be one of: {self.MEMORY_TYPES}"
            )

        # Validate order_by
        valid_order_fields = {"created_at", "access_count", "last_accessed"}
        if order_by not in valid_order_fields:
            raise ValueError(
                f"Invalid order_by: {order_by}. Must be one of: {valid_order_fields}"
            )

        # Validate order_dir
        order_dir_upper = order_dir.upper()
        if order_dir_upper not in {"ASC", "DESC"}:
            raise ValueError(f"Invalid order_dir: {order_dir}. Must be 'asc' or 'desc'")

        # Decode cursor
        cursor_data = self._decode_cursor(cursor)

        # If cursor provided, use order settings from cursor for consistency
        if cursor_data and "last_id" in cursor_data:
            effective_order_by = cursor_data.get("order_by", order_by)
            effective_order_dir = cursor_data.get("order_dir", order_dir_upper)
        else:
            effective_order_by = order_by
            effective_order_dir = order_dir_upper

        # Build base query
        base_query = """
            SELECT id, content, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories
        """
        count_query = "SELECT COUNT(*) FROM memories"
        params: List[Any] = []

        # Build WHERE clause - separate filter and cursor conditions
        filter_conditions = []
        cursor_conditions = []

        # Filter by memory_type
        if memory_type is not None:
            filter_conditions.append("memory_type = ?")
            params.append(memory_type)

        # Cursor-based pagination: filter by last value
        conn = await self._get_connection()
        if cursor_data and "last_id" in cursor_data:
            last_id = cursor_data["last_id"]

            # Get the value of the order_by field for the last_id
            cursor_value_query = f"SELECT {effective_order_by} FROM memories WHERE id = ?"
            async with conn.execute(cursor_value_query, (last_id,)) as cursor_result:
                cursor_value_row = await cursor_result.fetchone()

            if cursor_value_row:
                cursor_value = cursor_value_row[0]

                # Add cursor condition based on sort direction
                if effective_order_dir == "DESC":
                    # For DESC, we want values less than cursor_value
                    # or equal values with id > last_id (for stable ordering)
                    cursor_conditions.append(
                        f"({effective_order_by} < ? OR ({effective_order_by} = ? AND id > ?))"
                    )
                    params.extend([cursor_value, cursor_value, last_id])
                else:  # ASC
                    # For ASC, we want values greater than cursor_value
                    # or equal values with id > last_id (for stable ordering)
                    cursor_conditions.append(
                        f"({effective_order_by} > ? OR ({effective_order_by} = ? AND id > ?))"
                    )
                    params.extend([cursor_value, cursor_value, last_id])

        # Apply WHERE clause for main query (filters + cursor)
        all_conditions = filter_conditions + cursor_conditions
        if all_conditions:
            where_clause = " WHERE " + " AND ".join(all_conditions)
            base_query += where_clause

        # Apply WHERE clause for count query (filters only, no cursor)
        if filter_conditions:
            count_where_clause = " WHERE " + " AND ".join(filter_conditions)
            count_query += count_where_clause

        # Get total count (for filters only, not cursor)
        count_params = []
        if memory_type is not None:
            count_params.append(memory_type)

        async with conn.execute(count_query, count_params) as cursor_result:
            row = await cursor_result.fetchone()
            total_count = row[0]

        # Add ORDER BY and LIMIT
        base_query += f" ORDER BY {effective_order_by} {effective_order_dir}, id ASC"
        base_query += " LIMIT ?"
        params.append(limit + 1)  # Fetch one extra to determine has_more

        # Execute query
        async with conn.execute(base_query, params) as cursor_result:
            rows = await cursor_result.fetchall()

        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]  # Trim to requested limit

        # Convert rows to memory dictionaries (without embeddings)
        memories = []
        for row in rows:
            memory_dict = {
                "id": row[0],
                "content": row[1],
                "memory_type": row[2],
                "confidence": row[3],
                "created_at": row[4],
                "last_accessed": row[5],
                "access_count": row[6],
                "instance_ids": row[7] if row[7] else "[]",
                "content_hash": row[8]
            }
            memories.append(memory_dict)

        # Generate next cursor
        next_cursor = ""
        if has_more and memories:
            last_memory = memories[-1]
            next_cursor = self._encode_cursor({
                "last_id": last_memory["id"],
                "order_by": effective_order_by,
                "order_dir": effective_order_dir
            })

        return {
            "memories": memories,
            "total_count": total_count,
            "next_cursor": next_cursor,
            "has_more": has_more
        }

    async def list_beliefs(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        order_by: str = "created_at",
        order_dir: str = "desc"
    ) -> Dict[str, Any]:
        """
        List beliefs (memories with type='belief') with cursor-based pagination.

        Args:
            limit: Maximum number of beliefs to return (default 50, max 500)
            cursor: Pagination cursor from previous response (base64 encoded)
            order_by: Field to order by (confidence, created_at, access_count, last_accessed)
            order_dir: Order direction (asc, desc)

        Returns:
            Dictionary containing:
                - beliefs: List of belief dictionaries
                - total_count: Total number of beliefs
                - next_cursor: Cursor for next page (empty string if no more results)
                - has_more: Boolean indicating if more results exist

        Raises:
            ValueError: If invalid parameters provided
        """
        # Validate limit
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if limit > 500:
            raise ValueError(f"limit must be <= 500, got {limit}")

        # Validate order_by
        valid_order_fields = {"confidence", "created_at", "access_count", "last_accessed"}
        if order_by not in valid_order_fields:
            raise ValueError(
                f"Invalid order_by: {order_by}. Must be one of: {valid_order_fields}"
            )

        # Validate order_dir
        order_dir_upper = order_dir.upper()
        if order_dir_upper not in {"ASC", "DESC"}:
            raise ValueError(f"Invalid order_dir: {order_dir}. Must be 'asc' or 'desc'")

        # Decode cursor
        cursor_data = self._decode_cursor(cursor)

        # If cursor provided, use order settings from cursor for consistency
        if cursor_data and "last_id" in cursor_data:
            effective_order_by = cursor_data.get("order_by", order_by)
            effective_order_dir = cursor_data.get("order_dir", order_dir_upper)
        else:
            effective_order_by = order_by
            effective_order_dir = order_dir_upper

        # Build base query - filter for beliefs only
        base_query = """
            SELECT id, content, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories
            WHERE memory_type = 'belief'
        """
        count_query = "SELECT COUNT(*) FROM memories WHERE memory_type = 'belief'"
        params: List[Any] = []

        # Cursor-based pagination: filter by last value
        conn = await self._get_connection()
        if cursor_data and "last_id" in cursor_data:
            last_id = cursor_data["last_id"]

            # Get the value of the order_by field for the last_id
            cursor_value_query = f"SELECT {effective_order_by} FROM memories WHERE id = ?"
            async with conn.execute(cursor_value_query, (last_id,)) as cursor_result:
                cursor_value_row = await cursor_result.fetchone()

            if cursor_value_row:
                cursor_value = cursor_value_row[0]

                # Add cursor condition based on sort direction
                if effective_order_dir == "DESC":
                    # For DESC, we want values less than cursor_value
                    # or equal values with id > last_id (for stable ordering)
                    base_query += f" AND ({effective_order_by} < ? OR ({effective_order_by} = ? AND id > ?))"
                    params.extend([cursor_value, cursor_value, last_id])
                else:  # ASC
                    # For ASC, we want values greater than cursor_value
                    # or equal values with id > last_id (for stable ordering)
                    base_query += f" AND ({effective_order_by} > ? OR ({effective_order_by} = ? AND id > ?))"
                    params.extend([cursor_value, cursor_value, last_id])

        # Get total count (for beliefs only)
        async with conn.execute(count_query) as cursor_result:
            row = await cursor_result.fetchone()
            total_count = row[0]

        # Add ORDER BY and LIMIT
        base_query += f" ORDER BY {effective_order_by} {effective_order_dir}, id ASC"
        base_query += " LIMIT ?"
        params.append(limit + 1)  # Fetch one extra to determine has_more

        # Execute query
        async with conn.execute(base_query, params) as cursor_result:
            rows = await cursor_result.fetchall()

        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]  # Trim to requested limit

        # Convert rows to belief dictionaries
        beliefs = []
        for row in rows:
            belief_dict = {
                "id": row[0],
                "content": row[1],
                "memory_type": row[2],
                "confidence": row[3],
                "created_at": row[4],
                "last_accessed": row[5],
                "access_count": row[6],
                "instance_ids": row[7] if row[7] else "[]",
                "content_hash": row[8]
            }
            beliefs.append(belief_dict)

        # Generate next cursor
        next_cursor = ""
        if has_more and beliefs:
            last_belief = beliefs[-1]
            next_cursor = self._encode_cursor({
                "last_id": last_belief["id"],
                "order_by": effective_order_by,
                "order_dir": effective_order_dir
            })

        return {
            "beliefs": beliefs,
            "total_count": total_count,
            "next_cursor": next_cursor,
            "has_more": has_more
        }

    async def list_edges(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        edge_type: Optional[str] = None,
        order_by: str = "created_at",
        order_dir: str = "desc"
    ) -> Dict[str, Any]:
        """
        List edges with cursor-based pagination.

        Args:
            limit: Maximum number of edges to return (default 100, max 1000)
            cursor: Pagination cursor from previous response (base64 encoded)
            edge_type: Filter by edge type (SUPPORTS, CONTRADICTS, RELATED_TO, DEPENDS_ON, POSTED, DISCUSSED)
            order_by: Field to order by (created_at, strength)
            order_dir: Order direction (asc, desc)

        Returns:
            Dictionary containing:
                - edges: List of edge dictionaries
                - total_count: Total number of edges matching filters
                - next_cursor: Cursor for next page (empty string if no more results)
                - has_more: Boolean indicating if more results exist

        Raises:
            ValueError: If invalid parameters provided
        """
        # Validate limit
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        if limit > 1000:
            raise ValueError(f"limit must be <= 1000, got {limit}")

        # Validate edge_type if provided
        if edge_type is not None and edge_type not in self.EDGE_TYPES:
            raise ValueError(
                f"Invalid edge_type: {edge_type}. Must be one of: {self.EDGE_TYPES}"
            )

        # Validate order_by
        valid_order_fields = {"created_at", "strength"}
        if order_by not in valid_order_fields:
            raise ValueError(
                f"Invalid order_by: {order_by}. Must be one of: {valid_order_fields}"
            )

        # Validate order_dir
        order_dir_upper = order_dir.upper()
        if order_dir_upper not in {"ASC", "DESC"}:
            raise ValueError(f"Invalid order_dir: {order_dir}. Must be 'asc' or 'desc'")

        # Decode cursor
        cursor_data = self._decode_cursor(cursor)

        # If cursor provided, use order settings from cursor for consistency
        if cursor_data and "last_id" in cursor_data:
            effective_order_by = cursor_data.get("order_by", order_by)
            effective_order_dir = cursor_data.get("order_dir", order_dir_upper)
        else:
            effective_order_by = order_by
            effective_order_dir = order_dir_upper

        # Build base query
        base_query = """
            SELECT id, source_id, target_id, edge_type, strength, created_at
            FROM edges
        """
        count_query = "SELECT COUNT(*) FROM edges"
        params: List[Any] = []

        # Build WHERE clause - separate filter and cursor conditions
        filter_conditions = []
        cursor_conditions = []

        # Filter by edge_type
        if edge_type is not None:
            filter_conditions.append("edge_type = ?")
            params.append(edge_type)

        # Cursor-based pagination: filter by last value
        conn = await self._get_connection()
        if cursor_data and "last_id" in cursor_data:
            last_id = cursor_data["last_id"]

            # Get the value of the order_by field for the last_id
            cursor_value_query = f"SELECT {effective_order_by} FROM edges WHERE id = ?"
            async with conn.execute(cursor_value_query, (last_id,)) as cursor_result:
                cursor_value_row = await cursor_result.fetchone()

            if cursor_value_row:
                cursor_value = cursor_value_row[0]

                # Add cursor condition based on sort direction
                if effective_order_dir == "DESC":
                    # For DESC, we want values less than cursor_value
                    # or equal values with id > last_id (for stable ordering)
                    cursor_conditions.append(
                        f"({effective_order_by} < ? OR ({effective_order_by} = ? AND id > ?))"
                    )
                    params.extend([cursor_value, cursor_value, last_id])
                else:  # ASC
                    # For ASC, we want values greater than cursor_value
                    # or equal values with id > last_id (for stable ordering)
                    cursor_conditions.append(
                        f"({effective_order_by} > ? OR ({effective_order_by} = ? AND id > ?))"
                    )
                    params.extend([cursor_value, cursor_value, last_id])

        # Apply WHERE clause for main query (filters + cursor)
        all_conditions = filter_conditions + cursor_conditions
        if all_conditions:
            where_clause = " WHERE " + " AND ".join(all_conditions)
            base_query += where_clause

        # Apply WHERE clause for count query (filters only, no cursor)
        if filter_conditions:
            count_where_clause = " WHERE " + " AND ".join(filter_conditions)
            count_query += count_where_clause

        # Get total count (for filters only, not cursor)
        count_params = []
        if edge_type is not None:
            count_params.append(edge_type)

        async with conn.execute(count_query, count_params) as cursor_result:
            row = await cursor_result.fetchone()
            total_count = row[0]

        # Add ORDER BY and LIMIT
        base_query += f" ORDER BY {effective_order_by} {effective_order_dir}, id ASC"
        base_query += " LIMIT ?"
        params.append(limit + 1)  # Fetch one extra to determine has_more

        # Execute query
        async with conn.execute(base_query, params) as cursor_result:
            rows = await cursor_result.fetchall()

        # Check if there are more results
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]  # Trim to requested limit

        # Convert rows to edge dictionaries
        edges = []
        for row in rows:
            edge_dict = {
                "id": row[0],
                "source_id": row[1],
                "target_id": row[2],
                "edge_type": row[3],
                "strength": row[4],
                "created_at": row[5]
            }
            edges.append(edge_dict)

        # Generate next cursor
        next_cursor = ""
        if has_more and edges:
            last_edge = edges[-1]
            next_cursor = self._encode_cursor({
                "last_id": last_edge["id"],
                "order_by": effective_order_by,
                "order_dir": effective_order_dir
            })

        return {
            "edges": edges,
            "total_count": total_count,
            "next_cursor": next_cursor,
            "has_more": has_more
        }

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory and all its edges.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = await self._get_connection()

        # Remove from FTS index first
        await conn.execute("""
            DELETE FROM memories_fts WHERE memory_id = ?
        """, (memory_id,))
        async with conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,)) as cursor:
            await conn.commit()

            # Remove from cache
            if memory_id in self._embedding_cache:
                del self._embedding_cache[memory_id]

            return cursor.rowcount > 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dict with memory_count, edge_count, type_distribution
        """
        conn = await self._get_connection()

        async with conn.execute("SELECT COUNT(*) FROM memories") as cursor:
            memory_count = (await cursor.fetchone())[0]

        async with conn.execute("SELECT COUNT(*) FROM edges") as cursor:
            edge_count = (await cursor.fetchone())[0]

        type_distribution = {}
        async with conn.execute("""
            SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type
        """) as cursor:
            async for row in cursor:
                type_distribution[row[0]] = row[1]

        edge_distribution = {}
        async with conn.execute("""
            SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type
        """) as cursor:
            async for row in cursor:
                edge_distribution[row[0]] = row[1]

        return {
            "memory_count": memory_count,
            "edge_count": edge_count,
            "type_distribution": type_distribution,
            "edge_distribution": edge_distribution
        }

    async def get_compression_stats(self, threshold: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Calculate compression statistics for memories (optionally filtered by age).

        Used for dry-run reporting to estimate token savings before compression.

        Args:
            threshold: Optional datetime threshold - only include memories created before this.
                      If None, calculates stats for all memories.

        Returns:
            Dict with total_memories, total_chars, estimated_tokens, memories_by_type
        """
        conn = await self._get_connection()

        if threshold is not None:
            # Query memories before threshold
            cursor = await conn.execute("""
                SELECT content, memory_type FROM memories
                WHERE created_at < ?
            """, (threshold.isoformat(),))
        else:
            # Query all memories
            cursor = await conn.execute("""
                SELECT content, memory_type FROM memories
            """)

        total_memories = 0
        total_chars = 0
        memories_by_type: Dict[str, int] = {}

        async with cursor:
            async for row in cursor:
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

    async def find_contradictions(self, memory_id: str) -> List[Memory]:
        """
        Find memories that contradict a given memory.

        Args:
            memory_id: Memory to check

        Returns:
            List of contradicting memories
        """
        # Get memories connected via CONTRADICTS edge
        contradictions = await self.get_neighbors(memory_id, edge_type="CONTRADICTS")
        return contradictions

    async def get_supporting_evidence(self, memory_id: str) -> List[Memory]:
        """
        Get memories that support a given memory.

        Args:
            memory_id: Memory to get evidence for

        Returns:
            List of supporting memories
        """
        return await self.get_neighbors(memory_id, edge_type="SUPPORTS")

    async def get_memories_before(self, threshold: datetime, limit: Optional[int] = None) -> List[Memory]:
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

        conn = await self._get_connection()
        if limit is not None:
            async with conn.execute("""
                SELECT id, content, embedding, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                WHERE created_at < ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (threshold.isoformat(), limit)) as cursor:
                async for row in cursor:
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
        else:
            async with conn.execute("""
                SELECT id, content, embedding, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                WHERE created_at < ?
                ORDER BY created_at ASC
            """, (threshold.isoformat(),)) as cursor:
                async for row in cursor:
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

    async def vacuum(self) -> None:
        """Optimize database ( reclaim space )."""
        conn = await self._get_connection()
        await conn.execute("VACUUM")
        await conn.commit()

    async def close(self) -> None:
        """Close connection and cleanup."""
        if self._conn:
            await self._conn.close()
            self._conn = None
        self._embedding_cache.clear()

    async def __aenter__(self):
        await self._get_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
