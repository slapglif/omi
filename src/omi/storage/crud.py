"""
CRUD Operations for Graph Palace Memories

This module handles Create, Read, Update, Delete operations for memory nodes:
- store_memory: Create new memories with optional embeddings
- get_memory: Retrieve memories by ID (updates access stats)
- update_memory_content: Modify memory content and update FTS index
- update_embedding: Update vector embeddings
- delete_memory: Remove memories and associated FTS entries
"""

import sqlite3
import json
import hashlib
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

from .models import Memory
from .schema import init_database
from .embeddings import embed_to_blob, blob_to_embed


class MemoryCRUD:
    """
    Memory CRUD operations for Graph Palace.

    Handles basic create, read, update, delete operations for memory nodes
    with support for:
    - Vector embeddings (binary blob storage)
    - Full-text search index synchronization
    - Access tracking (count and timestamp)
    - Thread-safe database operations
    - In-memory embedding cache for performance
    """

    # Valid memory types
    MEMORY_TYPES = {"fact", "experience", "belief", "decision"}

    def __init__(self, db_path: str, enable_wal: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize Memory CRUD operations.

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

        # In-memory embedding cache for fast access
        self._embedding_cache: Dict[str, List[float]] = {}

    def _validate_memory_type(self, memory_type: str) -> None:
        """Validate memory type."""
        if memory_type not in self.MEMORY_TYPES:
            raise ValueError(f"Invalid memory_type: {memory_type}. Must be one of: {self.MEMORY_TYPES}")

    def store_memory(self,
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
        embedding_blob = embed_to_blob(embedding) if embedding else None

        # Use lock for thread-safe database access
        with self._db_lock:
            self._conn.execute("""
                INSERT INTO memories
                (id, content, embedding, memory_type, confidence, created_at,
                 last_accessed, access_count, instance_ids, content_hash, archived)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                content_hash,
                0  # archived = False (0)
            ))
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
                   created_at, last_accessed, access_count, instance_ids, content_hash, archived
            FROM memories WHERE id = ?
        """, (memory_id,))

        row = cursor.fetchone()
        if not row:
            return None

        # Parse embedding from blob
        embedding = blob_to_embed(row[2]) if row[2] else None
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
            content_hash=row[9],
            archived=bool(row[10])  # archived column (convert INTEGER to bool)
        )

        # Update cache
        if embedding:
            self._embedding_cache[memory_id] = embedding

        return memory

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

    def update_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding vector for a memory.

        Args:
            memory_id: Memory ID
            embedding: New embedding vector

        Returns:
            True if successful
        """
        embedding_blob = embed_to_blob(embedding) if embedding else None

        cursor = self._conn.execute("""
            UPDATE memories SET embedding = ? WHERE id = ?
        """, (embedding_blob, memory_id))
        self._conn.commit()

        if cursor.rowcount > 0 and embedding:
            self._embedding_cache[memory_id] = embedding

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

    def archive_memories(self, memory_ids: List[str]) -> int:
        """
        Mark memories as archived (excluded from default search).

        Args:
            memory_ids: List of memory IDs to archive

        Returns:
            Number of memories successfully archived
        """
        if not memory_ids:
            return 0

        archived_count = 0
        with self._db_lock:
            for memory_id in memory_ids:
                cursor = self._conn.execute("""
                    UPDATE memories SET archived = 1 WHERE id = ?
                """, (memory_id,))
                if cursor.rowcount > 0:
                    archived_count += 1
            self._conn.commit()

        return archived_count

    def close(self) -> None:
        """Close connection and cleanup."""
        if self._owns_connection and hasattr(self, '_conn') and self._conn:
            self._conn.close()
        self._embedding_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
