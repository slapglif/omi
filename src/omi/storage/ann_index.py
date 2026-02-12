"""
ANN (Approximate Nearest Neighbor) Index for Graph Palace

This module provides fast vector similarity search using HNSW (Hierarchical Navigable Small World):
- add: Incrementally add embeddings to the index
- search: Fast k-nearest neighbor search with cosine similarity
- save/load: Persist index to disk alongside palace.sqlite
- Multi-dimensional support: Handle both 768-dim (Ollama) and 1024-dim (NIM) embeddings
"""

import threading
from pathlib import Path
from typing import List, Tuple, Optional


class ANNIndex:
    """
    Approximate Nearest Neighbor index using HNSW.

    Provides fast vector similarity search with:
    - HNSW indexing for sub-100ms search on 100k+ embeddings
    - Disk persistence alongside SQLite database
    - Incremental updates as new memories are added
    - Multi-dimensional support (768 and 1024 dims)
    - Thread-safe operations

    Target performance: <50ms for top-10 search on 100k memories
    Memory footprint: <500MB for 100k 1024-dim embeddings
    Accuracy: 95%+ recall@10 vs exact brute-force
    """

    # HNSW parameters for good balance of speed/accuracy/memory
    M = 16  # Number of bi-directional links per element
    EF_CONSTRUCTION = 200  # Size of dynamic candidate list during construction
    EF_SEARCH = 50  # Size of dynamic candidate list during search

    def __init__(self, db_path: str, dim: Optional[int] = None, enable_persistence: bool = True):
        """
        Initialize ANN Index.

        Args:
            db_path: Path to SQLite database (or ':memory:' for in-memory)
            dim: Embedding dimension (768 or 1024). Auto-detected from first embedding if None.
            enable_persistence: Enable disk persistence (default: True)
        """
        self.db_path = Path(db_path) if db_path != ':memory:' else db_path
        self.dim = dim
        self.enable_persistence = enable_persistence and db_path != ':memory:'

        # Thread lock for serializing index operations
        self._lock = threading.Lock()

        # Index state (to be initialized in implementation phase)
        self._index = None
        self._id_map = {}  # Maps memory_id to index position
        self._reverse_id_map = {}  # Maps index position to memory_id
        self._current_size = 0

    def add(self, memory_id: str, embedding: List[float]) -> None:
        """
        Add a single embedding to the index.

        Args:
            memory_id: Unique memory identifier
            embedding: Vector embedding (768 or 1024 dims)

        Raises:
            ValueError: If embedding dimension doesn't match index dimension
        """
        # Implementation in phase 2
        pass

    def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of neighbors to return

        Returns:
            List of (memory_id, similarity_score) tuples, sorted by similarity descending
        """
        # Implementation in phase 2
        return []

    def save(self) -> None:
        """
        Persist index to disk.

        Saves to {db_path_stem}_{dim}.hnsw (e.g., palace_1024.hnsw)
        """
        # Implementation in phase 2
        pass

    def load(self) -> bool:
        """
        Load index from disk if it exists.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        # Implementation in phase 2
        return False

    def get_size(self) -> int:
        """
        Get current number of vectors in the index.

        Returns:
            Number of indexed vectors
        """
        return self._current_size

    def rebuild_from_embeddings(self, embeddings: List[Tuple[str, List[float]]]) -> None:
        """
        Rebuild index from scratch with a batch of embeddings.

        Args:
            embeddings: List of (memory_id, embedding) tuples

        Used by 'omi index rebuild' CLI command.
        """
        # Implementation in phase 2
        pass
