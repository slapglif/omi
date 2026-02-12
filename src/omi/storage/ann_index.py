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

import numpy as np
import hnswlib


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

    # Initial capacity for the index (will grow as needed)
    INITIAL_CAPACITY = 10000

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

        # Index state
        self._index: Optional[hnswlib.Index] = None
        self._id_map = {}  # Maps memory_id to index position
        self._reverse_id_map = {}  # Maps index position to memory_id
        self._current_size = 0
        self._max_capacity = 0

        # Try to load existing index if persistence is enabled
        if self.enable_persistence:
            self.load()

    def _init_index(self, dim: int, initial_capacity: Optional[int] = None) -> None:
        """
        Initialize a new HNSW index.

        Args:
            dim: Embedding dimension
            initial_capacity: Initial capacity (default: INITIAL_CAPACITY)
        """
        if initial_capacity is None:
            initial_capacity = self.INITIAL_CAPACITY

        self.dim = dim
        self._max_capacity = initial_capacity

        # Create HNSW index with cosine distance (equivalent to cosine similarity)
        self._index = hnswlib.Index(space='cosine', dim=dim)
        self._index.init_index(
            max_elements=initial_capacity,
            ef_construction=self.EF_CONSTRUCTION,
            M=self.M
        )
        self._index.set_ef(self.EF_SEARCH)

    def _resize_index(self, new_capacity: int) -> None:
        """
        Resize index to accommodate more elements.

        Args:
            new_capacity: New capacity
        """
        if self._index is not None:
            self._index.resize_index(new_capacity)
            self._max_capacity = new_capacity

    def add(self, memory_id: str, embedding: List[float]) -> None:
        """
        Add a single embedding to the index.

        Args:
            memory_id: Unique memory identifier
            embedding: Vector embedding (768 or 1024 dims)

        Raises:
            ValueError: If embedding dimension doesn't match index dimension
        """
        with self._lock:
            # Auto-detect dimension from first embedding
            if self._index is None:
                if self.dim is None:
                    self.dim = len(embedding)
                self._init_index(self.dim)

            # Validate dimension
            if len(embedding) != self.dim:
                raise ValueError(
                    f"Embedding dimension {len(embedding)} doesn't match index dimension {self.dim}"
                )

            # Skip if already indexed
            if memory_id in self._id_map:
                return

            # Resize if needed (double capacity)
            if self._current_size >= self._max_capacity:
                new_capacity = self._max_capacity * 2
                self._resize_index(new_capacity)

            # Add to index
            idx = self._current_size
            embedding_array = np.array(embedding, dtype=np.float32)
            self._index.add_items(embedding_array, idx)

            # Update mappings
            self._id_map[memory_id] = idx
            self._reverse_id_map[idx] = memory_id
            self._current_size += 1

    def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of neighbors to return

        Returns:
            List of (memory_id, similarity_score) tuples, sorted by similarity descending
        """
        with self._lock:
            if self._index is None or self._current_size == 0:
                return []

            # Validate dimension
            if len(query_embedding) != self.dim:
                raise ValueError(
                    f"Query dimension {len(query_embedding)} doesn't match index dimension {self.dim}"
                )

            # Limit k to actual number of elements
            k = min(k, self._current_size)

            # Convert query to numpy array
            query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

            # Search (returns indices and distances)
            # Note: hnswlib returns cosine distance (1 - cosine_similarity)
            indices, distances = self._index.knn_query(query_array, k=k)

            # Convert to results
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx in self._reverse_id_map:
                    memory_id = self._reverse_id_map[idx]
                    # Convert cosine distance back to similarity: similarity = 1 - distance
                    similarity = 1.0 - float(dist)
                    results.append((memory_id, similarity))

            return results

    def _save_unlocked(self) -> None:
        """
        Internal save method that doesn't acquire the lock.
        Must be called while holding self._lock.
        """
        if not self.enable_persistence or self._index is None:
            return

        # Determine save path
        if isinstance(self.db_path, Path):
            index_path = self.db_path.parent / f"{self.db_path.stem}_{self.dim}.hnsw"
        else:
            return

        # Save index
        self._index.save_index(str(index_path))

        # Save ID mappings as numpy archive
        mapping_path = index_path.with_suffix('.npz')
        np.savez(
            mapping_path,
            id_map_keys=list(self._id_map.keys()),
            id_map_values=list(self._id_map.values()),
            current_size=self._current_size,
            max_capacity=self._max_capacity,
            dim=self.dim
        )

    def save(self) -> None:
        """
        Persist index to disk.

        Saves to {db_path_stem}_{dim}.hnsw (e.g., palace_1024.hnsw)
        """
        with self._lock:
            self._save_unlocked()

    def load(self) -> bool:
        """
        Load index from disk if it exists.

        Returns:
            True if index was loaded successfully, False otherwise
        """
        if not self.enable_persistence:
            return False

        with self._lock:
            # Try to find existing index files
            if isinstance(self.db_path, Path):
                # If dimension is known, try that specific file
                if self.dim is not None:
                    index_path = self.db_path.parent / f"{self.db_path.stem}_{self.dim}.hnsw"
                    if index_path.exists():
                        return self._load_from_path(index_path)
                    return False

                # Otherwise, try to find any index file for this database
                # Try common dimensions: 1024 (NIM), 768 (Ollama)
                for dim in [1024, 768]:
                    index_path = self.db_path.parent / f"{self.db_path.stem}_{dim}.hnsw"
                    if index_path.exists():
                        return self._load_from_path(index_path)

            return False

    def _load_from_path(self, index_path: Path) -> bool:
        """
        Load index from a specific path.

        Args:
            index_path: Path to the .hnsw file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            mapping_path = index_path.with_suffix('.npz')
            if not mapping_path.exists():
                return False

            # Load metadata first
            data = np.load(mapping_path, allow_pickle=True)
            dim = int(data['dim'])
            current_size = int(data['current_size'])
            max_capacity = int(data['max_capacity'])

            # Initialize index with correct dimension
            self._index = hnswlib.Index(space='cosine', dim=dim)
            self._index.load_index(str(index_path), max_elements=max_capacity)
            self._index.set_ef(self.EF_SEARCH)

            # Restore ID mappings
            id_map_keys = data['id_map_keys']
            id_map_values = data['id_map_values']

            self._id_map = dict(zip(id_map_keys, id_map_values))
            self._reverse_id_map = {int(v): k for k, v in self._id_map.items()}
            self._current_size = current_size
            self._max_capacity = max_capacity
            self.dim = dim

            return True

        except Exception:
            # If loading fails, reset to empty state
            self._index = None
            self._id_map = {}
            self._reverse_id_map = {}
            self._current_size = 0
            self._max_capacity = 0
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
        with self._lock:
            if not embeddings:
                return

            # Detect dimension from first embedding
            dim = len(embeddings[0][1])
            initial_capacity = max(len(embeddings), self.INITIAL_CAPACITY)

            # Reset index
            self._init_index(dim, initial_capacity)
            self._id_map = {}
            self._reverse_id_map = {}
            self._current_size = 0

            # Batch add all embeddings
            embedding_matrix = np.array([emb for _, emb in embeddings], dtype=np.float32)
            indices = np.arange(len(embeddings))

            self._index.add_items(embedding_matrix, indices)

            # Update mappings
            for idx, (memory_id, _) in enumerate(embeddings):
                self._id_map[memory_id] = idx
                self._reverse_id_map[idx] = memory_id

            self._current_size = len(embeddings)

            # Save to disk if persistence is enabled
            if self.enable_persistence:
                self._save_unlocked()
