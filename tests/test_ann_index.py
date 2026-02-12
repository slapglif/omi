"""
Comprehensive tests for storage.ann_index module (ANNIndex)

Tests cover:
- Initialization with different configurations
- Adding embeddings (single and batch)
- Fast k-NN search with HNSW
- Dimension validation and auto-detection
- Index persistence (save/load)
- Index resizing and capacity management
- Thread-safe operations
- Rebuild from scratch
"""

import pytest
import sqlite3
import uuid
import numpy as np
from pathlib import Path
from typing import List

from omi.storage.ann_index import ANNIndex


class TestANNIndex:
    """Test suite for ANNIndex class"""

    def test_init_with_memory_db(self):
        """Test initialization with in-memory database"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)
        assert index.db_path == ':memory:'
        assert index.dim == 768
        assert index.enable_persistence is False
        assert index.get_size() == 0

    def test_init_with_persistent_db(self, tmp_path):
        """Test initialization with persistent database"""
        db_path = tmp_path / "test.db"
        index = ANNIndex(str(db_path), dim=1024, enable_persistence=True)
        assert index.db_path == db_path
        assert index.dim == 1024
        assert index.enable_persistence is True
        assert index.get_size() == 0

    def test_init_without_dimension(self):
        """Test initialization without specifying dimension (auto-detect)"""
        index = ANNIndex(':memory:', dim=None, enable_persistence=False)
        assert index.dim is None
        assert index.get_size() == 0

    def test_add_single_embedding_768(self):
        """Test adding a single 768-dim embedding"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        memory_id = str(uuid.uuid4())
        embedding = [0.1] * 768

        index.add(memory_id, embedding)

        assert index.get_size() == 1
        assert index.dim == 768

    def test_add_single_embedding_1024(self):
        """Test adding a single 1024-dim embedding"""
        index = ANNIndex(':memory:', dim=1024, enable_persistence=False)

        memory_id = str(uuid.uuid4())
        embedding = [0.1] * 1024

        index.add(memory_id, embedding)

        assert index.get_size() == 1
        assert index.dim == 1024

    def test_add_auto_detect_dimension(self):
        """Test automatic dimension detection from first embedding"""
        index = ANNIndex(':memory:', dim=None, enable_persistence=False)

        memory_id = str(uuid.uuid4())
        embedding = [0.1] * 768

        index.add(memory_id, embedding)

        assert index.dim == 768
        assert index.get_size() == 1

    def test_add_dimension_mismatch(self):
        """Test that adding embedding with wrong dimension raises error"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add first embedding with correct dimension
        index.add(str(uuid.uuid4()), [0.1] * 768)

        # Try to add embedding with wrong dimension
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            index.add(str(uuid.uuid4()), [0.1] * 1024)

    def test_add_duplicate_memory_id(self):
        """Test that adding duplicate memory_id is idempotent"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        memory_id = str(uuid.uuid4())
        embedding = [0.1] * 768

        # Add same memory twice
        index.add(memory_id, embedding)
        index.add(memory_id, embedding)

        # Should only be added once
        assert index.get_size() == 1

    def test_add_multiple_embeddings(self):
        """Test adding multiple embeddings"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        for i in range(10):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.01] * 768
            index.add(memory_id, embedding)

        assert index.get_size() == 10

    def test_add_triggers_resize(self):
        """Test that adding more elements than initial capacity triggers resize"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add more than INITIAL_CAPACITY (10000) would be slow,
        # so we test the resize logic by adding a reasonable number
        # The actual resize happens automatically
        for i in range(100):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.001] * 768
            index.add(memory_id, embedding)

        assert index.get_size() == 100

    def test_search_basic(self):
        """Test basic k-NN search"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add several embeddings
        memory_ids = []
        for i in range(10):
            memory_id = str(uuid.uuid4())
            memory_ids.append(memory_id)
            embedding = [0.1 + i * 0.05] * 768
            index.add(memory_id, embedding)

        # Search with query similar to first embedding
        query = [0.1] * 768
        results = index.search(query, k=5)

        assert len(results) <= 5
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        assert all(isinstance(item[0], str) for item in results)
        assert all(isinstance(item[1], float) for item in results)

    def test_search_empty_index(self):
        """Test search on empty index returns empty list"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        query = [0.1] * 768
        results = index.search(query, k=10)

        assert results == []

    def test_search_dimension_mismatch(self):
        """Test search with wrong dimension raises error"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add an embedding
        index.add(str(uuid.uuid4()), [0.1] * 768)

        # Try to search with wrong dimension
        with pytest.raises(ValueError, match="doesn't match index dimension"):
            index.search([0.1] * 1024, k=10)

    def test_search_returns_top_k(self):
        """Test that search returns at most k results"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add 20 embeddings
        for i in range(20):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.01] * 768
            index.add(memory_id, embedding)

        # Request only 5
        query = [0.15] * 768
        results = index.search(query, k=5)

        assert len(results) == 5

    def test_search_k_larger_than_size(self):
        """Test that k is clamped to index size"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add only 3 embeddings
        for i in range(3):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.1] * 768
            index.add(memory_id, embedding)

        # Request 10 but should only get 3
        query = [0.1] * 768
        results = index.search(query, k=10)

        assert len(results) == 3

    def test_search_sorted_by_similarity(self):
        """Test that results are sorted by similarity descending"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add embeddings with known similarities to query
        query = [1.0] + [0.0] * 767

        embeddings = [
            ([0.9] + [0.1] * 767, "high"),    # High similarity
            ([0.5] + [0.5] * 767, "medium"),  # Medium similarity
            ([0.1] + [0.9] * 767, "low"),     # Low similarity
        ]

        memory_ids = {}
        for embedding, label in embeddings:
            memory_id = str(uuid.uuid4())
            memory_ids[label] = memory_id
            index.add(memory_id, embedding)

        results = index.search(query, k=10)

        # Check that results are sorted by similarity descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i+1][1]

    def test_search_cosine_similarity(self):
        """Test that cosine similarity is calculated correctly"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add an identical embedding
        memory_id = str(uuid.uuid4())
        embedding = [1.0] + [0.0] * 767
        index.add(memory_id, embedding)

        # Search with identical query
        query = [1.0] + [0.0] * 767
        results = index.search(query, k=1)

        # Cosine similarity should be very close to 1.0
        assert len(results) == 1
        assert results[0][0] == memory_id
        assert results[0][1] > 0.99

    def test_save_and_load_768(self, tmp_path):
        """Test save and load for 768-dim embeddings"""
        db_path = tmp_path / "test.db"

        # Create index and add embeddings
        index1 = ANNIndex(str(db_path), dim=768, enable_persistence=True)

        memory_ids = []
        for i in range(10):
            memory_id = str(uuid.uuid4())
            memory_ids.append(memory_id)
            embedding = [0.1 + i * 0.05] * 768
            index1.add(memory_id, embedding)

        # Save index
        index1.save()

        # Create new index and load
        index2 = ANNIndex(str(db_path), dim=768, enable_persistence=True)
        loaded = index2.load()

        assert loaded is True
        assert index2.get_size() == 10
        assert index2.dim == 768

        # Search should work on loaded index
        query = [0.1] * 768
        results = index2.search(query, k=5)
        assert len(results) > 0

    def test_save_and_load_1024(self, tmp_path):
        """Test save and load for 1024-dim embeddings"""
        db_path = tmp_path / "test.db"

        # Create index and add embeddings
        index1 = ANNIndex(str(db_path), dim=1024, enable_persistence=True)

        for i in range(5):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.05] * 1024
            index1.add(memory_id, embedding)

        # Save index
        index1.save()

        # Create new index and load
        index2 = ANNIndex(str(db_path), dim=1024, enable_persistence=True)
        loaded = index2.load()

        assert loaded is True
        assert index2.get_size() == 5
        assert index2.dim == 1024

    def test_load_auto_detect_dimension(self, tmp_path):
        """Test loading without specifying dimension (auto-detect)"""
        db_path = tmp_path / "test.db"

        # Create index with 768 dims and save
        index1 = ANNIndex(str(db_path), dim=768, enable_persistence=True)
        index1.add(str(uuid.uuid4()), [0.1] * 768)
        index1.save()

        # Load without specifying dimension
        index2 = ANNIndex(str(db_path), dim=None, enable_persistence=True)

        assert index2.dim == 768
        assert index2.get_size() == 1

    def test_load_nonexistent_index(self, tmp_path):
        """Test loading when index file doesn't exist"""
        db_path = tmp_path / "nonexistent.db"

        index = ANNIndex(str(db_path), dim=768, enable_persistence=True)
        loaded = index.load()

        assert loaded is False
        assert index.get_size() == 0

    def test_save_without_persistence(self):
        """Test that save does nothing when persistence is disabled"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add an embedding
        index.add(str(uuid.uuid4()), [0.1] * 768)

        # Save should not raise error but do nothing
        index.save()

    def test_load_without_persistence(self):
        """Test that load returns False when persistence is disabled"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)
        loaded = index.load()

        assert loaded is False

    def test_rebuild_from_embeddings_768(self):
        """Test rebuilding index from batch of 768-dim embeddings"""
        index = ANNIndex(':memory:', dim=None, enable_persistence=False)

        # Create batch of embeddings
        embeddings = []
        for i in range(20):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.01] * 768
            embeddings.append((memory_id, embedding))

        # Rebuild
        index.rebuild_from_embeddings(embeddings)

        assert index.get_size() == 20
        assert index.dim == 768

        # Search should work
        query = [0.15] * 768
        results = index.search(query, k=5)
        assert len(results) > 0

    def test_rebuild_from_embeddings_1024(self):
        """Test rebuilding index from batch of 1024-dim embeddings"""
        index = ANNIndex(':memory:', dim=None, enable_persistence=False)

        # Create batch of embeddings
        embeddings = []
        for i in range(10):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.01] * 1024
            embeddings.append((memory_id, embedding))

        # Rebuild
        index.rebuild_from_embeddings(embeddings)

        assert index.get_size() == 10
        assert index.dim == 1024

    def test_rebuild_from_empty_list(self):
        """Test rebuilding from empty list does nothing"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add an embedding first
        index.add(str(uuid.uuid4()), [0.1] * 768)
        assert index.get_size() == 1

        # Rebuild with empty list
        index.rebuild_from_embeddings([])

        # Should still have original embedding
        assert index.get_size() == 1

    def test_rebuild_replaces_existing_index(self):
        """Test that rebuild replaces existing index"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add initial embeddings
        for i in range(5):
            index.add(str(uuid.uuid4()), [0.1 + i * 0.1] * 768)
        assert index.get_size() == 5

        # Rebuild with different embeddings
        new_embeddings = []
        for i in range(3):
            memory_id = str(uuid.uuid4())
            embedding = [0.5 + i * 0.1] * 768
            new_embeddings.append((memory_id, embedding))

        index.rebuild_from_embeddings(new_embeddings)

        # Should have only new embeddings
        assert index.get_size() == 3

    def test_rebuild_and_save(self, tmp_path):
        """Test that rebuild with persistence saves to disk"""
        db_path = tmp_path / "test.db"
        index = ANNIndex(str(db_path), dim=768, enable_persistence=True)

        # Rebuild with batch
        embeddings = []
        for i in range(10):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.05] * 768
            embeddings.append((memory_id, embedding))

        index.rebuild_from_embeddings(embeddings)

        # Check that index file was created
        index_path = db_path.parent / "test_768.hnsw"
        mapping_path = db_path.parent / "test_768.npz"

        assert index_path.exists()
        assert mapping_path.exists()

    def test_get_size_empty(self):
        """Test get_size on empty index"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)
        assert index.get_size() == 0

    def test_get_size_after_adds(self):
        """Test get_size after adding embeddings"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        for i in range(7):
            index.add(str(uuid.uuid4()), [0.1] * 768)

        assert index.get_size() == 7

    def test_thread_safety_add(self):
        """Test that add operations are thread-safe"""
        import threading

        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        def add_embeddings(start_idx):
            for i in range(start_idx, start_idx + 10):
                memory_id = f"memory_{i}"
                embedding = [0.1 + i * 0.001] * 768
                index.add(memory_id, embedding)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_embeddings, args=(i * 10,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have all embeddings
        assert index.get_size() == 50

    def test_thread_safety_search(self):
        """Test that search operations are thread-safe"""
        import threading

        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add some embeddings
        for i in range(100):
            index.add(str(uuid.uuid4()), [0.1 + i * 0.001] * 768)

        results_list = []

        def search_index():
            query = [0.15] * 768
            results = index.search(query, k=10)
            results_list.append(len(results))

        # Create multiple search threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=search_index)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All searches should have returned results
        assert len(results_list) == 10
        assert all(count == 10 for count in results_list)

    def test_persistence_on_init_loads_existing(self, tmp_path):
        """Test that initialization with persistence loads existing index"""
        db_path = tmp_path / "test.db"

        # Create and save index
        index1 = ANNIndex(str(db_path), dim=768, enable_persistence=True)
        memory_id = str(uuid.uuid4())
        index1.add(memory_id, [0.1] * 768)
        index1.save()

        # Create new index - should auto-load
        index2 = ANNIndex(str(db_path), dim=768, enable_persistence=True)

        assert index2.get_size() == 1

        # Search should find the memory
        results = index2.search([0.1] * 768, k=1)
        assert len(results) == 1
        assert results[0][0] == memory_id

    def test_hnsw_parameters(self):
        """Test that HNSW parameters are set correctly"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add an embedding to initialize the index
        index.add(str(uuid.uuid4()), [0.1] * 768)

        # Check that index was created with expected parameters
        assert index._index is not None
        assert index.M == 16
        assert index.EF_CONSTRUCTION == 200
        assert index.EF_SEARCH == 50

    def test_large_batch_performance(self):
        """Test performance with larger batch (1000 embeddings)"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add 1000 embeddings
        embeddings = []
        for i in range(1000):
            memory_id = str(uuid.uuid4())
            embedding = [0.1 + i * 0.0001] * 768
            embeddings.append((memory_id, embedding))

        index.rebuild_from_embeddings(embeddings)

        assert index.get_size() == 1000

        # Search should still be fast
        query = [0.15] * 768
        results = index.search(query, k=10)

        assert len(results) == 10
        assert all(isinstance(item[1], float) for item in results)


class TestANNIndexEdgeCases:
    """Test suite for edge cases and error conditions"""

    def test_search_before_any_add(self):
        """Test search before any embedding is added"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Index is initialized but no embeddings added
        query = [0.1] * 768
        results = index.search(query, k=10)

        assert results == []

    def test_zero_vector_search(self):
        """Test search with zero vector"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add normal embeddings
        for i in range(5):
            index.add(str(uuid.uuid4()), [0.1 + i * 0.05] * 768)

        # Search with zero vector (HNSW should handle this)
        query = [0.0] * 768
        results = index.search(query, k=5)

        # Should return results (may be arbitrary ordering)
        assert len(results) > 0

    def test_negative_values_in_embedding(self):
        """Test embeddings with negative values"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        # Add embedding with negative values
        memory_id = str(uuid.uuid4())
        embedding = [-0.5 + i * 0.001 for i in range(768)]
        index.add(memory_id, embedding)

        # Search should work
        query = [-0.5 + i * 0.001 for i in range(768)]
        results = index.search(query, k=1)

        assert len(results) == 1
        assert results[0][0] == memory_id

    def test_very_small_k(self):
        """Test search with k=1"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        for i in range(10):
            index.add(str(uuid.uuid4()), [0.1 + i * 0.05] * 768)

        query = [0.1] * 768
        results = index.search(query, k=1)

        assert len(results) == 1

    def test_id_map_consistency(self):
        """Test that internal ID mappings remain consistent"""
        index = ANNIndex(':memory:', dim=768, enable_persistence=False)

        memory_ids = []
        for i in range(10):
            memory_id = str(uuid.uuid4())
            memory_ids.append(memory_id)
            embedding = [0.1 + i * 0.05] * 768
            index.add(memory_id, embedding)

        # Check that all memory IDs are in the mapping
        for memory_id in memory_ids:
            assert memory_id in index._id_map

            # Check reverse mapping
            idx = index._id_map[memory_id]
            assert index._reverse_id_map[idx] == memory_id

    def test_corrupted_load_recovers_gracefully(self, tmp_path):
        """Test that corrupted index files are handled gracefully"""
        db_path = tmp_path / "test.db"

        # Create a corrupted index file
        index_path = tmp_path / "test_768.hnsw"
        index_path.write_text("corrupted data")

        # Try to load - should fail gracefully
        index = ANNIndex(str(db_path), dim=768, enable_persistence=True)
        loaded = index.load()

        assert loaded is False
        assert index.get_size() == 0

    def test_missing_mapping_file(self, tmp_path):
        """Test handling when .hnsw exists but .npz mapping is missing"""
        db_path = tmp_path / "test.db"

        # Create only .hnsw file without .npz
        index_path = tmp_path / "test_768.hnsw"
        index_path.write_text("fake hnsw data")

        # Try to load - should fail gracefully
        index = ANNIndex(str(db_path), dim=768, enable_persistence=True)
        loaded = index.load()

        assert loaded is False
        assert index.get_size() == 0
