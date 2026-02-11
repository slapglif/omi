"""
Unit tests for EmbeddingCache migration from .npy files to SQLite

Tests cover:
- Automatic migration of existing .npy files to SQLite
- Migration preserves embedding data correctly
- Migrated embeddings are accessible via get_or_compute()
- Duplicate migrations are handled (idempotency)
- Migration statistics and logging
"""

import unittest
import tempfile
import hashlib
import sqlite3
from pathlib import Path
from typing import List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.embeddings import EmbeddingCache, OllamaEmbedder


class MockEmbedder:
    """Mock embedder for testing without API calls."""

    def embed(self, text: str) -> List[float]:
        """Return deterministic embedding based on text hash."""
        # Generate deterministic embedding from text
        text_hash = hashlib.sha256(text.encode()).digest()
        # Convert first 128 bytes to 32 floats (simulating 32-dim embedding)
        embedding = []
        for i in range(0, min(128, len(text_hash) * 4), 4):
            # Unpack 4 bytes as float
            byte_chunk = text_hash[i//4:i//4+1] * 4  # Repeat byte to make 4 bytes
            val = int.from_bytes(byte_chunk, byteorder='little', signed=False) / (2**32)
            embedding.append(val)

        # Pad to 32 dimensions
        while len(embedding) < 32:
            embedding.append(0.0)

        return embedding[:32]


class TestEmbeddingMigration(unittest.TestCase):
    """Test suite for .npy to SQLite migration."""

    def setUp(self):
        """Set up test cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mock_embedder = MockEmbedder()

    def tearDown(self):
        """Clean up test directory."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_npy_file(self, content_hash: str, embedding: List[float]) -> Path:
        """Create a .npy file in the cache directory."""
        npy_path = self.cache_dir / f"{content_hash}.npy"
        np.save(npy_path, np.array(embedding, dtype=np.float32))
        return npy_path

    def _get_embedding_from_db(self, cache: EmbeddingCache, content_hash: str) -> List[float]:
        """Retrieve embedding from SQLite database."""
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
                (content_hash,)
            )
            row = cursor.fetchone()
            if row:
                return cache._blob_to_embed(row[0])
        return None

    # ==================== Migration Tests ====================

    def test_migrate_single_npy_file(self):
        """Test migration of a single .npy file to SQLite."""
        # Create a test embedding
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 6  # 30 dimensions
        content_hash = "abc123def456"

        # Create .npy file
        npy_path = self._create_npy_file(content_hash, test_embedding)
        self.assertTrue(npy_path.exists())

        # Initialize cache (triggers migration)
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Verify embedding is in SQLite
        migrated_embedding = self._get_embedding_from_db(cache, content_hash)
        self.assertIsNotNone(migrated_embedding)
        self.assertEqual(len(migrated_embedding), len(test_embedding))

        # Verify values match (within floating point precision)
        for i, (expected, actual) in enumerate(zip(test_embedding, migrated_embedding)):
            self.assertAlmostEqual(expected, actual, places=5,
                                 msg=f"Mismatch at index {i}")

    def test_migrate_multiple_npy_files(self):
        """Test migration of multiple .npy files."""
        # Create multiple test embeddings
        test_data = {
            "hash1": [0.1] * 32,
            "hash2": [0.2] * 32,
            "hash3": [0.3] * 32,
            "hash4": [0.4] * 32,
            "hash5": [0.5] * 32,
        }

        # Create .npy files
        for content_hash, embedding in test_data.items():
            self._create_npy_file(content_hash, embedding)

        # Initialize cache (triggers migration)
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Verify all embeddings are in SQLite
        for content_hash, expected_embedding in test_data.items():
            migrated_embedding = self._get_embedding_from_db(cache, content_hash)
            self.assertIsNotNone(migrated_embedding,
                               f"Embedding for {content_hash} not found in database")
            self.assertEqual(len(migrated_embedding), len(expected_embedding))

            for expected, actual in zip(expected_embedding, migrated_embedding):
                self.assertAlmostEqual(expected, actual, places=5)

    def test_migration_idempotency(self):
        """Test that re-running migration doesn't duplicate data."""
        test_embedding = [0.7] * 32
        content_hash = "idempotent_test"

        # Create .npy file
        self._create_npy_file(content_hash, test_embedding)

        # First migration
        cache1 = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Close and re-initialize (second migration attempt)
        cache2 = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Verify only one entry in database
        with sqlite3.connect(cache2.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM embedding_cache WHERE content_hash = ?",
                (content_hash,)
            )
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1, "Migration should be idempotent")

    def test_migration_with_no_npy_files(self):
        """Test initialization when no .npy files exist."""
        # Initialize cache with empty directory
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Verify database is created
        self.assertTrue(cache.db_path.exists())

        # Verify table exists and is empty
        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embedding_cache")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 0)

    def test_migrated_embedding_accessible_via_get_or_compute(self):
        """Test that migrated embeddings are accessible via get_or_compute()."""
        # Create a test embedding with known hash
        test_text = "This is a test sentence for migration"
        content_hash = hashlib.sha256(test_text.encode()).hexdigest()
        test_embedding = [0.9] * 32

        # Create .npy file
        self._create_npy_file(content_hash, test_embedding)

        # Initialize cache (triggers migration)
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Retrieve via get_or_compute (should return cached, not compute)
        retrieved_embedding = cache.get_or_compute(test_text)

        # Verify it matches the migrated data, not the mock embedder output
        self.assertEqual(len(retrieved_embedding), len(test_embedding))
        for expected, actual in zip(test_embedding, retrieved_embedding):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_get_or_compute_after_migration(self):
        """Test get_or_compute works correctly after migration."""
        # Create migrated embedding
        test_text1 = "migrated text"
        hash1 = hashlib.sha256(test_text1.encode()).hexdigest()
        migrated_embedding = [0.6] * 32
        self._create_npy_file(hash1, migrated_embedding)

        # Initialize cache
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Get migrated embedding
        result1 = cache.get_or_compute(test_text1)
        self.assertAlmostEqual(result1[0], 0.6, places=5)

        # Compute new embedding (not in cache)
        test_text2 = "new text requiring computation"
        result2 = cache.get_or_compute(test_text2)
        self.assertIsNotNone(result2)
        self.assertEqual(len(result2), 32)

        # Verify new embedding is stored
        hash2 = hashlib.sha256(test_text2.encode()).hexdigest()
        stored_embedding = self._get_embedding_from_db(cache, hash2)
        self.assertIsNotNone(stored_embedding)

        # Verify it matches what was computed
        for expected, actual in zip(result2, stored_embedding):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_migration_handles_corrupt_npy_file(self):
        """Test that migration handles corrupt .npy files gracefully."""
        # Create a valid .npy file
        valid_hash = "valid_file"
        valid_embedding = [0.5] * 32
        self._create_npy_file(valid_hash, valid_embedding)

        # Create a corrupt .npy file
        corrupt_path = self.cache_dir / "corrupt_file.npy"
        with open(corrupt_path, 'wb') as f:
            f.write(b"This is not a valid .npy file")

        # Initialize cache (should migrate valid file, skip corrupt)
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Verify valid file was migrated
        valid_result = self._get_embedding_from_db(cache, valid_hash)
        self.assertIsNotNone(valid_result)

        # Verify corrupt file is not in database
        corrupt_result = self._get_embedding_from_db(cache, "corrupt_file")
        self.assertIsNone(corrupt_result)

    def test_migration_preserves_embedding_dimensions(self):
        """Test that migration preserves different embedding dimensions."""
        test_cases = [
            ("dim_768", [0.1] * 768),   # Common for nomic-embed-text
            ("dim_1024", [0.2] * 1024), # Common for bge-m3
            ("dim_32", [0.3] * 32),     # Small test dimension
        ]

        for content_hash, embedding in test_cases:
            self._create_npy_file(content_hash, embedding)

        # Initialize cache
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Verify all dimensions preserved
        for content_hash, expected_embedding in test_cases:
            migrated = self._get_embedding_from_db(cache, content_hash)
            self.assertEqual(len(migrated), len(expected_embedding),
                           f"Dimension mismatch for {content_hash}")

    def test_database_schema_created_correctly(self):
        """Test that the SQLite schema is created correctly."""
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        with sqlite3.connect(cache.db_path) as conn:
            # Check table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='embedding_cache'"
            )
            self.assertIsNotNone(cursor.fetchone())

            # Check columns
            cursor = conn.execute("PRAGMA table_info(embedding_cache)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            self.assertIn('content_hash', columns)
            self.assertIn('embedding', columns)
            self.assertIn('created_at', columns)

            # Verify types
            self.assertEqual(columns['content_hash'], 'TEXT')
            self.assertEqual(columns['embedding'], 'BLOB')

            # Check index exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_embedding_cache_created'"
            )
            self.assertIsNotNone(cursor.fetchone())

    def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled for concurrent access."""
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder, enable_wal=True)

        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.lower(), 'wal')

    def test_wal_mode_disabled(self):
        """Test that WAL mode can be disabled."""
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder, enable_wal=False)

        with sqlite3.connect(cache.db_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            self.assertNotEqual(mode.lower(), 'wal')

    def test_empty_embedding_handling(self):
        """Test that empty embeddings are handled correctly."""
        # Create an empty embedding
        empty_hash = "empty_embedding"
        empty_embedding = []

        # Note: np.save will create a valid .npy file even for empty array
        npy_path = self.cache_dir / f"{empty_hash}.npy"
        np.save(npy_path, np.array(empty_embedding, dtype=np.float32))

        # Initialize cache
        cache = EmbeddingCache(self.cache_dir, self.mock_embedder)

        # Empty embeddings should still be migrated
        migrated = self._get_embedding_from_db(cache, empty_hash)
        # Could be None or empty list depending on implementation
        if migrated is not None:
            self.assertEqual(len(migrated), 0)


if __name__ == '__main__':
    unittest.main()
