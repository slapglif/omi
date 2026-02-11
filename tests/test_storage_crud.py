"""
Comprehensive tests for storage.crud module (MemoryCRUD)

Tests cover:
- Store memory with embeddings
- Get memory by ID (with access tracking)
- Update memory content
- Update embeddings
- Delete memory
- Memory type validation
- FTS index synchronization
- Embedding cache
"""

import pytest
import sqlite3
import uuid
from datetime import datetime

from omi.storage.crud import MemoryCRUD
from omi.storage.schema import init_database


class TestMemoryCRUD:
    """Test suite for MemoryCRUD class"""

    def test_init_with_memory_db(self):
        """Test initialization with in-memory database"""
        crud = MemoryCRUD(':memory:')
        assert crud.db_path == ':memory:'
        assert crud._owns_connection is True
        assert crud._embedding_cache == {}
        crud.close()

    def test_init_with_shared_connection(self, tmp_path):
        """Test initialization with shared connection"""
        db_path = tmp_path / "test_shared.db"
        conn = sqlite3.connect(str(db_path))
        init_database(conn, enable_wal=True)

        crud = MemoryCRUD(str(db_path), conn=conn)
        assert crud._owns_connection is False
        crud.close()
        conn.close()

    def test_memory_type_validation_valid(self):
        """Test that valid memory types are accepted"""
        crud = MemoryCRUD(':memory:')

        valid_types = ["fact", "experience", "belief", "decision"]

        for mem_type in valid_types:
            # Should not raise
            crud._validate_memory_type(mem_type)

        crud.close()

    def test_memory_type_validation_invalid(self):
        """Test that invalid memory types are rejected"""
        crud = MemoryCRUD(':memory:')

        with pytest.raises(ValueError, match="Invalid memory_type"):
            crud._validate_memory_type("invalid_type")

        crud.close()

    def test_store_memory_basic(self):
        """Test storing a basic memory"""
        crud = MemoryCRUD(':memory:')

        content = "Test memory content"
        memory_id = crud.store_memory(content, memory_type="fact")

        assert memory_id is not None
        assert len(memory_id) == 36  # UUID format

        # Verify in database
        cursor = crud._conn.execute("SELECT content, memory_type FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == content
        assert row[1] == "fact"

        crud.close()

    def test_store_memory_with_embedding(self):
        """Test storing memory with embedding"""
        crud = MemoryCRUD(':memory:')

        content = "Memory with embedding"
        embedding = [0.1] * 768
        memory_id = crud.store_memory(content, embedding=embedding, memory_type="experience")

        assert memory_id in crud._embedding_cache
        assert crud._embedding_cache[memory_id] == embedding

        crud.close()

    def test_store_memory_with_confidence(self):
        """Test storing belief with confidence"""
        crud = MemoryCRUD(':memory:')

        content = "Belief content"
        confidence = 0.85
        memory_id = crud.store_memory(content, memory_type="belief", confidence=confidence)

        cursor = crud._conn.execute("SELECT confidence FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        assert row[0] == confidence

        crud.close()

    def test_store_memory_invalid_confidence(self):
        """Test that invalid confidence values are rejected"""
        crud = MemoryCRUD(':memory:')

        with pytest.raises(ValueError, match="confidence must be between"):
            crud.store_memory("Content", memory_type="belief", confidence=1.5)

        with pytest.raises(ValueError, match="confidence must be between"):
            crud.store_memory("Content", memory_type="belief", confidence=-0.1)

        crud.close()

    def test_store_memory_invalid_type(self):
        """Test storing memory with invalid type raises error"""
        crud = MemoryCRUD(':memory:')

        with pytest.raises(ValueError, match="Invalid memory_type"):
            crud.store_memory("Content", memory_type="invalid")

        crud.close()

    def test_store_memory_fts_index(self):
        """Test that FTS index is populated"""
        crud = MemoryCRUD(':memory:')

        content = "Searchable content"
        memory_id = crud.store_memory(content, memory_type="fact")

        # Check FTS index
        cursor = crud._conn.execute("""
            SELECT memory_id FROM memories_fts WHERE memory_id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == memory_id

        crud.close()

    def test_get_memory_basic(self):
        """Test retrieving a memory"""
        crud = MemoryCRUD(':memory:')

        # Store memory
        content = "Test content"
        memory_id = crud.store_memory(content, memory_type="fact")

        # Retrieve it
        memory = crud.get_memory(memory_id)

        assert memory is not None
        assert memory.id == memory_id
        assert memory.content == content
        assert memory.memory_type == "fact"
        assert memory.access_count > 0  # Should be incremented

        crud.close()

    def test_get_memory_nonexistent(self):
        """Test getting non-existent memory returns None"""
        crud = MemoryCRUD(':memory:')

        fake_id = str(uuid.uuid4())
        memory = crud.get_memory(fake_id)

        assert memory is None

        crud.close()

    def test_get_memory_updates_access_count(self):
        """Test that getting memory updates access count"""
        crud = MemoryCRUD(':memory:')

        memory_id = crud.store_memory("Test", memory_type="fact")

        # Get multiple times
        memory1 = crud.get_memory(memory_id)
        memory2 = crud.get_memory(memory_id)

        # Access count should increase
        assert memory2.access_count > memory1.access_count

        crud.close()

    def test_get_memory_with_embedding(self):
        """Test retrieving memory with embedding"""
        crud = MemoryCRUD(':memory:')

        embedding = [0.2] * 768
        memory_id = crud.store_memory("Test", embedding=embedding, memory_type="fact")

        memory = crud.get_memory(memory_id)

        assert memory.embedding is not None
        assert len(memory.embedding) == 768
        assert memory_id in crud._embedding_cache

        crud.close()

    def test_update_memory_content(self):
        """Test updating memory content"""
        crud = MemoryCRUD(':memory:')

        memory_id = crud.store_memory("Original content", memory_type="fact")

        # Update content
        new_content = "Updated content"
        result = crud.update_memory_content(memory_id, new_content)

        assert result is True

        # Verify in database
        cursor = crud._conn.execute("SELECT content FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        assert row[0] == new_content

        crud.close()

    def test_update_memory_content_updates_fts(self):
        """Test that updating content also updates FTS index"""
        crud = MemoryCRUD(':memory:')

        memory_id = crud.store_memory("Original", memory_type="fact")

        # Update content
        new_content = "Updated searchable content"
        crud.update_memory_content(memory_id, new_content)

        # Check FTS was updated
        cursor = crud._conn.execute("""
            SELECT content FROM memories_fts WHERE memory_id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        assert row[0] == new_content

        crud.close()

    def test_update_memory_content_nonexistent(self):
        """Test updating non-existent memory returns False"""
        crud = MemoryCRUD(':memory:')

        fake_id = str(uuid.uuid4())
        result = crud.update_memory_content(fake_id, "New content")

        assert result is False

        crud.close()

    def test_update_memory_content_updates_hash(self):
        """Test that updating content recalculates hash"""
        crud = MemoryCRUD(':memory:')

        memory_id = crud.store_memory("Original", memory_type="fact")

        # Get original hash
        cursor = crud._conn.execute("SELECT content_hash FROM memories WHERE id = ?", (memory_id,))
        original_hash = cursor.fetchone()[0]

        # Update content
        crud.update_memory_content(memory_id, "Different content")

        # Get new hash
        cursor = crud._conn.execute("SELECT content_hash FROM memories WHERE id = ?", (memory_id,))
        new_hash = cursor.fetchone()[0]

        assert new_hash != original_hash

        crud.close()

    def test_update_embedding(self):
        """Test updating embedding vector"""
        crud = MemoryCRUD(':memory:')

        original_embedding = [0.1] * 768
        memory_id = crud.store_memory("Test", embedding=original_embedding, memory_type="fact")

        # Update embedding
        new_embedding = [0.9] * 768
        result = crud.update_embedding(memory_id, new_embedding)

        assert result is True
        assert crud._embedding_cache[memory_id] == new_embedding

        crud.close()

    def test_update_embedding_nonexistent(self):
        """Test updating embedding for non-existent memory"""
        crud = MemoryCRUD(':memory:')

        fake_id = str(uuid.uuid4())
        result = crud.update_embedding(fake_id, [0.1] * 768)

        assert result is False

        crud.close()

    def test_delete_memory_basic(self):
        """Test deleting a memory"""
        crud = MemoryCRUD(':memory:')

        memory_id = crud.store_memory("To be deleted", memory_type="fact")

        # Delete it
        result = crud.delete_memory(memory_id)

        assert result is True

        # Verify it's gone
        cursor = crud._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        assert cursor.fetchone() is None

        crud.close()

    def test_delete_memory_removes_fts_entry(self):
        """Test that deleting memory also removes FTS entry"""
        crud = MemoryCRUD(':memory:')

        memory_id = crud.store_memory("To be deleted", memory_type="fact")

        # Delete memory
        crud.delete_memory(memory_id)

        # Verify FTS entry is gone
        cursor = crud._conn.execute("""
            SELECT * FROM memories_fts WHERE memory_id = ?
        """, (memory_id,))
        assert cursor.fetchone() is None

        crud.close()

    def test_delete_memory_removes_from_cache(self):
        """Test that deleting memory removes from embedding cache"""
        crud = MemoryCRUD(':memory:')

        embedding = [0.3] * 768
        memory_id = crud.store_memory("Test", embedding=embedding, memory_type="fact")

        assert memory_id in crud._embedding_cache

        # Delete memory
        crud.delete_memory(memory_id)

        assert memory_id not in crud._embedding_cache

        crud.close()

    def test_delete_memory_nonexistent(self):
        """Test deleting non-existent memory returns False"""
        crud = MemoryCRUD(':memory:')

        fake_id = str(uuid.uuid4())
        result = crud.delete_memory(fake_id)

        assert result is False

        crud.close()

    def test_embedding_cache_behavior(self):
        """Test embedding cache is populated correctly"""
        crud = MemoryCRUD(':memory:')

        # Store without embedding - should not be cached
        mem_id_1 = crud.store_memory("No embedding", memory_type="fact")
        assert mem_id_1 not in crud._embedding_cache

        # Store with embedding - should be cached
        embedding = [0.4] * 768
        mem_id_2 = crud.store_memory("With embedding", embedding=embedding, memory_type="fact")
        assert mem_id_2 in crud._embedding_cache
        assert crud._embedding_cache[mem_id_2] == embedding

        crud.close()

    def test_close_clears_cache(self):
        """Test that close() clears embedding cache"""
        crud = MemoryCRUD(':memory:')

        embedding = [0.5] * 768
        memory_id = crud.store_memory("Test", embedding=embedding, memory_type="fact")

        assert len(crud._embedding_cache) > 0

        crud.close()

        assert len(crud._embedding_cache) == 0

    def test_context_manager(self, tmp_path):
        """Test context manager usage"""
        db_path = tmp_path / "test_cm.db"

        with MemoryCRUD(str(db_path)) as crud:
            memory_id = crud.store_memory("Test", memory_type="fact")
            assert memory_id is not None

    def test_memory_types_constant(self):
        """Test that MEMORY_TYPES constant is correct"""
        crud = MemoryCRUD(':memory:')

        expected_types = {"fact", "experience", "belief", "decision"}
        assert crud.MEMORY_TYPES == expected_types

        crud.close()

    def test_store_all_memory_types(self):
        """Test storing all valid memory types"""
        crud = MemoryCRUD(':memory:')

        for mem_type in ["fact", "experience", "belief", "decision"]:
            memory_id = crud.store_memory(f"Test {mem_type}", memory_type=mem_type)
            memory = crud.get_memory(memory_id)
            assert memory.memory_type == mem_type

        crud.close()
