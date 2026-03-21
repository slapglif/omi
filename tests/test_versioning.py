"""
Tests for memory versioning functionality.

Tests that memory updates create versions (append-only history) rather than overwriting.
"""

import pytest
import sqlite3
from pathlib import Path
from datetime import datetime
from omi.storage.graph_palace import GraphPalace
from omi.storage.schema import init_database


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_palace.db"
    conn = sqlite3.connect(db_path)
    init_database(conn, enable_wal=False)
    conn.close()
    return db_path


def test_store_creates_version(temp_db):
    """
    Test that store_memory creates a version entry when storing a new memory.

    When creating a new memory:
    - Should insert into memories table
    - Should insert version with operation_type='CREATE' and version_number=1
    """
    palace = GraphPalace(temp_db, enable_wal=False)

    # Store a new memory
    memory_id = palace.store_memory(
        content="Test memory content",
        memory_type="fact",
        confidence=0.9
    )

    # Verify memory was stored
    assert memory_id is not None

    # Verify version was created
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("""
        SELECT memory_id, content, version_number, operation_type
        FROM memory_versions
        WHERE memory_id = ?
    """, (memory_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Version should be created for new memory"
    assert row[0] == memory_id
    assert row[1] == "Test memory content"
    assert row[2] == 1, "First version should have version_number=1"
    assert row[3] == "CREATE", "First version should have operation_type='CREATE'"


def test_update_creates_new_version(temp_db):
    """
    Test that updating an existing memory creates a new version instead of overwriting.

    When updating a memory:
    - Should insert new version with operation_type='UPDATE' and incremented version_number
    - Should update memories table with new content
    - Should preserve old content in memory_versions
    """
    palace = GraphPalace(temp_db, enable_wal=False)

    # Store initial memory
    memory_id = palace.store_memory(
        content="Original content",
        memory_type="fact"
    )

    # Update the memory
    updated_id = palace.store_memory(
        content="Updated content",
        memory_type="fact",
        memory_id=memory_id  # Specify memory_id to update
    )

    # Should return same memory_id
    assert updated_id == memory_id

    # Verify current content in memories table
    memory = palace.get_memory(memory_id)
    assert memory.content == "Updated content"

    # Verify both versions exist
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("""
        SELECT version_number, operation_type, content
        FROM memory_versions
        WHERE memory_id = ?
        ORDER BY version_number
    """, (memory_id,))
    versions = cursor.fetchall()
    conn.close()

    assert len(versions) == 2, "Should have 2 versions"

    # First version
    assert versions[0][0] == 1
    assert versions[0][1] == "CREATE"
    assert versions[0][2] == "Original content"

    # Second version
    assert versions[1][0] == 2
    assert versions[1][1] == "UPDATE"
    assert versions[1][2] == "Updated content"


def test_multiple_updates_increment_version(temp_db):
    """
    Test that multiple updates correctly increment version_number.
    """
    palace = GraphPalace(temp_db, enable_wal=False)

    # Create initial memory
    memory_id = palace.store_memory(content="Version 1")

    # Update multiple times
    palace.store_memory(content="Version 2", memory_id=memory_id)
    palace.store_memory(content="Version 3", memory_id=memory_id)
    palace.store_memory(content="Version 4", memory_id=memory_id)

    # Verify version count and numbers
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("""
        SELECT version_number, operation_type
        FROM memory_versions
        WHERE memory_id = ?
        ORDER BY version_number
    """, (memory_id,))
    versions = cursor.fetchall()
    conn.close()

    assert len(versions) == 4
    assert versions[0][0] == 1 and versions[0][1] == "CREATE"
    assert versions[1][0] == 2 and versions[1][1] == "UPDATE"
    assert versions[2][0] == 3 and versions[2][1] == "UPDATE"
    assert versions[3][0] == 4 and versions[3][1] == "UPDATE"


def test_recall_at_timestamp(temp_db):
    """
    Test that recall_at returns memories as they existed at a specific timestamp.

    When querying at a point in time:
    - Should return memories that existed at that time
    - Should return the correct version (not the current one)
    - Should exclude memories created after the timestamp
    - Should exclude memories deleted before the timestamp
    """
    import time
    palace = GraphPalace(temp_db, enable_wal=False)

    # Store initial memory
    memory_id_1 = palace.store_memory(
        content="Original content v1",
        memory_type="fact"
    )

    # Get timestamp after first memory
    time.sleep(0.1)
    timestamp_after_v1 = datetime.now()
    time.sleep(0.1)

    # Update the memory (v2)
    palace.store_memory(
        content="Updated content v2",
        memory_type="fact",
        memory_id=memory_id_1
    )

    # Get timestamp after update
    time.sleep(0.1)
    timestamp_after_v2 = datetime.now()
    time.sleep(0.1)

    # Create a second memory
    memory_id_2 = palace.store_memory(
        content="Second memory",
        memory_type="experience"
    )

    # Test 1: Recall at timestamp_after_v1 should return v1 content
    memories_at_v1 = palace.recall_at(timestamp_after_v1)
    assert len(memories_at_v1) == 1, "Should have 1 memory at timestamp_after_v1"
    assert memories_at_v1[0].id == memory_id_1
    assert memories_at_v1[0].content == "Original content v1"

    # Test 2: Recall at timestamp_after_v2 should return v2 content and exclude second memory
    memories_at_v2 = palace.recall_at(timestamp_after_v2)
    assert len(memories_at_v2) == 1, "Should have 1 memory at timestamp_after_v2"
    assert memories_at_v2[0].id == memory_id_1
    assert memories_at_v2[0].content == "Updated content v2"

    # Test 3: Recall at current time should return both memories
    memories_now = palace.recall_at(datetime.now())
    assert len(memories_now) == 2, "Should have 2 memories at current time"
    memory_ids = {m.id for m in memories_now}
    assert memory_id_1 in memory_ids
    assert memory_id_2 in memory_ids
