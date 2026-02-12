"""
Tests for snapshot functionality.

Tests that snapshots capture point-in-time memory state with delta encoding.
"""

import pytest
import sqlite3
import json
import tarfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import time
from omi.storage.snapshots import SnapshotManager, SnapshotInfo, SnapshotDiff
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


@pytest.fixture
def snapshot_manager(temp_db):
    """Create a snapshot manager for testing."""
    return SnapshotManager(temp_db)


@pytest.fixture
def palace(temp_db):
    """Create a GraphPalace for testing."""
    return GraphPalace(temp_db, enable_wal=False)


def test_delta_encoding(temp_db, snapshot_manager, palace):
    """
    Test that delta encoding stores only changes between snapshots.

    Scenario:
    1. Create initial memories and first snapshot (full)
    2. Add, modify, and delete memories
    3. Create second snapshot (delta)
    4. Verify delta stores only changes
    5. Verify state reconstruction works correctly
    """
    # Step 1: Create initial memories
    memory_1 = palace.store_memory(content="Memory 1", memory_type="fact")
    memory_2 = palace.store_memory(content="Memory 2", memory_type="fact")
    memory_3 = palace.store_memory(content="Memory 3", memory_type="fact")

    # Create first snapshot (should be full)
    time.sleep(0.1)  # Ensure distinct timestamps
    snapshot1 = snapshot_manager.create_snapshot(description="Full snapshot")

    assert snapshot1.is_delta is False, "First snapshot should not be a delta"
    assert snapshot1.memory_count == 3, "First snapshot should have 3 memories"

    # Verify all memories stored with ADDED operation
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("""
        SELECT memory_id, operation_type
        FROM snapshot_memories
        WHERE snapshot_id = ?
    """, (snapshot1.snapshot_id,))
    snapshot1_memories = cursor.fetchall()
    conn.close()

    assert len(snapshot1_memories) == 3, "First snapshot should have 3 memory entries"
    for _, op_type in snapshot1_memories:
        assert op_type == "ADDED", "All operations in full snapshot should be ADDED"

    # Step 2: Make changes - add, modify, delete
    time.sleep(0.1)

    # Add a new memory
    memory_4 = palace.store_memory(content="Memory 4 - New", memory_type="fact")

    # Modify an existing memory (memory_2)
    palace.store_memory(content="Memory 2 - Updated", memory_type="fact", memory_id=memory_2)

    # Delete a memory (memory_3)
    conn = sqlite3.connect(temp_db)
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_3,))
    conn.commit()
    conn.close()

    # Step 3: Create second snapshot (should be delta)
    time.sleep(0.1)
    snapshot2 = snapshot_manager.create_snapshot(description="Delta snapshot")

    assert snapshot2.is_delta is True, "Second snapshot should be a delta"
    assert snapshot2.memory_count == 3, "Delta should have 3 changes (1 added, 1 modified, 1 deleted)"

    # Verify delta contains only changes
    conn = sqlite3.connect(temp_db)
    cursor = conn.execute("""
        SELECT memory_id, operation_type
        FROM snapshot_memories
        WHERE snapshot_id = ?
        ORDER BY operation_type
    """, (snapshot2.snapshot_id,))
    snapshot2_memories = cursor.fetchall()
    conn.close()

    # Group by operation type
    ops_by_type = {}
    for mem_id, op_type in snapshot2_memories:
        ops_by_type.setdefault(op_type, []).append(mem_id)

    # Verify we have the expected operations
    assert "ADDED" in ops_by_type, "Should have ADDED operation"
    assert "MODIFIED" in ops_by_type, "Should have MODIFIED operation"
    assert "DELETED" in ops_by_type, "Should have DELETED operation"

    assert len(ops_by_type["ADDED"]) == 1, "Should have 1 added memory"
    assert memory_4 in ops_by_type["ADDED"], "Memory 4 should be marked as ADDED"

    assert len(ops_by_type["MODIFIED"]) == 1, "Should have 1 modified memory"
    assert memory_2 in ops_by_type["MODIFIED"], "Memory 2 should be marked as MODIFIED"

    assert len(ops_by_type["DELETED"]) == 1, "Should have 1 deleted memory"
    assert memory_3 in ops_by_type["DELETED"], "Memory 3 should be marked as DELETED"

    # Step 4: Verify state reconstruction at snapshot1
    state1 = snapshot_manager._get_snapshot_memory_state(
        sqlite3.connect(temp_db),
        snapshot1.snapshot_id
    )
    assert len(state1) == 3, "Snapshot1 should have 3 memories"
    assert memory_1 in state1
    assert memory_2 in state1
    assert memory_3 in state1
    assert memory_4 not in state1, "Memory 4 should not exist at snapshot1"

    # Step 5: Verify state reconstruction at snapshot2
    state2 = snapshot_manager._get_snapshot_memory_state(
        sqlite3.connect(temp_db),
        snapshot2.snapshot_id
    )
    assert len(state2) == 3, "Snapshot2 should have 3 memories (1, 2, 4)"
    assert memory_1 in state2, "Memory 1 should still exist"
    assert memory_2 in state2, "Memory 2 should exist (modified)"
    assert memory_3 not in state2, "Memory 3 should be deleted"
    assert memory_4 in state2, "Memory 4 should be added"

    # Step 6: Verify diff between snapshots
    diff = snapshot_manager.diff_snapshots(snapshot1.snapshot_id, snapshot2.snapshot_id)

    assert len(diff.added) == 1, "Diff should show 1 added memory"
    assert memory_4 in diff.added, "Memory 4 should be in added list"

    assert len(diff.modified) == 1, "Diff should show 1 modified memory"
    assert memory_2 in diff.modified, "Memory 2 should be in modified list"

    assert len(diff.deleted) == 1, "Diff should show 1 deleted memory"
    assert memory_3 in diff.deleted, "Memory 3 should be in deleted list"

    assert diff.total_changes == 3, "Total changes should be 3"


def test_create_snapshot(temp_db, snapshot_manager, palace):
    """Test basic snapshot creation."""
    # Store some memories
    palace.store_memory(content="Test memory", memory_type="fact")

    # Create snapshot
    snapshot = snapshot_manager.create_snapshot(
        description="Test snapshot",
        metadata={"test": True}
    )

    assert snapshot.snapshot_id is not None
    assert snapshot.description == "Test snapshot"
    assert snapshot.metadata == {"test": True}
    assert snapshot.memory_count == 1


def test_list_snapshots(temp_db, snapshot_manager, palace):
    """Test listing snapshots."""
    palace.store_memory(content="Memory 1", memory_type="fact")

    # Create multiple snapshots
    snap1 = snapshot_manager.create_snapshot(description="Snapshot 1")
    time.sleep(0.1)
    snap2 = snapshot_manager.create_snapshot(description="Snapshot 2")
    time.sleep(0.1)
    snap3 = snapshot_manager.create_snapshot(description="Snapshot 3")

    # List all snapshots
    snapshots = snapshot_manager.list_snapshots()
    assert len(snapshots) == 3

    # Should be ordered newest first
    assert snapshots[0].snapshot_id == snap3.snapshot_id
    assert snapshots[1].snapshot_id == snap2.snapshot_id
    assert snapshots[2].snapshot_id == snap1.snapshot_id

    # Test limit
    limited = snapshot_manager.list_snapshots(limit=2)
    assert len(limited) == 2


def test_get_snapshot(temp_db, snapshot_manager, palace):
    """Test retrieving a specific snapshot."""
    palace.store_memory(content="Test memory", memory_type="fact")

    snapshot = snapshot_manager.create_snapshot(description="Test")

    # Retrieve snapshot
    retrieved = snapshot_manager.get_snapshot(snapshot.snapshot_id)

    assert retrieved is not None
    assert retrieved.snapshot_id == snapshot.snapshot_id
    assert retrieved.description == "Test"

    # Test non-existent snapshot
    not_found = snapshot_manager.get_snapshot("snap-nonexistent")
    assert not_found is None


def test_diff_snapshots(temp_db, snapshot_manager, palace):
    """Test diffing between two snapshots."""
    # Create initial state
    mem1 = palace.store_memory(content="Memory 1", memory_type="fact")
    mem2 = palace.store_memory(content="Memory 2", memory_type="fact")

    snap1 = snapshot_manager.create_snapshot(description="Before")

    # Make changes
    time.sleep(0.1)
    palace.store_memory(content="Memory 2 updated", memory_id=mem2)
    mem3 = palace.store_memory(content="Memory 3", memory_type="fact")

    snap2 = snapshot_manager.create_snapshot(description="After")

    # Diff
    diff = snapshot_manager.diff_snapshots(snap1.snapshot_id, snap2.snapshot_id)

    assert len(diff.added) == 1
    assert mem3 in diff.added
    assert len(diff.modified) == 1
    assert mem2 in diff.modified
    assert len(diff.deleted) == 0


def test_rollback_to_snapshot(temp_db, snapshot_manager, palace):
    """Test rolling back to a previous snapshot."""
    # Create initial state
    mem1 = palace.store_memory(content="Original 1", memory_type="fact")
    mem2 = palace.store_memory(content="Original 2", memory_type="fact")

    snap1 = snapshot_manager.create_snapshot(description="Before changes")

    # Make changes
    time.sleep(0.1)
    palace.store_memory(content="Modified 1", memory_id=mem1)
    palace.store_memory(content="New memory", memory_type="fact")

    # Verify current state is different
    current_mem1 = palace.get_memory(mem1)
    assert current_mem1.content == "Modified 1"

    # Rollback
    changes = snapshot_manager.rollback_to_snapshot(snap1.snapshot_id)

    assert changes > 0, "Should have made changes during rollback"

    # Verify state restored
    restored_mem1 = palace.get_memory(mem1)
    assert restored_mem1.content == "Original 1", "Memory 1 should be restored to original"

    restored_mem2 = palace.get_memory(mem2)
    assert restored_mem2.content == "Original 2", "Memory 2 should be unchanged"


def test_delete_snapshot(temp_db, snapshot_manager, palace):
    """Test deleting a snapshot."""
    palace.store_memory(content="Test", memory_type="fact")

    snapshot = snapshot_manager.create_snapshot(description="To delete")

    # Delete snapshot
    deleted = snapshot_manager.delete_snapshot(snapshot.snapshot_id)
    assert deleted is True

    # Verify it's gone
    retrieved = snapshot_manager.get_snapshot(snapshot.snapshot_id)
    assert retrieved is None

    # Try to delete non-existent
    deleted_again = snapshot_manager.delete_snapshot(snapshot.snapshot_id)
    assert deleted_again is False


def test_snapshot_with_no_changes(temp_db, snapshot_manager, palace):
    """Test creating a snapshot when there are no changes."""
    mem1 = palace.store_memory(content="Memory 1", memory_type="fact")

    # Create first snapshot
    snap1 = snapshot_manager.create_snapshot(description="First")
    assert snap1.memory_count == 1

    # Create second snapshot with no changes
    time.sleep(0.1)
    snap2 = snapshot_manager.create_snapshot(description="Second")

    # Should be a delta with 0 changes
    assert snap2.is_delta is True
    assert snap2.memory_count == 0, "No changes should result in 0 memories in delta"

    # State should be the same
    conn = sqlite3.connect(temp_db)
    state1 = snapshot_manager._get_snapshot_memory_state(conn, snap1.snapshot_id)
    state2 = snapshot_manager._get_snapshot_memory_state(conn, snap2.snapshot_id)
    conn.close()

    assert state1 == state2, "State should be identical"


def test_moltvault_integration(temp_db, palace, tmp_path):
    """Test integration with MoltVault for cloud backup."""
    # Create a mock MoltVault instance
    mock_moltvault = Mock()
    mock_backend = Mock()

    # Track uploaded files
    uploaded_files = {}

    def mock_upload(local_path, key, metadata=None):
        """Mock upload that stores file content for verification."""
        # Read and store the file content
        with open(local_path, 'rb') as f:
            uploaded_files[key] = {
                'content': f.read(),
                'metadata': metadata
            }
        return key

    mock_backend.upload = mock_upload
    mock_moltvault._get_backend.return_value = mock_backend

    # Create snapshot manager with MoltVault
    snapshot_manager = SnapshotManager(temp_db, moltvault=mock_moltvault)

    # Create some memories
    mem1 = palace.store_memory(content="Memory 1", memory_type="fact")
    mem2 = palace.store_memory(content="Memory 2", memory_type="fact")

    # Create snapshot with MoltVault backup
    snapshot = snapshot_manager.create_snapshot(
        description="Test snapshot with cloud backup",
        backup_to_moltvault=True
    )

    # Verify snapshot was created
    assert snapshot.snapshot_id is not None
    assert snapshot.description == "Test snapshot with cloud backup"
    assert snapshot.moltvault_backup_id is not None

    # Verify backend was called
    mock_moltvault._get_backend.assert_called_once()

    # Verify file was uploaded
    assert len(uploaded_files) == 1
    backup_key = snapshot.moltvault_backup_id
    assert backup_key in uploaded_files
    assert backup_key.startswith("snapshots/")
    assert backup_key.endswith(".tar.gz")

    # Verify uploaded metadata
    upload_metadata = uploaded_files[backup_key]['metadata']
    assert upload_metadata['snapshot_id'] == snapshot.snapshot_id
    assert upload_metadata['description'] == "Test snapshot with cloud backup"
    assert upload_metadata['type'] == "snapshot"

    # Extract and verify the tar.gz content
    uploaded_content = uploaded_files[backup_key]['content']
    tar_path = tmp_path / "uploaded.tar.gz"
    with open(tar_path, 'wb') as f:
        f.write(uploaded_content)

    # Extract tar and read JSON
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == 1, "Archive should contain one JSON file"

        json_member = members[0]
        assert json_member.name.endswith('.json')

        # Extract and parse JSON
        json_file = tar.extractfile(json_member)
        snapshot_data = json.load(json_file)

    # Verify exported snapshot data structure
    assert 'snapshot' in snapshot_data
    assert 'memories' in snapshot_data
    assert 'export_version' in snapshot_data

    # Verify snapshot metadata
    assert snapshot_data['snapshot']['snapshot_id'] == snapshot.snapshot_id
    assert snapshot_data['snapshot']['description'] == "Test snapshot with cloud backup"

    # Verify memory data
    assert len(snapshot_data['memories']) == 2
    memory_ids = {m['memory_id'] for m in snapshot_data['memories']}
    assert mem1 in memory_ids
    assert mem2 in memory_ids

    # Verify each memory has required fields
    for memory in snapshot_data['memories']:
        assert 'memory_id' in memory
        assert 'version_id' in memory
        assert 'operation_type' in memory
        assert 'content' in memory
        assert memory['operation_type'] == 'ADDED'  # First snapshot, all memories are ADDED


def test_moltvault_integration_without_instance(temp_db, palace):
    """Test that backup_to_moltvault=True fails without MoltVault instance."""
    # Create snapshot manager without MoltVault
    snapshot_manager = SnapshotManager(temp_db, moltvault=None)

    # Create a memory
    palace.store_memory(content="Test memory", memory_type="fact")

    # Try to create snapshot with backup - should fail
    with pytest.raises(ValueError, match="backup_to_moltvault=True but no MoltVault instance configured"):
        snapshot_manager.create_snapshot(
            description="Should fail",
            backup_to_moltvault=True
        )


def test_moltvault_integration_local_only_mode(temp_db, palace):
    """Test creating snapshots without MoltVault backup (local-only mode)."""
    # Create snapshot manager without MoltVault
    snapshot_manager = SnapshotManager(temp_db, moltvault=None)

    # Create a memory
    palace.store_memory(content="Test memory", memory_type="fact")

    # Create snapshot without backup - should work fine
    snapshot = snapshot_manager.create_snapshot(
        description="Local only snapshot",
        backup_to_moltvault=False
    )

    # Verify snapshot was created
    assert snapshot.snapshot_id is not None
    assert snapshot.description == "Local only snapshot"
    assert snapshot.moltvault_backup_id is None  # No cloud backup
