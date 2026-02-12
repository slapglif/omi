"""End-to-End Time Travel Tests for OMI

Tests the complete time-travel flow:
1. Store a memory
2. Update the memory (creates version)
3. Query at point-in-time before update
4. Verify historical version is returned

Verifies that memory versioning and point-in-time queries work
correctly through the entire pipeline.
"""
import pytest
from datetime import datetime, timedelta
from time import sleep


class TestE2ETimeTravel:
    """Test end-to-end time travel: version creation → point-in-time query."""

    def test_version_creation_through_time_travel_query(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        End-to-end verification:
        1. Store a memory via MemoryTools
        2. Record timestamp
        3. Update memory (creates version)
        4. Query with recall_at before update
        5. Verify old version returned
        6. Verify current version different
        """
        from omi.api import MemoryTools
        from omi import GraphPalace

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Step 1: Store initial memory
        original_content = "Initial project setup requires authentication"
        memory_id = memory_tools.store(
            content=original_content,
            memory_type="fact",
            confidence=0.85
        )

        assert memory_id is not None, "Memory should be stored successfully"

        # Step 2: Record timestamp after storage, before update
        # Add small delay to ensure distinct timestamps
        sleep(0.1)
        timestamp_before_update = datetime.now()
        sleep(0.1)

        # Step 3: Update the memory (creates a new version)
        # Use store_memory with existing memory_id to create a version
        updated_content = "Project setup now uses OAuth2 authentication"
        result_id = palace.store_memory(
            content=updated_content,
            memory_type="fact",
            confidence=0.85,
            memory_id=memory_id
        )
        assert result_id == memory_id, "Memory update should return same ID"

        # Step 4: Query using recall_at with timestamp before update
        historical_memories = palace.recall_at(timestamp_before_update)

        # Step 5: Verify old version is returned
        assert len(historical_memories) > 0, "Should retrieve at least one historical memory"

        # Find the specific memory we're testing
        historical_memory = None
        for mem in historical_memories:
            if mem.id == memory_id:
                historical_memory = mem
                break

        assert historical_memory is not None, f"Should find memory {memory_id} in historical results"
        assert historical_memory.content == original_content, \
            f"Historical content should be '{original_content}', got '{historical_memory.content}'"

        # Step 6: Verify current version is different
        current_memory = palace.get_memory(memory_id)
        assert current_memory is not None, "Should retrieve current memory"
        assert current_memory.content == updated_content, \
            f"Current content should be '{updated_content}', got '{current_memory.content}'"

        # Verify they are different
        assert historical_memory.content != current_memory.content, \
            "Historical and current content should be different"

    def test_multiple_version_history(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Test that multiple updates create correct version history.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Store initial memory
        version_1 = "Version 1: Initial thought"
        memory_id = memory_tools.store(
            content=version_1,
            memory_type="experience"
        )

        # Record timestamp after v1
        sleep(0.1)
        timestamp_after_v1 = datetime.now()
        sleep(0.1)

        # Update to v2 (use store_memory with existing ID to create version)
        version_2 = "Version 2: Revised thought"
        palace.store_memory(
            content=version_2,
            memory_type="experience",
            memory_id=memory_id
        )

        # Record timestamp after v2
        sleep(0.1)
        timestamp_after_v2 = datetime.now()
        sleep(0.1)

        # Update to v3 (use store_memory with existing ID to create version)
        version_3 = "Version 3: Final thought"
        palace.store_memory(
            content=version_3,
            memory_type="experience",
            memory_id=memory_id
        )

        # Query at different points in time
        memories_at_v1 = palace.recall_at(timestamp_after_v1)
        memories_at_v2 = palace.recall_at(timestamp_after_v2)
        memories_current = [palace.get_memory(memory_id)]

        # Find our memory in each timeline
        mem_v1 = next((m for m in memories_at_v1 if m.id == memory_id), None)
        mem_v2 = next((m for m in memories_at_v2 if m.id == memory_id), None)
        mem_v3 = memories_current[0]

        # Verify each version
        assert mem_v1 is not None, "Should find memory at v1 timestamp"
        assert mem_v1.content == version_1, f"Content at v1 should be '{version_1}'"

        assert mem_v2 is not None, "Should find memory at v2 timestamp"
        assert mem_v2.content == version_2, f"Content at v2 should be '{version_2}'"

        assert mem_v3 is not None, "Should find current memory"
        assert mem_v3.content == version_3, f"Current content should be '{version_3}'"

        # Verify all versions are different
        assert mem_v1.content != mem_v2.content
        assert mem_v2.content != mem_v3.content
        assert mem_v1.content != mem_v3.content

    def test_recall_at_with_multiple_memories(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Test recall_at returns correct state when multiple memories exist.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Store first memory
        mem1_id = memory_tools.store(
            content="Memory 1 initial",
            memory_type="fact"
        )

        sleep(0.1)
        timestamp_after_mem1 = datetime.now()
        sleep(0.1)

        # Store second memory
        mem2_id = memory_tools.store(
            content="Memory 2 initial",
            memory_type="fact"
        )

        sleep(0.1)
        timestamp_after_mem2 = datetime.now()
        sleep(0.1)

        # Update memory 1 (use store_memory with existing ID to create version)
        palace.store_memory(
            content="Memory 1 updated",
            memory_type="fact",
            memory_id=mem1_id
        )

        # Query at timestamp_after_mem1 (only mem1 should exist, with original content)
        memories_t1 = palace.recall_at(timestamp_after_mem1)
        assert len(memories_t1) == 1, "Should have 1 memory at timestamp_after_mem1"
        assert memories_t1[0].id == mem1_id
        assert memories_t1[0].content == "Memory 1 initial"

        # Query at timestamp_after_mem2 (both memories should exist, mem1 with original content)
        memories_t2 = palace.recall_at(timestamp_after_mem2)
        assert len(memories_t2) == 2, "Should have 2 memories at timestamp_after_mem2"

        mem1_t2 = next((m for m in memories_t2 if m.id == mem1_id), None)
        mem2_t2 = next((m for m in memories_t2 if m.id == mem2_id), None)

        assert mem1_t2 is not None
        assert mem1_t2.content == "Memory 1 initial", "Memory 1 should have original content at t2"

        assert mem2_t2 is not None
        assert mem2_t2.content == "Memory 2 initial"

        # Query current state (mem1 should be updated)
        current_mem1 = palace.get_memory(mem1_id)
        assert current_mem1.content == "Memory 1 updated"

    def test_recall_at_before_any_memories(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Test recall_at with timestamp before any memories exist.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Query in the past (before any memories)
        past_timestamp = datetime.now() - timedelta(days=1)
        memories = palace.recall_at(past_timestamp)

        assert len(memories) == 0, "Should have no memories before creation time"

        # Now store a memory
        memory_tools.store(
            content="New memory",
            memory_type="fact"
        )

        # Query again at the same past timestamp
        memories_still_empty = palace.recall_at(past_timestamp)
        assert len(memories_still_empty) == 0, "Past query should still return no memories"

        # Query at current time should return the memory
        current_memories = palace.recall_at(datetime.now() + timedelta(seconds=1))
        assert len(current_memories) == 1, "Current query should return 1 memory"

    def test_version_metadata_preservation(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Verify that metadata (type, confidence) is preserved in historical queries.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Store with specific metadata
        original_type = "decision"
        original_confidence = 0.92
        memory_id = memory_tools.store(
            content="Original decision",
            memory_type=original_type,
            confidence=original_confidence
        )

        sleep(0.1)
        timestamp_original = datetime.now()
        sleep(0.1)

        # Update content only (use store_memory with existing ID to create version)
        palace.store_memory(
            content="Updated decision",
            memory_type=original_type,
            confidence=original_confidence,
            memory_id=memory_id
        )

        # Query historical version
        historical_memories = palace.recall_at(timestamp_original)
        historical_memory = next((m for m in historical_memories if m.id == memory_id), None)

        assert historical_memory is not None
        assert historical_memory.content == "Original decision"
        # Metadata should be preserved (from main memories table)
        assert historical_memory.memory_type == original_type
        assert historical_memory.confidence == original_confidence


class TestE2ESnapshotWorkflow:
    """Test end-to-end snapshot workflow: create → modify → diff → rollback."""

    def test_snapshot_create_diff_rollback_workflow(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Complete end-to-end snapshot workflow:
        1. Create initial memories
        2. Create snapshot-1
        3. Modify memories (add, modify, delete)
        4. Create snapshot-2
        5. Diff shows all changes correctly
        6. Rollback to snapshot-1
        7. Verify state restored to snapshot-1
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.storage.snapshots import SnapshotManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)
        snapshot_manager = SnapshotManager(temp_omi_setup["db_path"])

        # Step 1: Create initial memories
        original_mem1 = "Initial memory 1: project uses Flask"
        original_mem2 = "Initial memory 2: database is PostgreSQL"
        original_mem3 = "Initial memory 3: deployment is Docker"

        mem1_id = memory_tools.store(
            content=original_mem1,
            memory_type="fact",
            confidence=0.9
        )
        mem2_id = memory_tools.store(
            content=original_mem2,
            memory_type="fact",
            confidence=0.85
        )
        mem3_id = memory_tools.store(
            content=original_mem3,
            memory_type="fact",
            confidence=0.8
        )

        assert mem1_id is not None
        assert mem2_id is not None
        assert mem3_id is not None

        # Step 2: Create snapshot-1 (full snapshot)
        sleep(0.1)  # Ensure distinct timestamp
        snapshot1 = snapshot_manager.create_snapshot(
            description="Before changes - initial state"
        )

        assert snapshot1 is not None
        assert snapshot1.snapshot_id is not None
        assert snapshot1.description == "Before changes - initial state"
        assert snapshot1.memory_count == 3, "Snapshot 1 should have 3 memories"
        assert snapshot1.is_delta is False, "First snapshot should be full, not delta"

        # Step 3: Modify memories (add, modify, delete)
        sleep(0.1)

        # Modify mem2 (update existing memory using palace.store_memory with memory_id)
        updated_mem2 = "Updated memory 2: database migrated to SQLite"
        palace.store_memory(
            content=updated_mem2,
            memory_type="fact",
            confidence=0.85,
            memory_id=mem2_id
        )

        # Add new memory
        new_mem4 = "New memory 4: added caching layer"
        mem4_id = memory_tools.store(
            content=new_mem4,
            memory_type="fact",
            confidence=0.75
        )

        # Delete mem3
        import sqlite3
        with sqlite3.connect(temp_omi_setup["db_path"]) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem3_id,))
            conn.commit()

        # Step 4: Create snapshot-2 (delta snapshot)
        sleep(0.1)
        snapshot2 = snapshot_manager.create_snapshot(
            description="After changes - modified state"
        )

        assert snapshot2 is not None
        assert snapshot2.snapshot_id is not None
        assert snapshot2.description == "After changes - modified state"
        assert snapshot2.is_delta is True, "Second snapshot should be delta"
        # Delta should capture: 1 added, 1 modified, 1 deleted = 3 changes
        assert snapshot2.memory_count == 3, f"Snapshot 2 should have 3 changes, got {snapshot2.memory_count}"

        # Step 5: Diff shows all changes correctly
        diff = snapshot_manager.diff_snapshots(snapshot1.snapshot_id, snapshot2.snapshot_id)

        assert diff is not None
        assert diff.snapshot1_id == snapshot1.snapshot_id
        assert diff.snapshot2_id == snapshot2.snapshot_id

        # Verify added memories
        assert len(diff.added) == 1, f"Should have 1 added memory, got {len(diff.added)}"
        assert mem4_id in diff.added, f"Memory 4 should be in added list"

        # Verify modified memories
        assert len(diff.modified) == 1, f"Should have 1 modified memory, got {len(diff.modified)}"
        assert mem2_id in diff.modified, f"Memory 2 should be in modified list"

        # Verify deleted memories
        assert len(diff.deleted) == 1, f"Should have 1 deleted memory, got {len(diff.deleted)}"
        assert mem3_id in diff.deleted, f"Memory 3 should be in deleted list"

        # Verify total changes
        assert diff.total_changes == 3, f"Total changes should be 3, got {diff.total_changes}"

        # Verify current state before rollback
        current_mem1 = palace.get_memory(mem1_id)
        current_mem2 = palace.get_memory(mem2_id)
        current_mem3 = palace.get_memory(mem3_id)  # Should be None (deleted)
        current_mem4 = palace.get_memory(mem4_id)

        assert current_mem1 is not None
        assert current_mem1.content == original_mem1, "Memory 1 should be unchanged"

        assert current_mem2 is not None
        assert current_mem2.content == updated_mem2, "Memory 2 should be updated"

        assert current_mem3 is None, "Memory 3 should be deleted"

        assert current_mem4 is not None
        assert current_mem4.content == new_mem4, "Memory 4 should exist"

        # Step 6: Rollback to snapshot-1
        changes = snapshot_manager.rollback_to_snapshot(snapshot1.snapshot_id)

        assert changes > 0, "Rollback should have made changes"

        # Step 7: Verify state restored to snapshot-1
        restored_mem1 = palace.get_memory(mem1_id)
        restored_mem2 = palace.get_memory(mem2_id)
        restored_mem3 = palace.get_memory(mem3_id)
        restored_mem4 = palace.get_memory(mem4_id)

        # Memory 1 should be unchanged (was never modified)
        assert restored_mem1 is not None
        assert restored_mem1.content == original_mem1, \
            f"Memory 1 should be restored to '{original_mem1}', got '{restored_mem1.content}'"

        # Memory 2 should be reverted to original
        assert restored_mem2 is not None
        assert restored_mem2.content == original_mem2, \
            f"Memory 2 should be restored to '{original_mem2}', got '{restored_mem2.content}'"

        # Memory 3 should be restored (was deleted)
        assert restored_mem3 is not None, "Memory 3 should be restored after rollback"
        assert restored_mem3.content == original_mem3, \
            f"Memory 3 should be restored to '{original_mem3}', got '{restored_mem3.content}'"

        # Memory 4 should be deleted (didn't exist in snapshot-1)
        assert restored_mem4 is None, \
            "Memory 4 should be deleted after rollback (didn't exist in snapshot-1)"

        # Verify we're back to exactly 3 memories
        import sqlite3
        with sqlite3.connect(temp_omi_setup["db_path"]) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]
            assert memory_count == 3, \
                f"Should have exactly 3 memories after rollback, got {memory_count}"

    def test_snapshot_workflow_with_multiple_snapshots(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Test snapshot workflow with multiple sequential snapshots.
        Verifies delta encoding and rollback to non-sequential snapshots.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.storage.snapshots import SnapshotManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)
        snapshot_manager = SnapshotManager(temp_omi_setup["db_path"])

        # Create initial memory
        mem1_id = memory_tools.store(
            content="Version 1",
            memory_type="fact"
        )

        # Create snapshot A
        sleep(0.1)
        snapshot_a = snapshot_manager.create_snapshot(description="Snapshot A")
        assert snapshot_a.memory_count == 1
        assert snapshot_a.is_delta is False

        # Modify and create snapshot B
        sleep(0.1)
        palace.store_memory(
            content="Version 2",
            memory_type="fact",
            memory_id=mem1_id
        )
        snapshot_b = snapshot_manager.create_snapshot(description="Snapshot B")
        assert snapshot_b.memory_count == 1, "Should have 1 modified memory"
        assert snapshot_b.is_delta is True

        # Modify and create snapshot C
        sleep(0.1)
        palace.store_memory(
            content="Version 3",
            memory_type="fact",
            memory_id=mem1_id
        )
        snapshot_c = snapshot_manager.create_snapshot(description="Snapshot C")
        assert snapshot_c.memory_count == 1, "Should have 1 modified memory"
        assert snapshot_c.is_delta is True

        # Verify current state is V3
        current = palace.get_memory(mem1_id)
        assert current.content == "Version 3"

        # Rollback to snapshot A (skip B)
        snapshot_manager.rollback_to_snapshot(snapshot_a.snapshot_id)
        restored = palace.get_memory(mem1_id)
        assert restored.content == "Version 1", \
            f"Should rollback to Version 1, got '{restored.content}'"

        # Diff between A and C should show modification
        diff_a_to_c = snapshot_manager.diff_snapshots(
            snapshot_a.snapshot_id,
            snapshot_c.snapshot_id
        )
        assert len(diff_a_to_c.modified) == 1
        assert mem1_id in diff_a_to_c.modified

    def test_snapshot_empty_state_handling(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Test snapshot workflow handles empty states correctly.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.storage.snapshots import SnapshotManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)
        snapshot_manager = SnapshotManager(temp_omi_setup["db_path"])

        # Create snapshot with no memories
        snapshot_empty = snapshot_manager.create_snapshot(description="Empty state")
        assert snapshot_empty.memory_count == 0
        assert snapshot_empty.is_delta is False

        # Add a memory
        sleep(0.1)
        mem_id = memory_tools.store(
            content="First memory",
            memory_type="fact"
        )

        # Create second snapshot
        sleep(0.1)
        snapshot_with_memory = snapshot_manager.create_snapshot(description="With memory")
        assert snapshot_with_memory.memory_count == 1
        assert snapshot_with_memory.is_delta is True

        # Diff should show one added
        diff = snapshot_manager.diff_snapshots(
            snapshot_empty.snapshot_id,
            snapshot_with_memory.snapshot_id
        )
        assert len(diff.added) == 1
        assert mem_id in diff.added
        assert len(diff.modified) == 0
        assert len(diff.deleted) == 0

        # Rollback to empty state
        snapshot_manager.rollback_to_snapshot(snapshot_empty.snapshot_id)

        # Verify memory is gone
        restored = palace.get_memory(mem_id)
        assert restored is None, "Memory should be deleted after rollback to empty state"

        import sqlite3
        with sqlite3.connect(temp_omi_setup["db_path"]) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
            assert count == 0, "Should have 0 memories after rollback to empty state"

    def test_snapshot_list_and_retrieval(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
    ):
        """
        Test listing and retrieving snapshots.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.storage.snapshots import SnapshotManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)
        snapshot_manager = SnapshotManager(temp_omi_setup["db_path"])

        # Create some memories and snapshots
        memory_tools.store(content="Test", memory_type="fact")

        sleep(0.1)
        snap1 = snapshot_manager.create_snapshot(description="First")
        sleep(0.1)
        snap2 = snapshot_manager.create_snapshot(description="Second")
        sleep(0.1)
        snap3 = snapshot_manager.create_snapshot(description="Third")

        # List all snapshots
        all_snapshots = snapshot_manager.list_snapshots()
        assert len(all_snapshots) == 3

        # Should be ordered newest first
        assert all_snapshots[0].snapshot_id == snap3.snapshot_id
        assert all_snapshots[1].snapshot_id == snap2.snapshot_id
        assert all_snapshots[2].snapshot_id == snap1.snapshot_id

        # Test limit
        limited = snapshot_manager.list_snapshots(limit=2)
        assert len(limited) == 2
        assert limited[0].snapshot_id == snap3.snapshot_id

        # Test retrieval
        retrieved = snapshot_manager.get_snapshot(snap2.snapshot_id)
        assert retrieved is not None
        assert retrieved.snapshot_id == snap2.snapshot_id
        assert retrieved.description == "Second"

        # Test non-existent snapshot
        not_found = snapshot_manager.get_snapshot("snap-nonexistent")
        assert not_found is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
