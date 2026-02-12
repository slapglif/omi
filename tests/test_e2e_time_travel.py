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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
