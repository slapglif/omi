"""End-to-End Event Flow Tests for OMI

Tests the complete event flow:
1. Operation (e.g., memory store) → EventBus
2. EventBus → EventHistory storage
3. Query events from EventHistory

Verifies that all event fields are correctly populated and persisted
through the entire pipeline.
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock


class TestE2EEventFlow:
    """Test end-to-end event flow: operation → EventBus → history."""

    def test_memory_store_event_flow(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        End-to-end verification:
        1. Store a memory via MemoryTools
        2. Verify MemoryStoredEvent was published to EventBus
        3. Verify event was recorded in EventHistory
        4. Query event via EventHistory.query_events()
        5. Verify all event fields are populated correctly
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.event_history import EventHistory
        from omi.events import MemoryStoredEvent

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Setup event history
        event_history_db = temp_omi_setup["base_path"] / "events.sqlite"
        event_history = EventHistory(event_history_db)

        # Track events published to EventBus
        captured_events = []

        def capture_and_store_event(event):
            """Callback to capture event and store in history."""
            captured_events.append(event)
            # Store event in history
            if hasattr(event, 'to_dict'):
                event_history.store_event(
                    event_type=event.event_type,
                    payload=event.to_dict(),
                    metadata={"source": "test"}
                )

        # Subscribe to EventBus to capture MemoryStoredEvent
        clean_event_bus.subscribe('memory.stored', capture_and_store_event)

        # Step 1: Store a memory via MemoryTools
        test_content = "Learned that Python list comprehensions are faster than loops"
        test_type = "experience"
        test_confidence = 0.85

        memory_id = memory_tools.store(
            content=test_content,
            memory_type=test_type,
            confidence=test_confidence
        )

        # Step 2: Verify MemoryStoredEvent was published to EventBus
        assert len(captured_events) == 1, "Should have captured 1 event"
        event = captured_events[0]

        assert isinstance(event, MemoryStoredEvent)
        assert event.memory_id == memory_id
        assert event.content == test_content
        assert event.memory_type == test_type
        assert event.confidence == test_confidence
        assert event.event_type == "memory.stored"
        assert isinstance(event.timestamp, datetime)

        # Step 3: Verify event was recorded in EventHistory
        stored_events = event_history.query_events(
            event_type="memory.stored",
            limit=10
        )

        assert len(stored_events) == 1, "Should have 1 event in history"
        stored_event = stored_events[0]

        # Step 4: Query event via EventHistory.query_events()
        # Already done above, now verify details

        # Step 5: Verify all event fields are populated correctly
        assert stored_event.event_type == "memory.stored"
        assert stored_event.payload["memory_id"] == memory_id
        assert stored_event.payload["content"] == test_content
        assert stored_event.payload["memory_type"] == test_type
        assert stored_event.payload["confidence"] == test_confidence
        assert stored_event.payload["event_type"] == "memory.stored"
        assert "timestamp" in stored_event.payload
        assert stored_event.metadata == {"source": "test"}
        assert isinstance(stored_event.timestamp, datetime)

    def test_multiple_memory_operations_event_flow(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Test multiple memory operations generate correct event sequence.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.event_history import EventHistory

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Setup event history
        event_history_db = temp_omi_setup["base_path"] / "events.sqlite"
        event_history = EventHistory(event_history_db)

        # Track events
        captured_events = []

        def capture_and_store_event(event):
            captured_events.append(event)
            if hasattr(event, 'to_dict'):
                event_history.store_event(
                    event_type=event.event_type,
                    payload=event.to_dict()
                )

        # Subscribe to all memory events
        clean_event_bus.subscribe('memory.stored', capture_and_store_event)
        clean_event_bus.subscribe('memory.recalled', capture_and_store_event)

        # Store multiple memories
        memory_ids = []
        for i in range(3):
            mid = memory_tools.store(
                content=f"Test memory {i}",
                memory_type="fact"
            )
            memory_ids.append(mid)

        # Recall memories
        results = memory_tools.recall("Test", limit=5)

        # Verify EventBus captured all events
        assert len(captured_events) == 4  # 3 stores + 1 recall

        # Verify event types
        stored_events = [e for e in captured_events if e.event_type == "memory.stored"]
        recalled_events = [e for e in captured_events if e.event_type == "memory.recalled"]

        assert len(stored_events) == 3
        assert len(recalled_events) == 1

        # Verify EventHistory has all events
        history_stored = event_history.query_events(
            event_type="memory.stored",
            limit=10
        )
        history_recalled = event_history.query_events(
            event_type="memory.recalled",
            limit=10
        )

        assert len(history_stored) == 3
        assert len(history_recalled) == 1

        # Verify stored memory IDs match
        stored_ids = [e.payload["memory_id"] for e in history_stored]
        for mid in memory_ids:
            assert mid in stored_ids

    def test_event_timestamp_consistency(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Verify timestamps are consistent across EventBus and EventHistory.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.event_history import EventHistory

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Setup event history
        event_history_db = temp_omi_setup["base_path"] / "events.sqlite"
        event_history = EventHistory(event_history_db)

        # Capture timestamp from EventBus
        bus_timestamps = []

        def capture_timestamp(event):
            bus_timestamps.append(event.timestamp)
            if hasattr(event, 'to_dict'):
                event_history.store_event(
                    event_type=event.event_type,
                    payload=event.to_dict()
                )

        clean_event_bus.subscribe('memory.stored', capture_timestamp)

        # Store memory
        before_time = datetime.now()
        memory_tools.store(
            content="Test timestamp consistency",
            memory_type="fact"
        )
        after_time = datetime.now()

        # Verify EventBus timestamp
        assert len(bus_timestamps) == 1
        bus_timestamp = bus_timestamps[0]
        assert before_time <= bus_timestamp <= after_time

        # Verify EventHistory timestamp
        history_events = event_history.query_events(
            event_type="memory.stored",
            limit=1
        )

        assert len(history_events) == 1
        history_event = history_events[0]

        # Timestamps should be close (within same second typically)
        # Note: EventHistory adds its own timestamp, but payload has original
        assert "timestamp" in history_event.payload
        payload_timestamp_str = history_event.payload["timestamp"]
        payload_timestamp = datetime.fromisoformat(payload_timestamp_str)

        # Payload timestamp should match bus timestamp
        # Allow small delta for serialization/deserialization
        time_delta = abs((payload_timestamp - bus_timestamp).total_seconds())
        assert time_delta < 1.0, "Timestamps should be very close"

    def test_event_query_filters(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Test EventHistory query filters work correctly.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.event_history import EventHistory
        from datetime import timedelta

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Setup event history
        event_history_db = temp_omi_setup["base_path"] / "events.sqlite"
        event_history = EventHistory(event_history_db)

        def store_event(event):
            if hasattr(event, 'to_dict'):
                event_history.store_event(
                    event_type=event.event_type,
                    payload=event.to_dict()
                )

        clean_event_bus.subscribe('memory.stored', store_event)
        clean_event_bus.subscribe('memory.recalled', store_event)

        # Record time before operations
        start_time = datetime.now()

        # Store memories
        memory_tools.store(content="Memory 1", memory_type="fact")
        memory_tools.store(content="Memory 2", memory_type="fact")

        # Recall
        memory_tools.recall("Memory")

        end_time = datetime.now()

        # Test filter by event_type
        stored_events = event_history.query_events(event_type="memory.stored")
        assert len(stored_events) == 2

        recalled_events = event_history.query_events(event_type="memory.recalled")
        assert len(recalled_events) == 1

        # Test filter by time range
        all_events = event_history.query_events(
            since=start_time - timedelta(seconds=1),
            until=end_time + timedelta(seconds=1)
        )
        assert len(all_events) == 3

        # Test limit
        limited_events = event_history.query_events(limit=2)
        assert len(limited_events) == 2

        # Test no filter (get all)
        all_events = event_history.query_events()
        assert len(all_events) == 3

    def test_event_metadata_preservation(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Verify event metadata is preserved through the entire flow.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.event_history import EventHistory

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Setup event history
        event_history_db = temp_omi_setup["base_path"] / "events.sqlite"
        event_history = EventHistory(event_history_db)

        # Capture event with metadata
        def store_with_metadata(event):
            if hasattr(event, 'to_dict'):
                event_history.store_event(
                    event_type=event.event_type,
                    payload=event.to_dict(),
                    metadata={
                        "test_metadata": "value",
                        "nested": {"key": "nested_value"},
                        "list": [1, 2, 3]
                    }
                )

        clean_event_bus.subscribe('memory.stored', store_with_metadata)

        # Store memory
        memory_tools.store(
            content="Test metadata preservation",
            memory_type="fact"
        )

        # Query from history
        events = event_history.query_events(event_type="memory.stored")

        assert len(events) == 1
        event = events[0]

        # Verify metadata preserved
        assert event.metadata is not None
        assert event.metadata["test_metadata"] == "value"
        assert event.metadata["nested"]["key"] == "nested_value"
        assert event.metadata["list"] == [1, 2, 3]

    def test_wildcard_subscription_captures_all_events(
        self,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache,
        clean_event_bus
    ):
        """
        Test wildcard subscription captures all event types.
        """
        from omi.api import MemoryTools
        from omi import GraphPalace
        from omi.event_history import EventHistory

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Setup event history
        event_history_db = temp_omi_setup["base_path"] / "events.sqlite"
        event_history = EventHistory(event_history_db)

        # Track all events with wildcard
        all_events = []

        def capture_all(event):
            all_events.append(event)
            if hasattr(event, 'to_dict'):
                event_history.store_event(
                    event_type=event.event_type,
                    payload=event.to_dict()
                )

        # Subscribe with wildcard
        clean_event_bus.subscribe('*', capture_all)

        # Perform operations
        memory_tools.store(content="Test wildcard", memory_type="fact")
        memory_tools.recall("Test")

        # Verify wildcard captured all events
        assert len(all_events) == 2

        event_types = [e.event_type for e in all_events]
        assert "memory.stored" in event_types
        assert "memory.recalled" in event_types

        # Verify all stored in history
        history_events = event_history.query_events()
        assert len(history_events) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
