"""Tests for OMI CLI Event Commands

Tests for 'omi events' command group including:
- omi events list (with filters)
- omi events subscribe (live streaming)
"""

import os
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, call
from typing import Dict, Any

import pytest
from click.testing import CliRunner


class TestCLIEventsGroup:
    """Tests for 'omi events' command group."""

    def test_events_group_exists(self):
        """Test that events command group exists and shows help."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["events", "--help"])

        assert result.exit_code == 0
        assert "Event history commands" in result.output

    def test_events_group_has_list_command(self):
        """Test that events group includes list command."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["events", "--help"])

        assert result.exit_code == 0
        assert "list" in result.output

    def test_events_group_has_subscribe_command(self):
        """Test that events group includes subscribe command."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["events", "--help"])

        assert result.exit_code == 0
        assert "subscribe" in result.output


class TestCLIEventsList:
    """Tests for 'omi events list' command."""

    def test_list_requires_init(self):
        """Test that events list requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["events", "list"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_list_shows_help(self):
        """Test that events list shows help with --help."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["events", "list", "--help"])

        assert result.exit_code == 0
        assert "Filter by event type" in result.output
        assert "--since" in result.output
        assert "--until" in result.output
        assert "--limit" in result.output
        assert "--json-output" in result.output

    def test_list_handles_no_events(self):
        """Test that events list handles empty event history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # List events (no events.sqlite file yet)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list"])

            assert result.exit_code == 0
            assert "No events found" in result.output or "Event history is empty" in result.output

    def test_list_handles_no_events_json(self):
        """Test that events list handles empty history with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # List events with JSON output (no events.sqlite file yet)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--json-output"])

            assert result.exit_code == 0
            # Should output empty JSON array
            output = json.loads(result.output.strip())
            assert output == []

    def test_list_displays_events(self):
        """Test that events list displays stored events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory, EventRecord

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history and store an event
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)
            event_id = history.store_event(
                event_type="memory.stored",
                payload={"memory_id": "test-123", "content": "Test memory"},
                metadata={"source": "test"}
            )

            # List events
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list"])

            assert result.exit_code == 0
            assert "memory.stored" in result.output
            assert "Event History" in result.output

    def test_list_outputs_json(self):
        """Test that events list outputs JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history and store events
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)
            history.store_event(
                event_type="memory.stored",
                payload={"memory_id": "test-123", "content": "Test memory"}
            )

            # List events with JSON output
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--json-output"])

            assert result.exit_code == 0
            # Parse JSON output
            output = json.loads(result.output.strip())
            assert isinstance(output, list)
            assert len(output) == 1
            assert output[0]["event_type"] == "memory.stored"
            assert output[0]["payload"]["memory_id"] == "test-123"

    def test_list_filters_by_type(self):
        """Test that events list filters by event type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history and store multiple event types
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)
            history.store_event(
                event_type="memory.stored",
                payload={"memory_id": "test-1"}
            )
            history.store_event(
                event_type="session.started",
                payload={"session_id": "test-session"}
            )

            # List events filtered by type
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--type", "memory.stored", "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output.strip())
            assert len(output) == 1
            assert output[0]["event_type"] == "memory.stored"

    def test_list_filters_by_since(self):
        """Test that events list filters by since timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)

            # Store an event
            history.store_event(
                event_type="memory.stored",
                payload={"memory_id": "test-1"}
            )

            # Get a future timestamp
            future_time = (datetime.now() + timedelta(days=1)).isoformat()

            # List events with since filter (should be empty)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--since", future_time, "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output.strip())
            assert len(output) == 0

    def test_list_filters_by_until(self):
        """Test that events list filters by until timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)

            # Store an event
            history.store_event(
                event_type="memory.stored",
                payload={"memory_id": "test-1"}
            )

            # Get a past timestamp
            past_time = (datetime.now() - timedelta(days=1)).isoformat()

            # List events with until filter (should be empty)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--until", past_time, "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output.strip())
            assert len(output) == 0

    def test_list_respects_limit(self):
        """Test that events list respects limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history and store multiple events
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)
            for i in range(10):
                history.store_event(
                    event_type="memory.stored",
                    payload={"memory_id": f"test-{i}"}
                )

            # List events with limit
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--limit", "5", "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output.strip())
            assert len(output) == 5

    def test_list_handles_invalid_since_format(self):
        """Test that events list handles invalid since timestamp format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create events.sqlite (empty is fine)
            events_db = base_path / "events.sqlite"
            from omi.event_history import EventHistory
            EventHistory(events_db)

            # List events with invalid since format
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--since", "not-a-timestamp"])

            assert result.exit_code == 1
            assert "Invalid" in result.output
            assert "timestamp" in result.output.lower()

    def test_list_handles_invalid_until_format(self):
        """Test that events list handles invalid until timestamp format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create events.sqlite (empty is fine)
            events_db = base_path / "events.sqlite"
            from omi.event_history import EventHistory
            EventHistory(events_db)

            # List events with invalid until format
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--until", "not-a-timestamp"])

            assert result.exit_code == 1
            assert "Invalid" in result.output
            assert "timestamp" in result.output.lower()


class TestCLIEventsSubscribe:
    """Tests for 'omi events subscribe' command."""

    def test_subscribe_requires_init(self):
        """Test that events subscribe requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["events", "subscribe"], input="\x03")  # Ctrl+C immediately

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_subscribe_shows_help(self):
        """Test that events subscribe shows help with --help."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["events", "subscribe", "--help"])

        assert result.exit_code == 0
        assert "Subscribe to live event stream" in result.output
        assert "--type" in result.output


class TestCLIEventsIntegration:
    """End-to-end integration tests for CLI event querying."""

    def test_memory_operations_event_query_integration(self):
        """Test end-to-end: perform memory operations and query events.

        This integration test:
        1. Sets up event history listener to persist events from event bus
        2. Performs multiple memory operations via MemoryTools API (which emits events)
        3. Queries events with --since today filter
        4. Verifies all operations are listed
        5. Queries events with --type filter
        6. Verifies only matching events are shown
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory
            from omi.event_bus import get_event_bus, reset_event_bus
            from omi.storage.graph_palace import GraphPalace
            from omi.api import MemoryTools
            from omi.embeddings import EmbeddingCache

            # Step 1: Set up base path and create directory structure
            base_path.mkdir(parents=True, exist_ok=True)

            # Step 2: Set up event history listener
            # Reset event bus to ensure clean state
            reset_event_bus()
            event_bus = get_event_bus()
            events_db_path = base_path / "events.sqlite"
            event_history = EventHistory(events_db_path)

            # Subscribe to all events and persist them
            def persist_event(event):
                """Persist events from event bus to event history."""
                if hasattr(event, 'to_dict'):
                    event_dict = event.to_dict()
                    event_type = event_dict.pop('event_type', event.event_type)
                    timestamp = event_dict.pop('timestamp', None)
                    event_history.store_event(
                        event_type=event_type,
                        payload=event_dict,
                        metadata={}
                    )

            event_bus.subscribe('*', persist_event)

            # Step 3: Initialize GraphPalace (this will create the proper schema)
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)

            # Step 4: Create a mock embedder that returns dummy embeddings
            class DummyEmbedder:
                def embed(self, text):
                    return [0.0] * 1024  # Return zero vector

            embedder = DummyEmbedder()
            cache_dir = base_path / "embeddings_cache"
            cache = EmbeddingCache(cache_dir, embedder)
            memory_tools = MemoryTools(palace, embedder, cache)

            # Step 5: Perform multiple memory operations using MemoryTools API (which emits events)
            memory_operations = [
                ("First test memory", "fact"),
                ("Second test memory", "experience"),
                ("Third test memory", "belief"),
                ("Fourth test memory", "fact"),
            ]

            for content, mem_type in memory_operations:
                memory_tools.store(content=content, memory_type=mem_type)

            # Step 6: Query all events with --since today (should show all memory.stored events)
            # Use a timestamp from yesterday to capture all today's events
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--since", yesterday, "--json-output"])

            assert result.exit_code == 0
            all_events = json.loads(result.output.strip())

            # Should have at least 4 memory.stored events (from our operations)
            memory_stored_events = [e for e in all_events if e["event_type"] == "memory.stored"]
            assert len(memory_stored_events) >= 4, f"Expected at least 4 memory.stored events, got {len(memory_stored_events)}"

            # Verify each memory operation generated an event
            memory_contents = [op[0] for op in memory_operations]
            for content in memory_contents:
                matching_events = [
                    e for e in memory_stored_events
                    if e["payload"].get("content") == content
                ]
                assert len(matching_events) >= 1, f"No event found for memory: {content}"

            # Step 7: Query events filtered by --type memory.stored
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--type", "memory.stored", "--json-output"])

            assert result.exit_code == 0
            filtered_events = json.loads(result.output.strip())

            # Step 8: Verify only memory.stored events are shown
            assert len(filtered_events) >= 4, f"Expected at least 4 events, got {len(filtered_events)}"

            # All filtered events should be memory.stored type
            for event in filtered_events:
                assert event["event_type"] == "memory.stored", \
                    f"Expected only memory.stored events, found {event['event_type']}"

            # Verify payloads contain expected data
            for event in filtered_events:
                assert "content" in event["payload"], "Event payload missing 'content' field"
                assert "memory_type" in event["payload"], "Event payload missing 'memory_type' field"
                assert "memory_id" in event["payload"], "Event payload missing 'memory_id' field"

    def test_event_query_with_multiple_filters(self):
        """Test querying events with multiple filters combined."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.event_history import EventHistory
            import time

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create event history and store events at different times
            events_db = base_path / "events.sqlite"
            history = EventHistory(events_db)

            # Store first batch of events
            history.store_event(
                event_type="memory.stored",
                payload={"test_id": "test-0"},
                metadata={"source": "integration_test"}
            )
            history.store_event(
                event_type="session.started",
                payload={"test_id": "test-1"},
                metadata={"source": "integration_test"}
            )

            # Wait a bit to ensure timestamp difference
            time.sleep(0.1)

            # Record the "since" time
            since_time = datetime.now().isoformat()

            # Wait a bit more
            time.sleep(0.1)

            # Store second batch of events (after since_time)
            history.store_event(
                event_type="memory.stored",
                payload={"test_id": "test-2"},
                metadata={"source": "integration_test"}
            )
            history.store_event(
                event_type="session.ended",
                payload={"test_id": "test-3"},
                metadata={"source": "integration_test"}
            )

            # Query with type filter and since filter
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "events", "list",
                    "--type", "memory.stored",
                    "--since", since_time,
                    "--json-output"
                ])

            assert result.exit_code == 0
            events = json.loads(result.output.strip())

            # Should only get test-2 (memory.stored after since_time)
            assert len(events) >= 1, f"Expected at least 1 event, got {len(events)}"
            assert events[0]["event_type"] == "memory.stored"
            assert events[0]["payload"]["test_id"] == "test-2"

            # Verify the first batch is NOT included
            event_ids = [e["payload"]["test_id"] for e in events]
            assert "test-0" not in event_ids, "Event test-0 should not be included (before since_time)"
            assert "test-1" not in event_ids, "Event test-1 should not be included (wrong type)"

    def test_event_query_limit_parameter(self):
        """Test that event query respects limit parameter in integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Perform many memory operations
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                for i in range(20):
                    result = runner.invoke(cli, ["store", f"Memory {i}", "--type", "fact"])
                    assert result.exit_code == 0

            # Query with limit
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["events", "list", "--limit", "5", "--json-output"])

            assert result.exit_code == 0
            events = json.loads(result.output.strip())

            # Should return exactly 5 events (or less if fewer exist)
            assert len(events) <= 5

            # If we have enough events, should be exactly 5
            if len(events) == 5:
                # All should be memory.stored
                for event in events:
                    assert event["event_type"] == "memory.stored"
