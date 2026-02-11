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
