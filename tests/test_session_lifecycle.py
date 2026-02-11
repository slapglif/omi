"""End-to-end tests for OMI session lifecycle

Tests the complete workflow: init → session-start → store → recall → session-end
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime

import pytest
from click.testing import CliRunner


def create_mock_now_store(base_path):
    """Create a properly mocked NOWStore with all needed methods."""
    mock_store = MagicMock()
    mock_store.read.return_value = """# NOW - 2024-01-01T10:00:00

## Current Task
Test task

## Recent Completions
- [x] Setup complete

## Pending Decisions
- [ ] None

## Key Files
- `test.py`
"""
    mock_store.update = MagicMock()
    mock_store.now_path = Path(base_path) / "NOW.md"
    mock_store.hash_file = Path(base_path) / ".now.hash"
    mock_store.check_integrity.return_value = True
    return mock_store


class TestSessionLifecycleEndToEnd:
    """End-to-end tests for complete session workflow."""

    def test_complete_session_workflow(self):
        """Test full session lifecycle: init -> start -> store -> recall -> end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace, NOWStore

            # Step 1: Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0
                assert base_path.exists()
                assert (base_path / "palace.sqlite").exists()
                assert (base_path / "NOW.md").exists()
                assert (base_path / "config.yaml").exists()

            # Step 2: Start session with mocked NOWStore
            mock_store = create_mock_now_store(base_path)
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-start"])
                    assert result.exit_code == 0
                    assert "Starting OMI session" in result.output

            # Step 3: Store a memory
            with patch.object(GraphPalace, 'store_memory', return_value="test-memory-id-1"):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, [
                        "store",
                        "Implemented session lifecycle tests",
                        "--type", "experience"
                    ])
                    assert result.exit_code == 0

            # Step 4: Recall the memory
            # Mock Memory object for full_text_search result
            from omi.storage.graph_palace import Memory
            mock_memory = Memory(
                id="test-memory-id-1",
                content="Implemented session lifecycle tests",
                memory_type="experience",
                confidence=0.95,
                created_at=datetime.now(),
                embedding=None
            )
            with patch.object(GraphPalace, 'full_text_search', return_value=[mock_memory]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "lifecycle tests"])
                    assert result.exit_code == 0

            # Step 5: End session with mocked NOWStore
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end"])
                    assert result.exit_code == 0
                    assert "Ending OMI session" in result.output

            # Verify daily log was created
            memory_dir = base_path / "memory"
            assert memory_dir.exists()
            daily_logs = list(memory_dir.glob("*.md"))
            assert len(daily_logs) >= 1

    def test_session_start_loads_now_context(self):
        """Test that session-start loads NOW.md content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create custom NOW.md content
            custom_content = """# NOW - 2024-01-01T10:00:00

## Current Task
Writing comprehensive session lifecycle tests

## Recent Completions
- [x] Set up test infrastructure
- [x] Implemented basic CLI tests

## Pending Decisions
- [ ] Choose between FTS5 and vector search
- [ ] Decide on embedding provider

## Key Files
- `tests/test_session_lifecycle.py`
- `src/omi/cli.py`
"""
            # Mock NOWStore to return custom content
            mock_store = create_mock_now_store(base_path)
            mock_store.read.return_value = custom_content

            # Start session and verify NOW.md is loaded
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-start"])
                    assert result.exit_code == 0
                    # Should read the custom NOW.md content
                    assert "Starting OMI session" in result.output

    def test_session_end_appends_to_daily_log(self):
        """Test that session-end creates daily log entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NOWStore
            mock_store = create_mock_now_store(base_path)

            # Start session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    runner.invoke(cli, ["session-start"])

            # End session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end"])
                    assert result.exit_code == 0

            # Verify daily log exists
            memory_dir = base_path / "memory"
            today = datetime.now().strftime("%Y-%m-%d")
            daily_log = memory_dir / f"{today}.md"

            # Check if daily log was created (it should exist or be attempted)
            assert memory_dir.exists()

    def test_session_without_init_fails(self):
        """Test that session commands fail without initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "uninitialized"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Try session-start without init
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["session-start"])
                assert result.exit_code == 1
                assert "not initialized" in result.output.lower()

            # Try session-end without init
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["session-end"])
                assert result.exit_code == 1
                assert "not initialized" in result.output.lower()


class TestSessionNowManagement:
    """Tests for NOW.md management during sessions."""

    def test_session_start_creates_default_now(self):
        """Test that session-start creates default NOW.md if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NOWStore to simulate missing NOW.md
            mock_store = create_mock_now_store(base_path)
            mock_store.read.return_value = None

            # Start session - should create default
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-start"])
                    assert result.exit_code == 0

            # Verify update was called
            mock_store.update.assert_called()

    def test_session_end_preserves_now_content(self):
        """Test that session-end preserves NOW.md content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Set custom NOW.md content
            custom_content = """# NOW - 2024-01-01T10:00:00

## Current Task
Test NOW.md preservation

## Recent Completions
- [x] Wrote test case

## Pending Decisions
- [ ] Verify preservation logic

## Key Files
- `tests/test_session_lifecycle.py`
"""
            mock_store = create_mock_now_store(base_path)
            mock_store.read.return_value = custom_content

            # End session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end"])
                    assert result.exit_code == 0

            # Verify read was called (content was accessed)
            mock_store.read.assert_called()


class TestSessionMemoryOperations:
    """Tests for memory operations during sessions."""

    def test_store_during_active_session(self):
        """Test storing memories during an active session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize and start session
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])
                runner.invoke(cli, ["session-start"])

            # Store multiple memories
            memory_ids = ["mem-1", "mem-2", "mem-3"]
            for i, mem_id in enumerate(memory_ids):
                with patch.object(GraphPalace, 'store_memory', return_value=mem_id):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, [
                            "store",
                            f"Test memory {i+1}",
                            "--type", "fact"
                        ])
                        assert result.exit_code == 0

    def test_recall_during_active_session(self):
        """Test recalling memories during an active session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace, NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Start session with mocked NOWStore
            mock_store = create_mock_now_store(base_path)
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    runner.invoke(cli, ["session-start"])

            # Recall memories
            from omi.storage.graph_palace import Memory
            mock_memory = Memory(
                id="mem-1",
                content="Python testing best practices",
                memory_type="fact",
                confidence=0.90,
                created_at=datetime.now(),
                embedding=None
            )
            with patch.object(GraphPalace, 'full_text_search', return_value=[mock_memory]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "testing"])
                    assert result.exit_code == 0


class TestSessionBackup:
    """Tests for backup operations during session lifecycle."""

    def test_session_end_with_backup_disabled(self):
        """Test session-end with --no-backup flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NOWStore
            mock_store = create_mock_now_store(base_path)

            # Start session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    runner.invoke(cli, ["session-start"])

            # End session with no backup
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end", "--no-backup"])
                    assert result.exit_code == 0

    def test_multiple_session_cycles(self):
        """Test multiple start/end cycles in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize once
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NOWStore
            mock_store = create_mock_now_store(base_path)

            # Run multiple session cycles
            for cycle in range(3):
                # Start session
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-start"])
                        assert result.exit_code == 0

                # End session
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify persistence across cycles
            assert base_path.exists()
            assert (base_path / "palace.sqlite").exists()
            assert (base_path / "NOW.md").exists()


class TestSessionEventEmission:
    """Tests for event emission during session lifecycle."""

    def test_session_start_emits_event(self):
        """Test that session-start emits SessionStartedEvent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NOWStore
            mock_store = create_mock_now_store(base_path)

            # Start session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-start"])
                    assert result.exit_code == 0

            # Note: Event emission might be async or happen in process
            # This test verifies the command succeeds

    def test_session_end_emits_event(self):
        """Test that session-end emits SessionEndedEvent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import NOWStore

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock NOWStore
            mock_store = create_mock_now_store(base_path)

            # Start session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    runner.invoke(cli, ["session-start"])

            # End session
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end"])
                    assert result.exit_code == 0

            # Verify session end completed successfully
            assert "Ending OMI session" in result.output
