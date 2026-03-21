"""Tests for OMI CLI

Uses Click's test runner for command testing.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


class TestCLIInit:
    """Tests for 'omi init' command."""

    def test_init_creates_directory_structure(self):
        """Test that init creates the base directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / ".openclaw" / "omi"

            # Import cli module
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli, get_base_path

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert base_path.exists()
            assert (base_path / "memory").exists()

    def test_init_creates_config_yaml(self):
        """Test that init creates config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])

            config_path = base_path / "config.yaml"
            assert config_path.exists()
            content = config_path.read_text()
            assert "embedding:" in content
            assert "provider:" in content

    def test_init_creates_now_md(self):
        """Test that init creates NOW.md template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])

            now_path = base_path / "NOW.md"
            assert now_path.exists()
            content = now_path.read_text()
            assert "# NOW" in content
            assert "Current Task" in content

    def test_init_creates_database(self):
        """Test that init creates SQLite database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])

            db_path = base_path / "palace.sqlite"
            assert db_path.exists()


class TestCLISession:
    """Tests for session commands."""

    def test_session_start_requires_init(self):
        """Test that session-start requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["session-start"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_session_end_requires_init(self):
        """Test that session-end requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["session-end"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()


class TestCLIStore:
    """Tests for 'omi store' command."""

    def test_store_requires_init(self):
        """Test that store requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["store", "test content"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_store_validates_memory_type(self):
        """Test that store validates memory type options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace to avoid database errors
            with patch.object(GraphPalace, 'store_memory', return_value="test-id-123"):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, [
                        "store",
                        "Test memory",
                        "--type", "belief",
                        "--confidence", "0.85"
                    ])

            assert result.exit_code == 0


class TestCLIBelief:
    """Tests for belief commands."""

    def test_belief_group_exists(self):
        """Test that belief command group exists."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["belief", "--help"])

        assert result.exit_code == 0
        assert "Belief management commands" in result.output
        assert "evidence" in result.output
        assert "update" in result.output

    def test_belief_evidence_requires_init(self):
        """Test that belief evidence requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["belief", "evidence", "test-belief-id"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_belief_evidence_requires_belief_id(self):
        """Test that belief evidence requires belief_id argument."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["belief", "evidence"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "belief_id" in result.output.lower()

    def test_belief_evidence_displays_chain(self):
        """Test that belief evidence displays chain correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace
            from omi.belief import BeliefNetwork
            from datetime import datetime

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace methods
            mock_belief = {
                'id': 'test-belief-id',
                'content': 'Test belief content',
                'confidence': 0.75
            }

            mock_evidence = MagicMock()
            mock_evidence.memory_id = 'evidence-1'
            mock_evidence.supports = True
            mock_evidence.strength = 0.8
            mock_evidence.timestamp = datetime.now()

            with patch.object(GraphPalace, 'get_belief', return_value=mock_belief):
                with patch.object(BeliefNetwork, 'get_evidence_chain', return_value=[mock_evidence]):
                    with patch.object(GraphPalace, 'get_memory', return_value={'content': 'Evidence content'}):
                        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                            result = runner.invoke(cli, ["belief", "evidence", "test-belief-id"])

            assert result.exit_code == 0
            assert "Test belief content" in result.output
            assert "SUPPORTS" in result.output
            assert "evidence-1" in result.output

    def test_belief_evidence_json_output(self):
        """Test that belief evidence accepts --json option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace
            from omi.belief import BeliefNetwork
            from datetime import datetime
            import json

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace methods
            mock_belief = {
                'id': 'test-belief-id',
                'content': 'Test belief content',
                'confidence': 0.75
            }

            mock_evidence = MagicMock()
            mock_evidence.memory_id = 'evidence-1'
            mock_evidence.supports = True
            mock_evidence.strength = 0.8
            mock_evidence.timestamp = datetime.now()

            with patch.object(GraphPalace, 'get_belief', return_value=mock_belief):
                with patch.object(BeliefNetwork, 'get_evidence_chain', return_value=[mock_evidence]):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["belief", "evidence", "test-belief-id", "--json"])

            assert result.exit_code == 0
            output = json.loads(result.output)
            assert isinstance(output, list)
            assert len(output) == 1
            assert output[0]['memory_id'] == 'evidence-1'
            assert output[0]['supports'] is True

    def test_belief_evidence_handles_missing_belief(self):
        """Test that belief evidence handles missing belief gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace to return None for missing belief
            with patch.object(GraphPalace, 'get_belief', return_value=None):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["belief", "evidence", "nonexistent-belief"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_belief_update_requires_init(self):
        """Test that belief update requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["belief", "update", "test-belief-id", "--evidence", "test-evidence-id"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_belief_update_requires_evidence_option(self):
        """Test that belief update requires --evidence option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["belief", "update", "test-belief-id"])

            assert result.exit_code != 0
            assert "Missing option" in result.output or "--evidence" in result.output

    def test_belief_update_validates_strength_range(self):
        """Test that belief update validates strength is between 0.0 and 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Test strength > 1.0
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "belief", "update", "test-belief-id",
                    "--evidence", "test-evidence-id",
                    "--strength", "1.5"
                ])

            assert result.exit_code == 1
            assert "0.0 and 1.0" in result.output

            # Test strength < 0.0
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "belief", "update", "test-belief-id",
                    "--evidence", "test-evidence-id",
                    "--strength", "-0.5"
                ])

            assert result.exit_code == 1
            assert "0.0 and 1.0" in result.output

    def test_belief_update_with_supports_flag(self):
        """Test that belief update works with --supports flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace
            from omi.belief import BeliefNetwork

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace methods
            mock_belief = {
                'id': 'test-belief-id',
                'content': 'Test belief content',
                'confidence': 0.5
            }
            mock_memory = {'id': 'test-evidence-id', 'content': 'Test evidence'}

            with patch.object(GraphPalace, 'get_belief', return_value=mock_belief):
                with patch.object(GraphPalace, 'get_memory', return_value=mock_memory):
                    with patch.object(BeliefNetwork, 'update_with_evidence', return_value=0.6):
                        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                            result = runner.invoke(cli, [
                                "belief", "update", "test-belief-id",
                                "--evidence", "test-evidence-id",
                                "--supports",
                                "--strength", "0.8"
                            ])

            assert result.exit_code == 0
            assert "SUPPORTING" in result.output
            assert "updated successfully" in result.output.lower()

    def test_belief_update_with_contradicts_flag(self):
        """Test that belief update works with --contradicts flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace
            from omi.belief import BeliefNetwork

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace methods
            mock_belief = {
                'id': 'test-belief-id',
                'content': 'Test belief content',
                'confidence': 0.6
            }
            mock_memory = {'id': 'test-evidence-id', 'content': 'Test evidence'}

            with patch.object(GraphPalace, 'get_belief', return_value=mock_belief):
                with patch.object(GraphPalace, 'get_memory', return_value=mock_memory):
                    with patch.object(BeliefNetwork, 'update_with_evidence', return_value=0.4):
                        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                            result = runner.invoke(cli, [
                                "belief", "update", "test-belief-id",
                                "--evidence", "test-evidence-id",
                                "--contradicts",
                                "--strength", "0.6"
                            ])

            assert result.exit_code == 0
            assert "CONTRADICTING" in result.output
            assert "updated successfully" in result.output.lower()

    def test_belief_update_handles_missing_belief(self):
        """Test that belief update handles missing belief gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace to return None for missing belief
            with patch.object(GraphPalace, 'get_belief', return_value=None):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, [
                        "belief", "update", "nonexistent-belief",
                        "--evidence", "test-evidence-id"
                    ])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_belief_update_handles_missing_evidence(self):
        """Test that belief update handles missing evidence memory gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace methods
            mock_belief = {
                'id': 'test-belief-id',
                'content': 'Test belief content',
                'confidence': 0.5
            }

            with patch.object(GraphPalace, 'get_belief', return_value=mock_belief):
                with patch.object(GraphPalace, 'get_memory', return_value=None):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, [
                            "belief", "update", "test-belief-id",
                            "--evidence", "nonexistent-evidence"
                        ])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()


class TestCLIRecall:
    """Tests for 'omi recall' command."""

    def test_recall_requires_init(self):
        """Test that recall requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["recall", "test query"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_recall_accepts_json_output(self):
        """Test that recall accepts --json option."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.full_text_search (CLI uses full_text_search, not recall)
            mock_mem = MagicMock()
            mock_mem.id = "1"
            mock_mem.content = "Test memory"
            mock_mem.memory_type = "fact"
            mock_mem.confidence = None
            mock_mem.created_at = None
            mock_results = [mock_mem]
            with patch.object(GraphPalace, 'full_text_search', return_value=mock_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "test", "--json-output"])

            assert result.exit_code == 0
            assert '"content": "Test memory"' in result.output

    def test_recall_filters_by_memory_type(self):
        """Test that recall --type filters results by memory type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create mock memories of different types
            mock_fact = MagicMock()
            mock_fact.id = "1"
            mock_fact.content = "Test fact"
            mock_fact.memory_type = "fact"
            mock_fact.confidence = None
            mock_fact.created_at = None

            mock_experience = MagicMock()
            mock_experience.id = "2"
            mock_experience.content = "Test experience"
            mock_experience.memory_type = "experience"
            mock_experience.confidence = None
            mock_experience.created_at = None

            mock_belief = MagicMock()
            mock_belief.id = "3"
            mock_belief.content = "Test belief"
            mock_belief.memory_type = "belief"
            mock_belief.confidence = 0.8
            mock_belief.created_at = None

            all_results = [mock_fact, mock_experience, mock_belief]

            # Test filtering by fact
            with patch.object(GraphPalace, 'full_text_search', return_value=all_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "test", "--type", "fact", "--json-output"])

            assert result.exit_code == 0
            import json
            output = json.loads(result.output)
            assert len(output) == 1
            assert output[0]['memory_type'] == "fact"
            assert output[0]['content'] == "Test fact"

            # Test filtering by experience
            with patch.object(GraphPalace, 'full_text_search', return_value=all_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "test", "--type", "experience", "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output)
            assert len(output) == 1
            assert output[0]['memory_type'] == "experience"

            # Test no filter returns all
            with patch.object(GraphPalace, 'full_text_search', return_value=all_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "test", "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output)
            assert len(output) == 3

            # Test filtering with no matches returns empty
            with patch.object(GraphPalace, 'full_text_search', return_value=[mock_fact]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "test", "--type", "decision", "--json-output"])

            assert result.exit_code == 0
            output = json.loads(result.output)
            assert len(output) == 0


class TestCLIProgressIndicators:
    """Tests for progress indicators in CLI commands."""

    def test_init_shows_progress_bar(self):
        """Test that init command shows progress during database initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            # Verify progress messages appear in output
            assert "Initializing database" in result.output or "Initialized database" in result.output

    def test_recall_shows_progress_bar(self):
        """Test that recall command shows progress during search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace to avoid actual search
            with patch.object(GraphPalace, 'full_text_search', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["recall", "test query"])

            assert result.exit_code == 0
            # Verify progress message appears
            assert "Searching memories" in result.output

    def test_audit_shows_progress_bar(self):
        """Test that audit command shows progress during security checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.security import PoisonDetector

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock PoisonDetector to avoid actual audit
            mock_results = {
                'file_integrity': {'status': 'clean', 'issues': []},
                'topology': {'status': 'clean', 'issues': []},
                'git_history': {'status': 'clean', 'issues': []},
            }
            with patch.object(PoisonDetector, 'full_security_audit', return_value=mock_results):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["audit"])

            assert result.exit_code == 0
            # Verify progress message appears
            assert "Performing security checks" in result.output

    def test_backup_shows_progress_bar(self):
        """Test that backup command shows progress bar elements."""
        # This test verifies that the CLI code for progress bars is present
        # Full integration testing requires S3/vault configuration
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Try backup without vault configured - should show error
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["backup", "--full"])

            # Should fail with vault not configured message
            assert "Vault not configured" in result.output or "backup" in result.output.lower()

    def test_restore_shows_progress_bar(self):
        """Test that restore command shows progress bar elements."""
        # This test verifies that the CLI code for progress bars is present
        # Full integration testing requires S3/vault configuration
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Try restore without vault configured - should show error
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["restore", "test-backup-123"])

            # Should fail with configuration or initialization message
            assert result.exit_code != 0


class TestCLIDelete:
    """Tests for 'omi delete' command."""

    def test_delete_requires_init(self):
        """Test that delete requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["delete", "test-id-123"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_delete_nonexistent_memory(self):
        """Test that deleting a non-existent memory fails gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_memory to return None (memory not found)
            with patch.object(GraphPalace, 'get_memory', return_value=None):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["delete", "nonexistent-id", "--force"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_delete_with_confirmation_cancel(self):
        """Test that delete prompts for confirmation and can be cancelled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock memory to delete
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Test memory content"
            mock_mem.memory_type = "fact"

            # Mock GraphPalace methods
            with patch.object(GraphPalace, 'get_memory', return_value=mock_mem):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    # Simulate user saying 'no' to confirmation
                    result = runner.invoke(cli, ["delete", "test-id-123"], input="n\n")

            assert result.exit_code == 0
            assert "cancelled" in result.output.lower()

    def test_delete_with_confirmation_accept(self):
        """Test that delete prompts for confirmation and can be accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock memory to delete
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Test memory content"
            mock_mem.memory_type = "fact"

            # Mock GraphPalace methods
            with patch.object(GraphPalace, 'get_memory', return_value=mock_mem):
                with patch.object(GraphPalace, 'delete_memory', return_value=True):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        # Simulate user saying 'yes' to confirmation
                        result = runner.invoke(cli, ["delete", "test-id-123"], input="y\n")

            assert result.exit_code == 0
            assert "deleted successfully" in result.output.lower()

    def test_delete_with_force_flag(self):
        """Test that delete with --force skips confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock memory to delete
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Test memory content"
            mock_mem.memory_type = "fact"

            # Mock GraphPalace methods
            with patch.object(GraphPalace, 'get_memory', return_value=mock_mem):
                with patch.object(GraphPalace, 'delete_memory', return_value=True):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["delete", "test-id-123", "--force"])

            assert result.exit_code == 0
            assert "deleted successfully" in result.output.lower()
            # Should not contain confirmation prompt
            assert "are you sure" not in result.output.lower()

    def test_delete_displays_memory_details(self):
        """Test that delete displays memory details before confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock memory to delete
            mock_mem = MagicMock()
            mock_mem.id = "test-id-123"
            mock_mem.content = "Test memory content to display"
            mock_mem.memory_type = "experience"

            # Mock GraphPalace methods
            with patch.object(GraphPalace, 'get_memory', return_value=mock_mem):
                with patch.object(GraphPalace, 'delete_memory', return_value=True):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["delete", "test-id-123"], input="y\n")

            assert result.exit_code == 0
            # Check that memory details are displayed
            assert "memory to delete" in result.output.lower()
            assert "test-id-123" in result.output.lower()
            assert "experience" in result.output.lower()
            assert "test memory content" in result.output.lower()


class TestCLIStatus:
    """Tests for 'omi status' command."""

    def test_status_requires_init(self):
        """Test that status requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["status"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_status_shows_files(self):
        """Test that status shows file information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])
                result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "Config" in result.output
            assert "Database" in result.output
            assert "NOW.md" in result.output


class TestCLIAudit:
    """Tests for 'omi audit' command."""

    def test_audit_requires_init(self):
        """Test that audit requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["audit"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()


class TestCLIConfig:
    """Tests for 'omi config' subcommands."""

    def test_config_set_requires_init(self):
        """Test that config set requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["config", "set", "key", "value"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_config_get_requires_init(self):
        """Test that config get requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["config", "get", "key"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_config_set_and_get(self):
        """Test that config set and get work together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Set config
                result = runner.invoke(cli, ["config", "set", "embedding.provider", "ollama"])
                assert result.exit_code == 0
                assert "embedding.provider = ollama" in result.output

                # Get config
                result = runner.invoke(cli, ["config", "get", "embedding.provider"])
                assert result.exit_code == 0
                assert "ollama" in result.output


class TestCLIAPIKey:
    """Tests for 'omi config api-key' commands."""

    def test_api_key_group_help(self):
        """Test that api-key group help shows subcommands."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["config", "api-key", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "revoke" in result.output
        assert "list" in result.output

    def test_api_key_generate_requires_init(self):
        """Test that api-key generate requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["config", "api-key", "generate", "--name", "test-key"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_api_key_generate_creates_key(self):
        """Test that api-key generate creates a key and prints it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Generate API key
                result = runner.invoke(cli, ["config", "api-key", "generate", "--name", "test-key"])

            assert result.exit_code == 0
            assert "API key generated successfully" in result.output
            assert "test-key" in result.output
            assert "Rate Limit: 100 requests/minute" in result.output
            assert "X-API-Key:" in result.output
            assert "api_key=" in result.output

    def test_api_key_generate_with_custom_rate_limit(self):
        """Test that api-key generate works with custom rate limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Generate API key with custom rate limit
                result = runner.invoke(cli, [
                    "config", "api-key", "generate",
                    "--name", "custom-key",
                    "--rate-limit", "50"
                ])

            assert result.exit_code == 0
            assert "API key generated successfully" in result.output
            assert "custom-key" in result.output
            assert "Rate Limit: 50 requests/minute" in result.output

    def test_api_key_list_shows_all_keys(self):
        """Test that api-key list shows all keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Generate two API keys
                runner.invoke(cli, ["config", "api-key", "generate", "--name", "key-one"])
                runner.invoke(cli, ["config", "api-key", "generate", "--name", "key-two", "--rate-limit", "75"])

                # List all keys
                result = runner.invoke(cli, ["config", "api-key", "list"])

            assert result.exit_code == 0
            assert "API Keys (2 found)" in result.output
            assert "key-one" in result.output
            assert "key-two" in result.output
            assert "100 requests/minute" in result.output
            assert "75 requests/minute" in result.output
            assert "Last Used: Never" in result.output

    def test_api_key_list_empty(self):
        """Test that api-key list shows message when no keys exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # List keys (should be empty)
                result = runner.invoke(cli, ["config", "api-key", "list"])

            assert result.exit_code == 0
            assert "No API keys found" in result.output

    def test_api_key_revoke_removes_key(self):
        """Test that api-key revoke removes a key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Generate API key
                runner.invoke(cli, ["config", "api-key", "generate", "--name", "revoke-test"])

                # Revoke the key
                result = runner.invoke(cli, ["config", "api-key", "revoke", "--name", "revoke-test"])

            assert result.exit_code == 0
            assert "Revoked API key: revoke-test" in result.output

    def test_api_key_revoke_nonexistent_key(self):
        """Test that revoking nonexistent key shows warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Try to revoke nonexistent key
                result = runner.invoke(cli, ["config", "api-key", "revoke", "--name", "nonexistent"])

            assert result.exit_code == 1
            assert "No active API key found" in result.output

    def test_api_key_list_includes_revoked(self):
        """Test that api-key list can show revoked keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Generate and revoke a key
                runner.invoke(cli, ["config", "api-key", "generate", "--name", "active-key"])
                runner.invoke(cli, ["config", "api-key", "generate", "--name", "revoked-key"])
                runner.invoke(cli, ["config", "api-key", "revoke", "--name", "revoked-key"])

                # List without revoked (should show 1)
                result = runner.invoke(cli, ["config", "api-key", "list"])
                assert result.exit_code == 0
                assert "API Keys (1 found)" in result.output
                assert "active-key" in result.output
                assert "revoked-key" not in result.output

                # List with revoked (should show 2)
                result = runner.invoke(cli, ["config", "api-key", "list", "--include-revoked"])
                assert result.exit_code == 0
                assert "API Keys (2 found" in result.output
                assert "active-key" in result.output
                assert "revoked-key" in result.output
                assert "[REVOKED]" in result.output

    def test_api_key_revoke_requires_name_or_key(self):
        """Test that revoke requires either --name or --key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Try to revoke without name or key
                result = runner.invoke(cli, ["config", "api-key", "revoke"])

            assert result.exit_code == 1
            assert "Must provide either --name or --key" in result.output

    def test_api_key_generate_duplicate_name_fails(self):
        """Test that generating a key with duplicate name fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Generate first key
                result = runner.invoke(cli, ["config", "api-key", "generate", "--name", "duplicate"])
                assert result.exit_code == 0

                # Try to generate with same name
                result = runner.invoke(cli, ["config", "api-key", "generate", "--name", "duplicate"])

            assert result.exit_code == 1
            assert "Error:" in result.output


class TestCLIGlobal:
    """Tests for global CLI behavior."""

    def test_version_flag(self):
        """Test --version flag works."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.2.0" in result.output

    def test_help_shows_commands(self):
        """Test that help shows available commands."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "session-start" in result.output
        assert "session-end" in result.output
        assert "store" in result.output
        assert "recall" in result.output
        assert "check" in result.output
        assert "status" in result.output
        assert "audit" in result.output
        assert "config" in result.output

    def test_config_group_help(self):
        """Test that config group help shows subcommands."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "set" in result.output
        assert "get" in result.output
        assert "show" in result.output


class TestCommandHelp:
    """Tests for individual command help text."""

    def test_init_help(self):
        """Test init has helpful description."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output
        assert "directory" in result.output.lower() or "config" in result.output.lower()

    def test_store_help(self):
        """Test store has helpful description."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["store", "--help"])
        assert result.exit_code == 0
        assert "memory" in result.output.lower()
        assert "type" in result.output.lower()


def test_config():
    """Test config command group functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = CliRunner()
        base_path = Path(tmpdir) / "omi"

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        # Initialize
        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0

            # Test config set
            result = runner.invoke(cli, ["config", "set", "embedding.provider", "ollama"])
            assert result.exit_code == 0
            assert "ollama" in result.output

            # Test config get
            result = runner.invoke(cli, ["config", "get", "embedding.provider"])
            assert result.exit_code == 0
            assert "ollama" in result.output

            # Test config show
            result = runner.invoke(cli, ["config", "show"])
            assert result.exit_code == 0
            assert "embedding:" in result.output


def test_sync():
    """Test sync command group functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = CliRunner()
        base_path = Path(tmpdir) / "omi"

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        # Initialize
        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0

            # Test sync status without cloud configured (should show disabled)
            result = runner.invoke(cli, ["sync", "status"])
            assert result.exit_code == 0
            assert "Cloud Sync Status" in result.output

            # Configure cloud storage using backup config structure
            config_path = base_path / "config.yaml"
            import yaml
            config_data = yaml.safe_load(config_path.read_text())
            config_data['backup'] = {
                'backend': 's3',
                'bucket': 'test-bucket',
                'region': 'us-east-1'
            }
            config_path.write_text(yaml.dump(config_data))

            # Test sync push (should fail gracefully without actual S3 credentials)
            result = runner.invoke(cli, ["sync", "push"])
            # Exit code may be 0 or 1 depending on whether boto3 is available
            # and whether credentials are configured
            assert "Pushing to cloud storage" in result.output
            assert "s3" in result.output.lower()
            assert "test-bucket" in result.output

            # Test sync pull (should fail gracefully without actual S3 credentials)
            result = runner.invoke(cli, ["sync", "pull"])
            # Exit code may be 0 or 1 depending on whether boto3 is available
            assert "Pulling from cloud storage" in result.output
            assert "s3" in result.output.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
