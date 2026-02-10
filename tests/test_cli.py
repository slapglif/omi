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
            from omi.persistence import GraphPalace

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
            from omi.persistence import GraphPalace

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
        assert "0.1.0" in result.output

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
