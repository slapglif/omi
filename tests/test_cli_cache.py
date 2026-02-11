"""Tests for OMI cache CLI commands.

Uses Click's test runner for command testing.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner


class TestCLICacheStats:
    """Tests for 'omi cache stats' command."""

    def test_cache_stats_requires_init(self):
        """Test that cache stats requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["cache", "stats"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_cache_stats_empty_cache_no_directory(self):
        """Test cache stats with no cache directory (0 entries)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Delete embeddings directory if it exists
                embeddings_dir = base_path / "embeddings"
                if embeddings_dir.exists():
                    import shutil
                    shutil.rmtree(embeddings_dir)

                result = runner.invoke(cli, ["cache", "stats"])

            assert result.exit_code == 0
            assert "0" in result.output
            assert "0 KB" in result.output
            assert "Cache directory does not exist" in result.output

    def test_cache_stats_empty_cache_with_directory(self):
        """Test cache stats with empty cache directory (0 entries)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create empty embeddings directory
                embeddings_dir = base_path / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                result = runner.invoke(cli, ["cache", "stats"])

            assert result.exit_code == 0
            assert "Entries: 0" in result.output or "0" in result.output
            assert "0 KB" in result.output
            assert "Cache is empty" in result.output

    def test_cache_stats_with_files(self):
        """Test cache stats with populated cache (show count and size)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create embeddings directory and sample .npy files
                embeddings_dir = base_path / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                # Create sample embedding files
                for i in range(3):
                    embedding = np.array([1.0, 2.0, 3.0, 4.0])
                    np.save(embeddings_dir / f"test_embedding_{i}.npy", embedding)

                result = runner.invoke(cli, ["cache", "stats"])

            assert result.exit_code == 0
            assert "3" in result.output  # 3 entries
            assert "KB" in result.output or "MB" in result.output  # Size should be shown
            assert "Cache is empty" not in result.output


class TestCLICacheClear:
    """Tests for 'omi cache clear' command."""

    def test_cache_clear_requires_init(self):
        """Test that cache clear requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["cache", "clear", "--force"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_cache_clear_with_force(self):
        """Test cache clear with --force flag (no confirmation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create embeddings directory and sample .npy files
                embeddings_dir = base_path / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                # Create sample embedding files
                for i in range(3):
                    embedding = np.array([1.0, 2.0, 3.0, 4.0])
                    np.save(embeddings_dir / f"test_embedding_{i}.npy", embedding)

                # Verify files exist
                assert len(list(embeddings_dir.glob('*.npy'))) == 3

                result = runner.invoke(cli, ["cache", "clear", "--force"])

            assert result.exit_code == 0
            assert "Cleared 3 cache entries" in result.output
            # Verify files were deleted
            assert len(list(embeddings_dir.glob('*.npy'))) == 0

    def test_cache_clear_with_confirmation(self):
        """Test cache clear without force (with confirmation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create embeddings directory and sample .npy files
                embeddings_dir = base_path / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                # Create sample embedding files
                for i in range(2):
                    embedding = np.array([1.0, 2.0, 3.0, 4.0])
                    np.save(embeddings_dir / f"test_embedding_{i}.npy", embedding)

                # Test with user confirming yes
                result = runner.invoke(cli, ["cache", "clear"], input="y\n")

            assert result.exit_code == 0
            assert "Delete 2 cache entries?" in result.output
            assert "Cleared 2 cache entries" in result.output
            assert len(list(embeddings_dir.glob('*.npy'))) == 0

    def test_cache_clear_cancelled(self):
        """Test cache clear when user cancels confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create embeddings directory and sample .npy files
                embeddings_dir = base_path / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                # Create sample embedding files
                for i in range(2):
                    embedding = np.array([1.0, 2.0, 3.0, 4.0])
                    np.save(embeddings_dir / f"test_embedding_{i}.npy", embedding)

                # Test with user cancelling
                result = runner.invoke(cli, ["cache", "clear"], input="n\n")

            assert result.exit_code == 0
            assert "Cancelled" in result.output
            # Verify files were NOT deleted
            assert len(list(embeddings_dir.glob('*.npy'))) == 2

    def test_cache_clear_no_directory(self):
        """Test cache clear when cache directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Delete embeddings directory if it exists
                embeddings_dir = base_path / "embeddings"
                if embeddings_dir.exists():
                    import shutil
                    shutil.rmtree(embeddings_dir)

                result = runner.invoke(cli, ["cache", "clear", "--force"])

            assert result.exit_code == 0
            assert "Cache directory does not exist" in result.output

    def test_cache_clear_already_empty(self):
        """Test cache clear when cache is already empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create empty embeddings directory
                embeddings_dir = base_path / "embeddings"
                embeddings_dir.mkdir(parents=True, exist_ok=True)

                result = runner.invoke(cli, ["cache", "clear", "--force"])

            assert result.exit_code == 0
            assert "Cache is already empty" in result.output


class TestCLICacheHelp:
    """Tests for cache command help text."""

    def test_cache_group_help(self):
        """Test that cache group help shows subcommands."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["cache", "--help"])
        assert result.exit_code == 0
        assert "stats" in result.output
        assert "clear" in result.output

    def test_cache_stats_help(self):
        """Test cache stats has helpful description."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["cache", "stats", "--help"])
        assert result.exit_code == 0
        assert "cache" in result.output.lower() or "statistics" in result.output.lower()

    def test_cache_clear_help(self):
        """Test cache clear has helpful description."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["cache", "clear", "--help"])
        assert result.exit_code == 0
        assert "clear" in result.output.lower() or "delete" in result.output.lower()
        assert "--force" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
