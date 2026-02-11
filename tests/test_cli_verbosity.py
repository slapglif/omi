"""Tests for CLI verbosity flags and utilities.

Tests the --verbose/-v and --quiet/-q flags, and the verbosity utility functions.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

import pytest
from click.testing import CliRunner


class TestVerbosityUtilities:
    """Tests for verbosity utility functions in common.py."""

    def test_should_print_quiet_level(self):
        """Test should_print with quiet verbosity level."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import should_print, VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE

        # Quiet mode should only print messages at level 0
        assert should_print(VERBOSITY_QUIET, VERBOSITY_QUIET) is True
        assert should_print(VERBOSITY_QUIET, VERBOSITY_NORMAL) is False
        assert should_print(VERBOSITY_QUIET, VERBOSITY_VERBOSE) is False

    def test_should_print_normal_level(self):
        """Test should_print with normal verbosity level."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import should_print, VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE

        # Normal mode should print quiet and normal messages
        assert should_print(VERBOSITY_NORMAL, VERBOSITY_QUIET) is True
        assert should_print(VERBOSITY_NORMAL, VERBOSITY_NORMAL) is True
        assert should_print(VERBOSITY_NORMAL, VERBOSITY_VERBOSE) is False

    def test_should_print_verbose_level(self):
        """Test should_print with verbose verbosity level."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import should_print, VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE

        # Verbose mode should print all messages
        assert should_print(VERBOSITY_VERBOSE, VERBOSITY_QUIET) is True
        assert should_print(VERBOSITY_VERBOSE, VERBOSITY_NORMAL) is True
        assert should_print(VERBOSITY_VERBOSE, VERBOSITY_VERBOSE) is True

    def test_echo_verbose_in_verbose_mode(self):
        """Test echo_verbose prints in verbose mode."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_verbose, VERBOSITY_VERBOSE

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Capture output
            from io import StringIO
            import click

            output = StringIO()
            with patch('click.echo') as mock_echo:
                echo_verbose("Test verbose message", VERBOSITY_VERBOSE)
                mock_echo.assert_called_once_with("Test verbose message", err=False)

    def test_echo_verbose_in_normal_mode(self):
        """Test echo_verbose does not print in normal mode."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_verbose, VERBOSITY_NORMAL

        with patch('click.echo') as mock_echo:
            echo_verbose("Test verbose message", VERBOSITY_NORMAL)
            mock_echo.assert_not_called()

    def test_echo_verbose_in_quiet_mode(self):
        """Test echo_verbose does not print in quiet mode."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_verbose, VERBOSITY_QUIET

        with patch('click.echo') as mock_echo:
            echo_verbose("Test verbose message", VERBOSITY_QUIET)
            mock_echo.assert_not_called()

    def test_echo_normal_in_verbose_mode(self):
        """Test echo_normal prints in verbose mode."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_normal, VERBOSITY_VERBOSE

        with patch('click.echo') as mock_echo:
            echo_normal("Test normal message", VERBOSITY_VERBOSE)
            mock_echo.assert_called_once_with("Test normal message", err=False)

    def test_echo_normal_in_normal_mode(self):
        """Test echo_normal prints in normal mode."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_normal, VERBOSITY_NORMAL

        with patch('click.echo') as mock_echo:
            echo_normal("Test normal message", VERBOSITY_NORMAL)
            mock_echo.assert_called_once_with("Test normal message", err=False)

    def test_echo_normal_in_quiet_mode(self):
        """Test echo_normal does not print in quiet mode."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_normal, VERBOSITY_QUIET

        with patch('click.echo') as mock_echo:
            echo_normal("Test normal message", VERBOSITY_QUIET)
            mock_echo.assert_not_called()

    def test_echo_quiet_always_prints(self):
        """Test echo_quiet prints in all modes."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import echo_quiet, VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE

        # Test all verbosity levels
        for verbosity in [VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE]:
            with patch('click.echo') as mock_echo:
                echo_quiet("Critical message", verbosity)
                mock_echo.assert_called_once_with("Critical message", err=False)


class TestCLIVerbosityFlags:
    """Tests for CLI --verbose and --quiet flags."""

    def test_verbose_flag_sets_verbosity_level(self):
        """Test that --verbose flag sets verbosity to VERBOSITY_VERBOSE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Test with init command - verbose mode should show output
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['--verbose', 'init'])

            # Should succeed
            assert result.exit_code == 0

    def test_quiet_flag_sets_verbosity_level(self):
        """Test that --quiet flag sets verbosity to VERBOSITY_QUIET."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Test with init command - quiet mode should suppress output
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['--quiet', 'init'])

            # Should succeed
            assert result.exit_code == 0

    def test_no_flag_sets_normal_verbosity(self):
        """Test that no flag sets verbosity to VERBOSITY_NORMAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Test with init command - normal mode should show standard output
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['init'])

            # Should succeed
            assert result.exit_code == 0

    def test_verbose_and_quiet_are_mutually_exclusive(self):
        """Test that --verbose and --quiet cannot be used together."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        # Don't use --help as it might override validation; use status command instead
        result = runner.invoke(cli, ['--verbose', '--quiet', 'status'])

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_short_verbose_flag(self):
        """Test that -v short flag works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Test with init command using short flag
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['-v', 'init'])

            # Should succeed
            assert result.exit_code == 0

    def test_short_quiet_flag(self):
        """Test that -q short flag works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Test with init command using short flag
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['-q', 'init'])

            # Should succeed
            assert result.exit_code == 0

    def test_help_shows_verbose_flag(self):
        """Test that --help shows the --verbose/-v flag."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert '--verbose' in result.output
        assert '-v' in result.output
        assert 'verbose output' in result.output.lower()

    def test_help_shows_quiet_flag(self):
        """Test that --help shows the --quiet/-q flag."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert '--quiet' in result.output
        assert '-q' in result.output
        assert 'quiet' in result.output.lower() or 'suppress' in result.output.lower()


class TestVerbosityIntegration:
    """Tests for verbosity integration with actual commands."""

    def test_init_verbose_mode(self):
        """Test that init command respects verbose mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['--verbose', 'init'])

            # In verbose mode, should see additional output
            assert result.exit_code == 0
            # Check that some output was produced (verbose mode should show details)
            assert len(result.output) > 0

    def test_init_quiet_mode(self):
        """Test that init command respects quiet mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ['--quiet', 'init'])

            # In quiet mode, should have minimal output
            assert result.exit_code == 0

    def test_status_verbose_mode(self):
        """Test that status command respects verbose mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize first
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ['init'])

            # Mock GraphPalace.get_stats to avoid database errors
            with patch.object(GraphPalace, 'get_stats', return_value={
                'total_memories': 0,
                'memory_types': {},
                'total_edges': 0,
                'edge_types': {}
            }):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ['--verbose', 'status'])

            # Should succeed and show output
            assert result.exit_code == 0

    def test_verbosity_context_passed_to_commands(self):
        """Test that verbosity level is available in command context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ['init'])

            # Store command with verbose flag
            from omi import GraphPalace
            with patch.object(GraphPalace, 'store_memory', return_value="test-id-123"):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, [
                        '--verbose',
                        'store',
                        'Test memory',
                        '--type', 'fact'
                    ])

            assert result.exit_code == 0


class TestVerbosityConstants:
    """Tests for verbosity constant values."""

    def test_verbosity_level_values(self):
        """Test that verbosity levels have expected integer values."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli.common import VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE

        # Test values are in expected order
        assert VERBOSITY_QUIET < VERBOSITY_NORMAL < VERBOSITY_VERBOSE

        # Test specific expected values
        assert VERBOSITY_QUIET == 0
        assert VERBOSITY_NORMAL == 1
        assert VERBOSITY_VERBOSE == 2

    def test_verbosity_constants_importable(self):
        """Test that verbosity constants can be imported."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

        # Should not raise ImportError
        from omi.cli.common import VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE
        from omi.cli.common import should_print, echo_verbose, echo_normal, echo_quiet

        # Basic sanity check
        assert callable(should_print)
        assert callable(echo_verbose)
        assert callable(echo_normal)
        assert callable(echo_quiet)
