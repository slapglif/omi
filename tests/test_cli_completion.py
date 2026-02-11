"""Tests for OMI CLI completion command

Tests shell completion script generation for bash and zsh.
"""

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner


class TestCLICompletion:
    """Tests for 'omi completion' command."""

    def test_completion_command_exists(self):
        """Test that completion command is registered."""
        # Import cli module
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "completion" in result.output

    def test_completion_requires_shell_argument(self):
        """Test that completion command requires shell argument."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "required" in result.output.lower()

    def test_completion_bash_generates_script(self):
        """Test that completion bash generates a valid bash script."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"])

        assert result.exit_code == 0
        assert len(result.output) > 0
        # Check for bash-specific markers
        assert "bash completion" in result.output
        assert "complete" in result.output
        assert "COMPREPLY" in result.output

    def test_completion_bash_contains_omi_references(self):
        """Test that bash completion script references omi command."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"])

        assert result.exit_code == 0
        # Should contain references to the omi program
        assert "omi" in result.output.lower()
        assert "_OMI_COMPLETE" in result.output or "omi" in result.output

    def test_completion_zsh_generates_script(self):
        """Test that completion zsh generates a valid zsh script."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "zsh"])

        assert result.exit_code == 0
        assert len(result.output) > 0
        # Check for zsh-specific markers
        assert "#compdef" in result.output
        assert "compdef" in result.output

    def test_completion_zsh_contains_completion_function(self):
        """Test that zsh completion script contains completion function."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "zsh"])

        assert result.exit_code == 0
        # Should define a completion function
        assert "_omi_completion" in result.output or "completion" in result.output

    def test_completion_invalid_shell_rejected(self):
        """Test that completion rejects invalid shell types."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "fish"])

        assert result.exit_code != 0
        # Click should reject invalid choice
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    def test_completion_bash_script_structure(self):
        """Test bash completion script has expected structure."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"])

        assert result.exit_code == 0
        output = result.output

        # Check for essential bash completion components
        assert "COMP_WORDS" in output
        assert "COMP_CWORD" in output
        assert "_OMI_COMPLETE=bash_complete" in output or "bash_complete" in output
        assert "complete -F" in output or "complete" in output

    def test_completion_zsh_script_structure(self):
        """Test zsh completion script has expected structure."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "zsh"])

        assert result.exit_code == 0
        output = result.output

        # Check for essential zsh completion components
        assert "#compdef omi" in output or "#compdef" in output
        assert "COMP_WORDS" in output or "words" in output
        assert "_OMI_COMPLETE=zsh_complete" in output or "zsh_complete" in output
        assert "compdef" in output

    def test_completion_bash_output_is_single_script(self):
        """Test bash completion outputs a single cohesive script."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"])

        assert result.exit_code == 0
        # Should be a non-empty script
        assert len(result.output.strip()) > 100
        # Should not have error messages
        assert "error" not in result.output.lower()
        assert "traceback" not in result.output.lower()

    def test_completion_zsh_output_is_single_script(self):
        """Test zsh completion outputs a single cohesive script."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "zsh"])

        assert result.exit_code == 0
        # Should be a non-empty script
        assert len(result.output.strip()) > 100
        # Should not have error messages
        assert "error" not in result.output.lower()
        assert "traceback" not in result.output.lower()

    def test_completion_help_text(self):
        """Test completion command has helpful documentation."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "--help"])

        assert result.exit_code == 0
        # Should explain what the command does
        assert "completion" in result.output.lower()
        assert "shell" in result.output.lower()
        # Should mention bash and zsh
        assert "bash" in result.output.lower()
        assert "zsh" in result.output.lower()
