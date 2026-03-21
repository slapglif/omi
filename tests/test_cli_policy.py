"""Tests for OMI CLI policy commands

Tests for 'omi policy' command group including show, dry-run, and execute commands.
Uses Click's test runner for command testing.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


class TestCLIPolicyShow:
    """Tests for 'omi policy show' command."""

    def test_policy_show_requires_init(self):
        """Test that policy show requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["policy", "show"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_policy_show_displays_default_policies(self):
        """Test that policy show displays default policies when no config exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Run policy show
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "show"])

            assert result.exit_code == 0
            assert "Policy Configuration" in result.output
            assert "defaults" in result.output.lower()
            # Should show at least one policy
            assert "ENABLED" in result.output or "DISABLED" in result.output

    def test_policy_show_loads_from_config(self):
        """Test that policy show loads policies from config.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create a config with policies
            config_path = base_path / "config.yaml"
            config_content = """
embedding:
  provider: ollama

policies:
  - name: test-policy
    enabled: true
    description: Test policy for testing
    type: retention
    action: archive
    conditions:
      max_age_days: 30
"""
            config_path.write_text(config_content)

            # Run policy show
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "show"])

            assert result.exit_code == 0
            assert "test-policy" in result.output
            assert "Test policy for testing" in result.output
            assert "ENABLED" in result.output

    def test_policy_show_handles_empty_config(self):
        """Test that policy show handles config.yaml with no policies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create a config without policies
            config_path = base_path / "config.yaml"
            config_content = """
embedding:
  provider: ollama
"""
            config_path.write_text(config_content)

            # Run policy show - should fall back to defaults
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "show"])

            assert result.exit_code == 0
            assert "defaults" in result.output.lower()

    def test_policy_show_handles_invalid_config(self):
        """Test that policy show handles invalid config.yaml gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create invalid YAML
            config_path = base_path / "config.yaml"
            config_path.write_text("invalid: yaml: content: [\n")

            # Run policy show - should error gracefully
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "show"])

            assert result.exit_code == 1
            assert "error" in result.output.lower()


class TestCLIPolicyDryRun:
    """Tests for 'omi policy dry-run' command."""

    def test_policy_dry_run_requires_init(self):
        """Test that policy dry-run requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["policy", "dry-run"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_policy_dry_run_requires_database(self):
        """Test that policy dry-run requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path but no database
            (base_path / "config.yaml").write_text("embedding:\n  provider: ollama\n")

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "dry-run"])

            assert result.exit_code == 1
            assert "database not found" in result.output.lower() or "not initialized" in result.output.lower()

    def test_policy_dry_run_executes_successfully(self):
        """Test that policy dry-run executes without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Run dry-run
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "dry-run"])

            assert result.exit_code == 0
            assert "Policy Dry Run" in result.output
            assert "dry-run mode" in result.output.lower() or "dry run" in result.output.lower()

    def test_policy_dry_run_with_policy_filter(self):
        """Test that policy dry-run can filter to specific policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create config with multiple policies
            config_path = base_path / "config.yaml"
            config_content = """
embedding:
  provider: ollama

policies:
  - name: policy-one
    enabled: true
    type: retention
    action: archive
    conditions:
      max_age_days: 30

  - name: policy-two
    enabled: true
    type: usage
    action: delete
    conditions:
      min_access_count: 1
      max_age_days: 90
"""
            config_path.write_text(config_content)

            # Run dry-run with policy filter
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "dry-run", "--policy", "policy-one"])

            assert result.exit_code == 0
            assert "policy-one" in result.output.lower()

    def test_policy_dry_run_with_nonexistent_policy(self):
        """Test that policy dry-run errors on nonexistent policy name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Run dry-run with nonexistent policy
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "dry-run", "--policy", "nonexistent"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_policy_dry_run_shows_affected_memories(self):
        """Test that policy dry-run shows count of affected memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.policies import PolicyEngine, PolicyExecutionResult, PolicyAction

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock PolicyEngine to return some results
            mock_result = PolicyExecutionResult(
                policy_name="test-policy",
                action=PolicyAction.ARCHIVE,
                affected_memory_ids=["mem-1", "mem-2", "mem-3"],
                dry_run=True,
                error=None
            )

            with patch.object(PolicyEngine, 'execute', return_value=[mock_result]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["policy", "dry-run"])

            assert result.exit_code == 0
            assert "3" in result.output  # Should show count
            assert "ARCHIVE" in result.output


class TestCLIPolicyExecute:
    """Tests for 'omi policy execute' command."""

    def test_policy_execute_requires_init(self):
        """Test that policy execute requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["policy", "execute", "--yes"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_policy_execute_requires_database(self):
        """Test that policy execute requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base path but no database
            (base_path / "config.yaml").write_text("embedding:\n  provider: ollama\n")

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "execute", "--yes"])

            assert result.exit_code == 1
            assert "database not found" in result.output.lower() or "not initialized" in result.output.lower()

    def test_policy_execute_with_yes_flag(self):
        """Test that policy execute works with --yes flag (skip confirmation)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.policies import PolicyEngine

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock PolicyEngine.execute to avoid actual database operations
            with patch.object(PolicyEngine, 'execute', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["policy", "execute", "--yes"])

            assert result.exit_code == 0
            assert "Policy Execution" in result.output
            # Should not prompt for confirmation with --yes
            assert "Proceed with policy execution?" not in result.output

    def test_policy_execute_requires_confirmation(self):
        """Test that policy execute requires confirmation without --yes flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Run without --yes, answer 'n' to confirmation
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "execute"], input="n\n")

            assert result.exit_code == 0
            assert "Proceed with policy execution?" in result.output
            assert "cancelled" in result.output.lower()

    def test_policy_execute_with_confirmation_yes(self):
        """Test that policy execute proceeds when user confirms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.policies import PolicyEngine

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock PolicyEngine to avoid actual execution
            with patch.object(PolicyEngine, 'execute', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["policy", "execute"], input="y\n")

            assert result.exit_code == 0
            assert "Execution complete" in result.output

    def test_policy_execute_with_policy_filter(self):
        """Test that policy execute can filter to specific policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.policies import PolicyEngine

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create config with specific policy
            config_path = base_path / "config.yaml"
            config_content = """
embedding:
  provider: ollama

policies:
  - name: my-policy
    enabled: true
    type: retention
    action: archive
    conditions:
      max_age_days: 30
"""
            config_path.write_text(config_content)

            # Mock PolicyEngine
            with patch.object(PolicyEngine, 'execute', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["policy", "execute", "--policy", "my-policy", "--yes"])

            assert result.exit_code == 0
            assert "my-policy" in result.output.lower()

    def test_policy_execute_with_nonexistent_policy(self):
        """Test that policy execute errors on nonexistent policy name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Run execute with nonexistent policy
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "execute", "--policy", "nonexistent", "--yes"])

            assert result.exit_code == 1
            assert "not found" in result.output.lower()

    def test_policy_execute_skips_disabled_policies(self):
        """Test that policy execute skips disabled policies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Create config with only disabled policies
            config_path = base_path / "config.yaml"
            config_content = """
embedding:
  provider: ollama

policies:
  - name: disabled-policy
    enabled: false
    type: retention
    action: archive
    conditions:
      max_age_days: 30
"""
            config_path.write_text(config_content)

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["policy", "execute", "--yes"])

            assert result.exit_code == 0
            assert "No enabled policies" in result.output

    def test_policy_execute_shows_results(self):
        """Test that policy execute shows execution results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.policies import PolicyEngine, PolicyExecutionResult, PolicyAction

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock PolicyEngine to return results
            mock_result = PolicyExecutionResult(
                policy_name="test-policy",
                action=PolicyAction.ARCHIVE,
                affected_memory_ids=["mem-1", "mem-2"],
                dry_run=False,
                error=None
            )

            with patch.object(PolicyEngine, 'execute', return_value=[mock_result]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["policy", "execute", "--yes"])

            assert result.exit_code == 0
            assert "2 memories" in result.output or "2" in result.output
            assert "ARCHIVE" in result.output


class TestCLIPolicyGroup:
    """Tests for policy command group."""

    def test_policy_group_help(self):
        """Test that policy command group has help text."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["policy", "--help"])

        assert result.exit_code == 0
        assert "Policy management commands" in result.output or "policy" in result.output.lower()
        assert "show" in result.output
        assert "dry-run" in result.output
        assert "execute" in result.output

    def test_policy_show_help(self):
        """Test that policy show command has help text."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["policy", "show", "--help"])

        assert result.exit_code == 0
        assert "show" in result.output.lower() or "display" in result.output.lower()

    def test_policy_dry_run_help(self):
        """Test that policy dry-run command has help text."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["policy", "dry-run", "--help"])

        assert result.exit_code == 0
        assert "dry-run" in result.output.lower() or "preview" in result.output.lower()

    def test_policy_execute_help(self):
        """Test that policy execute command has help text."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["policy", "execute", "--help"])

        assert result.exit_code == 0
        assert "execute" in result.output.lower()
        assert "--yes" in result.output or "-y" in result.output
