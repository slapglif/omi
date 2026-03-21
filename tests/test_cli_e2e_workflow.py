"""End-to-end CLI workflow tests for user management and RBAC.

Tests the complete user management workflow from initialization through
user creation, role assignment, API key generation, and permission verification.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


class TestCLIE2EWorkflow:
    """End-to-end tests for CLI user management workflow."""

    def test_complete_user_workflow(self):
        """Test the complete user management workflow end-to-end.

        Workflow:
        1. omi init - Initialize fresh OMI instance
        2. omi user create --role admin alice - Create admin user
        3. omi user create-api-key alice - Generate API key
        4. omi user create --role reader bob - Create reader user
        5. omi user list - Verify both users exist
        6. omi user permissions alice - Verify admin permissions
        7. omi user permissions bob - Verify reader permissions
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            # Import CLI
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Step 1: Initialize OMI instance
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0, f"omi init failed: {result.output}"
                assert base_path.exists(), "Base path not created"
                assert (base_path / "palace.sqlite").exists(), "Database not created"

                # Step 2: Create admin user 'alice'
                result = runner.invoke(cli, ["user", "create", "--role", "admin", "alice"])
                assert result.exit_code == 0, f"Failed to create admin user: {result.output}"
                assert "User created" in result.output or "✓" in result.output
                assert "alice" in result.output

                # Step 3: Generate API key for alice
                result = runner.invoke(cli, ["user", "create-api-key", "alice"])
                assert result.exit_code == 0, f"Failed to create API key: {result.output}"
                assert "API key created" in result.output or "API Key:" in result.output

                # Step 4: Create reader user 'bob'
                result = runner.invoke(cli, ["user", "create", "--role", "reader", "bob"])
                assert result.exit_code == 0, f"Failed to create reader user: {result.output}"
                assert "User created" in result.output or "✓" in result.output
                assert "bob" in result.output

                # Step 5: List all users - verify both exist
                result = runner.invoke(cli, ["user", "list"])
                assert result.exit_code == 0, f"Failed to list users: {result.output}"
                assert "alice" in result.output, "Alice not in user list"
                assert "bob" in result.output, "Bob not in user list"

                # Step 6: Verify alice has admin permissions
                result = runner.invoke(cli, ["user", "permissions", "alice"])
                assert result.exit_code == 0, f"Failed to get alice permissions: {result.output}"
                assert "admin" in result.output.lower(), "Alice doesn't have admin role"

                # Step 7: Verify bob has reader permissions
                result = runner.invoke(cli, ["user", "permissions", "bob"])
                assert result.exit_code == 0, f"Failed to get bob permissions: {result.output}"
                assert "reader" in result.output.lower(), "Bob doesn't have reader role"

    def test_user_lifecycle(self):
        """Test creating, modifying, and deleting users."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Initialize
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

                # Create developer user
                result = runner.invoke(cli, ["user", "create", "--role", "developer", "charlie"])
                assert result.exit_code == 0
                assert "charlie" in result.output

                # Verify user exists
                result = runner.invoke(cli, ["user", "list"])
                assert result.exit_code == 0
                assert "charlie" in result.output
                assert "developer" in result.output.lower()

                # Change role to admin
                result = runner.invoke(cli, ["user", "set-role", "charlie", "admin"])
                assert result.exit_code == 0
                assert "admin" in result.output.lower()

                # Verify role changed
                result = runner.invoke(cli, ["user", "permissions", "charlie"])
                assert result.exit_code == 0
                assert "admin" in result.output.lower()

                # Delete user (with force flag to skip confirmation)
                result = runner.invoke(cli, ["user", "delete", "--force", "charlie"])
                assert result.exit_code == 0
                assert "deleted" in result.output.lower()

                # Verify user is gone
                result = runner.invoke(cli, ["user", "list"])
                assert result.exit_code == 0
                # Charlie should not be in the list anymore
                # (or "No users found" if it was the only user)

    def test_api_key_lifecycle(self):
        """Test creating, listing, and revoking API keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Initialize
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

                # Create user
                result = runner.invoke(cli, ["user", "create", "--role", "developer", "dave"])
                assert result.exit_code == 0

                # Create API key
                result = runner.invoke(cli, ["user", "create-api-key", "dave"])
                assert result.exit_code == 0
                assert "API key created" in result.output or "API Key:" in result.output

                # Extract key ID from output (format varies, but should contain UUID)
                # For now, we'll just verify the command succeeded

                # List API keys for user
                result = runner.invoke(cli, ["user", "list-api-keys", "dave"])
                assert result.exit_code == 0
                # Should show at least one API key

    def test_namespace_specific_roles(self):
        """Test creating users with namespace-specific roles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Initialize
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

                # Create developer with namespace-specific access
                result = runner.invoke(cli, [
                    "user", "create",
                    "--role", "developer",
                    "--namespace", "acme/teamA",
                    "--namespace", "acme/teamB",
                    "eve"
                ])
                assert result.exit_code == 0
                assert "eve" in result.output
                assert "acme/teamA" in result.output or "Namespaces" in result.output

                # Verify namespaces are assigned
                result = runner.invoke(cli, ["user", "permissions", "eve"])
                assert result.exit_code == 0

    def test_json_output(self):
        """Test JSON output format for user list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Initialize
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

                # Create users
                result = runner.invoke(cli, ["user", "create", "--role", "admin", "frank"])
                assert result.exit_code == 0

                result = runner.invoke(cli, ["user", "create", "--role", "reader", "grace"])
                assert result.exit_code == 0

                # List users in JSON format
                result = runner.invoke(cli, ["user", "list", "--json-output"])
                assert result.exit_code == 0

                # Parse JSON output
                import json
                try:
                    users = json.loads(result.output)
                    assert isinstance(users, list)
                    assert len(users) >= 2

                    # Verify user data structure
                    user_names = [u['username'] for u in users]
                    assert 'frank' in user_names
                    assert 'grace' in user_names

                    # Verify roles are included
                    for user in users:
                        assert 'roles' in user
                        assert isinstance(user['roles'], list)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Failed to parse JSON output: {e}\nOutput: {result.output}")

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Initialize
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

                # Try to create API key for non-existent user
                result = runner.invoke(cli, ["user", "create-api-key", "nonexistent"])
                assert result.exit_code != 0
                assert "not found" in result.output.lower() or "error" in result.output.lower()

                # Try to delete non-existent user
                result = runner.invoke(cli, ["user", "delete", "--force", "nonexistent"])
                assert result.exit_code != 0
                assert "not found" in result.output.lower() or "error" in result.output.lower()

                # Try to view permissions for non-existent user
                result = runner.invoke(cli, ["user", "permissions", "nonexistent"])
                assert result.exit_code != 0
                assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_duplicate_username(self):
        """Test that duplicate usernames are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi_test"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                # Initialize
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

                # Create first user
                result = runner.invoke(cli, ["user", "create", "--role", "developer", "henry"])
                assert result.exit_code == 0

                # Try to create duplicate user
                result = runner.invoke(cli, ["user", "create", "--role", "reader", "henry"])
                assert result.exit_code != 0
                # Should fail with duplicate error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
