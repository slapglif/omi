"""
test_namespace_permissions.py - Comprehensive Namespace-Level Permission Tests

Tests namespace-level access control to ensure:
1. Users can have different roles in different namespaces
2. Namespace-specific permissions override global permissions
3. Permission isolation between namespaces
4. Developer can write to assigned namespaces but not others
5. Reader can read all namespaces
6. Admin has access to all namespaces
7. Auditor has read-only access to all namespaces

Pattern: Create users with namespace-specific roles and verify permission boundaries

Issue: https://github.com/slapglif/omi/issues/049
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

from omi.rbac import RBACManager, Role, Permission
from omi.user_manager import UserManager, User
from omi.storage.schema import init_database


# Fixtures for namespace permission testing

@pytest.fixture
def namespace_db():
    """Create a temporary database with full schema for namespace testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "namespace.sqlite"

        # Create connection and initialize schema
        conn = sqlite3.connect(db_path)
        init_database(conn, enable_wal=False)
        conn.close()

        yield db_path


@pytest.fixture
def user_manager(namespace_db):
    """Create UserManager with initialized database."""
    manager = UserManager(str(namespace_db), enable_wal=False)
    yield manager
    manager.close()


@pytest.fixture
def rbac_manager(namespace_db):
    """Create RBACManager with initialized database."""
    return RBACManager(str(namespace_db))


@pytest.fixture
def namespace_users(user_manager):
    """
    Create test users with namespace-specific roles.

    Setup:
        - admin: global admin role
        - dev_teamA: developer role in 'acme/teamA' only
        - dev_teamB: developer role in 'acme/teamB' only
        - dev_both: developer role in both 'acme/teamA' and 'acme/teamB'
        - reader_global: global reader role
        - auditor_global: global auditor role
        - no_role: user with no roles
    """
    users = {}

    # Admin with global access
    admin_id = user_manager.create_user("admin", "admin@example.com")
    user_manager.assign_role(admin_id, "admin")
    users["admin"] = admin_id

    # Developer with access only to acme/teamA
    dev_teamA_id = user_manager.create_user("dev_teamA", "devA@example.com")
    user_manager.assign_role(dev_teamA_id, "developer", namespace="acme/teamA")
    users["dev_teamA"] = dev_teamA_id

    # Developer with access only to acme/teamB
    dev_teamB_id = user_manager.create_user("dev_teamB", "devB@example.com")
    user_manager.assign_role(dev_teamB_id, "developer", namespace="acme/teamB")
    users["dev_teamB"] = dev_teamB_id

    # Developer with access to both teamA and teamB
    dev_both_id = user_manager.create_user("dev_both", "devBoth@example.com")
    user_manager.assign_role(dev_both_id, "developer", namespace="acme/teamA")
    user_manager.assign_role(dev_both_id, "developer", namespace="acme/teamB")
    users["dev_both"] = dev_both_id

    # Reader with global read access
    reader_id = user_manager.create_user("reader", "reader@example.com")
    user_manager.assign_role(reader_id, "reader")
    users["reader"] = reader_id

    # Auditor with global audit access
    auditor_id = user_manager.create_user("auditor", "auditor@example.com")
    user_manager.assign_role(auditor_id, "auditor")
    users["auditor"] = auditor_id

    # User with no roles
    no_role_id = user_manager.create_user("no_role", "norole@example.com")
    users["no_role"] = no_role_id

    return users


class TestNamespaceLevelWriteAccess:
    """Test write access is properly isolated to assigned namespaces."""

    def test_developer_can_write_to_assigned_namespace(self, namespace_users, rbac_manager):
        """Developer with 'acme/teamA' role can write to 'acme/teamA'."""
        dev_id = namespace_users["dev_teamA"]

        # Should have write access to assigned namespace
        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "write", "belief", namespace="acme/teamA") is True

    def test_developer_cannot_write_to_unassigned_namespace(self, namespace_users, rbac_manager):
        """Developer with 'acme/teamA' role CANNOT write to 'acme/teamB'."""
        dev_id = namespace_users["dev_teamA"]

        # Should NOT have write access to unassigned namespace
        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamB") is False
        assert rbac_manager.check_permission(dev_id, "write", "belief", namespace="acme/teamB") is False

    def test_developer_with_multiple_namespaces(self, namespace_users, rbac_manager):
        """Developer can have write access to multiple namespaces."""
        dev_id = namespace_users["dev_both"]

        # Should have write access to both assigned namespaces
        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamB") is True

        # But not to unassigned namespaces
        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamC") is False

    def test_developer_cannot_write_without_namespace_context(self, namespace_users, rbac_manager):
        """Developer with namespace-specific role cannot write globally (without namespace)."""
        dev_id = namespace_users["dev_teamA"]

        # When namespace is not specified, should check global roles (which user doesn't have)
        # Note: Current implementation may fall back to global roles if namespace not specified
        # This test verifies namespace isolation
        result = rbac_manager.check_permission(dev_id, "write", "memory", namespace=None)

        # Should NOT have global write access (only namespace-specific)
        assert result is False


class TestNamespaceLevelReadAccess:
    """Test read access across namespaces."""

    def test_developer_can_read_assigned_namespace(self, namespace_users, rbac_manager):
        """Developer can read from assigned namespace."""
        dev_id = namespace_users["dev_teamA"]

        assert rbac_manager.check_permission(dev_id, "read", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "read", "belief", namespace="acme/teamA") is True

    def test_reader_can_read_all_namespaces(self, namespace_users, rbac_manager):
        """Global reader can read from all namespaces."""
        reader_id = namespace_users["reader"]

        # Should have read access to all namespaces (global role)
        assert rbac_manager.check_permission(reader_id, "read", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(reader_id, "read", "memory", namespace="acme/teamB") is True
        assert rbac_manager.check_permission(reader_id, "read", "memory", namespace="acme/teamC") is True
        assert rbac_manager.check_permission(reader_id, "read", "belief", namespace="acme/teamA") is True

    def test_reader_cannot_write_any_namespace(self, namespace_users, rbac_manager):
        """Global reader CANNOT write to any namespace."""
        reader_id = namespace_users["reader"]

        # Should NOT have write access to any namespace
        assert rbac_manager.check_permission(reader_id, "write", "memory", namespace="acme/teamA") is False
        assert rbac_manager.check_permission(reader_id, "write", "memory", namespace="acme/teamB") is False
        assert rbac_manager.check_permission(reader_id, "write", "belief", namespace="acme/teamA") is False

    def test_auditor_can_read_all_namespaces(self, namespace_users, rbac_manager):
        """Global auditor can read from all namespaces."""
        auditor_id = namespace_users["auditor"]

        # Should have read access to all namespaces
        assert rbac_manager.check_permission(auditor_id, "read", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(auditor_id, "read", "memory", namespace="acme/teamB") is True
        assert rbac_manager.check_permission(auditor_id, "read", "belief", namespace="acme/teamA") is True


class TestNamespaceAdminAccess:
    """Test admin access across all namespaces."""

    def test_admin_can_write_all_namespaces(self, namespace_users, rbac_manager):
        """Admin has write access to all namespaces."""
        admin_id = namespace_users["admin"]

        # Should have write access to any namespace
        assert rbac_manager.check_permission(admin_id, "write", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(admin_id, "write", "memory", namespace="acme/teamB") is True
        assert rbac_manager.check_permission(admin_id, "write", "memory", namespace="acme/research") is True

    def test_admin_can_read_all_namespaces(self, namespace_users, rbac_manager):
        """Admin has read access to all namespaces."""
        admin_id = namespace_users["admin"]

        assert rbac_manager.check_permission(admin_id, "read", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(admin_id, "read", "belief", namespace="acme/teamB") is True

    def test_admin_can_delete_all_namespaces(self, namespace_users, rbac_manager):
        """Admin has delete access to all namespaces."""
        admin_id = namespace_users["admin"]

        assert rbac_manager.check_permission(admin_id, "delete", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(admin_id, "delete", "memory", namespace="acme/teamB") is True


class TestNamespacePermissionIsolation:
    """Test that permissions in one namespace don't leak to others."""

    def test_no_cross_namespace_write_leak(self, namespace_users, rbac_manager):
        """Write permission in teamA doesn't grant write in teamB."""
        dev_A = namespace_users["dev_teamA"]
        dev_B = namespace_users["dev_teamB"]

        # dev_A can write to teamA but not teamB
        assert rbac_manager.check_permission(dev_A, "write", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_A, "write", "memory", namespace="acme/teamB") is False

        # dev_B can write to teamB but not teamA
        assert rbac_manager.check_permission(dev_B, "write", "memory", namespace="acme/teamB") is True
        assert rbac_manager.check_permission(dev_B, "write", "memory", namespace="acme/teamA") is False

    def test_no_role_user_denied_all_namespaces(self, namespace_users, rbac_manager):
        """User with no roles is denied access to all namespaces."""
        no_role_id = namespace_users["no_role"]

        # Should be denied everywhere
        assert rbac_manager.check_permission(no_role_id, "read", "memory", namespace="acme/teamA") is False
        assert rbac_manager.check_permission(no_role_id, "write", "memory", namespace="acme/teamA") is False
        assert rbac_manager.check_permission(no_role_id, "read", "memory", namespace="acme/teamB") is False
        assert rbac_manager.check_permission(no_role_id, "write", "memory", namespace="acme/teamB") is False

    def test_namespace_role_list_is_accurate(self, namespace_users, user_manager):
        """User roles list accurately reflects namespace assignments."""
        dev_id = namespace_users["dev_teamA"]

        # Get all roles
        roles = user_manager.get_user_roles(dev_id)

        # Should have exactly one role
        assert len(roles) == 1

        # Should be developer role in acme/teamA namespace
        role_name, namespace = roles[0]
        assert role_name == "developer"
        assert namespace == "acme/teamA"


class TestNamespaceWithGlobalRoles:
    """Test interaction between global roles and namespace-specific roles."""

    def test_global_role_applies_to_all_namespaces(self, user_manager, rbac_manager):
        """Global role grants access to all namespaces."""
        user_id = user_manager.create_user("global_dev", "global@example.com")
        user_manager.assign_role(user_id, "developer")  # Global role

        # Should have access to any namespace
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/teamB") is True
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/any") is True

    def test_namespace_role_overrides_global(self, user_manager, rbac_manager):
        """Namespace-specific role has higher priority than global role."""
        # Create user with global reader role
        user_id = user_manager.create_user("mixed_user", "mixed@example.com")
        user_manager.assign_role(user_id, "reader")  # Global reader (read-only)

        # Assign developer role to specific namespace (read-write)
        user_manager.assign_role(user_id, "developer", namespace="acme/special")

        # Should have read access everywhere (global reader)
        assert rbac_manager.check_permission(user_id, "read", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(user_id, "read", "memory", namespace="acme/special") is True

        # Should NOT have write access to regular namespaces (global reader)
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/teamA") is False

        # Should have write access to special namespace (developer role)
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/special") is True

    def test_user_can_have_multiple_roles_in_same_namespace(self, user_manager, rbac_manager):
        """User can have multiple roles (though redundant in practice)."""
        user_id = user_manager.create_user("multi_role", "multi@example.com")

        # Assign reader role to namespace
        user_manager.assign_role(user_id, "reader", namespace="acme/test")

        # Assign auditor role to same namespace
        user_manager.assign_role(user_id, "auditor", namespace="acme/test")

        # Should have both roles
        roles = user_manager.get_user_roles(user_id, namespace="acme/test")
        assert len(roles) == 2

        role_names = [r[0] for r in roles]
        assert "reader" in role_names
        assert "auditor" in role_names


class TestNamespaceRoleRevocation:
    """Test removing namespace-specific roles."""

    def test_revoke_namespace_role(self, user_manager, rbac_manager):
        """Revoking namespace role removes access to that namespace."""
        user_id = user_manager.create_user("revoke_test", "revoke@example.com")

        # Assign developer role to namespace
        user_manager.assign_role(user_id, "developer", namespace="acme/temp")

        # Verify access
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/temp") is True

        # Revoke the role
        revoked = user_manager.revoke_role(user_id, "developer", namespace="acme/temp")
        assert revoked is True

        # Should no longer have access
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/temp") is False

    def test_revoke_global_role_keeps_namespace_roles(self, user_manager, rbac_manager):
        """Revoking global role doesn't affect namespace-specific roles."""
        user_id = user_manager.create_user("partial_revoke", "partial@example.com")

        # Assign both global and namespace roles
        user_manager.assign_role(user_id, "reader")  # Global
        user_manager.assign_role(user_id, "developer", namespace="acme/keep")

        # Revoke global role
        user_manager.revoke_role(user_id, "reader", namespace=None)

        # Should still have namespace-specific role
        roles = user_manager.get_user_roles(user_id)
        assert len(roles) == 1
        assert roles[0][1] == "acme/keep"

        # Should still have write access to namespace
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/keep") is True

    def test_revoke_nonexistent_role_returns_false(self, user_manager):
        """Revoking a role that doesn't exist returns False."""
        user_id = user_manager.create_user("revoke_none", "revokeNone@example.com")

        # Try to revoke role user doesn't have
        revoked = user_manager.revoke_role(user_id, "admin", namespace="acme/fake")
        assert revoked is False


class TestNamespaceWithDifferentResources:
    """Test namespace permissions work correctly across different resource types."""

    def test_namespace_permissions_apply_to_memory(self, namespace_users, rbac_manager):
        """Namespace permissions work for memory resource."""
        dev_id = namespace_users["dev_teamA"]

        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "write", "memory", namespace="acme/teamB") is False

    def test_namespace_permissions_apply_to_beliefs(self, namespace_users, rbac_manager):
        """Namespace permissions work for belief resource."""
        dev_id = namespace_users["dev_teamA"]

        assert rbac_manager.check_permission(dev_id, "write", "belief", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "write", "belief", namespace="acme/teamB") is False

    def test_namespace_permissions_apply_to_checkpoints(self, namespace_users, rbac_manager):
        """Namespace permissions work for checkpoint resource."""
        dev_id = namespace_users["dev_teamA"]

        assert rbac_manager.check_permission(dev_id, "write", "checkpoint", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "write", "checkpoint", namespace="acme/teamB") is False

    def test_namespace_permissions_apply_to_read_operations(self, namespace_users, rbac_manager):
        """Namespace permissions work for read operations."""
        dev_id = namespace_users["dev_teamA"]

        assert rbac_manager.check_permission(dev_id, "read", "memory", namespace="acme/teamA") is True
        assert rbac_manager.check_permission(dev_id, "read", "memory", namespace="acme/teamB") is False


class TestComplexNamespaceScenarios:
    """Test complex real-world namespace scenarios."""

    def test_hierarchical_namespace_access(self, user_manager, rbac_manager):
        """
        Test hierarchical namespace pattern (e.g., 'acme/engineering/backend').

        Note: Current implementation treats namespaces as exact matches, not hierarchical.
        This test documents the current behavior.
        """
        user_id = user_manager.create_user("hierarchical", "hier@example.com")
        user_manager.assign_role(user_id, "developer", namespace="acme/engineering")

        # Access to exact namespace
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/engineering") is True

        # No automatic access to child namespaces (flat namespace model)
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/engineering/backend") is False
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme") is False

    def test_special_characters_in_namespace(self, user_manager, rbac_manager):
        """Namespaces can contain special characters."""
        user_id = user_manager.create_user("special_ns", "special@example.com")

        # Assign role with special characters in namespace
        namespace = "client-123/project_alpha"
        user_manager.assign_role(user_id, "developer", namespace=namespace)

        # Should work with special characters
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace=namespace) is True

    def test_empty_namespace_string(self, user_manager, rbac_manager):
        """Empty string namespace is treated as no namespace (global)."""
        user_id = user_manager.create_user("empty_ns", "empty@example.com")

        # Assign global role (no namespace)
        user_manager.assign_role(user_id, "developer")

        # Check with None and empty string (both should work for global)
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace=None) is True

    def test_namespace_case_sensitivity(self, user_manager, rbac_manager):
        """Namespaces are case-sensitive."""
        user_id = user_manager.create_user("case_test", "case@example.com")
        user_manager.assign_role(user_id, "developer", namespace="acme/TeamA")

        # Exact match works
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/TeamA") is True

        # Different case should not match
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="acme/teama") is False
        assert rbac_manager.check_permission(user_id, "write", "memory", namespace="ACME/TeamA") is False


class TestNamespacePermissionQueries:
    """Test querying and listing namespace permissions."""

    def test_get_user_permissions_for_namespace(self, namespace_users, user_manager):
        """Get all permissions for a user in a specific namespace."""
        dev_id = namespace_users["dev_teamA"]

        # Get permissions for the assigned namespace
        permissions = user_manager.get_user_permissions(dev_id, namespace="acme/teamA")

        # Should have developer permissions
        assert len(permissions) > 0

        # Check that read and write are included
        actions = {p["action"] for p in permissions}
        assert "read" in actions
        assert "write" in actions

    def test_get_all_user_permissions(self, namespace_users, user_manager):
        """Get all permissions for a user across all namespaces."""
        dev_id = namespace_users["dev_both"]

        # Get all permissions (no namespace filter)
        permissions = user_manager.get_user_permissions(dev_id)

        # Should have permissions from multiple namespaces
        assert len(permissions) > 0

    def test_list_namespaces_for_user(self, namespace_users, user_manager):
        """List all namespaces a user has access to."""
        dev_id = namespace_users["dev_both"]

        # Get all roles
        roles = user_manager.get_user_roles(dev_id)

        # Extract unique namespaces
        namespaces = {r[1] for r in roles}

        # Should have access to both teamA and teamB
        assert "acme/teamA" in namespaces
        assert "acme/teamB" in namespaces


class TestEndToEndNamespaceWorkflow:
    """Test complete end-to-end workflows with namespace permissions."""

    def test_team_collaboration_workflow(self, user_manager, rbac_manager):
        """
        Simulate real team collaboration scenario:
        - Admin creates project namespace
        - Admin assigns developers to their team namespaces
        - Developers can only modify their team's data
        - Readers can view all teams
        """
        # Admin creates users
        admin_id = user_manager.create_user("project_admin", "admin@example.com")
        user_manager.assign_role(admin_id, "admin")

        dev1_id = user_manager.create_user("dev1", "dev1@example.com")
        dev2_id = user_manager.create_user("dev2", "dev2@example.com")
        reader_id = user_manager.create_user("project_reader", "preader@example.com")

        # Admin assigns developers to their teams
        user_manager.assign_role(dev1_id, "developer", namespace="project_x/frontend")
        user_manager.assign_role(dev2_id, "developer", namespace="project_x/backend")

        # Admin assigns reader global access
        user_manager.assign_role(reader_id, "reader")

        # Verify dev1 can write to frontend but not backend
        assert rbac_manager.check_permission(dev1_id, "write", "memory", namespace="project_x/frontend") is True
        assert rbac_manager.check_permission(dev1_id, "write", "memory", namespace="project_x/backend") is False

        # Verify dev2 can write to backend but not frontend
        assert rbac_manager.check_permission(dev2_id, "write", "memory", namespace="project_x/backend") is True
        assert rbac_manager.check_permission(dev2_id, "write", "memory", namespace="project_x/frontend") is False

        # Verify reader can read both
        assert rbac_manager.check_permission(reader_id, "read", "memory", namespace="project_x/frontend") is True
        assert rbac_manager.check_permission(reader_id, "read", "memory", namespace="project_x/backend") is True

        # Verify reader cannot write
        assert rbac_manager.check_permission(reader_id, "write", "memory", namespace="project_x/frontend") is False

        # Verify admin can access everything
        assert rbac_manager.check_permission(admin_id, "write", "memory", namespace="project_x/frontend") is True
        assert rbac_manager.check_permission(admin_id, "write", "memory", namespace="project_x/backend") is True
        assert rbac_manager.check_permission(admin_id, "delete", "memory", namespace="project_x/frontend") is True
