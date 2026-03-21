"""
Unit tests for PermissionManager - Permission-based access control

Tests cover:
- Permission granting and revocation
- Permission hierarchy (READ < WRITE < ADMIN)
- Permission checks (has_permission, can_read, can_write, can_admin)
- List operations (by agent, by namespace)
- PermissionLevel enum validation
- Thread safety (WAL mode)
- Error handling (invalid levels, non-existent namespaces)
- Foreign key constraints
"""

import unittest
import tempfile
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.permissions import (
    PermissionManager,
    PermissionLevel,
    PermissionInfo,
    validate_permission_level
)
from omi.shared_namespace import SharedNamespace
from omi.storage.schema import init_database
import sqlite3


class TestPermissionLevel(unittest.TestCase):
    """Test suite for PermissionLevel enum."""

    def test_from_string_read(self):
        """Test converting 'read' string to PermissionLevel."""
        level = PermissionLevel.from_string("read")
        self.assertEqual(level, PermissionLevel.READ)

    def test_from_string_write(self):
        """Test converting 'write' string to PermissionLevel."""
        level = PermissionLevel.from_string("write")
        self.assertEqual(level, PermissionLevel.WRITE)

    def test_from_string_admin(self):
        """Test converting 'admin' string to PermissionLevel."""
        level = PermissionLevel.from_string("admin")
        self.assertEqual(level, PermissionLevel.ADMIN)

    def test_from_string_case_insensitive(self):
        """Test case-insensitive string conversion."""
        self.assertEqual(PermissionLevel.from_string("READ"), PermissionLevel.READ)
        self.assertEqual(PermissionLevel.from_string("Write"), PermissionLevel.WRITE)
        self.assertEqual(PermissionLevel.from_string("ADMIN"), PermissionLevel.ADMIN)

    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PermissionLevel.from_string("invalid")

        self.assertIn("Invalid permission level", str(ctx.exception))

    def test_can_read(self):
        """Test can_read method for all levels."""
        self.assertTrue(PermissionLevel.READ.can_read())
        self.assertTrue(PermissionLevel.WRITE.can_read())
        self.assertTrue(PermissionLevel.ADMIN.can_read())

    def test_can_write(self):
        """Test can_write method for all levels."""
        self.assertFalse(PermissionLevel.READ.can_write())
        self.assertTrue(PermissionLevel.WRITE.can_write())
        self.assertTrue(PermissionLevel.ADMIN.can_write())

    def test_can_admin(self):
        """Test can_admin method for all levels."""
        self.assertFalse(PermissionLevel.READ.can_admin())
        self.assertFalse(PermissionLevel.WRITE.can_admin())
        self.assertTrue(PermissionLevel.ADMIN.can_admin())

    def test_string_representation(self):
        """Test string conversion."""
        self.assertEqual(str(PermissionLevel.READ), "read")
        self.assertEqual(str(PermissionLevel.WRITE), "write")
        self.assertEqual(str(PermissionLevel.ADMIN), "admin")

    def test_validate_permission_level(self):
        """Test validate_permission_level utility function."""
        self.assertTrue(validate_permission_level("read"))
        self.assertTrue(validate_permission_level("write"))
        self.assertTrue(validate_permission_level("admin"))
        self.assertTrue(validate_permission_level("READ"))
        self.assertFalse(validate_permission_level("invalid"))
        self.assertFalse(validate_permission_level(""))


class TestPermissionManager(unittest.TestCase):
    """Test suite for PermissionManager."""

    def setUp(self):
        """Set up test database with shared namespaces."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_permissions.sqlite"

        # Initialize database schema
        conn = sqlite3.connect(self.db_path)
        init_database(conn, enable_wal=True)
        conn.close()

        # Create managers
        self.perm_mgr = PermissionManager(self.db_path)
        self.shared_ns = SharedNamespace(self.db_path)

        # Create test namespaces
        self.shared_ns.create("acme/team-alpha", "agent-1")
        self.shared_ns.create("acme/team-beta", "agent-2")

    def tearDown(self):
        """Clean up test database."""
        self.perm_mgr.close()
        self.shared_ns.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ==================== Grant/Revoke Operations ====================

    def test_grant_permission(self):
        """Test granting permission to an agent."""
        info = self.perm_mgr.grant(
            namespace="acme/team-alpha",
            agent_id="agent-1",
            permission_level=PermissionLevel.READ
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.namespace, "acme/team-alpha")
        self.assertEqual(info.agent_id, "agent-1")
        self.assertEqual(info.permission_level, PermissionLevel.READ)
        self.assertIsInstance(info.created_at, datetime)

    def test_grant_permission_nonexistent_namespace(self):
        """Test granting permission to non-existent namespace raises error."""
        with self.assertRaises(ValueError) as ctx:
            self.perm_mgr.grant(
                namespace="nonexistent/namespace",
                agent_id="agent-1",
                permission_level=PermissionLevel.READ
            )

        self.assertIn("Failed to grant permission", str(ctx.exception))

    def test_grant_permission_update_existing(self):
        """Test granting permission updates existing permission."""
        # Grant READ first
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)

        # Upgrade to WRITE
        info = self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.WRITE)

        self.assertEqual(info.permission_level, PermissionLevel.WRITE)

        # Verify only one permission exists
        perms = self.perm_mgr.list_for_namespace("acme/team-alpha")
        self.assertEqual(len(perms), 1)
        self.assertEqual(perms[0].permission_level, PermissionLevel.WRITE)

    def test_revoke_permission(self):
        """Test revoking permission from an agent."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)

        result = self.perm_mgr.revoke("acme/team-alpha", "agent-1")
        self.assertTrue(result)

        # Verify permission is gone
        level = self.perm_mgr.get("acme/team-alpha", "agent-1")
        self.assertIsNone(level)

    def test_revoke_nonexistent_permission(self):
        """Test revoking non-existent permission returns False."""
        result = self.perm_mgr.revoke("acme/team-alpha", "agent-99")
        self.assertFalse(result)

    # ==================== Get Permission ====================

    def test_get_permission(self):
        """Test getting permission level."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.WRITE)

        level = self.perm_mgr.get("acme/team-alpha", "agent-1")
        self.assertEqual(level, PermissionLevel.WRITE)

    def test_get_nonexistent_permission(self):
        """Test getting non-existent permission returns None."""
        level = self.perm_mgr.get("acme/team-alpha", "agent-99")
        self.assertIsNone(level)

    # ==================== Permission Hierarchy ====================

    def test_has_permission_read(self):
        """Test has_permission with READ requirement."""
        self.perm_mgr.grant("acme/team-alpha", "agent-read", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-write", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-admin", PermissionLevel.ADMIN)

        # All levels satisfy READ requirement
        self.assertTrue(self.perm_mgr.has_permission("acme/team-alpha", "agent-read", PermissionLevel.READ))
        self.assertTrue(self.perm_mgr.has_permission("acme/team-alpha", "agent-write", PermissionLevel.READ))
        self.assertTrue(self.perm_mgr.has_permission("acme/team-alpha", "agent-admin", PermissionLevel.READ))

    def test_has_permission_write(self):
        """Test has_permission with WRITE requirement."""
        self.perm_mgr.grant("acme/team-alpha", "agent-read", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-write", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-admin", PermissionLevel.ADMIN)

        # Only WRITE and ADMIN satisfy WRITE requirement
        self.assertFalse(self.perm_mgr.has_permission("acme/team-alpha", "agent-read", PermissionLevel.WRITE))
        self.assertTrue(self.perm_mgr.has_permission("acme/team-alpha", "agent-write", PermissionLevel.WRITE))
        self.assertTrue(self.perm_mgr.has_permission("acme/team-alpha", "agent-admin", PermissionLevel.WRITE))

    def test_has_permission_admin(self):
        """Test has_permission with ADMIN requirement."""
        self.perm_mgr.grant("acme/team-alpha", "agent-read", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-write", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-admin", PermissionLevel.ADMIN)

        # Only ADMIN satisfies ADMIN requirement
        self.assertFalse(self.perm_mgr.has_permission("acme/team-alpha", "agent-read", PermissionLevel.ADMIN))
        self.assertFalse(self.perm_mgr.has_permission("acme/team-alpha", "agent-write", PermissionLevel.ADMIN))
        self.assertTrue(self.perm_mgr.has_permission("acme/team-alpha", "agent-admin", PermissionLevel.ADMIN))

    def test_has_permission_no_permission(self):
        """Test has_permission with no permission granted."""
        self.assertFalse(self.perm_mgr.has_permission("acme/team-alpha", "agent-none", PermissionLevel.READ))

    # ==================== Convenience Methods ====================

    def test_can_read(self):
        """Test can_read convenience method."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.WRITE)

        self.assertTrue(self.perm_mgr.can_read("acme/team-alpha", "agent-1"))
        self.assertTrue(self.perm_mgr.can_read("acme/team-alpha", "agent-2"))
        self.assertFalse(self.perm_mgr.can_read("acme/team-alpha", "agent-99"))

    def test_can_write(self):
        """Test can_write convenience method."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-3", PermissionLevel.ADMIN)

        self.assertFalse(self.perm_mgr.can_write("acme/team-alpha", "agent-1"))
        self.assertTrue(self.perm_mgr.can_write("acme/team-alpha", "agent-2"))
        self.assertTrue(self.perm_mgr.can_write("acme/team-alpha", "agent-3"))

    def test_can_admin(self):
        """Test can_admin convenience method."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-3", PermissionLevel.ADMIN)

        self.assertFalse(self.perm_mgr.can_admin("acme/team-alpha", "agent-1"))
        self.assertFalse(self.perm_mgr.can_admin("acme/team-alpha", "agent-2"))
        self.assertTrue(self.perm_mgr.can_admin("acme/team-alpha", "agent-3"))

    # ==================== List Operations ====================

    def test_list_for_agent_empty(self):
        """Test listing permissions when agent has none."""
        perms = self.perm_mgr.list_for_agent("agent-99")
        self.assertEqual(len(perms), 0)

    def test_list_for_agent(self):
        """Test listing all permissions for an agent."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        time.sleep(0.1)  # Increased sleep for timestamp precision
        self.perm_mgr.grant("acme/team-beta", "agent-1", PermissionLevel.WRITE)

        perms = self.perm_mgr.list_for_agent("agent-1")

        self.assertEqual(len(perms), 2)

        # Verify both permissions exist
        namespaces = {p.namespace for p in perms}
        self.assertEqual(namespaces, {"acme/team-alpha", "acme/team-beta"})

        # Verify permission levels
        perm_dict = {p.namespace: p.permission_level for p in perms}
        self.assertEqual(perm_dict["acme/team-alpha"], PermissionLevel.READ)
        self.assertEqual(perm_dict["acme/team-beta"], PermissionLevel.WRITE)

    def test_list_for_namespace_empty(self):
        """Test listing permissions when namespace has none."""
        perms = self.perm_mgr.list_for_namespace("acme/team-alpha")
        self.assertEqual(len(perms), 0)

    def test_list_for_namespace(self):
        """Test listing all permissions for a namespace."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        time.sleep(0.1)  # Increased sleep for timestamp precision
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.WRITE)
        time.sleep(0.1)
        self.perm_mgr.grant("acme/team-alpha", "agent-3", PermissionLevel.ADMIN)

        perms = self.perm_mgr.list_for_namespace("acme/team-alpha")

        self.assertEqual(len(perms), 3)

        # Verify all agents are present
        agent_ids = {p.agent_id for p in perms}
        self.assertEqual(agent_ids, {"agent-1", "agent-2", "agent-3"})

        # Verify permission levels
        perm_dict = {p.agent_id: p.permission_level for p in perms}
        self.assertEqual(perm_dict["agent-1"], PermissionLevel.READ)
        self.assertEqual(perm_dict["agent-2"], PermissionLevel.WRITE)
        self.assertEqual(perm_dict["agent-3"], PermissionLevel.ADMIN)

    def test_get_agents_with_level(self):
        """Test getting agents with specific permission level."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-3", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-4", PermissionLevel.ADMIN)

        read_agents = self.perm_mgr.get_agents_with_level("acme/team-alpha", PermissionLevel.READ)
        self.assertEqual(read_agents, {"agent-1", "agent-2"})

        write_agents = self.perm_mgr.get_agents_with_level("acme/team-alpha", PermissionLevel.WRITE)
        self.assertEqual(write_agents, {"agent-3"})

        admin_agents = self.perm_mgr.get_agents_with_level("acme/team-alpha", PermissionLevel.ADMIN)
        self.assertEqual(admin_agents, {"agent-4"})

    def test_get_agents_with_level_empty(self):
        """Test getting agents when none have the level."""
        agents = self.perm_mgr.get_agents_with_level("acme/team-alpha", PermissionLevel.ADMIN)
        self.assertEqual(len(agents), 0)

    # ==================== PermissionInfo ====================

    def test_permission_info_to_dict(self):
        """Test PermissionInfo to_dict conversion."""
        info = self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.WRITE)

        info_dict = info.to_dict()

        self.assertEqual(info_dict["namespace"], "acme/team-alpha")
        self.assertEqual(info_dict["agent_id"], "agent-1")
        self.assertEqual(info_dict["permission_level"], "write")
        self.assertIsInstance(info_dict["created_at"], str)  # ISO format

    # ==================== Foreign Key Constraints ====================

    def test_cascade_delete_on_namespace_delete(self):
        """Test permissions are deleted when namespace is deleted."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.WRITE)

        # Verify permissions exist
        perms = self.perm_mgr.list_for_namespace("acme/team-alpha")
        self.assertEqual(len(perms), 2)

        # Delete namespace
        self.shared_ns.delete("acme/team-alpha")

        # Permissions should be cascade deleted
        perms = self.perm_mgr.list_for_namespace("acme/team-alpha")
        self.assertEqual(len(perms), 0)

    # ==================== Context Manager ====================

    def test_context_manager(self):
        """Test using PermissionManager as context manager."""
        with PermissionManager(self.db_path) as pm:
            pm.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
            self.assertTrue(pm.can_read("acme/team-alpha", "agent-1"))

        # Verify data persisted
        with PermissionManager(self.db_path) as pm:
            self.assertTrue(pm.can_read("acme/team-alpha", "agent-1"))

    # ==================== Thread Safety ====================

    def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled."""
        cursor = self.perm_mgr._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        self.assertEqual(mode.upper(), "WAL")

    def test_foreign_keys_enabled(self):
        """Test that foreign key constraints are enabled."""
        cursor = self.perm_mgr._conn.execute("PRAGMA foreign_keys")
        enabled = cursor.fetchone()[0]
        self.assertEqual(enabled, 1)

    def test_concurrent_grants(self):
        """Test concurrent permission grants (WAL mode)."""
        import threading

        errors = []

        def grant_permission(agent_id: str, level: PermissionLevel):
            try:
                self.perm_mgr.grant("acme/team-alpha", agent_id, level)
            except Exception as e:
                errors.append(e)

        # Grant permissions concurrently
        threads = []
        for i in range(5):
            level = [PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.ADMIN][i % 3]
            t = threading.Thread(
                target=grant_permission,
                args=(f"agent-{i}", level)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        self.assertEqual(len(errors), 0)

        # All permissions should exist
        perms = self.perm_mgr.list_for_namespace("acme/team-alpha")
        self.assertEqual(len(perms), 5)

    # ==================== Edge Cases ====================

    def test_multiple_agents_same_namespace(self):
        """Test multiple agents with different permissions in same namespace."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-alpha", "agent-2", PermissionLevel.WRITE)
        self.perm_mgr.grant("acme/team-alpha", "agent-3", PermissionLevel.ADMIN)

        self.assertEqual(self.perm_mgr.get("acme/team-alpha", "agent-1"), PermissionLevel.READ)
        self.assertEqual(self.perm_mgr.get("acme/team-alpha", "agent-2"), PermissionLevel.WRITE)
        self.assertEqual(self.perm_mgr.get("acme/team-alpha", "agent-3"), PermissionLevel.ADMIN)

    def test_same_agent_multiple_namespaces(self):
        """Test same agent with different permissions in multiple namespaces."""
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)
        self.perm_mgr.grant("acme/team-beta", "agent-1", PermissionLevel.ADMIN)

        self.assertEqual(self.perm_mgr.get("acme/team-alpha", "agent-1"), PermissionLevel.READ)
        self.assertEqual(self.perm_mgr.get("acme/team-beta", "agent-1"), PermissionLevel.ADMIN)

    def test_permission_downgrade(self):
        """Test downgrading permission level."""
        # Start with ADMIN
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.ADMIN)
        self.assertTrue(self.perm_mgr.can_admin("acme/team-alpha", "agent-1"))

        # Downgrade to READ
        self.perm_mgr.grant("acme/team-alpha", "agent-1", PermissionLevel.READ)

        self.assertTrue(self.perm_mgr.can_read("acme/team-alpha", "agent-1"))
        self.assertFalse(self.perm_mgr.can_write("acme/team-alpha", "agent-1"))
        self.assertFalse(self.perm_mgr.can_admin("acme/team-alpha", "agent-1"))


if __name__ == '__main__':
    unittest.main()
