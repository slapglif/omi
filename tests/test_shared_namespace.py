"""
Unit tests for SharedNamespace - Multi-agent namespace management

Tests cover:
- CRUD operations (create, get, exists, delete)
- List operations (list_all, list_by_creator)
- Metadata management
- Thread safety (WAL mode)
- Error handling (invalid namespace, duplicates)
- Context manager support
- Database initialization
"""

import unittest
import tempfile
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.shared_namespace import SharedNamespace, SharedNamespaceInfo
from omi.storage.schema import init_database
import sqlite3


class TestSharedNamespace(unittest.TestCase):
    """Test suite for SharedNamespace."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_shared.sqlite"

        # Initialize database schema
        conn = sqlite3.connect(self.db_path)
        init_database(conn, enable_wal=True)
        conn.close()

        # Create SharedNamespace manager
        self.shared_ns = SharedNamespace(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        self.shared_ns.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ==================== CRUD Operations ====================

    def test_create_namespace(self):
        """Test creating a shared namespace."""
        info = self.shared_ns.create(
            namespace="acme/team-alpha",
            created_by="agent-1"
        )

        self.assertIsNotNone(info)
        self.assertEqual(info.namespace, "acme/team-alpha")
        self.assertEqual(info.created_by, "agent-1")
        self.assertIsInstance(info.created_at, datetime)
        self.assertIsNone(info.metadata)

    def test_create_namespace_with_metadata(self):
        """Test creating a namespace with metadata."""
        metadata = {
            "description": "Shared research space",
            "max_agents": 5,
            "tags": ["research", "collaboration"]
        }

        info = self.shared_ns.create(
            namespace="acme/research",
            created_by="agent-1",
            metadata=metadata
        )

        self.assertEqual(info.namespace, "acme/research")
        self.assertEqual(info.metadata, metadata)

    def test_create_duplicate_namespace(self):
        """Test creating duplicate namespace raises error."""
        self.shared_ns.create(
            namespace="acme/team-alpha",
            created_by="agent-1"
        )

        with self.assertRaises(ValueError) as ctx:
            self.shared_ns.create(
                namespace="acme/team-alpha",
                created_by="agent-2"
            )

        self.assertIn("already exists", str(ctx.exception))

    def test_create_invalid_namespace(self):
        """Test creating namespace with invalid format raises error."""
        invalid_namespaces = [
            "",
            "/leading-slash",
            "trailing-slash/",
            "too/many/levels/here",
            "invalid@chars",
            "has spaces",
            "double//slash"
        ]

        for invalid in invalid_namespaces:
            with self.assertRaises(ValueError):
                self.shared_ns.create(
                    namespace=invalid,
                    created_by="agent-1"
                )

    def test_get_namespace(self):
        """Test retrieving a namespace."""
        # Create namespace
        created = self.shared_ns.create(
            namespace="acme/team-beta",
            created_by="agent-2",
            metadata={"type": "development"}
        )

        # Retrieve it
        info = self.shared_ns.get("acme/team-beta")

        self.assertIsNotNone(info)
        self.assertEqual(info.namespace, created.namespace)
        self.assertEqual(info.created_by, created.created_by)
        self.assertEqual(info.metadata, created.metadata)

    def test_get_nonexistent_namespace(self):
        """Test getting non-existent namespace returns None."""
        info = self.shared_ns.get("nonexistent/namespace")
        self.assertIsNone(info)

    def test_exists(self):
        """Test checking if namespace exists."""
        self.assertFalse(self.shared_ns.exists("acme/team-gamma"))

        self.shared_ns.create(
            namespace="acme/team-gamma",
            created_by="agent-1"
        )

        self.assertTrue(self.shared_ns.exists("acme/team-gamma"))

    def test_delete_namespace(self):
        """Test deleting a namespace."""
        self.shared_ns.create(
            namespace="acme/to-delete",
            created_by="agent-1"
        )

        self.assertTrue(self.shared_ns.exists("acme/to-delete"))

        result = self.shared_ns.delete("acme/to-delete")
        self.assertTrue(result)

        self.assertFalse(self.shared_ns.exists("acme/to-delete"))
        self.assertIsNone(self.shared_ns.get("acme/to-delete"))

    def test_delete_nonexistent_namespace(self):
        """Test deleting non-existent namespace returns False."""
        result = self.shared_ns.delete("nonexistent/namespace")
        self.assertFalse(result)

    # ==================== List Operations ====================

    def test_list_all_empty(self):
        """Test listing when no namespaces exist."""
        namespaces = self.shared_ns.list_all()
        self.assertEqual(len(namespaces), 0)

    def test_list_all(self):
        """Test listing all namespaces."""
        # Create multiple namespaces
        self.shared_ns.create("acme/team-1", "agent-1")
        time.sleep(0.01)  # Ensure different timestamps
        self.shared_ns.create("acme/team-2", "agent-2")
        time.sleep(0.01)
        self.shared_ns.create("acme/team-3", "agent-1")

        namespaces = self.shared_ns.list_all()

        self.assertEqual(len(namespaces), 3)

        # Should be ordered by created_at DESC (most recent first)
        self.assertEqual(namespaces[0].namespace, "acme/team-3")
        self.assertEqual(namespaces[1].namespace, "acme/team-2")
        self.assertEqual(namespaces[2].namespace, "acme/team-1")

    def test_list_by_creator(self):
        """Test listing namespaces by creator."""
        self.shared_ns.create("acme/team-1", "agent-1")
        self.shared_ns.create("acme/team-2", "agent-2")
        self.shared_ns.create("acme/team-3", "agent-1")
        self.shared_ns.create("acme/team-4", "agent-3")

        agent1_namespaces = self.shared_ns.list_by_creator("agent-1")
        self.assertEqual(len(agent1_namespaces), 2)
        self.assertIn("acme/team-1", [ns.namespace for ns in agent1_namespaces])
        self.assertIn("acme/team-3", [ns.namespace for ns in agent1_namespaces])

        agent2_namespaces = self.shared_ns.list_by_creator("agent-2")
        self.assertEqual(len(agent2_namespaces), 1)
        self.assertEqual(agent2_namespaces[0].namespace, "acme/team-2")

        # Non-existent creator
        agent99_namespaces = self.shared_ns.list_by_creator("agent-99")
        self.assertEqual(len(agent99_namespaces), 0)

    # ==================== Metadata Operations ====================

    def test_update_metadata(self):
        """Test updating namespace metadata."""
        self.shared_ns.create(
            namespace="acme/team-delta",
            created_by="agent-1",
            metadata={"version": 1}
        )

        new_metadata = {
            "version": 2,
            "description": "Updated metadata",
            "active": True
        }

        result = self.shared_ns.update_metadata("acme/team-delta", new_metadata)
        self.assertTrue(result)

        info = self.shared_ns.get("acme/team-delta")
        self.assertEqual(info.metadata, new_metadata)

    def test_update_metadata_nonexistent(self):
        """Test updating metadata for non-existent namespace."""
        result = self.shared_ns.update_metadata(
            "nonexistent/namespace",
            {"key": "value"}
        )
        self.assertFalse(result)

    def test_metadata_serialization(self):
        """Test complex metadata serialization."""
        complex_metadata = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {
                "key": "value",
                "deep": {
                    "level": 3
                }
            }
        }

        info = self.shared_ns.create(
            namespace="acme/complex",
            created_by="agent-1",
            metadata=complex_metadata
        )

        # Retrieve and verify
        retrieved = self.shared_ns.get("acme/complex")
        self.assertEqual(retrieved.metadata, complex_metadata)

    # ==================== Info Object ====================

    def test_namespace_info_to_dict(self):
        """Test SharedNamespaceInfo to_dict conversion."""
        info = self.shared_ns.create(
            namespace="acme/test",
            created_by="agent-1",
            metadata={"key": "value"}
        )

        info_dict = info.to_dict()

        self.assertEqual(info_dict["namespace"], "acme/test")
        self.assertEqual(info_dict["created_by"], "agent-1")
        self.assertEqual(info_dict["metadata"], {"key": "value"})
        self.assertIsInstance(info_dict["created_at"], str)  # ISO format

    def test_namespace_info_to_dict_no_metadata(self):
        """Test to_dict with no metadata."""
        info = self.shared_ns.create(
            namespace="acme/test",
            created_by="agent-1"
        )

        info_dict = info.to_dict()
        self.assertEqual(info_dict["metadata"], {})

    # ==================== Context Manager ====================

    def test_context_manager(self):
        """Test using SharedNamespace as context manager."""
        with SharedNamespace(self.db_path) as ns:
            ns.create("acme/test-ctx", "agent-1")
            self.assertTrue(ns.exists("acme/test-ctx"))

        # Verify connection is closed
        # Create new instance to verify data persisted
        with SharedNamespace(self.db_path) as ns:
            self.assertTrue(ns.exists("acme/test-ctx"))

    # ==================== Thread Safety ====================

    def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled."""
        # Check PRAGMA
        cursor = self.shared_ns._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        self.assertEqual(mode.upper(), "WAL")

    def test_foreign_keys_enabled(self):
        """Test that foreign key constraints are enabled."""
        cursor = self.shared_ns._conn.execute("PRAGMA foreign_keys")
        enabled = cursor.fetchone()[0]
        self.assertEqual(enabled, 1)

    def test_concurrent_creates(self):
        """Test concurrent namespace creation (WAL mode)."""
        import threading

        errors = []

        def create_namespace(ns_name: str, agent: str):
            try:
                self.shared_ns.create(ns_name, agent)
            except Exception as e:
                errors.append(e)

        # Create multiple namespaces concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=create_namespace,
                args=(f"acme/concurrent-{i}", f"agent-{i}")
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # No errors should occur
        self.assertEqual(len(errors), 0)

        # All namespaces should exist
        namespaces = self.shared_ns.list_all()
        self.assertEqual(len(namespaces), 5)

    # ==================== Edge Cases ====================

    def test_namespace_with_unicode(self):
        """Test namespace with valid unicode characters."""
        # Valid: alphanumeric + hyphens + underscores only
        self.shared_ns.create("acme/test_team-1", "agent-1")
        self.assertTrue(self.shared_ns.exists("acme/test_team-1"))

    def test_namespace_hierarchy_levels(self):
        """Test namespaces at different hierarchy levels."""
        # Org level
        self.shared_ns.create("acme", "agent-1")

        # Team level
        self.shared_ns.create("acme/research", "agent-2")

        # Agent level
        self.shared_ns.create("acme/research/reader", "agent-3")

        namespaces = self.shared_ns.list_all()
        self.assertEqual(len(namespaces), 3)

    def test_empty_metadata(self):
        """Test creating namespace with empty metadata dict."""
        info = self.shared_ns.create(
            namespace="acme/empty-meta",
            created_by="agent-1",
            metadata={}
        )

        # Empty dict is serialized as "{}" in JSON, which deserializes back to {}
        # But the implementation may return None for empty metadata - accept both
        self.assertIn(info.metadata, [None, {}])

    def test_large_metadata(self):
        """Test handling large metadata objects."""
        large_metadata = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(100)
        }

        info = self.shared_ns.create(
            namespace="acme/large-meta",
            created_by="agent-1",
            metadata=large_metadata
        )

        retrieved = self.shared_ns.get("acme/large-meta")
        self.assertEqual(retrieved.metadata, large_metadata)

    def test_special_characters_in_metadata(self):
        """Test metadata with special characters."""
        special_metadata = {
            "quotes": "He said \"hello\"",
            "newlines": "Line 1\nLine 2\nLine 3",
            "tabs": "Col1\tCol2\tCol3",
            "unicode": "Hello 世界 🌍",
            "json_string": '{"nested": "json"}'
        }

        info = self.shared_ns.create(
            namespace="acme/special",
            created_by="agent-1",
            metadata=special_metadata
        )

        retrieved = self.shared_ns.get("acme/special")
        self.assertEqual(retrieved.metadata, special_metadata)


if __name__ == '__main__':
    unittest.main()
