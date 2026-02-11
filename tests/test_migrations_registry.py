"""
Comprehensive tests for migration registry (MigrationRegistry)
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import importlib

from omi.migrations.registry import MigrationRegistry
from omi.migrations.migration_base import MigrationBase


class TestMigration1(MigrationBase):
    """Test migration v1"""
    version = 1
    description = "Test migration 1"

    def upgrade(self, db):
        pass

    def downgrade(self, db):
        pass


class TestMigration2(MigrationBase):
    """Test migration v2"""
    version = 2
    description = "Test migration 2"

    def upgrade(self, db):
        pass

    def downgrade(self, db):
        pass


class TestMigration3(MigrationBase):
    """Test migration v3"""
    version = 3
    description = "Test migration 3"

    def upgrade(self, db):
        pass

    def downgrade(self, db):
        pass


class TestMigrationRegistry:
    """Test MigrationRegistry functionality"""

    def test_init(self):
        """Test registry initialization"""
        registry = MigrationRegistry()
        assert registry.versions_package == "omi.migrations.versions"
        assert registry._discovered is False
        assert len(registry._migrations) == 0

    def test_init_custom_package(self):
        """Test initialization with custom package"""
        registry = MigrationRegistry(versions_package="custom.migrations")
        assert registry.versions_package == "custom.migrations"

    def test_discover_no_package(self):
        """Test discovery when versions package doesn't exist"""
        registry = MigrationRegistry(versions_package="nonexistent.package")

        with patch('importlib.import_module', side_effect=ImportError):
            registry.discover()

        assert registry._discovered is True
        assert len(registry._migrations) == 0

    def test_discover_empty_package(self):
        """Test discovery with empty package"""
        mock_package = Mock()
        mock_package.__file__ = "/tmp/test/__init__.py"

        with patch('importlib.import_module', return_value=mock_package):
            with patch('pathlib.Path.glob', return_value=[]):
                registry = MigrationRegistry()
                registry.discover()

        assert registry._discovered is True

    def test_register_migration(self):
        """Test manually registering a migration"""
        registry = MigrationRegistry()
        migration = TestMigration1()

        registry.register(migration)

        assert 1 in registry._migrations
        assert registry._migrations[1] == migration

    def test_register_duplicate_version(self):
        """Test registering duplicate version fails"""
        registry = MigrationRegistry()
        migration1 = TestMigration1()
        migration2 = TestMigration1()  # Same version

        registry.register(migration1)

        with pytest.raises(ValueError, match="Duplicate migration version"):
            registry.register(migration2)

    def test_validate_sequence_valid(self):
        """Test validation of valid migration sequence"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry.register(TestMigration3())

        # Should not raise
        registry._validate_sequence()

    def test_validate_sequence_not_starting_at_one(self):
        """Test validation fails if not starting at version 1"""
        registry = MigrationRegistry()
        registry._migrations[2] = TestMigration2()

        with pytest.raises(ValueError, match="must start at 1"):
            registry._validate_sequence()

    def test_validate_sequence_gap(self):
        """Test validation fails with version gap"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration3())  # Skip version 2

        with pytest.raises(ValueError, match="version gap detected"):
            registry._validate_sequence()

    def test_validate_sequence_empty(self):
        """Test validation passes for empty registry"""
        registry = MigrationRegistry()
        # Should not raise
        registry._validate_sequence()

    def test_get_migration(self):
        """Test getting a specific migration"""
        registry = MigrationRegistry()
        migration = TestMigration1()
        registry.register(migration)
        registry._discovered = True

        result = registry.get_migration(1)

        assert result == migration

    def test_get_migration_not_found(self):
        """Test getting non-existent migration"""
        registry = MigrationRegistry()
        registry._discovered = True

        result = registry.get_migration(999)

        assert result is None

    def test_get_migration_triggers_discovery(self):
        """Test get_migration triggers discovery if needed"""
        registry = MigrationRegistry()

        with patch.object(registry, 'discover') as mock_discover:
            registry.get_migration(1)

        mock_discover.assert_called_once()

    def test_get_all_migrations(self):
        """Test getting all migrations in order"""
        registry = MigrationRegistry()
        m3 = TestMigration3()
        m1 = TestMigration1()
        m2 = TestMigration2()

        # Register out of order
        registry.register(m3)
        registry.register(m1)
        registry.register(m2)
        registry._discovered = True

        migrations = registry.get_all_migrations()

        assert len(migrations) == 3
        assert migrations[0] == m1  # Should be sorted
        assert migrations[1] == m2
        assert migrations[2] == m3

    def test_get_all_migrations_empty(self):
        """Test getting all migrations from empty registry"""
        registry = MigrationRegistry()
        registry._discovered = True

        migrations = registry.get_all_migrations()

        assert migrations == []

    def test_get_pending_migrations(self):
        """Test getting pending migrations"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry.register(TestMigration3())
        registry._discovered = True

        pending = registry.get_pending_migrations(current_version=1)

        assert len(pending) == 2
        assert pending[0].version == 2
        assert pending[1].version == 3

    def test_get_pending_migrations_all_applied(self):
        """Test pending when all migrations applied"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry._discovered = True

        pending = registry.get_pending_migrations(current_version=2)

        assert len(pending) == 0

    def test_get_pending_migrations_none_applied(self):
        """Test pending when no migrations applied"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry._discovered = True

        pending = registry.get_pending_migrations(current_version=0)

        assert len(pending) == 2

    def test_get_latest_version(self):
        """Test getting latest version"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry.register(TestMigration3())
        registry._discovered = True

        latest = registry.get_latest_version()

        assert latest == 3

    def test_get_latest_version_empty(self):
        """Test latest version when no migrations"""
        registry = MigrationRegistry()
        registry._discovered = True

        latest = registry.get_latest_version()

        assert latest == 0

    def test_has_migrations_true(self):
        """Test has_migrations returns True"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry._discovered = True

        result = registry.has_migrations()

        assert result is True

    def test_has_migrations_false(self):
        """Test has_migrations returns False"""
        registry = MigrationRegistry()
        registry._discovered = True

        result = registry.has_migrations()

        assert result is False

    def test_get_migration_count(self):
        """Test getting migration count"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry._discovered = True

        count = registry.get_migration_count()

        assert count == 2

    def test_get_migration_count_empty(self):
        """Test count when empty"""
        registry = MigrationRegistry()
        registry._discovered = True

        count = registry.get_migration_count()

        assert count == 0

    def test_clear(self):
        """Test clearing registry"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry._discovered = True

        registry.clear()

        assert len(registry._migrations) == 0
        assert registry._discovered is False

    def test_repr(self):
        """Test string representation"""
        registry = MigrationRegistry()
        registry.register(TestMigration1())
        registry.register(TestMigration2())
        registry._discovered = True

        repr_str = repr(registry)

        assert "MigrationRegistry" in repr_str
        assert "2 migrations" in repr_str
        assert "v2" in repr_str

    def test_discover_with_modules(self):
        """Test discovery finds migrations in modules"""
        registry = MigrationRegistry(versions_package="test.migrations")

        # Mock the package structure
        mock_package = Mock()
        mock_package.__file__ = "/tmp/test/migrations/__init__.py"

        mock_module = Mock()
        mock_module.TestMigration = TestMigration1

        with patch('importlib.import_module') as mock_import:
            def import_side_effect(name):
                if name == "test.migrations":
                    return mock_package
                elif name.endswith(".v1"):
                    return mock_module
                raise ImportError()

            mock_import.side_effect = import_side_effect

            with patch('pathlib.Path.glob', return_value=[Path("/tmp/test/migrations/v1.py")]):
                with patch('inspect.getmembers', return_value=[("TestMigration", TestMigration1)]):
                    with patch('inspect.isclass', return_value=True):
                        registry.discover()

        assert registry._discovered is True
