"""
Migration Registry for OMI

Discovers and registers available database migrations.

Features:
- Auto-discovery of migrations from versions directory
- Migration version validation (no gaps, duplicates)
- Ordered migration sequencing
- Query available and pending migrations
"""

import importlib
import inspect
from pathlib import Path
from typing import List, Optional, Dict, Type
from .migration_base import MigrationBase


class MigrationRegistry:
    """
    Migration Registry - Discovers and manages available migrations

    Pattern: Auto-discovery from versions directory with validation
    Lifetime: Created on-demand for migration operations

    Features:
    - Discovers migrations from omi.migrations.versions package
    - Validates migration version sequence (no gaps or duplicates)
    - Provides ordered list of migrations
    - Filters pending migrations based on current schema version
    - Supports manual registration for testing

    Example:
        registry = MigrationRegistry()
        registry.discover()
        pending = registry.get_pending_migrations(current_version=5)
        for migration in pending:
            print(f"Apply {migration}")
    """

    def __init__(self, versions_package: str = "omi.migrations.versions"):
        """
        Initialize Migration Registry.

        Args:
            versions_package: Python package containing migration modules
                            (default: "omi.migrations.versions")
        """
        self.versions_package = versions_package
        self._migrations: Dict[int, MigrationBase] = {}
        self._discovered = False

    def discover(self) -> None:
        """
        Discover and register all migrations from versions package.

        Scans the versions package for Python modules, imports them, and
        registers any MigrationBase subclasses found.

        Validates:
        - All migrations inherit from MigrationBase
        - No duplicate version numbers
        - Version sequence is valid

        Raises:
            ImportError: If versions package cannot be imported
            ValueError: If migration validation fails
        """
        if self._discovered:
            return  # Already discovered

        # Try to import the versions package
        try:
            package = importlib.import_module(self.versions_package)
        except ImportError:
            # Package doesn't exist yet (no migrations created)
            self._discovered = True
            return

        # Get package directory path
        package_path = Path(package.__file__).parent

        # Find all Python modules in the package
        for module_file in sorted(package_path.glob("*.py")):
            # Skip __init__.py and non-migration files
            if module_file.name.startswith("_"):
                continue

            # Import the module
            module_name = f"{self.versions_package}.{module_file.stem}"
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            # Find MigrationBase subclasses in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Check if it's a MigrationBase subclass (but not MigrationBase itself)
                if (issubclass(obj, MigrationBase) and
                    obj is not MigrationBase and
                    obj.__module__ == module_name):
                    # Instantiate and register the migration
                    try:
                        migration = obj()
                        self.register(migration)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to instantiate migration {name} in {module_name}: {e}"
                        )

        # Validate the migration sequence
        self._validate_sequence()
        self._discovered = True

    def register(self, migration: MigrationBase) -> None:
        """
        Manually register a migration.

        Useful for testing or programmatic migration registration.

        Args:
            migration: Migration instance to register

        Raises:
            ValueError: If migration version already registered
        """
        if migration.version in self._migrations:
            existing = self._migrations[migration.version]
            raise ValueError(
                f"Duplicate migration version {migration.version}: "
                f"{migration} conflicts with {existing}"
            )

        self._migrations[migration.version] = migration

    def _validate_sequence(self) -> None:
        """
        Validate migration version sequence.

        Ensures migrations form a valid sequence with no gaps.
        Version numbers must start at 1 and be consecutive.

        Raises:
            ValueError: If sequence is invalid
        """
        if not self._migrations:
            return  # No migrations to validate

        versions = sorted(self._migrations.keys())

        # Check that versions start at 1
        if versions[0] != 1:
            raise ValueError(
                f"Migration versions must start at 1, found {versions[0]}"
            )

        # Check for gaps in sequence
        for i, version in enumerate(versions, start=1):
            if version != i:
                raise ValueError(
                    f"Migration version gap detected: expected {i}, found {version}"
                )

    def get_migration(self, version: int) -> Optional[MigrationBase]:
        """
        Get a specific migration by version.

        Args:
            version: Migration version number

        Returns:
            Migration instance or None if not found
        """
        if not self._discovered:
            self.discover()

        return self._migrations.get(version)

    def get_all_migrations(self) -> List[MigrationBase]:
        """
        Get all registered migrations in version order.

        Returns:
            List of migrations sorted by version (ascending)
        """
        if not self._discovered:
            self.discover()

        versions = sorted(self._migrations.keys())
        return [self._migrations[v] for v in versions]

    def get_pending_migrations(self, current_version: int) -> List[MigrationBase]:
        """
        Get migrations that need to be applied.

        Returns all migrations with version > current_version,
        ordered by version (ascending).

        Args:
            current_version: Current schema version from database

        Returns:
            List of pending migrations in order to apply
        """
        if not self._discovered:
            self.discover()

        pending = []
        for version in sorted(self._migrations.keys()):
            if version > current_version:
                pending.append(self._migrations[version])

        return pending

    def get_latest_version(self) -> int:
        """
        Get the highest migration version available.

        Returns:
            Latest migration version, or 0 if no migrations
        """
        if not self._discovered:
            self.discover()

        if not self._migrations:
            return 0

        return max(self._migrations.keys())

    def has_migrations(self) -> bool:
        """
        Check if any migrations are registered.

        Returns:
            True if migrations exist, False otherwise
        """
        if not self._discovered:
            self.discover()

        return len(self._migrations) > 0

    def get_migration_count(self) -> int:
        """
        Get total number of registered migrations.

        Returns:
            Count of registered migrations
        """
        if not self._discovered:
            self.discover()

        return len(self._migrations)

    def clear(self) -> None:
        """
        Clear all registered migrations.

        Useful for testing or resetting the registry.
        """
        self._migrations.clear()
        self._discovered = False

    def __repr__(self) -> str:
        """String representation for debugging."""
        count = len(self._migrations)
        latest = self.get_latest_version()
        return f"<MigrationRegistry: {count} migrations, latest v{latest}>"
