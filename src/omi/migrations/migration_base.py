"""
Migration Base Class

Abstract base class for database migrations. All migrations must inherit from
this class and implement the up() and down() methods.

Pattern:
- Each migration has a unique version number
- up() applies the migration forward
- down() rolls back the migration
- Migrations are applied in version order
"""

import sqlite3
from abc import ABC, abstractmethod
from typing import Optional


class MigrationBase(ABC):
    """
    Abstract base class for database migrations.

    All migrations must inherit from this class and implement:
    - version: Unique integer version number (e.g., 1, 2, 3)
    - description: Human-readable description of the migration
    - up(): Method to apply the migration
    - down(): Method to roll back the migration

    Example:
        class AddUserTable(MigrationBase):
            version = 1
            description = "Add users table"

            def up(self, conn: sqlite3.Connection) -> None:
                conn.execute('''
                    CREATE TABLE users (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL
                    )
                ''')

            def down(self, conn: sqlite3.Connection) -> None:
                conn.execute('DROP TABLE users')
    """

    # Subclasses must define these
    version: int
    description: str

    def __init__(self):
        """Initialize migration and validate required attributes."""
        if not hasattr(self, 'version') or not isinstance(self.version, int):
            raise ValueError(
                f"{self.__class__.__name__} must define 'version' as an integer"
            )
        if not hasattr(self, 'description') or not isinstance(self.description, str):
            raise ValueError(
                f"{self.__class__.__name__} must define 'description' as a string"
            )
        if self.version < 1:
            raise ValueError(
                f"Migration version must be >= 1, got {self.version}"
            )

    @abstractmethod
    def up(self, conn: sqlite3.Connection) -> None:
        """
        Apply the migration forward.

        Args:
            conn: SQLite database connection with autocommit disabled.
                  The caller handles transaction commit/rollback.

        Raises:
            Exception: If migration fails. Will trigger automatic rollback.
        """
        pass

    @abstractmethod
    def down(self, conn: sqlite3.Connection) -> None:
        """
        Roll back the migration.

        Args:
            conn: SQLite database connection with autocommit disabled.
                  The caller handles transaction commit/rollback.

        Raises:
            Exception: If rollback fails. Manual intervention may be required.
        """
        pass

    def validate(self) -> None:
        """
        Optional validation before migration runs.

        Override this to add pre-migration validation checks.
        Called before up() runs during migration.

        Raises:
            Exception: If validation fails, migration will not run.
        """
        pass

    def __repr__(self) -> str:
        """String representation for logging."""
        return f"<Migration v{self.version}: {self.description}>"

    def __eq__(self, other) -> bool:
        """Compare migrations by version."""
        if not isinstance(other, MigrationBase):
            return False
        return self.version == other.version

    def __lt__(self, other) -> bool:
        """Order migrations by version."""
        if not isinstance(other, MigrationBase):
            return NotImplemented
        return self.version < other.version

    def __hash__(self) -> int:
        """Hash by version for use in sets/dicts."""
        return hash(self.version)
