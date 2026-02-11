"""
Example Migration v001

This is an example migration that demonstrates the migration pattern.
Use this as a template for creating new migrations.

Pattern:
1. Create a class named 'Migration' that inherits from MigrationBase
2. Define version number (unique, sequential)
3. Define description (clear, concise)
4. Implement up() for forward migration
5. Implement down() for rollback
6. (Optional) Implement validate() for pre-migration checks
"""

import sqlite3
from omi.migrations.migration_base import MigrationBase


class Migration(MigrationBase):
    """
    Example migration demonstrating the pattern.

    This migration adds an example_tags table to demonstrate
    how to add a new table with proper schema.
    """

    # Required: Unique version number (sequential from 1)
    version = 1

    # Required: Human-readable description
    description = "Add example_tags table to demonstrate migration pattern"

    def up(self, conn: sqlite3.Connection) -> None:
        """
        Apply the migration forward.

        This example creates a simple tags table with:
        - id: Primary key
        - name: Tag name (unique)
        - created_at: Timestamp

        Args:
            conn: SQLite database connection
        """
        conn.execute('''
            CREATE TABLE IF NOT EXISTS example_tags (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create index for performance
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_example_tags_name
            ON example_tags(name)
        ''')

    def down(self, conn: sqlite3.Connection) -> None:
        """
        Roll back the migration.

        This example drops the tags table and its index.

        Args:
            conn: SQLite database connection
        """
        # Drop index first
        conn.execute('DROP INDEX IF EXISTS idx_example_tags_name')

        # Then drop table
        conn.execute('DROP TABLE IF EXISTS example_tags')

    def validate(self) -> None:
        """
        Optional: Pre-migration validation.

        This example doesn't need validation, but you can use this
        to check preconditions before running the migration.

        Example validations:
        - Check if required data exists
        - Verify compatible schema state
        - Validate configuration
        """
        pass
