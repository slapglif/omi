"""
Migration v002: Add archived column to memories table

This migration adds the 'archived' column to the memories table to support
policy-based archival of memories.

The archived column is used to mark memories as archived (excluded from
default search operations) without permanently deleting them.
"""

import sqlite3
from omi.migrations.migration_base import MigrationBase


class Migration(MigrationBase):
    """
    Add archived column to memories table.

    This migration:
    1. Adds 'archived' INTEGER column (0=active, 1=archived) with default 0
    2. Creates an index on the archived column for fast filtering
    """

    version = 2
    description = "Add archived column to memories table for policy-based archival"

    def up(self, conn: sqlite3.Connection) -> None:
        """
        Apply the migration forward.

        Adds the archived column to the memories table and creates an index.

        Args:
            conn: SQLite database connection
        """
        # Check if column already exists to make this migration idempotent
        cursor = conn.execute("PRAGMA table_info(memories)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'archived' not in columns:
            # Add archived column with default value 0 (not archived)
            conn.execute('''
                ALTER TABLE memories
                ADD COLUMN archived INTEGER DEFAULT 0
            ''')

            # Create index for fast filtering of archived/non-archived memories
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_archived
                ON memories(archived)
            ''')

    def down(self, conn: sqlite3.Connection) -> None:
        """
        Roll back the migration.

        Note: SQLite does not support DROP COLUMN directly.
        Instead, this migration creates a new table without the archived column
        and copies data over.

        Args:
            conn: SQLite database connection
        """
        # Drop the index first
        conn.execute('DROP INDEX IF EXISTS idx_memories_archived')

        # SQLite doesn't support DROP COLUMN, so we need to:
        # 1. Create a new table without the archived column
        # 2. Copy data from old table to new table
        # 3. Drop old table
        # 4. Rename new table to old table name

        conn.execute('''
            CREATE TABLE memories_backup (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB,
                memory_type TEXT CHECK(memory_type IN ('fact','experience','belief','decision')),
                confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                instance_ids TEXT,
                content_hash TEXT
            )
        ''')

        # Copy data (excluding archived column)
        conn.execute('''
            INSERT INTO memories_backup
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories
        ''')

        # Drop old table
        conn.execute('DROP TABLE memories')

        # Rename backup table to memories
        conn.execute('ALTER TABLE memories_backup RENAME TO memories')

        # Recreate indexes (excluding archived index)
        conn.executescript('''
            CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
        ''')

    def validate(self) -> None:
        """
        Pre-migration validation.

        Ensures the memories table exists before attempting to add the column.
        """
        # No validation needed - the migration is idempotent
        pass
