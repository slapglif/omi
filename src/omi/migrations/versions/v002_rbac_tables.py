"""
Migration v002: Add RBAC Tables

This migration adds Role-Based Access Control (RBAC) support to OMI.
It creates tables for users, roles, permissions, API keys, and audit logging.

Tables Added:
- users: User accounts with username and email
- roles: Four built-in roles (admin, developer, reader, auditor)
- user_roles: Many-to-many mapping with namespace-level permissions
- permissions: Role-based action permissions
- api_keys: API key authentication linked to users
- audit_log: Audit trail for all user operations

Pattern:
1. Create all RBAC tables with proper foreign keys
2. Add indexes for performance
3. Rollback drops all tables and indexes in reverse order
"""

import sqlite3
from omi.migrations.migration_base import MigrationBase


class Migration(MigrationBase):
    """
    Add RBAC tables for multi-user access control.

    This migration implements the database schema for role-based access control,
    enabling multi-user deployments with granular permissions.

    Features:
    - User management with email
    - Four built-in roles: admin, developer, reader, auditor
    - Namespace-level permission scoping
    - API key authentication with hashed storage
    - Complete audit trail of user operations
    """

    # Required: Unique version number (sequential from 1)
    version = 2

    # Required: Human-readable description
    description = "Add RBAC tables for multi-user access control"

    def up(self, conn: sqlite3.Connection) -> None:
        """
        Apply the RBAC migration forward.

        Creates six tables:
        1. users - User accounts
        2. roles - Role definitions with CHECK constraint
        3. user_roles - User-role assignments with namespace support
        4. permissions - Role-based permissions
        5. api_keys - API key authentication with hashed keys
        6. audit_log - Audit trail for operations

        Args:
            conn: SQLite database connection
        """
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS roles (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL CHECK(name IN ('admin','developer','reader','auditor')),
                description TEXT
            );

            CREATE TABLE IF NOT EXISTS user_roles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role_id TEXT NOT NULL,
                namespace TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS permissions (
                id TEXT PRIMARY KEY,
                role_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource TEXT NOT NULL,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                namespace TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,  -- JSON for additional context
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            );
        """)

        # Create indexes for RBAC performance
        conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id);
            CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id);
            CREATE INDEX IF NOT EXISTS idx_user_roles_namespace ON user_roles(namespace);
            CREATE INDEX IF NOT EXISTS idx_permissions_role ON permissions(role_id);
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
        """)

    def down(self, conn: sqlite3.Connection) -> None:
        """
        Roll back the RBAC migration.

        Drops all RBAC tables and indexes in reverse dependency order:
        1. Drop indexes first (for cleaner rollback)
        2. Drop tables with foreign keys (user_roles, permissions, api_keys, audit_log)
        3. Drop parent tables (users, roles)

        Args:
            conn: SQLite database connection
        """
        # Drop indexes first
        conn.executescript("""
            DROP INDEX IF EXISTS idx_audit_log_action;
            DROP INDEX IF EXISTS idx_audit_log_timestamp;
            DROP INDEX IF EXISTS idx_audit_log_user;
            DROP INDEX IF EXISTS idx_api_keys_user;
            DROP INDEX IF EXISTS idx_api_keys_hash;
            DROP INDEX IF EXISTS idx_permissions_role;
            DROP INDEX IF EXISTS idx_user_roles_namespace;
            DROP INDEX IF EXISTS idx_user_roles_role;
            DROP INDEX IF EXISTS idx_user_roles_user;
            DROP INDEX IF EXISTS idx_users_username;
        """)

        # Drop tables in reverse dependency order
        conn.executescript("""
            DROP TABLE IF EXISTS audit_log;
            DROP TABLE IF EXISTS api_keys;
            DROP TABLE IF EXISTS permissions;
            DROP TABLE IF EXISTS user_roles;
            DROP TABLE IF EXISTS roles;
            DROP TABLE IF EXISTS users;
        """)

    def validate(self) -> None:
        """
        Pre-migration validation.

        This migration doesn't require validation because:
        - It creates new tables (doesn't modify existing data)
        - All tables use IF NOT EXISTS (idempotent)
        - No data migration or transformation needed

        Future versions might validate:
        - Database is not already at version 2+
        - Foreign key constraints are enabled
        - Sufficient disk space for new tables
        """
        pass


# Expose migration instance for direct import
migrate = Migration()
