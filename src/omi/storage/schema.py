"""
Database schema management for Graph Palace.

This module handles:
- Table creation (memories, edges, users, roles, permissions, api_keys, audit_log)
- Index creation for performance
- FTS5 virtual table setup
- WAL mode configuration
- Foreign key constraints
- Multi-user access control (RBAC)
"""

import sqlite3
from typing import Optional


def init_database(conn: sqlite3.Connection, enable_wal: bool = True) -> None:
    """
    Initialize database schema with indexes and FTS5.

    Args:
        conn: SQLite connection object
        enable_wal: Enable WAL mode for concurrent writes (default: True)
    """
    # Enable WAL mode for concurrent writes
    if enable_wal:
        conn.execute("PRAGMA journal_mode=WAL")

    # Foreign key constraints
    conn.execute("PRAGMA foreign_keys=ON")

    # Create memories table with vector support
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            embedding BLOB,  -- 1024-dim float32 for bge-m3
            memory_type TEXT CHECK(memory_type IN ('fact','experience','belief','decision')),
            confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            instance_ids TEXT,  -- JSON array
            content_hash TEXT  -- SHA-256 for integrity
        );

        CREATE TABLE IF NOT EXISTS edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT CHECK(edge_type IN ('SUPPORTS','CONTRADICTS','RELATED_TO','DEPENDS_ON','POSTED','DISCUSSED')),
            strength REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
        CREATE INDEX IF NOT EXISTS idx_edges_bidirectional ON edges(source_id, target_id);
    """)

    # Create standalone FTS5 virtual table for full-text search
    # Note: Using standalone FTS5 (no content= sync) because memories.id
    # is TEXT (UUID), and FTS5 content_rowid requires INTEGER.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id,
            content
        )
    """)

    # Create RBAC tables for multi-user access control
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

        -- Indexes for RBAC performance
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

    conn.commit()
