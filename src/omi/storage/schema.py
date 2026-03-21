"""
Database schema management for Graph Palace.

This module handles:
- Table creation (memories, memory_versions, edges, snapshots, snapshot_memories)
- Table creation (users, roles, permissions, api_keys, audit_log) for RBAC
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
            content_hash TEXT,  -- SHA-256 for integrity
            archived INTEGER DEFAULT 0,  -- 0=active, 1=archived (excluded from default search)
            locked INTEGER DEFAULT 0  -- 0=unlocked, 1=locked (exempt from policy actions)
        );

        CREATE TABLE IF NOT EXISTS memory_versions (
            version_id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            content TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            operation_type TEXT CHECK(operation_type IN ('CREATE','UPDATE','DELETE')) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            previous_version_id TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (previous_version_id) REFERENCES memory_versions(version_id)
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

        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT,
            metadata_json TEXT,  -- JSON metadata for snapshot
            moltvault_backup_id TEXT  -- Optional MoltVault backup reference
        );

        CREATE TABLE IF NOT EXISTS snapshot_memories (
            snapshot_id TEXT NOT NULL,
            memory_id TEXT NOT NULL,
            version_id TEXT,  -- Reference to specific version
            operation_type TEXT CHECK(operation_type IN ('ADDED','MODIFIED','DELETED')),
            PRIMARY KEY (snapshot_id, memory_id),
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (version_id) REFERENCES memory_versions(version_id) ON DELETE SET NULL
        );

        -- Indexes for performance
        CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
        CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
        CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
        CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);
        CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
        CREATE INDEX IF NOT EXISTS idx_memories_locked ON memories(locked);
        CREATE INDEX IF NOT EXISTS idx_memory_versions_memory_id ON memory_versions(memory_id);
        CREATE INDEX IF NOT EXISTS idx_memory_versions_created_at ON memory_versions(created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_versions_version_number ON memory_versions(version_number);
        CREATE INDEX IF NOT EXISTS idx_memory_versions_operation_type ON memory_versions(operation_type);
        CREATE INDEX IF NOT EXISTS idx_memory_versions_composite ON memory_versions(memory_id, version_number);
        CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
        CREATE INDEX IF NOT EXISTS idx_edges_bidirectional ON edges(source_id, target_id);
        CREATE INDEX IF NOT EXISTS idx_snapshots_created_at ON snapshots(created_at);
        CREATE INDEX IF NOT EXISTS idx_snapshots_moltvault_backup_id ON snapshots(moltvault_backup_id);
        CREATE INDEX IF NOT EXISTS idx_snapshot_memories_snapshot_id ON snapshot_memories(snapshot_id);
        CREATE INDEX IF NOT EXISTS idx_snapshot_memories_memory_id ON snapshot_memories(memory_id);
        CREATE INDEX IF NOT EXISTS idx_snapshot_memories_version_id ON snapshot_memories(version_id);
        CREATE INDEX IF NOT EXISTS idx_snapshot_memories_operation_type ON snapshot_memories(operation_type);
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

    # Migration: Add locked column if it doesn't exist (for existing databases)
    try:
        conn.execute("SELECT locked FROM memories LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE memories ADD COLUMN locked INTEGER DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_locked ON memories(locked)")

    # Create distributed sync metadata tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS instance_registry (
            instance_id TEXT PRIMARY KEY,
            hostname TEXT,
            topology_type TEXT CHECK(topology_type IN ('leader','follower','multi-leader')),
            status TEXT CHECK(status IN ('active','inactive','partitioned')) DEFAULT 'active',
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id TEXT PRIMARY KEY,
            instance_id TEXT NOT NULL,
            memory_id TEXT,
            operation TEXT CHECK(operation IN ('store','update','delete','bulk_sync')),
            status TEXT CHECK(status IN ('success','failure','pending')) DEFAULT 'pending',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (instance_id) REFERENCES instance_registry(instance_id) ON DELETE CASCADE,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS conflict_queue (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            instance_id_source TEXT NOT NULL,
            instance_id_target TEXT NOT NULL,
            conflict_data TEXT,  -- JSON with conflict details
            resolution_status TEXT CHECK(resolution_status IN ('pending','resolved','ignored')) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (instance_id_source) REFERENCES instance_registry(instance_id) ON DELETE CASCADE,
            FOREIGN KEY (instance_id_target) REFERENCES instance_registry(instance_id) ON DELETE CASCADE
        );

        -- Indexes for sync operations
        CREATE INDEX IF NOT EXISTS idx_instance_registry_status ON instance_registry(status);
        CREATE INDEX IF NOT EXISTS idx_instance_registry_last_seen ON instance_registry(last_seen);
        CREATE INDEX IF NOT EXISTS idx_sync_log_instance ON sync_log(instance_id);
        CREATE INDEX IF NOT EXISTS idx_sync_log_memory ON sync_log(memory_id);
        CREATE INDEX IF NOT EXISTS idx_sync_log_created ON sync_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_sync_log_status ON sync_log(status);
        CREATE INDEX IF NOT EXISTS idx_conflict_queue_memory ON conflict_queue(memory_id);
        CREATE INDEX IF NOT EXISTS idx_conflict_queue_status ON conflict_queue(resolution_status);
        CREATE INDEX IF NOT EXISTS idx_conflict_queue_created ON conflict_queue(created_at);
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
