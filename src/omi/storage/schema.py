"""
Database schema management for Graph Palace.

This module handles:
- Table creation (memories, edges)
- Index creation for performance
- FTS5 virtual table setup
- WAL mode configuration
- Foreign key constraints
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

        CREATE TABLE IF NOT EXISTS shared_namespaces (
            namespace TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT NOT NULL,
            metadata TEXT  -- JSON for configuration and settings
        );

        CREATE TABLE IF NOT EXISTS namespace_permissions (
            namespace TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            permission_level TEXT CHECK(permission_level IN ('read','write','admin')) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (namespace, agent_id),
            FOREIGN KEY (namespace) REFERENCES shared_namespaces(namespace) ON DELETE CASCADE
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
        CREATE INDEX IF NOT EXISTS idx_shared_namespaces_created_by ON shared_namespaces(created_by);
        CREATE INDEX IF NOT EXISTS idx_shared_namespaces_created_at ON shared_namespaces(created_at);
        CREATE INDEX IF NOT EXISTS idx_namespace_permissions_agent_id ON namespace_permissions(agent_id);
        CREATE INDEX IF NOT EXISTS idx_namespace_permissions_namespace ON namespace_permissions(namespace);
        CREATE INDEX IF NOT EXISTS idx_namespace_permissions_level ON namespace_permissions(permission_level);
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

    conn.commit()
