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

    conn.commit()
