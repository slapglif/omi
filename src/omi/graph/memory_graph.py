"""Graph-based memory storage with topology verification.

Based on SandyBlake's memory-palace architecture.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryNode:
    """A memory node in the graph."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    memory_type: str = "fact"  # fact | experience | belief | decision
    confidence: Optional[float] = None
    hash: str = ""
    instance_ids: List[str] = None


class MemoryGraph:
    """SQLite-backed graph storage with embeddings and topology verification."""
    
    def __init__(self, db_path: Path, embeddings_dim: int = 768):
        self.db_path = Path(db_path)
        self.embeddings_dim = embeddings_dim
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB,  -- JSON array of floats
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    memory_type TEXT NOT NULL,
                    confidence REAL,
                    hash TEXT NOT NULL,
                    instance_ids TEXT  -- JSON array
                );
                
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES memories(id),
                    FOREIGN KEY (target_id) REFERENCES memories(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
                CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            """)
    
    def store(self, node: MemoryNode) -> str:
        """Store a memory node. Returns node ID."""
        node.created_at = node.created_at or datetime.utcnow().isoformat()
        node.last_accessed = node.last_accessed or node.created_at
        node.hash = hashlib.sha256(node.content.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, embedding, created_at, last_accessed, access_count, 
                 memory_type, confidence, hash, instance_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id, node.content, 
                json.dumps(node.embedding) if node.embedding else None,
                node.created_at, node.last_accessed, node.access_count,
                node.memory_type, node.confidence, node.hash,
                json.dumps(node.instance_ids or [])
            ))
        return node.id
    
    def semantic_search(self, query_embedding: List[float], 
                       limit: int = 10,
                       min_similarity: float = 0.7) -> List[Tuple[MemoryNode, float]]:
        """Search memories by embedding similarity."""
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT id, content, embedding, created_at, last_accessed, access_count, "
                "memory_type, confidence, hash, instance_ids FROM memories WHERE embedding IS NOT NULL"
            )
            
            for row in cursor:
                embedding = json.loads(row[2])
                similarity = self._cosine_similarity(query_embedding, embedding)
                if similarity >= min_similarity:
                    node = MemoryNode(
                        id=row[0], content=row[1], embedding=embedding,
                        created_at=row[3], last_accessed=row[4], access_count=row[5],
                        memory_type=row[6], confidence=row[7], hash=row[8],
                        instance_ids=json.loads(row[9])
                    )
                    results.append((node, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def get_centrality(self, node_id: str) -> float:
        """Calculate PageRank-like centrality for a memory."""
        # Simplified: access_count + edge_count
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT access_count FROM memories WHERE id = ?", (node_id,)
            )
            access = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            cursor = conn.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
            edges = cursor.fetchone()[0]
            
            return (access * 0.7) + (edges * 0.3)
    
    def verify_topology(self) -> Dict[str, List[str]]:
        """Verify graph topology - return anomalies."""
        anomalies = {
            "orphan_nodes": [],
            "sudden_cores": [],
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Find orphan nodes (no edges, type="foundational")
            cursor = conn.execute("""
                SELECT m.id FROM memories m
                LEFT JOIN edges e ON m.id = e.source_id OR m.id = e.target_id
                WHERE e.id IS NULL AND m.memory_type = 'foundational'
            """)
            anomalies["orphan_nodes"] = [row[0] for row in cursor]
            
            # Find sudden core memories (claimed foundational but low access)
            cursor = conn.execute("""
                SELECT id FROM memories
                WHERE memory_type = 'foundational' 
                AND access_count < 5
                AND (julianday('now') - julianday(created_at)) < 1
            """)
            anomalies["sudden_cores"] = [row[0] for row in cursor]
        
        return anomalies
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
