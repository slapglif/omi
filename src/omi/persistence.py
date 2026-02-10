"""
Tiered persistence layer: NOW.md / Daily Logs / Graph Palace / Vault
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class NOWEntry:
    """Hot context - <1k tokens, loaded first on session start"""
    current_task: str
    recent_completions: List[str]
    pending_decisions: List[str]
    key_files: List[str]
    timestamp: datetime
    
    def to_markdown(self) -> str:
        """Serialize to NOW.md format"""
        lines = [
            f"# NOW - {self.timestamp.isoformat()}",
            "",
            "## Current Task",
            self.current_task,
            "",
            "## Recent Completions",
        ]
        for item in self.recent_completions:
            lines.append(f"- [x] {item}")
        lines.extend(["", "## Pending Decisions"])
        for item in self.pending_decisions:
            lines.append(f"- [ ] {item}")
        lines.extend(["", "## Key Files"])
        for item in self.key_files:
            lines.append(f"- `{item}`")
        return "\n".join(lines)
    
    @classmethod
    def from_markdown(cls, content: str) -> "NOWEntry":
        """Parse from NOW.md format"""
        current_task = ""
        recent_completions: List[str] = []
        pending_decisions: List[str] = []
        key_files: List[str] = []
        timestamp = datetime.now()

        # Extract timestamp from header line: # NOW - 2024-01-01T00:00:00
        lines = content.split("\n")
        section = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# NOW - "):
                ts_str = stripped.replace("# NOW - ", "").strip()
                try:
                    timestamp = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    pass
            elif stripped == "## Current Task":
                section = "task"
            elif stripped == "## Recent Completions":
                section = "completions"
            elif stripped == "## Pending Decisions":
                section = "pending"
            elif stripped == "## Key Files":
                section = "files"
            elif stripped.startswith("## "):
                section = None
            elif stripped and section == "task":
                current_task = stripped
            elif stripped.startswith("- [x] ") and section == "completions":
                recent_completions.append(stripped[6:])
            elif stripped.startswith("- [ ] ") and section == "pending":
                pending_decisions.append(stripped[6:])
            elif stripped.startswith("- `") and stripped.endswith("`") and section == "files":
                key_files.append(stripped[3:-1])

        return cls(
            current_task=current_task,
            recent_completions=recent_completions,
            pending_decisions=pending_decisions,
            key_files=key_files,
            timestamp=timestamp,
        )


class NOWStore:
    """
    Tier 1: Hot context storage
    
    Pattern: Read FIRST on session start, update after major context shifts
    Trigger: 70% context threshold, task completion, session end
    """
    
    def __init__(self, base_path: Path):
        self.now_path = base_path / "NOW.md"
        self.base_path = base_path
    
    def read(self) -> Optional[NOWEntry]:
        """Load hot context"""
        if not self.now_path.exists():
            return None
        content = self.now_path.read_text()
        return NOWEntry.from_markdown(content)
    
    def write(self, entry: NOWEntry) -> None:
        """Update hot context"""
        content = entry.to_markdown()
        self.now_path.write_text(content)
        
        # Also update hash for integrity checking
        self._update_hash(content)
    
    def _update_hash(self, content: str) -> None:
        """Track hash for tamper detection"""
        hash_path = self.base_path / ".now.hash"
        hash_value = hashlib.sha256(content.encode()).hexdigest()
        hash_path.write_text(hash_value)
    
    def check_integrity(self) -> bool:
        """Verify NOW.md hasn't been tampered"""
        if not self.now_path.exists():
            return True  # No file = nothing to tamper
        
        content = self.now_path.read_text()
        current_hash = hashlib.sha256(content.encode()).hexdigest()
        
        hash_path = self.base_path / ".now.hash"
        if not hash_path.exists():
            return False  # No stored hash = can't verify
        
        stored_hash = hash_path.read_text().strip()
        return current_hash == stored_hash


class DailyLogStore:
    """
    Tier 2: Daily logs - raw timeline
    
    Pattern: Append-only, chronological, human-readable
    Lifetime: Weeks (deprioritized in retrieval)
    """
    
    def __init__(self, base_path: Path):
        self.log_path = base_path / "memory"
        self.log_path.mkdir(exist_ok=True)
    
    def append(self, content: str) -> Path:
        """Append entry to today's log"""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self.log_path / f"{today}.md"
        
        timestamp = datetime.now().isoformat()
        entry = f"\n\n## [{timestamp}]\n\n{content}\n"
        
        with open(file_path, "a") as f:
            f.write(entry)
        
        return file_path
    
    def read_daily(self, date: Optional[datetime] = None) -> str:
        """Read specific day's log"""
        if date is None:
            date = datetime.now()
        
        file_path = self.log_path / f"{date.strftime('%Y-%m-%d')}.md"
        if file_path.exists():
            return file_path.read_text()
        return ""
    
    def list_days(self, days: int = 30) -> List[Path]:
        """List recent daily log files, sorted newest first"""
        cutoff = datetime.now() - timedelta(days=days)
        results = []
        for p in sorted(self.log_path.glob("*.md"), reverse=True):
            # Parse date from filename (YYYY-MM-DD.md)
            try:
                file_date = datetime.strptime(p.stem, "%Y-%m-%d")
                if file_date >= cutoff:
                    results.append(p)
            except ValueError:
                continue
        return results


class GraphPalace:
    """
    Tier 3: Graph Palace - semantic memories, relationships, beliefs

    Pattern: Structured, queryable, centrality-weighted
    Lifetime: Indefinite (with decay)

    This is a minimal stub implementation. The full implementation lives in
    omi.storage.graph_palace.GraphPalace with FTS5, vector search, and more.
    """

    def __init__(self, db_path: Path):
        import sqlite3
        import uuid
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create minimal schema for stub persistence."""
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                confidence REAL,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    def store_memory(self, content: str, memory_type: str,
                     confidence: Optional[float] = None) -> str:
        """
        Store memory with embedding

        Args:
            content: Memory text
            memory_type: 'fact' | 'experience' | 'belief' | 'decision'
            confidence: For beliefs only, 0.0-1.0

        Returns:
            memory_id: UUID for created memory
        """
        import uuid
        memory_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        self._conn.execute(
            """INSERT INTO memories (id, content, memory_type, confidence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (memory_id, content, memory_type, confidence, now, now)
        )
        self._conn.commit()
        return memory_id

    def recall(self, query: str, limit: int = 10,
               min_relevance: float = 0.7) -> List[Dict[str, Any]]:
        """
        Semantic search with recency weighting

        Recency formula: exp(-days_ago / half_life)
        Default half_life: 30 days

        Note: Stub returns simple LIKE match. Full implementation uses vector similarity.
        """
        cursor = self._conn.execute(
            """SELECT id, content, memory_type, confidence, created_at
               FROM memories
               WHERE content LIKE ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (f"%{query}%", limit)
        )
        return [dict(row) for row in cursor]

    def create_edge(self, source_id: str, target_id: str,
                   edge_type: str, strength: float) -> None:
        """Create relationship between memories"""
        self._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, strength)
               VALUES (?, ?, ?, ?)""",
            (source_id, target_id, edge_type, strength)
        )
        self._conn.commit()

    def get_centrality(self, memory_id: str) -> float:
        """
        Calculate centrality score for a memory

        Used for: hub detection, poisoning resistance
        """
        cursor = self._conn.execute(
            """SELECT COUNT(*) FROM edges
               WHERE source_id = ? OR target_id = ?""",
            (memory_id, memory_id)
        )
        count = cursor.fetchone()[0]
        return float(count)

    def get_belief(self, belief_id: str) -> Dict[str, Any]:
        """Retrieve a belief memory by ID."""
        cursor = self._conn.execute(
            """SELECT id, content, memory_type, confidence, created_at
               FROM memories WHERE id = ?""",
            (belief_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {'confidence': 0.5}

    def update_belief_confidence(self, belief_id: str, new_confidence: float) -> None:
        """Update the confidence value of a belief."""
        now = datetime.now().isoformat()
        self._conn.execute(
            """UPDATE memories SET confidence = ?, updated_at = ?
               WHERE id = ?""",
            (new_confidence, now, belief_id)
        )
        self._conn.commit()

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory by ID."""
        cursor = self._conn.execute(
            """SELECT id, content, memory_type, confidence, created_at
               FROM memories WHERE id = ?""",
            (memory_id,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_edges(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get all edges connected to a memory."""
        cursor = self._conn.execute(
            """SELECT source_id, target_id, edge_type, strength, created_at
               FROM edges
               WHERE source_id = ? OR target_id = ?""",
            (memory_id, memory_id)
        )
        results = []
        for row in cursor:
            results.append({
                'source_id': row[0],
                'target_id': row[1],
                'target_type': 'memory',
                'edge_type': row[2],
                'strength': row[3],
                'timestamp': datetime.fromisoformat(row[4]) if row[4] else datetime.now()
            })
        return results

    def full_text_search(self, query: str, limit: int = 10) -> List[Any]:
        """Simple text search. Returns list of dict-like objects with attribute access."""
        cursor = self._conn.execute(
            """SELECT id, content, memory_type, confidence, created_at
               FROM memories
               WHERE content LIKE ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (f"%{query}%", limit)
        )
        results = []
        for row in cursor:
            # Return simple namespace objects for attribute access compatibility
            from types import SimpleNamespace
            results.append(SimpleNamespace(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                confidence=row[3],
                created_at=row[4]
            ))
        return results

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class VaultBackup:
    """
    Tier 4: Full snapshots for disaster recovery
    
    Pattern: POST to molt-vault.com, base64 encode
    Frequency: Daily or session-end
    """
    
    VAULT_API = "https://molt-vault.com/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def backup(self, memory_content: str) -> str:
        """
        Full backup to MoltVault
        
        Returns:
            backup_id: Identifier for restore
        """
        import base64
        
        encoded = base64.b64encode(memory_content.encode()).decode()
        
        # TODO: POST to vault API
        # response = requests.post(f"{self.VAULT_API}/vault/backup", ...)
        
        return "backup_id_placeholder"
    
    def restore(self, backup_id: str) -> str:
        """Restore from vault"""
        # TODO: POST to vault API
        return ""
