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

    Local filesystem backup to ~/.openclaw/omi/vault/ as .tar.gz archives.
    Each backup is a timestamped archive containing critical OMI data files.
    """

    def __init__(self, api_key: Optional[str] = None,
                 base_path: Optional[Path] = None):
        """
        Args:
            api_key: Unused, kept for interface compatibility
            base_path: OMI data directory (default: ~/.openclaw/omi)
        """
        self.api_key = api_key
        self.base_path = Path(base_path) if base_path else Path.home() / ".openclaw" / "omi"
        self.vault_dir = self.base_path / "vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def backup(self, memory_content: str) -> str:
        """
        Create a local .tar.gz backup of OMI data.

        The archive contains palace.sqlite, NOW.md, config.yaml, MEMORY.md,
        and all daily log files. The memory_content argument is written to a
        snapshot.txt file inside the archive for session-level context.

        Args:
            memory_content: Additional text content to include in the backup

        Returns:
            backup_id: Timestamped identifier for the created backup
        """
        import tarfile
        import tempfile
        import uuid

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"omi_backup_{timestamp}_{uuid.uuid4().hex[:8]}"
        archive_path = self.vault_dir / f"{backup_id}.tar.gz"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the session snapshot
            snapshot_path = Path(tmpdir) / "snapshot.txt"
            snapshot_path.write_text(memory_content)

            with tarfile.open(archive_path, "w:gz") as tar:
                # Include the session snapshot
                tar.add(snapshot_path, arcname="snapshot.txt")

                # Include critical OMI files
                critical_files = [
                    self.base_path / "palace.sqlite",
                    self.base_path / "NOW.md",
                    self.base_path / "config.yaml",
                    self.base_path / "MEMORY.md",
                ]
                for file_path in critical_files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.name)

                # Include daily logs
                memory_dir = self.base_path / "memory"
                if memory_dir.exists() and memory_dir.is_dir():
                    for log_file in memory_dir.glob("*.md"):
                        tar.add(log_file, arcname=f"memory/{log_file.name}")

        # Write metadata alongside the archive
        metadata = {
            "backup_id": backup_id,
            "created_at": datetime.now().isoformat(),
            "archive_path": str(archive_path),
            "checksum": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        }
        meta_path = self.vault_dir / f"{backup_id}.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        return backup_id

    def restore(self, backup_id: str) -> str:
        """
        Restore from a local vault backup.

        Extracts the archive back into the OMI base_path, overwriting
        existing files. Returns the snapshot.txt content that was saved
        during the backup.

        Args:
            backup_id: Identifier returned by backup()

        Returns:
            Content of the snapshot.txt from the backup, or empty string
            if not found.
        """
        import tarfile

        archive_path = self.vault_dir / f"{backup_id}.tar.gz"
        if not archive_path.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")

        # Verify integrity via metadata
        meta_path = self.vault_dir / f"{backup_id}.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            expected_checksum = metadata.get("checksum", "")
            actual_checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if expected_checksum and actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {backup_id}: "
                    f"expected {expected_checksum}, got {actual_checksum}"
                )

        snapshot_content = ""
        with tarfile.open(archive_path, "r:gz") as tar:
            # Security: validate all members to prevent path traversal
            for member in tar.getmembers():
                member_path = self.base_path / member.name
                try:
                    member_path.resolve().relative_to(self.base_path.resolve())
                except ValueError:
                    raise ValueError(
                        f"Path traversal detected in archive member: {member.name}"
                    )

            # Extract snapshot.txt content before overwriting
            try:
                snapshot_file = tar.extractfile("snapshot.txt")
                if snapshot_file:
                    snapshot_content = snapshot_file.read().decode("utf-8")
            except (KeyError, AttributeError):
                snapshot_content = ""

            # Extract everything except snapshot.txt to base_path
            for member in tar.getmembers():
                if member.name == "snapshot.txt":
                    continue
                tar.extract(member, path=self.base_path)

        return snapshot_content

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all local vault backups.

        Returns:
            List of backup metadata dicts, sorted newest first.
        """
        backups = []
        for meta_file in sorted(self.vault_dir.glob("*.json"), reverse=True):
            try:
                metadata = json.loads(meta_file.read_text())
                backups.append(metadata)
            except (json.JSONDecodeError, OSError):
                continue
        return backups
