"""
Core persistence helpers: NOWEntry and DailyLogStore

This module contains lightweight persistence classes without dedicated submodules:
- NOWEntry: Dataclass for hot context (<1k tokens, loaded on session start)
- DailyLogStore: Append-only daily log manager

Other persistence components (NowStorage, GraphPalace, MoltVault) are in storage/ modules.
"""

import json
import tarfile
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


# Backward compatibility imports - preserve old import paths
# GraphPalace moved to storage.graph_palace in v0.3.0
from .storage.graph_palace import GraphPalace  # noqa: F401


# NOWStore is now a compatibility wrapper for NowStorage
class NOWStore:
    """Backward compatibility wrapper for NowStorage.

    Provides the old NOWStore API (read/write with NOWEntry objects)
    while delegating to the new NowStorage implementation.

    DEPRECATED: Use NowStorage directly instead.
    """
    def __init__(self, base_path: str) -> None:
        from .storage.now import NowStorage
        self.base_path = Path(base_path)
        self._storage = NowStorage(self.base_path)
        self.now_path = self._storage.now_file
        self.hash_file = self.base_path / ".now.hash"

    def read(self) -> Optional["NOWEntry"]:
        """Read NOW.md and return NOWEntry object or None."""
        content = self._storage.read()
        if not content or "No active task" in content:
            return None
        try:
            return NOWEntry.from_markdown(content)
        except Exception:
            return None

    def write(self, entry: "NOWEntry") -> None:
        """Write a NOWEntry object to NOW.md."""
        if not isinstance(entry, NOWEntry):
            raise TypeError(f"Expected NOWEntry, got {type(entry)}")
        self._storage.update(
            current_task=entry.current_task,
            recent_completions=entry.recent_completions,
            pending_decisions=entry.pending_decisions,
            key_files=entry.key_files
        )
        # Create hash file for integrity checking
        import hashlib
        current_content = self._storage.now_file.read_text()
        hash_value = hashlib.sha256(current_content.encode()).hexdigest()
        self.hash_file.write_text(hash_value)

    def check_integrity(self) -> bool:
        """Check if NOW.md hash matches stored hash."""
        import hashlib
        if not self.hash_file.exists():
            return False
        if not self._storage.now_file.exists():
            return False

        content = self._storage.now_file.read_text()
        current_hash = hashlib.sha256(content.encode()).hexdigest()
        stored_hash = self.hash_file.read_text().strip()
        return current_hash == stored_hash


# VaultBackup is now MoltVault in moltvault.py
# But keep VaultBackup class for backward compatibility
class VaultBackup:
    """Local filesystem vault backup.

    Stores archives in <base_path>/vault/ without requiring cloud credentials.
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.vault_dir = self.base_path / "vault"
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def backup(self, content: str) -> str:
        """Create a local tar.gz backup archive and return the backup_id."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"omi_backup_{timestamp}"

        archive_path = self.vault_dir / f"{backup_id}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            for file_path in self.base_path.iterdir():
                if file_path.is_file():
                    try:
                        tar.add(file_path, arcname=file_path.name)
                    except (OSError, tarfile.TarError):
                        pass

        checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        meta = {
            "backup_id": backup_id,
            "created_at": datetime.now().isoformat(),
            "content": content,
            "checksum": checksum,
        }
        (self.vault_dir / f"{backup_id}.json").write_text(json.dumps(meta, indent=2))

        return backup_id

    def restore(self, backup_id: str) -> str:
        """Restore files from a backup archive and return the stored content string."""
        archive_path = self.vault_dir / f"{backup_id}.tar.gz"
        meta_path = self.vault_dir / f"{backup_id}.json"

        if not archive_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_id}")

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.base_path)

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            return meta.get("content", "")
        return ""

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups sorted newest first."""
        backups = []
        for meta_file in self.vault_dir.glob("*.json"):
            try:
                meta = json.loads(meta_file.read_text())
                backups.append(meta)
            except (json.JSONDecodeError, OSError):
                pass
        backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return backups
