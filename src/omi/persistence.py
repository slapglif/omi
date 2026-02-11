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
