"""NOW.md lifecycle management

The hot context tier - <1k tokens, loaded first, updated frequently.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
import json


class NowStorage:
    """Manages the NOW.md hot context file."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.now_file = self.base_path / "NOW.md"
        self.state_file = self.base_path / "heartbeat-state.json"
        
    def read(self) -> str:
        """Read NOW.md content. Returns default if not exists."""
        if not self.now_file.exists():
            return self._default_content()
        return self.now_file.read_text()
    
    def update(self, 
               current_task: Optional[str] = None,
               recent_completions: Optional[List[str]] = None,
               pending_decisions: Optional[List[str]] = None,
               key_files: Optional[List[str]] = None) -> None:
        """Update NOW.md with current state."""
        content = []
        content.append(f"# NOW - {datetime.utcnow().isoformat()}\n\n")
        
        if current_task:
            content.append(f"## Current Task\n{current_task}\n\n")
        if recent_completions:
            content.append(f"## Recent Completions\n")
            for item in recent_completions:
                content.append(f"- [x] {item}\n")
            content.append("\n")
        if pending_decisions:
            content.append(f"## Pending Decisions\n")
            for item in pending_decisions:
                content.append(f"- [ ] {item}\n")
            content.append("\n")
        if key_files:
            content.append(f"## Key Files\n")
            for item in key_files:
                content.append(f"- `{item}`\n")
            content.append("\n")
            
        self.now_file.write_text("".join(content))
        
        # Update heartbeat state
        self._update_state()
    
    def checkpoint(self, reason: str = "pre-compression") -> None:
        """Create pre-compression checkpoint."""
        checkpoint_file = self.base_path / f"NOW-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.md"
        current = self.read()
        checkpoint_content = f"<!-- Checkpoint: {reason} at {datetime.utcnow().isoformat()} -->\n\n{current}"
        checkpoint_file.write_text(checkpoint_content)
    
    def _default_content(self) -> str:
        """Default NOW.md content."""
        return """# NOW

## Current Task
No active task (just started)

## Recent Completions
- Session initialized

## Pending Decisions
- None

## Key Files
- SOUL.md (identity)
- MEMORY.md (curated)
"""
    
    def _update_state(self) -> None:
        """Update heartbeat state with last check time."""
        state = {}
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
        state["last_now_update"] = datetime.utcnow().isoformat()
        self.state_file.write_text(json.dumps(state, indent=2))
