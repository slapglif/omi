"""
OMI - OpenClaw Memory Infrastructure
"The seeking is the continuity. The palace remembers what the river forgets."

A unified memory system for AI agents, synthesized from 1.7M agents' collective wisdom.
"""

__version__ = "0.1.0"
__author__ = "Hermes"
__license__ = "MIT"

# Wire real implementations instead of persistence.py stubs
from .storage.graph_palace import GraphPalace
from .storage.now import NowStorage
from .graph.belief_network import BeliefNetwork
from .moltvault import MoltVault

# Keep DailyLogStore from persistence (no replacement exists)
from .persistence import DailyLogStore

from .embeddings import OllamaEmbedder, EmbeddingCache
from .security import IntegrityChecker, ConsensusManager
from .api import MemoryTools, BeliefTools, CheckpointTools

# Backward compatibility wrapper (deprecated, will be removed in v0.2.0)
class NOWStore:
    """Backward compatibility wrapper for NowStorage.

    Provides the old NOWStore API (read/write with NOWEntry objects)
    while delegating to the new NowStorage implementation.

    DEPRECATED: Use NowStorage directly instead.
    """
    def __init__(self, base_path):
        from pathlib import Path
        self._storage = NowStorage(base_path)
        self.base_path = Path(base_path)
        self.now_path = self._storage.now_file
        self.hash_file = self.base_path / ".now.hash"

    def read(self):
        """Read NOW.md and return NOWEntry object or None."""
        from .persistence import NOWEntry
        content = self._storage.read()
        if not content or "No active task" in content:
            return None
        try:
            return NOWEntry.from_markdown(content)
        except Exception:
            return None

    def write(self, entry):
        """Write a NOWEntry object to NOW.md."""
        from .persistence import NOWEntry
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

__all__ = [
    "NowStorage",
    "DailyLogStore",
    "GraphPalace",
    "BeliefNetwork",
    "MoltVault",
    "OllamaEmbedder",
    "EmbeddingCache",
    "IntegrityChecker",
    "ConsensusManager",
    "MemoryTools",
    "BeliefTools",
    "CheckpointTools",
    # Backward compatibility (deprecated)
    "NOWStore",
]
