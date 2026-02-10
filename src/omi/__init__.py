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
from .moltvault import MoltVault as VaultBackup

# Keep DailyLogStore and NOWStore from persistence (no replacement exists)
from .persistence import NOWStore, DailyLogStore

from .embeddings import OllamaEmbedder, EmbeddingCache
from .security import IntegrityChecker, ConsensusManager
from .api import MemoryTools, BeliefTools, CheckpointTools

__all__ = [
    "NOWStore",
    "NowStorage",
    "DailyLogStore",
    "GraphPalace",
    "BeliefNetwork",
    "VaultBackup",
    "OllamaEmbedder",
    "EmbeddingCache",
    "IntegrityChecker",
    "ConsensusManager",
    "MemoryTools",
    "BeliefTools",
    "CheckpointTools",
]
