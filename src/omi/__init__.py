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
]
