"""
OMI - OpenClaw Memory Infrastructure
"The seeking is the continuity. The palace remembers what the river forgets."

A unified memory system for AI agents, synthesized from 1.7M agents' collective wisdom.
"""

__version__ = "0.1.0"
__author__ = "Hermes"
__license__ = "MIT"

from .persistence import NOWStore, DailyLogStore, GraphPalace
from .embeddings import OllamaEmbedder, EmbeddingCache
from .security import IntegrityChecker, ConsensusManager
from .api import MemoryTools, BeliefTools, CheckpointTools

__all__ = [
    "NOWStore",
    "DailyLogStore", 
    "GraphPalace",
    "OllamaEmbedder",
    "EmbeddingCache",
    "IntegrityChecker",
    "ConsensusManager",
    "MemoryTools",
    "BeliefTools",
    "CheckpointTools",
]
