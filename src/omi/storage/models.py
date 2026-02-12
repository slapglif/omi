"""
Data models for Graph Palace storage.

This module contains the core dataclasses representing memories and relationships
in the graph-based storage system.
"""

import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Memory:
    """A memory node in the graph palace."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    memory_type: str = "experience"  # fact | experience | belief | decision
    confidence: Optional[float] = None  # 0.0-1.0 for beliefs
    created_at: datetime = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    instance_ids: Optional[List[str]] = None
    content_hash: Optional[str] = None  # SHA-256 for integrity
    version_number: Optional[int] = None  # Version tracking
    version_id: Optional[str] = None  # Version identifier
    previous_version_id: Optional[str] = None  # Link to previous version

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.content_hash is None and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        if self.instance_ids is None:
            self.instance_ids = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "instance_ids": self.instance_ids,
            "content_hash": self.content_hash,
            "version_number": self.version_number,
            "version_id": self.version_id,
            "previous_version_id": self.previous_version_id
        }


@dataclass
class Edge:
    """A relationship edge between memories."""
    id: str
    source_id: str
    target_id: str
    edge_type: str  # SUPPORTS | CONTRADICTS | RELATED_TO | DEPENDS_ON | POSTED | DISCUSSED
    strength: Optional[float] = None  # 0.0-1.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
