"""
Event type definitions for OMI event streaming.

This module defines typed events emitted during memory operations:
- MemoryStoredEvent: When a memory is stored
- MemoryRecalledEvent: When memories are recalled
- BeliefUpdatedEvent: When a belief's confidence is updated
- ContradictionDetectedEvent: When a contradiction is detected
- SessionStartedEvent: When a session starts
- SessionEndedEvent: When a session ends

All events include full context (memory content, metadata, timestamps).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class MemoryStoredEvent:
    """Event emitted when a memory is stored."""
    memory_id: str
    content: str
    memory_type: str  # fact | experience | belief | decision
    confidence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "memory.stored"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class MemoryRecalledEvent:
    """Event emitted when memories are recalled."""
    query: str
    result_count: int
    top_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "memory.recalled"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "query": self.query,
            "result_count": self.result_count,
            "top_results": self.top_results,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class BeliefUpdatedEvent:
    """Event emitted when a belief's confidence is updated."""
    belief_id: str
    old_confidence: float
    new_confidence: float
    evidence_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "belief.updated"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "belief_id": self.belief_id,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "evidence_id": self.evidence_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class ContradictionDetectedEvent:
    """Event emitted when a contradiction is detected."""
    memory_id_1: str
    memory_id_2: str
    contradiction_pattern: str
    confidence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "belief.contradiction_detected"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "memory_id_1": self.memory_id_1,
            "memory_id_2": self.memory_id_2,
            "contradiction_pattern": self.contradiction_pattern,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class SessionStartedEvent:
    """Event emitted when a session starts."""
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "session.started"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class SessionEndedEvent:
    """Event emitted when a session ends."""
    session_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "session.ended"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class BeliefPropagatedEvent:
    """Event emitted when a belief is propagated between agents."""
    belief_id: str
    source_agent_id: str
    target_agent_id: str
    content: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "belief.propagated"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "belief_id": self.belief_id,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


@dataclass
class SharedMemoryStoredEvent:
    """Event emitted when a memory is stored for sharing across agents."""
    memory_id: str
    content: str
    memory_type: str  # fact | experience | belief | decision
    source_agent_id: str
    target_agent_ids: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "memory.shared_stored"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source_agent_id": self.source_agent_id,
            "target_agent_ids": self.target_agent_ids,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }


# Export all event types
__all__ = [
    "MemoryStoredEvent",
    "MemoryRecalledEvent",
    "BeliefUpdatedEvent",
    "ContradictionDetectedEvent",
    "SessionStartedEvent",
    "SessionEndedEvent",
    "BeliefPropagatedEvent",
    "SharedMemoryStoredEvent",
]
