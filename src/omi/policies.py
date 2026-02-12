"""
Policy engine for automatic memory lifecycle management.

This module defines the core policy types, actions, rules, and engine for managing
memory retention, archival, and deletion based on configurable criteria.

Policy Types:
- Retention: Age-based policies (auto-archive or delete after N days)
- Usage: Recall-based policies (archive unused memories)
- Confidence: Threshold-based policies (delete low-confidence beliefs)
- Size: Tier-limit policies (compress or archive when tier exceeds size)

Policy Actions:
- Archive: Mark memories as archived (excluded from search by default)
- Delete: Permanently remove memories
- Compress: Reduce memory storage footprint
- Promote: Move to higher tier (e.g., to NOW.md)
- Demote: Move to lower tier (e.g., from NOW.md to daily log)

All policy execution is auditable with PolicyExecutionLog tracking what was done,
when, why, and to which memories.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Callable


class PolicyType(Enum):
    """Types of policies for memory management."""
    RETENTION = "retention"  # Age-based policies
    USAGE = "usage"  # Recall/access-based policies
    CONFIDENCE = "confidence"  # Confidence threshold-based policies
    SIZE = "size"  # Tier size limit-based policies


class PolicyAction(Enum):
    """Actions that can be taken by policies."""
    ARCHIVE = "archive"  # Mark as archived (hidden from default search)
    DELETE = "delete"  # Permanently remove
    COMPRESS = "compress"  # Reduce storage footprint
    PROMOTE = "promote"  # Move to higher tier
    DEMOTE = "demote"  # Move to lower tier


@dataclass
class PolicyRule:
    """A single policy rule with conditions and actions."""
    name: str
    policy_type: PolicyType
    action: PolicyAction
    conditions: Dict[str, Any]  # e.g., {"max_age_days": 90, "min_confidence": 0.3}
    enabled: bool = True
    description: Optional[str] = None
    memory_type_filter: Optional[str] = None  # Filter by memory type (fact, experience, etc.)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "policy_type": self.policy_type.value,
            "action": self.action.value,
            "conditions": self.conditions,
            "enabled": self.enabled,
            "description": self.description,
            "memory_type_filter": self.memory_type_filter,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata or {}
        }


@dataclass
class Policy:
    """A complete policy with multiple rules and metadata."""
    name: str
    rules: List[PolicyRule] = field(default_factory=list)
    enabled: bool = True
    description: Optional[str] = None
    priority: int = 0  # Higher priority policies execute first
    schedule: Optional[str] = None  # Cron-style schedule or "on_event"
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
            "enabled": self.enabled,
            "description": self.description,
            "priority": self.priority,
            "schedule": self.schedule,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "execution_count": self.execution_count,
            "metadata": self.metadata or {}
        }


class RetentionPolicy:
    """
    Age-based retention policy for memory lifecycle management.

    Evaluates memories against a maximum age threshold and identifies
    memories that should be acted upon (archived, deleted, etc.).

    Example:
        policy = RetentionPolicy(max_age_days=90)
        old_memories = policy.evaluate(all_memories)
        # Returns memories older than 90 days
    """

    def __init__(self, max_age_days: int, memory_type_filter: Optional[str] = None):
        """
        Initialize retention policy.

        Args:
            max_age_days: Maximum age in days before memory is considered for action
            memory_type_filter: Optional filter for specific memory types (fact, experience, belief, decision)
        """
        if max_age_days <= 0:
            raise ValueError("max_age_days must be positive")

        self.max_age_days = max_age_days
        self.memory_type_filter = memory_type_filter

    def evaluate(self, memories: List[Any]) -> List[str]:
        """
        Evaluate memories against retention policy.

        Args:
            memories: List of Memory objects to evaluate

        Returns:
            List of memory IDs that exceed the age threshold
        """
        now = datetime.now()
        matching_ids = []

        for memory in memories:
            # Skip if memory type doesn't match filter
            if self.memory_type_filter and hasattr(memory, 'memory_type'):
                if memory.memory_type != self.memory_type_filter:
                    continue

            # Check age
            if hasattr(memory, 'created_at') and memory.created_at:
                age_days = (now - memory.created_at).days
                if age_days > self.max_age_days:
                    memory_id = memory.id if hasattr(memory, 'id') else str(memory)
                    matching_ids.append(memory_id)

        return matching_ids

    def is_expired(self, memory: Any) -> bool:
        """
        Check if a single memory exceeds the age threshold.

        Args:
            memory: Memory object to check

        Returns:
            True if memory exceeds max_age_days, False otherwise
        """
        if not hasattr(memory, 'created_at') or not memory.created_at:
            return False

        # Check memory type filter
        if self.memory_type_filter and hasattr(memory, 'memory_type'):
            if memory.memory_type != self.memory_type_filter:
                return False

        now = datetime.now()
        age_days = (now - memory.created_at).days
        return age_days > self.max_age_days

    def days_until_expiry(self, memory: Any) -> Optional[int]:
        """
        Calculate days until memory expires under this policy.

        Args:
            memory: Memory object to check

        Returns:
            Days until expiry (negative if already expired), None if no created_at
        """
        if not hasattr(memory, 'created_at') or not memory.created_at:
            return None

        now = datetime.now()
        age_days = (now - memory.created_at).days
        return self.max_age_days - age_days


# Export all policy types, actions, and classes
__all__ = [
    "PolicyType",
    "PolicyAction",
    "PolicyRule",
    "Policy",
    "RetentionPolicy",
]
