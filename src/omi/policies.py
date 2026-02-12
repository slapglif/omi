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
import math


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


class UsagePolicy:
    """
    Usage-based (recall-based) policy for memory lifecycle management.

    Evaluates memories against access patterns to identify underused memories
    that should be acted upon (archived, deleted, etc.).

    A memory is considered unused if:
    - It has been accessed fewer than min_access_count times
    - AND it's older than max_age_days

    Example:
        policy = UsagePolicy(min_access_count=2, max_age_days=30)
        unused_memories = policy.evaluate(all_memories)
        # Returns memories older than 30 days with <2 accesses
    """

    def __init__(
        self,
        min_access_count: int,
        max_age_days: int,
        memory_type_filter: Optional[str] = None
    ):
        """
        Initialize usage policy.

        Args:
            min_access_count: Minimum access count threshold (memories below this are candidates)
            max_age_days: Minimum age in days (only consider memories older than this)
            memory_type_filter: Optional filter for specific memory types (fact, experience, belief, decision)
        """
        if min_access_count < 0:
            raise ValueError("min_access_count must be non-negative")
        if max_age_days <= 0:
            raise ValueError("max_age_days must be positive")

        self.min_access_count = min_access_count
        self.max_age_days = max_age_days
        self.memory_type_filter = memory_type_filter

    def evaluate(self, memories: List[Any]) -> List[str]:
        """
        Evaluate memories against usage policy.

        Args:
            memories: List of Memory objects to evaluate

        Returns:
            List of memory IDs that are underused (low access count AND old enough)
        """
        now = datetime.now()
        matching_ids = []

        for memory in memories:
            # Skip if memory type doesn't match filter
            if self.memory_type_filter and hasattr(memory, 'memory_type'):
                if memory.memory_type != self.memory_type_filter:
                    continue

            # Check age threshold first (must be old enough)
            if hasattr(memory, 'created_at') and memory.created_at:
                age_days = (now - memory.created_at).days
                if age_days <= self.max_age_days:
                    continue  # Too new, skip

            # Check access count
            access_count = getattr(memory, 'access_count', 0)
            if access_count < self.min_access_count:
                memory_id = memory.id if hasattr(memory, 'id') else str(memory)
                matching_ids.append(memory_id)

        return matching_ids

    def is_unused(self, memory: Any) -> bool:
        """
        Check if a single memory is underused based on access patterns.

        Args:
            memory: Memory object to check

        Returns:
            True if memory is old enough AND has insufficient access count
        """
        # Check memory type filter
        if self.memory_type_filter and hasattr(memory, 'memory_type'):
            if memory.memory_type != self.memory_type_filter:
                return False

        # Check age threshold
        if not hasattr(memory, 'created_at') or not memory.created_at:
            return False

        now = datetime.now()
        age_days = (now - memory.created_at).days
        if age_days <= self.max_age_days:
            return False  # Too new

        # Check access count
        access_count = getattr(memory, 'access_count', 0)
        return access_count < self.min_access_count

    def days_since_last_access(self, memory: Any) -> Optional[int]:
        """
        Calculate days since memory was last accessed.

        Args:
            memory: Memory object to check

        Returns:
            Days since last access, None if no last_accessed timestamp
        """
        if not hasattr(memory, 'last_accessed') or not memory.last_accessed:
            # Fall back to created_at if no last_accessed
            if hasattr(memory, 'created_at') and memory.created_at:
                now = datetime.now()
                return (now - memory.created_at).days
            return None

        now = datetime.now()
        return (now - memory.last_accessed).days

    def usage_score(self, memory: Any) -> Optional[float]:
        """
        Calculate a usage score for the memory (0.0 = unused, 1.0 = well-used).

        Score factors:
        - Access count relative to threshold
        - Recency of last access

        Args:
            memory: Memory object to score

        Returns:
            Usage score 0.0-1.0, None if insufficient data
        """
        access_count = getattr(memory, 'access_count', 0)
        days_since_access = self.days_since_last_access(memory)

        if days_since_access is None:
            return None

        # Score based on access count (capped at 2x threshold = score 1.0)
        access_score = min(1.0, access_count / (self.min_access_count * 2)) if self.min_access_count > 0 else 1.0

        # Score based on recency (exponential decay with 30-day half-life)
        recency_score = math.exp(-days_since_access / 30.0)

        # Combine: 60% access frequency, 40% recency
        return (access_score * 0.6) + (recency_score * 0.4)


# Export all policy types, actions, and classes
__all__ = [
    "PolicyType",
    "PolicyAction",
    "PolicyRule",
    "Policy",
    "RetentionPolicy",
    "UsagePolicy",
]
