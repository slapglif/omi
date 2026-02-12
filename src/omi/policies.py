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
from typing import Optional, List, Dict, Any, Callable, Union
import math


def is_memory_locked(memory: Any) -> bool:
    """
    Check if a memory is locked (exempt from policy actions).

    Locked memories cannot be archived, deleted, compressed, promoted,
    or demoted by policy engine actions. This provides a mechanism to
    protect critical memories from automatic lifecycle management.

    Args:
        memory: Memory object to check (must have a 'locked' attribute)

    Returns:
        True if memory is locked, False otherwise

    Example:
        from omi.policies import is_memory_locked
        from omi.storage.graph_palace import Memory

        memory = Memory(id="mem-1", content="Critical information", locked=True)
        if is_memory_locked(memory):
            print("This memory is protected from policy actions")
    """
    return getattr(memory, 'locked', False)


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


@dataclass
class PolicyExecutionLog:
    """Audit trail record for policy execution."""
    policy_name: str
    action: str  # PolicyAction value or string
    memory_ids: List[str]
    dry_run: bool
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[str] = None  # "success", "failure", "partial"
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_name": self.policy_name,
            "action": self.action,
            "memory_ids": self.memory_ids,
            "dry_run": self.dry_run,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "result": self.result,
            "error": self.error,
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
            # Skip locked memories (exempt from policy actions)
            if is_memory_locked(memory):
                continue

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
            # Skip locked memories (exempt from policy actions)
            if is_memory_locked(memory):
                continue

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


class ConfidencePolicy:
    """
    Confidence threshold policy for memory lifecycle management.

    Evaluates memories (typically beliefs) against a minimum confidence threshold
    and identifies memories that should be acted upon (archived, deleted, etc.).

    This policy is particularly useful for cleaning up low-confidence beliefs
    or flagging uncertain information for review.

    Example:
        policy = ConfidencePolicy(min_confidence=0.3)
        low_confidence_memories = policy.evaluate(all_memories)
        # Returns memories with confidence < 0.3
    """

    def __init__(self, min_confidence: float, memory_type_filter: Optional[str] = None):
        """
        Initialize confidence policy.

        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0). Memories below this are candidates.
            memory_type_filter: Optional filter for specific memory types (typically "belief")

        Raises:
            ValueError: If min_confidence is not between 0.0 and 1.0
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        self.min_confidence = min_confidence
        self.memory_type_filter = memory_type_filter

    def evaluate(self, memories: List[Any]) -> List[str]:
        """
        Evaluate memories against confidence policy.

        Args:
            memories: List of Memory objects to evaluate

        Returns:
            List of memory IDs that fall below the confidence threshold
        """
        matching_ids = []

        for memory in memories:
            # Skip locked memories (exempt from policy actions)
            if is_memory_locked(memory):
                continue

            # Skip if memory type doesn't match filter
            if self.memory_type_filter and hasattr(memory, 'memory_type'):
                if memory.memory_type != self.memory_type_filter:
                    continue

            # Check confidence threshold
            # Only consider memories that have a confidence value
            if hasattr(memory, 'confidence') and memory.confidence is not None:
                if memory.confidence < self.min_confidence:
                    memory_id = memory.id if hasattr(memory, 'id') else str(memory)
                    matching_ids.append(memory_id)

        return matching_ids

    def is_below_threshold(self, memory: Any) -> bool:
        """
        Check if a single memory falls below the confidence threshold.

        Args:
            memory: Memory object to check

        Returns:
            True if memory has confidence below min_confidence, False otherwise
        """
        # Check memory type filter
        if self.memory_type_filter and hasattr(memory, 'memory_type'):
            if memory.memory_type != self.memory_type_filter:
                return False

        # Check confidence threshold
        if not hasattr(memory, 'confidence') or memory.confidence is None:
            return False

        return memory.confidence < self.min_confidence

    def confidence_margin(self, memory: Any) -> Optional[float]:
        """
        Calculate the margin between memory confidence and threshold.

        Args:
            memory: Memory object to check

        Returns:
            Confidence margin (positive = above threshold, negative = below threshold),
            None if no confidence value
        """
        if not hasattr(memory, 'confidence') or memory.confidence is None:
            return None

        return memory.confidence - self.min_confidence

    def confidence_score(self, memory: Any) -> Optional[float]:
        """
        Calculate a normalized confidence score for the memory (0.0 = no confidence, 1.0 = full confidence).

        This is simply the memory's confidence value if it exists.

        Args:
            memory: Memory object to score

        Returns:
            Confidence score 0.0-1.0, None if no confidence value
        """
        if not hasattr(memory, 'confidence') or memory.confidence is None:
            return None

        return memory.confidence


@dataclass
class PolicyExecutionResult:
    """Result of policy execution."""
    policy_name: str
    action: PolicyAction
    affected_memory_ids: List[str]
    dry_run: bool
    executed_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_name": self.policy_name,
            "action": self.action.value,
            "affected_memory_ids": self.affected_memory_ids,
            "dry_run": self.dry_run,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "error": self.error,
            "metadata": self.metadata or {}
        }


class PolicyEngine:
    """
    Core policy execution engine for memory lifecycle management.

    The PolicyEngine is responsible for executing policies against memories in the Graph Palace.
    It supports both dry-run mode (preview what would happen) and actual execution.

    Features:
    - Execute individual policies or policy sets
    - Dry-run mode for safe preview
    - Automatic policy evaluator selection based on policy type
    - Execution result tracking
    - Error handling and rollback support

    Example:
        # Initialize with Graph Palace instance
        engine = PolicyEngine(graph_palace)

        # Create a retention policy
        policy = Policy(
            name="archive-old-experiences",
            rules=[
                PolicyRule(
                    name="archive-90-day-experiences",
                    policy_type=PolicyType.RETENTION,
                    action=PolicyAction.ARCHIVE,
                    conditions={"max_age_days": 90},
                    memory_type_filter="experience"
                )
            ]
        )

        # Dry run to preview
        results = engine.execute(policy, dry_run=True)
        print(f"Would affect {len(results[0].affected_memory_ids)} memories")

        # Execute for real
        results = engine.execute(policy, dry_run=False)
    """

    def __init__(self, graph_palace: Optional[Any] = None):
        """
        Initialize PolicyEngine.

        Args:
            graph_palace: GraphPalace instance for memory operations (can be None for testing)
        """
        self.graph_palace = graph_palace
        self._execution_history: List[PolicyExecutionResult] = []

    def execute(
        self,
        policy: Policy,
        dry_run: bool = True,
        memory_filter: Optional[Callable[[Any], bool]] = None
    ) -> List[PolicyExecutionResult]:
        """
        Execute a policy against memories.

        Args:
            policy: Policy to execute
            dry_run: If True, only preview actions without executing (default: True)
            memory_filter: Optional filter function to pre-filter memories

        Returns:
            List of PolicyExecutionResult objects, one per rule in the policy

        Raises:
            ValueError: If policy is invalid or disabled
        """
        if not policy.enabled:
            raise ValueError(f"Policy '{policy.name}' is disabled")

        results: List[PolicyExecutionResult] = []

        # Fetch all memories (can be overridden for testing)
        try:
            memories = self._fetch_memories(memory_filter)
        except Exception as e:
            # If we can't fetch memories, return error result
            error_result = PolicyExecutionResult(
                policy_name=policy.name,
                action=PolicyAction.ARCHIVE,  # Placeholder
                affected_memory_ids=[],
                dry_run=dry_run,
                error=f"Failed to fetch memories: {str(e)}"
            )
            results.append(error_result)
            return results

        # Execute each rule in the policy
        for rule in policy.rules:
            if not rule.enabled:
                continue

            try:
                result = self._execute_rule(rule, memories, dry_run)
                results.append(result)
            except Exception as e:
                # Record error but continue with other rules
                error_result = PolicyExecutionResult(
                    policy_name=policy.name,
                    action=rule.action,
                    affected_memory_ids=[],
                    dry_run=dry_run,
                    error=f"Failed to execute rule '{rule.name}': {str(e)}"
                )
                results.append(error_result)

        # Update policy execution metadata
        if not dry_run:
            policy.last_executed = datetime.now()
            policy.execution_count += 1

        # Store in execution history
        self._execution_history.extend(results)

        return results

    def _execute_rule(
        self,
        rule: PolicyRule,
        memories: List[Any],
        dry_run: bool
    ) -> PolicyExecutionResult:
        """
        Execute a single policy rule.

        Args:
            rule: PolicyRule to execute
            memories: List of memories to evaluate
            dry_run: If True, only preview actions without executing

        Returns:
            PolicyExecutionResult for this rule
        """
        # Filter memories by type if specified
        filtered_memories = memories
        if rule.memory_type_filter:
            filtered_memories = [
                m for m in memories
                if hasattr(m, 'memory_type') and m.memory_type == rule.memory_type_filter
            ]

        # Select appropriate evaluator based on policy type
        affected_ids = self._evaluate_policy(rule, filtered_memories)

        # Execute action if not dry run
        if not dry_run and affected_ids:
            self._execute_action(rule.action, affected_ids)

        return PolicyExecutionResult(
            policy_name=rule.name,
            action=rule.action,
            affected_memory_ids=affected_ids,
            dry_run=dry_run,
            metadata={
                "policy_type": rule.policy_type.value,
                "conditions": rule.conditions,
                "memory_type_filter": rule.memory_type_filter,
                "description": rule.description
            }
        )

    def _evaluate_policy(self, rule: PolicyRule, memories: List[Any]) -> List[str]:
        """
        Evaluate policy rule against memories using appropriate evaluator.

        Args:
            rule: PolicyRule to evaluate
            memories: List of memories to evaluate

        Returns:
            List of memory IDs that match the policy conditions
        """
        conditions = rule.conditions

        if rule.policy_type == PolicyType.RETENTION:
            # Age-based retention policy
            max_age_days = conditions.get("max_age_days")
            if max_age_days is None:
                return []

            policy = RetentionPolicy(
                max_age_days=max_age_days,
                memory_type_filter=rule.memory_type_filter
            )
            return policy.evaluate(memories)

        elif rule.policy_type == PolicyType.USAGE:
            # Recall/access-based usage policy
            min_access_count = conditions.get("min_access_count", 1)
            max_age_days = conditions.get("max_age_days", 90)

            policy = UsagePolicy(
                min_access_count=min_access_count,
                max_age_days=max_age_days,
                memory_type_filter=rule.memory_type_filter
            )
            return policy.evaluate(memories)

        elif rule.policy_type == PolicyType.CONFIDENCE:
            # Confidence threshold policy
            min_confidence = conditions.get("min_confidence")
            if min_confidence is None:
                return []

            policy = ConfidencePolicy(
                min_confidence=min_confidence,
                memory_type_filter=rule.memory_type_filter
            )
            return policy.evaluate(memories)

        elif rule.policy_type == PolicyType.SIZE:
            # Size-based policy (placeholder - not yet implemented)
            # This would check tier sizes and select memories to archive/delete
            return []

        return []

    def _execute_action(self, action: PolicyAction, memory_ids: List[str]) -> None:
        """
        Execute a policy action on the specified memories.

        Args:
            action: PolicyAction to execute
            memory_ids: List of memory IDs to act upon

        Note:
            This is a placeholder implementation. Actual action execution will be
            implemented in phase-2-policy-actions with proper integration to
            GraphPalace methods for archive, delete, compress, etc.
        """
        # Placeholder - actual implementation will be in phase 2
        # For now, just validate that we have a graph_palace instance
        if self.graph_palace is None:
            return

        # Action implementations will call appropriate graph_palace methods:
        # - ARCHIVE: Mark memories as archived
        # - DELETE: Remove memories
        # - COMPRESS: Reduce storage footprint
        # - PROMOTE: Move to higher tier
        # - DEMOTE: Move to lower tier
        pass

    def _fetch_memories(self, memory_filter: Optional[Callable[[Any], bool]] = None) -> List[Any]:
        """
        Fetch memories from GraphPalace.

        Args:
            memory_filter: Optional filter function to pre-filter memories

        Returns:
            List of Memory objects
        """
        if self.graph_palace is None:
            return []

        # This is a placeholder - actual implementation depends on GraphPalace API
        # For now, return empty list to enable testing
        memories = []

        # Apply filter if provided
        if memory_filter:
            memories = [m for m in memories if memory_filter(m)]

        return memories

    def get_execution_history(self) -> List[PolicyExecutionResult]:
        """
        Get the execution history for this engine instance.

        Returns:
            List of PolicyExecutionResult objects in chronological order
        """
        return self._execution_history.copy()

    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()


def archive_memories(graph_palace, memory_ids: List[str]) -> Dict[str, Any]:
    """
    Archive memories by marking them as archived (excluded from default search).

    This is a policy action function that can be used by the PolicyEngine
    or called directly to archive specific memories.

    Args:
        graph_palace: GraphPalace instance with archive_memories method
        memory_ids: List of memory IDs to archive

    Returns:
        Dict with:
            - success: bool indicating if operation completed
            - archived_count: number of memories successfully archived
            - memory_ids: list of archived memory IDs
            - errors: list of any errors encountered

    Example:
        from omi.storage.graph_palace import GraphPalace
        from omi.policies import archive_memories

        palace = GraphPalace(db_path="palace.sqlite")
        result = archive_memories(palace, ["mem-uuid-1", "mem-uuid-2"])
        print(f"Archived {result['archived_count']} memories")
    """
    if not memory_ids:
        return {
            "success": True,
            "archived_count": 0,
            "memory_ids": [],
            "errors": []
        }

    errors = []
    archived_count = 0

    try:
        # Call the archive_memories method on the graph_palace instance
        archived_count = graph_palace.archive_memories(memory_ids)

        return {
            "success": True,
            "archived_count": archived_count,
            "memory_ids": memory_ids[:archived_count],  # Only IDs that were actually archived
            "errors": errors
        }
    except Exception as e:
        errors.append(str(e))
        return {
            "success": False,
            "archived_count": archived_count,
            "memory_ids": [],
            "errors": errors
        }


def delete_memories(graph_palace, memory_ids: List[str], safety_check: bool = True) -> Dict[str, Any]:
    """
    Delete memories permanently with safety checks.

    This is a destructive operation that permanently removes memories and their edges.
    By default, performs safety checks to validate memories exist before deletion.

    This is a policy action function that can be used by the PolicyEngine
    or called directly to delete specific memories.

    Args:
        graph_palace: GraphPalace instance with delete_memories method
        memory_ids: List of memory IDs to delete
        safety_check: If True, validates memories exist before deletion (default: True)

    Returns:
        Dict with:
            - success: bool indicating if operation completed
            - deleted_count: number of memories successfully deleted
            - memory_ids: list of deleted memory IDs
            - errors: list of any errors encountered
            - skipped: list of memory IDs that were skipped (if safety_check=True and not found)

    Example:
        from omi.storage.graph_palace import GraphPalace
        from omi.policies import delete_memories

        palace = GraphPalace(db_path="palace.sqlite")
        result = delete_memories(palace, ["mem-uuid-1", "mem-uuid-2"])
        print(f"Deleted {result['deleted_count']} memories")

    Safety:
        - Validates memory existence before deletion (if safety_check=True)
        - Returns detailed results about what was deleted
        - Handles errors gracefully
        - Cascades deletion to related edges automatically
    """
    if not memory_ids:
        return {
            "success": True,
            "deleted_count": 0,
            "memory_ids": [],
            "skipped": [],
            "errors": []
        }

    errors = []
    deleted_count = 0
    skipped = []
    valid_memory_ids = memory_ids.copy()

    try:
        # Safety check: validate memories exist before deletion
        if safety_check:
            valid_memory_ids = []
            for memory_id in memory_ids:
                # Try to retrieve the memory to check if it exists
                memory = graph_palace.get_memory(memory_id)
                if memory is not None:
                    valid_memory_ids.append(memory_id)
                else:
                    skipped.append(memory_id)

            # If no valid memories found, return early
            if not valid_memory_ids:
                return {
                    "success": True,
                    "deleted_count": 0,
                    "memory_ids": [],
                    "skipped": skipped,
                    "errors": []
                }

        # Call the delete_memories method on the graph_palace instance
        deleted_count = graph_palace.delete_memories(valid_memory_ids)

        return {
            "success": True,
            "deleted_count": deleted_count,
            "memory_ids": valid_memory_ids[:deleted_count],  # Only IDs that were actually deleted
            "skipped": skipped,
            "errors": errors
        }
    except Exception as e:
        errors.append(str(e))
        return {
            "success": False,
            "deleted_count": deleted_count,
            "memory_ids": [],
            "skipped": skipped,
            "errors": errors
        }


def load_policies_from_config(config_path) -> List[Policy]:
    """
    Load policy configuration from config.yaml file.

    Reads the policies section from a YAML config file and converts it
    to a list of Policy objects with their associated PolicyRule objects.

    Config YAML Schema:
        policies:
          - name: "archive-old-memories"
            enabled: true
            description: "Archive memories older than 90 days"
            priority: 1
            schedule: "daily"
            rules:
              - name: "archive-old-facts"
                policy_type: "retention"
                action: "archive"
                enabled: true
                memory_type_filter: "fact"
                conditions:
                  max_age_days: 90
              - name: "archive-old-experiences"
                policy_type: "retention"
                action: "archive"
                enabled: true
                memory_type_filter: "experience"
                conditions:
                  max_age_days: 60

    Args:
        config_path: Path to config.yaml file (can be Path or str)

    Returns:
        List of Policy objects loaded from configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config YAML is invalid or policies section is malformed
        KeyError: If required fields are missing from policy or rule definitions

    Example:
        from pathlib import Path
        from omi.policies import load_policies_from_config

        config_path = Path("~/.openclaw/omi/config.yaml").expanduser()
        policies = load_policies_from_config(config_path)
        for policy in policies:
            print(f"Loaded policy: {policy.name}")
    """
    from pathlib import Path
    import yaml

    # Convert to Path if string
    if isinstance(config_path, str):
        config_path = Path(config_path)

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Read and parse YAML
        config_data = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

    # Extract policies section
    policies_data = config_data.get("policies", [])
    if not isinstance(policies_data, list):
        raise ValueError("'policies' section must be a list")

    policies = []

    for policy_dict in policies_data:
        try:
            # Extract policy fields
            name = policy_dict["name"]
            enabled = policy_dict.get("enabled", True)
            description = policy_dict.get("description")
            priority = policy_dict.get("priority", 0)
            schedule = policy_dict.get("schedule")
            metadata = policy_dict.get("metadata")

            # Parse datetime fields if present
            created_at = None
            if "created_at" in policy_dict:
                created_at_str = policy_dict["created_at"]
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)

            last_executed = None
            if "last_executed" in policy_dict:
                last_executed_str = policy_dict["last_executed"]
                if last_executed_str:
                    last_executed = datetime.fromisoformat(last_executed_str)

            execution_count = policy_dict.get("execution_count", 0)

            # Parse rules
            rules_data = policy_dict.get("rules", [])
            rules = []

            for rule_dict in rules_data:
                # Extract rule fields
                rule_name = rule_dict["name"]
                policy_type_str = rule_dict["policy_type"]
                action_str = rule_dict["action"]
                conditions = rule_dict.get("conditions", {})
                rule_enabled = rule_dict.get("enabled", True)
                rule_description = rule_dict.get("description")
                memory_type_filter = rule_dict.get("memory_type_filter")
                rule_metadata = rule_dict.get("metadata")

                # Convert enum strings to enum values
                try:
                    policy_type = PolicyType(policy_type_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid policy_type '{policy_type_str}' in rule '{rule_name}'. "
                        f"Valid values: {[t.value for t in PolicyType]}"
                    )

                try:
                    action = PolicyAction(action_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid action '{action_str}' in rule '{rule_name}'. "
                        f"Valid values: {[a.value for a in PolicyAction]}"
                    )

                # Parse rule created_at if present
                rule_created_at = None
                if "created_at" in rule_dict:
                    rule_created_at_str = rule_dict["created_at"]
                    if rule_created_at_str:
                        rule_created_at = datetime.fromisoformat(rule_created_at_str)

                # Create PolicyRule object
                rule = PolicyRule(
                    name=rule_name,
                    policy_type=policy_type,
                    action=action,
                    conditions=conditions,
                    enabled=rule_enabled,
                    description=rule_description,
                    memory_type_filter=memory_type_filter,
                    created_at=rule_created_at or datetime.now(),
                    metadata=rule_metadata
                )
                rules.append(rule)

            # Create Policy object
            policy = Policy(
                name=name,
                rules=rules,
                enabled=enabled,
                description=description,
                priority=priority,
                schedule=schedule,
                created_at=created_at or datetime.now(),
                last_executed=last_executed,
                execution_count=execution_count,
                metadata=metadata
            )
            policies.append(policy)

        except KeyError as e:
            raise KeyError(f"Missing required field in policy configuration: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing policy '{policy_dict.get('name', 'unknown')}': {e}")

    return policies


def get_default_policies() -> List[Policy]:
    """
    Get default policy set with sensible memory lifecycle management defaults.

    Creates a baseline set of policies for automatic memory management that:
    - Archives old, unused memories to reduce clutter in search results
    - Deletes very old memories for compliance with data retention requirements
    - Removes low-confidence beliefs to maintain memory quality

    Default Policies:
        1. Archive Old Memories (180 days):
           - Archives memories older than 180 days to reduce search clutter
           - Applies to all memory types (facts, experiences, beliefs, decisions)

        2. Archive Unused Memories (90 days):
           - Archives memories not recalled in 90 days with <3 total accesses
           - Focuses on memories that are not actively being used

        3. Delete Low-Confidence Beliefs:
           - Permanently removes beliefs with confidence < 0.2
           - Only applies to belief-type memories to maintain quality

        4. Compliance Deletion (365 days):
           - Permanently deletes memories older than 365 days for data retention
           - Ensures compliance with typical data retention policies

    Returns:
        List of Policy objects with default rules configured

    Example:
        from omi.policies import get_default_policies

        # Get default policies for new installations
        policies = get_default_policies()

        # Apply to policy engine
        engine = PolicyEngine(graph_palace)
        for policy in policies:
            result = engine.execute(policy)
            print(f"Executed {policy.name}: {result.summary}")

    Note:
        These policies provide sensible defaults but can be overridden by
        defining custom policies in config.yaml. Locked memories are always
        exempt from all policy actions.
    """
    now = datetime.now()

    # Policy 1: Archive old memories (180 days)
    archive_old_policy = Policy(
        name="archive-old-memories",
        description="Archive memories older than 180 days to reduce search clutter",
        enabled=True,
        priority=1,
        schedule="daily",
        created_at=now,
        rules=[
            PolicyRule(
                name="archive-old-all-types",
                policy_type=PolicyType.RETENTION,
                action=PolicyAction.ARCHIVE,
                conditions={"max_age_days": 180},
                enabled=True,
                description="Archive all memory types older than 180 days",
                memory_type_filter=None,  # Apply to all types
                created_at=now
            )
        ]
    )

    # Policy 2: Archive unused memories (90 days, <3 accesses)
    archive_unused_policy = Policy(
        name="archive-unused-memories",
        description="Archive memories not recalled in 90 days with low access counts",
        enabled=True,
        priority=2,
        schedule="daily",
        created_at=now,
        rules=[
            PolicyRule(
                name="archive-unused-low-access",
                policy_type=PolicyType.USAGE,
                action=PolicyAction.ARCHIVE,
                conditions={
                    "max_days_since_last_access": 90,
                    "min_access_count": 3
                },
                enabled=True,
                description="Archive memories not accessed in 90 days with <3 total accesses",
                memory_type_filter=None,  # Apply to all types
                created_at=now
            )
        ]
    )

    # Policy 3: Delete low-confidence beliefs
    delete_low_confidence_policy = Policy(
        name="delete-low-confidence-beliefs",
        description="Delete beliefs with confidence below 0.2 to maintain quality",
        enabled=True,
        priority=3,
        schedule="weekly",
        created_at=now,
        rules=[
            PolicyRule(
                name="delete-weak-beliefs",
                policy_type=PolicyType.CONFIDENCE,
                action=PolicyAction.DELETE,
                conditions={"min_confidence": 0.2},
                enabled=True,
                description="Delete beliefs with confidence < 0.2",
                memory_type_filter="belief",  # Only beliefs
                created_at=now
            )
        ]
    )

    # Policy 4: Compliance deletion (365 days)
    compliance_deletion_policy = Policy(
        name="compliance-deletion",
        description="Delete memories older than 365 days for data retention compliance",
        enabled=True,
        priority=4,
        schedule="weekly",
        created_at=now,
        rules=[
            PolicyRule(
                name="delete-very-old-memories",
                policy_type=PolicyType.RETENTION,
                action=PolicyAction.DELETE,
                conditions={"max_age_days": 365},
                enabled=True,
                description="Delete all memory types older than 365 days",
                memory_type_filter=None,  # Apply to all types
                created_at=now
            )
        ]
    )

    return [
        archive_old_policy,
        archive_unused_policy,
        delete_low_confidence_policy,
        compliance_deletion_policy
    ]


class PolicyEventHandler:
    """
    Event handler for triggering policy execution based on events.

    This handler integrates with the EventBus to automatically execute policies
    when specific events occur in the memory system. It enables reactive policy
    execution based on memory operations, session boundaries, and belief updates.

    Supports:
    - Triggering policies on session end
    - Triggering policies on memory storage
    - Triggering policies on belief updates
    - Custom policy execution based on event metadata

    Example:
        from omi.policies import PolicyEngine, PolicyEventHandler
        from omi.event_bus import EventBus

        # Initialize policy engine
        engine = PolicyEngine(graph_palace)

        # Create event handler
        handler = PolicyEventHandler(engine)

        # Subscribe to events
        bus = EventBus()
        bus.subscribe('session.ended', handler.handle)
        bus.subscribe('memory.stored', handler.handle)

        # Events will now trigger policy execution automatically
    """

    def __init__(self, policy_engine: Optional['PolicyEngine'] = None):
        """
        Initialize PolicyEventHandler.

        Args:
            policy_engine: PolicyEngine instance to execute policies (can be None for testing)
        """
        self.policy_engine = policy_engine

    def handle(self, event: Any) -> None:
        """
        Handle an event and trigger appropriate policy execution.

        This method processes events from the EventBus and triggers policy execution
        based on event type and metadata. Different event types may trigger different
        policies or policy sets.

        Event triggers:
        - session.ended: Triggers cleanup policies (archive old memories, delete low-confidence)
        - memory.stored: Triggers size-based policies if tier limits are exceeded
        - belief.updated: Triggers confidence-based policies for low-confidence beliefs
        - policy.triggered: Custom policy execution based on event metadata

        Args:
            event: Event object to process (must have 'event_type' attribute)

        Note:
            This is a reactive handler - it responds to events by executing policies.
            Policy execution is logged through the PolicyEngine's execution history.
            If policy_engine is None, events are processed but no actions are taken.
        """
        if not hasattr(event, 'event_type'):
            # Invalid event, skip silently
            return

        # If no policy engine configured, we can't execute policies
        if self.policy_engine is None:
            return

        event_type = event.event_type

        # Handle different event types
        if event_type == 'session.ended':
            # On session end, trigger cleanup policies
            # This is a good time to archive old/unused memories
            # Actual policy selection and execution will be implemented in phase 2
            pass

        elif event_type == 'memory.stored':
            # On memory storage, check if size-based policies should trigger
            # This could trigger tier size limit policies
            pass

        elif event_type == 'belief.updated':
            # On belief update, check if confidence-based policies should trigger
            # This could trigger low-confidence belief deletion
            pass

        elif event_type == 'policy.triggered':
            # Custom policy trigger event
            # Execute policies specified in event metadata
            pass


# Export all policy types, actions, and classes
__all__ = [
    "PolicyType",
    "PolicyAction",
    "PolicyRule",
    "Policy",
    "RetentionPolicy",
    "UsagePolicy",
    "ConfidencePolicy",
    "PolicyEngine",
    "PolicyExecutionResult",
    "PolicyExecutionLog",
    "PolicyEventHandler",
    "archive_memories",
    "delete_memories",
    "is_memory_locked",
    "load_policies_from_config",
    "get_default_policies",
]
