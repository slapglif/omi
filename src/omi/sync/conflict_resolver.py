"""
Conflict Resolution for Distributed Memory Sync

Handles conflicts when multiple instances modify the same memory concurrently.
Supports multiple strategies: last-writer-wins, merge, and manual queue.

Last-Writer-Wins: The most recently modified memory wins based on last_accessed timestamp.
                  If timestamps are equal, higher version number wins.
                  If versions are equal, compare vector clocks.

Merge: Attempts intelligent merging using vector clock comparison.
       Falls back to manual queue for true conflicts.

Manual Queue: Unresolvable conflicts are queued for manual resolution.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import logging

from ..storage.models import Memory

logger = logging.getLogger(__name__)


class ConflictStrategy(Enum):
    """
    Conflict resolution strategies for distributed sync.

    LAST_WRITER_WINS: Most recent modification wins.
                      Simple, fast, but may lose data.
                      Best for: Non-critical updates, caches.

    MERGE: Intelligent merging using vector clocks.
           Falls back to manual queue if unresolvable.
           Best for: Collaborative editing, belief updates.

    MANUAL_QUEUE: Queue all conflicts for manual resolution.
                  Preserves all data but requires human intervention.
                  Best for: Critical data, compliance requirements.
    """
    LAST_WRITER_WINS = "last_writer_wins"
    MERGE = "merge"
    MANUAL_QUEUE = "manual_queue"


@dataclass
class ConflictResolution:
    """
    Result of conflict resolution.

    Attributes:
        winner: The memory that won the conflict (or merged result)
        strategy_used: Which strategy was applied
        needs_manual_review: Whether this requires human review
        conflict_metadata: Additional details about the conflict
    """
    winner: Memory
    strategy_used: ConflictStrategy
    needs_manual_review: bool = False
    conflict_metadata: Optional[Dict[str, Any]] = None


class ConflictResolver:
    """
    Resolves conflicts between memory instances using configurable strategies.

    When two instances modify the same memory concurrently, this resolver
    determines which version should be kept based on the configured strategy.

    Thread-safe for concurrent resolution operations.
    """

    def __init__(self, strategy: ConflictStrategy = ConflictStrategy.LAST_WRITER_WINS):
        """
        Initialize conflict resolver.

        Args:
            strategy: Resolution strategy to use (default: LAST_WRITER_WINS)
        """
        self.strategy = strategy
        logger.info(f"ConflictResolver initialized with strategy: {strategy.value}")

    def resolve(self, memory1: Memory, memory2: Memory) -> ConflictResolution:
        """
        Resolve conflict between two memory versions.

        Args:
            memory1: First memory version
            memory2: Second memory version

        Returns:
            ConflictResolution with winner and metadata

        Raises:
            ValueError: If memories have different IDs
        """
        if memory1.id != memory2.id:
            raise ValueError(
                f"Cannot resolve conflict: memory IDs don't match "
                f"({memory1.id} != {memory2.id})"
            )

        logger.debug(
            f"Resolving conflict for memory {memory1.id} using {self.strategy.value}"
        )

        # Dispatch to strategy-specific handler
        if self.strategy == ConflictStrategy.LAST_WRITER_WINS:
            return self._resolve_last_writer_wins(memory1, memory2)
        elif self.strategy == ConflictStrategy.MERGE:
            return self._resolve_merge(memory1, memory2)
        elif self.strategy == ConflictStrategy.MANUAL_QUEUE:
            return self._resolve_manual_queue(memory1, memory2)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _resolve_last_writer_wins(
        self, memory1: Memory, memory2: Memory
    ) -> ConflictResolution:
        """
        Resolve using last-writer-wins strategy.

        Decision order:
        1. Compare last_accessed timestamps (most recent wins)
        2. If equal, compare version numbers (higher wins)
        3. If equal, compare vector clock sums (higher wins)
        4. If still equal, prefer memory1 (arbitrary but deterministic)

        Args:
            memory1: First memory version
            memory2: Second memory version

        Returns:
            ConflictResolution with the winner
        """
        # Extract timestamps, handling None values
        ts1 = memory1.last_accessed if memory1.last_accessed else memory1.created_at
        ts2 = memory2.last_accessed if memory2.last_accessed else memory2.created_at

        # Compare timestamps
        if ts1 and ts2:
            if ts1 > ts2:
                winner = memory1
                reason = f"memory1 more recent ({ts1} > {ts2})"
            elif ts2 > ts1:
                winner = memory2
                reason = f"memory2 more recent ({ts2} > {ts1})"
            else:
                # Timestamps equal, compare versions
                if memory1.version > memory2.version:
                    winner = memory1
                    reason = f"memory1 higher version ({memory1.version} > {memory2.version})"
                elif memory2.version > memory1.version:
                    winner = memory2
                    reason = f"memory2 higher version ({memory2.version} > {memory1.version})"
                else:
                    # Versions equal, compare vector clock sums
                    vc1_sum = sum(memory1.vector_clock.values()) if memory1.vector_clock else 0
                    vc2_sum = sum(memory2.vector_clock.values()) if memory2.vector_clock else 0

                    if vc1_sum > vc2_sum:
                        winner = memory1
                        reason = f"memory1 higher vector clock sum ({vc1_sum} > {vc2_sum})"
                    elif vc2_sum > vc1_sum:
                        winner = memory2
                        reason = f"memory2 higher vector clock sum ({vc2_sum} > {vc1_sum})"
                    else:
                        # Complete tie, prefer memory1 (deterministic)
                        winner = memory1
                        reason = "complete tie, preferring memory1 (deterministic)"
        elif ts1:
            winner = memory1
            reason = "memory1 has timestamp, memory2 doesn't"
        elif ts2:
            winner = memory2
            reason = "memory2 has timestamp, memory1 doesn't"
        else:
            # No timestamps, fall back to version comparison
            if memory1.version >= memory2.version:
                winner = memory1
                reason = f"no timestamps, memory1 version >= memory2 ({memory1.version} >= {memory2.version})"
            else:
                winner = memory2
                reason = f"no timestamps, memory2 version > memory1 ({memory2.version} > {memory1.version})"

        logger.info(f"Last-writer-wins: {winner.id} won ({reason})")

        return ConflictResolution(
            winner=winner,
            strategy_used=ConflictStrategy.LAST_WRITER_WINS,
            needs_manual_review=False,
            conflict_metadata={
                "reason": reason,
                "memory1_timestamp": str(ts1) if ts1 else None,
                "memory2_timestamp": str(ts2) if ts2 else None,
                "memory1_version": memory1.version,
                "memory2_version": memory2.version
            }
        )

    def _resolve_merge(
        self, memory1: Memory, memory2: Memory
    ) -> ConflictResolution:
        """
        Resolve using merge strategy with vector clock comparison.

        Uses vector clocks to determine causal relationships:
        - If one memory's clock dominates the other, it's causally later (wins)
        - If clocks are concurrent (neither dominates), queue for manual review

        Vector clock A dominates B if:
        - All counters in A >= corresponding counters in B
        - At least one counter in A > corresponding counter in B

        Args:
            memory1: First memory version
            memory2: Second memory version

        Returns:
            ConflictResolution with winner or manual queue flag
        """
        # Get vector clocks, defaulting to empty dicts
        vc1 = memory1.vector_clock if memory1.vector_clock else {}
        vc2 = memory2.vector_clock if memory2.vector_clock else {}

        # If both are empty, fall back to last-writer-wins
        if not vc1 and not vc2:
            logger.info(
                f"Merge strategy: both vector clocks empty for {memory1.id}, "
                "falling back to last-writer-wins"
            )
            return self._resolve_last_writer_wins(memory1, memory2)

        # Compare vector clocks to determine causal relationship
        comparison = self._compare_vector_clocks(vc1, vc2)

        if comparison == "memory1_dominates":
            # memory1 is causally later, it wins
            logger.info(
                f"Merge strategy: memory1 dominates for {memory1.id}, "
                f"vc1={vc1}, vc2={vc2}"
            )
            return ConflictResolution(
                winner=memory1,
                strategy_used=ConflictStrategy.MERGE,
                needs_manual_review=False,
                conflict_metadata={
                    "reason": "memory1 vector clock dominates memory2",
                    "memory1_vector_clock": vc1,
                    "memory2_vector_clock": vc2,
                    "causal_relationship": "memory1_later"
                }
            )

        elif comparison == "memory2_dominates":
            # memory2 is causally later, it wins
            logger.info(
                f"Merge strategy: memory2 dominates for {memory2.id}, "
                f"vc1={vc1}, vc2={vc2}"
            )
            return ConflictResolution(
                winner=memory2,
                strategy_used=ConflictStrategy.MERGE,
                needs_manual_review=False,
                conflict_metadata={
                    "reason": "memory2 vector clock dominates memory1",
                    "memory1_vector_clock": vc1,
                    "memory2_vector_clock": vc2,
                    "causal_relationship": "memory2_later"
                }
            )

        else:
            # Concurrent modification - true conflict, needs manual review
            logger.warning(
                f"Merge strategy: concurrent conflict detected for {memory1.id}, "
                f"vc1={vc1}, vc2={vc2}, queuing for manual review"
            )
            # Prefer newer version as temporary winner while waiting for review
            temp_winner = self._resolve_last_writer_wins(memory1, memory2).winner

            return ConflictResolution(
                winner=temp_winner,
                strategy_used=ConflictStrategy.MERGE,
                needs_manual_review=True,
                conflict_metadata={
                    "reason": "concurrent modification detected",
                    "memory1_vector_clock": vc1,
                    "memory2_vector_clock": vc2,
                    "causal_relationship": "concurrent",
                    "temporary_winner": temp_winner.id,
                    "requires_manual_merge": True
                }
            )

    def _compare_vector_clocks(
        self, vc1: Dict[str, int], vc2: Dict[str, int]
    ) -> str:
        """
        Compare two vector clocks to determine causal relationship.

        Args:
            vc1: First vector clock (instance_id -> counter)
            vc2: Second vector clock (instance_id -> counter)

        Returns:
            "memory1_dominates" if vc1 >= vc2 and vc1 > vc2 for at least one counter
            "memory2_dominates" if vc2 >= vc1 and vc2 > vc1 for at least one counter
            "concurrent" if neither dominates (true conflict)
        """
        # Get all instance IDs from both clocks
        all_instances = set(vc1.keys()) | set(vc2.keys())

        if not all_instances:
            # Both empty
            return "concurrent"

        # Track comparison results
        vc1_greater_or_equal = True  # vc1 >= vc2 for all counters
        vc1_strictly_greater = False  # vc1 > vc2 for at least one counter

        vc2_greater_or_equal = True  # vc2 >= vc1 for all counters
        vc2_strictly_greater = False  # vc2 > vc1 for at least one counter

        for instance_id in all_instances:
            count1 = vc1.get(instance_id, 0)
            count2 = vc2.get(instance_id, 0)

            if count1 > count2:
                vc1_strictly_greater = True
                vc2_greater_or_equal = False
            elif count2 > count1:
                vc2_strictly_greater = True
                vc1_greater_or_equal = False

        # Determine dominance
        if vc1_greater_or_equal and vc1_strictly_greater:
            return "memory1_dominates"
        elif vc2_greater_or_equal and vc2_strictly_greater:
            return "memory2_dominates"
        else:
            return "concurrent"

    def _resolve_manual_queue(
        self, memory1: Memory, memory2: Memory
    ) -> ConflictResolution:
        """
        Queue conflict for manual resolution.

        TODO: Implement in subtask-3-3
        This will add the conflict to a queue in the database for human review.

        Args:
            memory1: First memory version
            memory2: Second memory version

        Returns:
            ConflictResolution marking need for manual review
        """
        raise NotImplementedError(
            "MANUAL_QUEUE strategy not yet implemented. "
            "Use LAST_WRITER_WINS for now."
        )

    def set_strategy(self, strategy: ConflictStrategy) -> None:
        """
        Change the conflict resolution strategy.

        Args:
            strategy: New strategy to use
        """
        logger.info(f"Changing conflict resolution strategy: {self.strategy.value} -> {strategy.value}")
        self.strategy = strategy

    def get_strategy(self) -> ConflictStrategy:
        """
        Get current conflict resolution strategy.

        Returns:
            Current strategy
        """
        return self.strategy
