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

        TODO: Implement in subtask-3-2
        This will use vector clocks to determine causal relationships
        and merge non-conflicting changes.

        Args:
            memory1: First memory version
            memory2: Second memory version

        Returns:
            ConflictResolution with merged result or manual queue
        """
        raise NotImplementedError(
            "MERGE strategy not yet implemented. "
            "Use LAST_WRITER_WINS or MANUAL_QUEUE for now."
        )

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
