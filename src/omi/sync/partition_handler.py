"""
Network Partition Handler for Distributed Sync

Detects and manages network partitions between instances. When a partition is detected,
instances continue operating independently and track changes that need reconciliation
when the partition heals.

Pattern: Failure detection via heartbeat timeout, explicit partition tracking,
eventual reconciliation when connectivity is restored.

Usage:
    handler = PartitionHandler('instance-1')

    # Detect partition by checking heartbeat failures
    if handler.detect_partition('instance-2'):
        handler.mark_partition_start('instance-2')
        # Instance continues operating independently

    # When connectivity restored
    if handler.check_connectivity('instance-2'):
        handler.mark_partition_end('instance-2')
        # Trigger reconciliation process
"""

from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
import logging

from .protocol import SyncState

logger = logging.getLogger(__name__)


@dataclass
class PartitionEvent:
    """
    Record of a network partition event.

    Attributes:
        instance_id: ID of the partitioned instance
        started_at: When the partition was detected
        ended_at: When connectivity was restored (None if still partitioned)
        memory_ids_during: Memory IDs created/modified during partition
        reconciled: Whether changes have been reconciled
    """
    instance_id: str
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    memory_ids_during: Set[str] = field(default_factory=set)
    reconciled: bool = False

    def duration(self) -> Optional[timedelta]:
        """
        Calculate partition duration.

        Returns:
            timedelta if partition has ended, None if still active
        """
        if self.ended_at:
            return self.ended_at - self.started_at
        return None

    def is_active(self) -> bool:
        """
        Check if partition is still active.

        Returns:
            True if partition has not been resolved, False otherwise
        """
        return self.ended_at is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "memory_ids_during": list(self.memory_ids_during),
            "reconciled": self.reconciled,
            "duration_seconds": self.duration().total_seconds() if self.duration() else None,
            "is_active": self.is_active()
        }


class PartitionHandler:
    """
    Manages network partition detection and recovery.

    Responsibilities:
    - Detect network partitions via heartbeat monitoring
    - Track which instances are partitioned
    - Record changes made during partition for reconciliation
    - Manage partition lifecycle (start, end, reconciliation)
    - Provide partition status queries

    Partition Detection Strategy:
    - Monitor heartbeat failures from instances
    - Multiple consecutive failures trigger partition detection
    - Configurable timeout and failure threshold

    Pattern: Conservative detection (avoid false positives), optimistic reconciliation
    (assume eventual connectivity restoration).

    Thread Safety:
    All public methods are thread-safe using a lock, similar to EventBus pattern.

    Example:
        handler = PartitionHandler('instance-1', failure_threshold=3)

        # Check for partition
        if handler.detect_partition('instance-2', last_heartbeat):
            handler.mark_partition_start('instance-2')
            logger.warning("Partition detected with instance-2")

        # Track changes during partition
        handler.track_change_during_partition('instance-2', 'memory-123')

        # Check connectivity restoration
        if handler.check_connectivity('instance-2'):
            handler.mark_partition_end('instance-2')
            changes = handler.get_partition_changes('instance-2')
            # Trigger reconciliation with changes
    """

    def __init__(
        self,
        instance_id: str,
        heartbeat_timeout: int = 30,
        failure_threshold: int = 3
    ):
        """
        Initialize partition handler.

        Args:
            instance_id: Unique identifier for this instance
            heartbeat_timeout: Seconds without heartbeat before considering failure
            failure_threshold: Number of consecutive failures to trigger partition detection
        """
        self.instance_id = instance_id
        self.heartbeat_timeout = heartbeat_timeout
        self.failure_threshold = failure_threshold

        # Track active and historical partition events
        self._active_partitions: Dict[str, PartitionEvent] = {}
        self._partition_history: List[PartitionEvent] = []

        # Track consecutive failures for each instance
        self._failure_counts: Dict[str, int] = {}
        self._last_heartbeats: Dict[str, datetime] = {}

        # Thread safety
        self._lock = Lock()

        logger.info(
            f"PartitionHandler initialized for {instance_id}: "
            f"timeout={heartbeat_timeout}s, threshold={failure_threshold}"
        )

    def update_heartbeat(self, instance_id: str) -> None:
        """
        Record successful heartbeat from an instance.

        This resets failure count and may trigger partition end detection.

        Args:
            instance_id: ID of the instance sending heartbeat
        """
        with self._lock:
            self._last_heartbeats[instance_id] = datetime.now()

            # Reset failure count on successful heartbeat
            if instance_id in self._failure_counts:
                self._failure_counts[instance_id] = 0

            # Check if this ends an active partition
            if instance_id in self._active_partitions:
                logger.info(
                    f"Connectivity restored with {instance_id} "
                    f"(heartbeat received during partition)"
                )

    def record_heartbeat_failure(self, instance_id: str) -> bool:
        """
        Record a heartbeat failure for an instance.

        Increments failure count and triggers partition detection if threshold reached.

        Args:
            instance_id: ID of the instance that failed heartbeat

        Returns:
            True if partition detected (threshold reached), False otherwise
        """
        with self._lock:
            # Increment failure count
            current_count = self._failure_counts.get(instance_id, 0) + 1
            self._failure_counts[instance_id] = current_count

            logger.debug(
                f"Heartbeat failure from {instance_id}: "
                f"{current_count}/{self.failure_threshold}"
            )

            # Check if threshold reached
            if current_count >= self.failure_threshold:
                if instance_id not in self._active_partitions:
                    # New partition detected
                    self.mark_partition_start(instance_id)
                    return True

            return False

    def detect_partition(
        self,
        instance_id: str,
        last_heartbeat: Optional[datetime] = None
    ) -> bool:
        """
        Check if instance is partitioned based on heartbeat timing.

        Args:
            instance_id: ID of the instance to check
            last_heartbeat: Optional explicit last heartbeat time.
                          If None, uses tracked heartbeat.

        Returns:
            True if partition detected, False otherwise
        """
        with self._lock:
            # Use provided heartbeat or tracked one
            if last_heartbeat:
                heartbeat = last_heartbeat
            else:
                heartbeat = self._last_heartbeats.get(instance_id)

            if not heartbeat:
                # No heartbeat recorded yet
                return False

            # Calculate time since last heartbeat
            elapsed = (datetime.now() - heartbeat).total_seconds()

            # Partition detected if timeout exceeded
            is_partitioned = elapsed > self.heartbeat_timeout

            if is_partitioned and instance_id not in self._active_partitions:
                logger.warning(
                    f"Partition detected: {instance_id} "
                    f"(last heartbeat {elapsed:.1f}s ago, timeout={self.heartbeat_timeout}s)"
                )

            return is_partitioned

    def mark_partition_start(self, instance_id: str) -> None:
        """
        Mark the start of a partition with an instance.

        Creates a PartitionEvent and transitions to PARTITIONED state.

        Args:
            instance_id: ID of the partitioned instance
        """
        with self._lock:
            if instance_id in self._active_partitions:
                logger.debug(f"Partition with {instance_id} already active")
                return

            # Create partition event
            event = PartitionEvent(instance_id=instance_id)
            self._active_partitions[instance_id] = event

            logger.warning(
                f"Partition started: {self.instance_id} <-> {instance_id}"
            )

    def mark_partition_end(self, instance_id: str) -> bool:
        """
        Mark the end of a partition with an instance.

        Moves partition event to history and prepares for reconciliation.

        Args:
            instance_id: ID of the instance with restored connectivity

        Returns:
            True if partition was active and ended, False if no active partition
        """
        with self._lock:
            if instance_id not in self._active_partitions:
                logger.debug(f"No active partition with {instance_id} to end")
                return False

            # End the partition event
            event = self._active_partitions.pop(instance_id)
            event.ended_at = datetime.now()

            # Move to history
            self._partition_history.append(event)

            # Reset failure count
            self._failure_counts[instance_id] = 0

            duration = event.duration()
            logger.info(
                f"Partition ended: {self.instance_id} <-> {instance_id} "
                f"(duration: {duration.total_seconds():.1f}s, "
                f"changes: {len(event.memory_ids_during)})"
            )

            return True

    def check_connectivity(self, instance_id: str) -> bool:
        """
        Actively check if connectivity has been restored with an instance.

        This is a manual check, typically called after failed heartbeats
        to confirm partition healing.

        Args:
            instance_id: ID of the instance to check

        Returns:
            True if connectivity restored, False if still partitioned
        """
        with self._lock:
            # Check if heartbeat is recent
            heartbeat = self._last_heartbeats.get(instance_id)
            if not heartbeat:
                return False

            elapsed = (datetime.now() - heartbeat).total_seconds()
            return elapsed <= self.heartbeat_timeout

    def track_change_during_partition(
        self,
        instance_id: str,
        memory_id: str
    ) -> None:
        """
        Track a memory change made during an active partition.

        This records changes that need reconciliation when partition heals.

        Args:
            instance_id: ID of the partitioned instance
            memory_id: ID of the memory that was created/modified
        """
        with self._lock:
            if instance_id in self._active_partitions:
                self._active_partitions[instance_id].memory_ids_during.add(memory_id)
                logger.debug(
                    f"Tracked change during partition with {instance_id}: {memory_id}"
                )

    def get_partition_changes(self, instance_id: str) -> Set[str]:
        """
        Get memory IDs changed during the last partition with an instance.

        Args:
            instance_id: ID of the instance

        Returns:
            Set of memory IDs changed during partition
        """
        with self._lock:
            # Check active partition first
            if instance_id in self._active_partitions:
                return self._active_partitions[instance_id].memory_ids_during.copy()

            # Check recent history
            for event in reversed(self._partition_history):
                if event.instance_id == instance_id:
                    return event.memory_ids_during.copy()

            return set()

    def is_partitioned(self, instance_id: Optional[str] = None) -> bool:
        """
        Check if currently partitioned.

        Args:
            instance_id: Optional specific instance to check.
                        If None, returns True if any partition is active.

        Returns:
            True if partitioned, False otherwise
        """
        with self._lock:
            if instance_id:
                return instance_id in self._active_partitions
            else:
                return len(self._active_partitions) > 0

    def get_active_partitions(self) -> List[str]:
        """
        Get list of currently partitioned instances.

        Returns:
            List of instance IDs with active partitions
        """
        with self._lock:
            return list(self._active_partitions.keys())

    def get_partition_info(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a partition.

        Args:
            instance_id: ID of the instance

        Returns:
            Partition event dictionary, or None if no partition found
        """
        with self._lock:
            # Check active partitions
            if instance_id in self._active_partitions:
                return self._active_partitions[instance_id].to_dict()

            # Check history
            for event in reversed(self._partition_history):
                if event.instance_id == instance_id:
                    return event.to_dict()

            return None

    def get_partition_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive partition statistics.

        Returns:
            Dictionary with partition metrics and history
        """
        with self._lock:
            active_count = len(self._active_partitions)
            historical_count = len(self._partition_history)

            # Calculate average partition duration from history
            durations = [
                event.duration().total_seconds()
                for event in self._partition_history
                if event.duration()
            ]
            avg_duration = sum(durations) / len(durations) if durations else 0

            # Count unreconciled partitions
            unreconciled = sum(
                1 for event in self._partition_history
                if not event.reconciled
            )

            return {
                "instance_id": self.instance_id,
                "active_partitions": active_count,
                "partitioned_instances": list(self._active_partitions.keys()),
                "historical_partitions": historical_count,
                "unreconciled_partitions": unreconciled,
                "average_duration_seconds": avg_duration,
                "heartbeat_timeout": self.heartbeat_timeout,
                "failure_threshold": self.failure_threshold,
                "active_partition_details": [
                    event.to_dict() for event in self._active_partitions.values()
                ]
            }

    def mark_reconciled(self, instance_id: str) -> bool:
        """
        Mark a partition as reconciled after conflict resolution.

        Args:
            instance_id: ID of the instance that was reconciled

        Returns:
            True if partition was found and marked, False otherwise
        """
        with self._lock:
            # Find the most recent partition event for this instance
            for event in reversed(self._partition_history):
                if event.instance_id == instance_id and not event.reconciled:
                    event.reconciled = True
                    logger.info(f"Marked partition with {instance_id} as reconciled")
                    return True

            return False

    def clear_history(self, keep_unreconciled: bool = True) -> int:
        """
        Clear partition history.

        Args:
            keep_unreconciled: If True, only clear reconciled partitions

        Returns:
            Number of entries cleared
        """
        with self._lock:
            if keep_unreconciled:
                before = len(self._partition_history)
                self._partition_history = [
                    event for event in self._partition_history
                    if not event.reconciled
                ]
                cleared = before - len(self._partition_history)
            else:
                cleared = len(self._partition_history)
                self._partition_history.clear()

            if cleared > 0:
                logger.info(f"Cleared {cleared} partition history entries")

            return cleared
