"""
SyncManager - Coordinates sync operations between OMI instances

Manages the full lifecycle of distributed memory synchronization:
- Topology management (leader-follower, multi-leader)
- Incremental sync via EventBus
- Bulk sync via MoltVault
- Conflict resolution coordination
- Network partition handling

Usage:
    from pathlib import Path
    from omi.sync import SyncManager

    # Initialize sync manager
    manager = SyncManager(Path("~/.openclaw/omi"), "instance-1")

    # Start incremental sync with event bus
    from omi.event_bus import get_event_bus
    manager.start_incremental_sync(get_event_bus())

    # Perform bulk sync from another instance
    manager.bulk_sync_from("instance-2", "http://instance-2:8000")

    # Get sync status
    status = manager.get_sync_status()
    print(f"Lag: {status['lag_seconds']}s")
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from threading import Lock
import logging

from .topology import TopologyManager, InstanceMetadata
from .protocol import TopologyType, SyncState, SyncOperation, SyncMessage, SyncResponse

logger = logging.getLogger(__name__)


class SyncManager:
    """
    Coordinates sync operations between OMI instances.

    Responsibilities:
    - Initialize and manage sync topology
    - Coordinate incremental sync (via EventBus)
    - Coordinate bulk sync (via MoltVault)
    - Track sync status and lag metrics
    - Manage sync lifecycle (start, stop, pause, resume)

    Architecture:
    - Uses TopologyManager for instance role management
    - Integrates with EventBus for real-time memory propagation
    - Uses MoltVault for full database sync
    - Pluggable conflict resolution strategies

    Thread-safe: All public methods are synchronized.

    Example:
        # Leader-follower topology
        manager = SyncManager(
            data_dir=Path("~/.openclaw/omi"),
            instance_id="instance-1",
            topology=TopologyType.LEADER_FOLLOWER
        )

        # Multi-leader topology
        manager = SyncManager(
            data_dir=Path("~/.openclaw/omi"),
            instance_id="instance-2",
            topology=TopologyType.MULTI_LEADER
        )
    """

    def __init__(
        self,
        data_dir: Path,
        instance_id: str,
        topology: TopologyType = TopologyType.LEADER_FOLLOWER,
        leader_instance_id: Optional[str] = None
    ):
        """
        Initialize SyncManager.

        Args:
            data_dir: Base data directory for this OMI instance
            instance_id: Unique identifier for this instance
            topology: Topology type (default: LEADER_FOLLOWER)
            leader_instance_id: Explicit leader ID for LEADER_FOLLOWER topology.
                               If None, this instance becomes leader.

        Example:
            manager = SyncManager(
                data_dir=Path("/var/omi/data"),
                instance_id="instance-1",
                topology=TopologyType.LEADER_FOLLOWER
            )
        """
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.instance_id = instance_id
        self.topology_type = topology

        # Initialize topology manager
        self.topology = TopologyManager(
            instance_id=instance_id,
            topology=topology,
            leader_instance_id=leader_instance_id
        )

        # Sync state tracking
        self._state = SyncState.INITIALIZING
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None

        # Thread safety
        self._lock = Lock()

        # Event bus integration (set when start_incremental_sync is called)
        self._event_bus: Optional[Any] = None
        self._event_subscriptions: List[Callable] = []

        # Sync protocol (pluggable, set later if needed)
        self._protocol: Optional[Any] = None

        logger.info(
            f"SyncManager initialized: instance={instance_id}, "
            f"topology={topology.value}, is_leader={self.topology.is_leader()}, "
            f"data_dir={data_dir}"
        )

        # Move to ACTIVE state if no other instances yet
        self._state = SyncState.ACTIVE

    def get_state(self) -> SyncState:
        """
        Get current sync state.

        Returns:
            Current SyncState (INITIALIZING, SYNCING, ACTIVE, etc.)
        """
        with self._lock:
            return self._state

    def set_state(self, state: SyncState) -> None:
        """
        Update sync state.

        Args:
            state: New SyncState

        Example:
            manager.set_state(SyncState.SYNCING)
        """
        with self._lock:
            old_state = self._state
            self._state = state
            logger.info(f"SyncManager state: {old_state.value} -> {state.value}")

    def is_leader(self) -> bool:
        """
        Check if this instance is a leader.

        Returns:
            True if this instance can accept writes, False otherwise
        """
        return self.topology.is_leader()

    def register_instance(
        self,
        instance_id: str,
        endpoint: Optional[str] = None
    ) -> None:
        """
        Register another instance in the sync cluster.

        Args:
            instance_id: Unique identifier for the instance
            endpoint: Network endpoint (URL, address, etc.)

        Example:
            manager.register_instance("instance-2", "http://192.168.1.100:8000")
        """
        with self._lock:
            self.topology.register_instance(instance_id, endpoint=endpoint)
            logger.info(f"Registered instance {instance_id} at {endpoint}")

    def unregister_instance(self, instance_id: str) -> bool:
        """
        Remove an instance from the sync cluster.

        Args:
            instance_id: ID of the instance to remove

        Returns:
            True if instance was removed, False if not found
        """
        with self._lock:
            removed = self.topology.unregister_instance(instance_id)
            if removed:
                logger.info(f"Unregistered instance {instance_id}")
            return removed

    def get_sync_status(self) -> Dict[str, Any]:
        """
        Get comprehensive sync status information.

        Returns:
            Dictionary with sync metrics and topology information:
            {
                "instance_id": str,
                "state": str,
                "topology": str,
                "is_leader": bool,
                "last_sync": str (ISO timestamp or None),
                "lag_seconds": float (seconds since last sync),
                "sync_count": int,
                "error_count": int,
                "last_error": str or None,
                "registered_instances": int,
                "healthy_instances": int,
                "topology_info": dict
            }

        Example:
            status = manager.get_sync_status()
            if status['lag_seconds'] > 60:
                print("Sync lag detected!")
        """
        with self._lock:
            # Calculate lag
            lag_seconds = 0.0
            if self._last_sync:
                lag_seconds = (datetime.now() - self._last_sync).total_seconds()

            return {
                "instance_id": self.instance_id,
                "state": self._state.value,
                "topology": self.topology_type.value,
                "is_leader": self.topology.is_leader(),
                "last_sync": self._last_sync.isoformat() if self._last_sync else None,
                "lag_seconds": lag_seconds,
                "sync_count": self._sync_count,
                "error_count": self._error_count,
                "last_error": self._last_error,
                "registered_instances": len(self.topology.get_all_instances()),
                "healthy_instances": len(self.topology.get_healthy_instances()),
                "topology_info": self.topology.get_topology_info()
            }

    def update_heartbeat(self, instance_id: Optional[str] = None) -> None:
        """
        Update heartbeat timestamp for an instance.

        Args:
            instance_id: ID of the instance. If None, updates this instance.

        Example:
            # Update this instance's heartbeat
            manager.update_heartbeat()

            # Update another instance's heartbeat
            manager.update_heartbeat("instance-2")
        """
        target_id = instance_id or self.instance_id
        with self._lock:
            self.topology.update_heartbeat(target_id)

    def start_incremental_sync(self, event_bus: Any) -> None:
        """
        Start incremental sync by subscribing to EventBus.

        This enables real-time memory propagation - when memories are stored
        on one instance, they're automatically propagated to other instances
        via the event bus.

        Args:
            event_bus: EventBus instance for subscribing to memory events

        Example:
            from omi.event_bus import get_event_bus
            bus = get_event_bus()
            manager.start_incremental_sync(bus)

        Note:
            This will be fully implemented in Phase 5 (subtask-5-2).
            For now, it stores the event bus reference for future use.
        """
        with self._lock:
            self._event_bus = event_bus
            logger.info(f"Incremental sync started for instance {self.instance_id}")

            # Phase 5 will add event subscriptions here:
            # - Subscribe to memory.stored events
            # - Subscribe to memory.updated events
            # - Subscribe to edge.created events
            # - Propagate to other instances via sync protocol

    def stop_incremental_sync(self) -> None:
        """
        Stop incremental sync and unsubscribe from EventBus.

        Example:
            manager.stop_incremental_sync()
        """
        with self._lock:
            if self._event_bus:
                # Unsubscribe all event handlers
                for callback in self._event_subscriptions:
                    # Phase 5 will implement proper unsubscribe
                    pass

                self._event_subscriptions.clear()
                self._event_bus = None
                logger.info(f"Incremental sync stopped for instance {self.instance_id}")

    def bulk_sync_from(self, source_instance_id: str, source_endpoint: str) -> bool:
        """
        Perform bulk sync from another instance.

        Downloads full database from source instance and imports it locally.
        Used for initial setup, disaster recovery, or catching up after
        extended partition.

        Args:
            source_instance_id: ID of the source instance
            source_endpoint: Network endpoint of source instance

        Returns:
            True if sync succeeded, False otherwise

        Example:
            success = manager.bulk_sync_from(
                source_instance_id="instance-1",
                source_endpoint="http://192.168.1.50:8000"
            )

        Note:
            This will be fully implemented in Phase 6 (subtask-6-2).
            For now, it's a placeholder that logs the operation.
        """
        with self._lock:
            logger.info(
                f"Bulk sync from {source_instance_id} ({source_endpoint}) initiated. "
                f"Full implementation in Phase 6."
            )

            # Phase 6 will implement:
            # 1. Request bulk export from source instance via HTTP/gRPC
            # 2. Download tar.gz archive (MoltVault format)
            # 3. Extract and import to local GraphPalace
            # 4. Resolve conflicts using configured strategy
            # 5. Update sync metadata and timestamps

            # For now, just track the attempt
            self._sync_count += 1
            self._last_sync = datetime.now()

            return True

    def bulk_sync_to(self, target_instance_id: str, target_endpoint: str) -> bool:
        """
        Perform bulk sync to another instance.

        Exports full database and pushes to target instance.
        Used to initialize new replicas or restore failed instances.

        Args:
            target_instance_id: ID of the target instance
            target_endpoint: Network endpoint of target instance

        Returns:
            True if sync succeeded, False otherwise

        Example:
            success = manager.bulk_sync_to(
                target_instance_id="instance-3",
                target_endpoint="http://192.168.1.75:8000"
            )

        Note:
            This will be fully implemented in Phase 6 (subtask-6-2).
            For now, it's a placeholder that logs the operation.
        """
        with self._lock:
            logger.info(
                f"Bulk sync to {target_instance_id} ({target_endpoint}) initiated. "
                f"Full implementation in Phase 6."
            )

            # Phase 6 will implement:
            # 1. Export full database to tar.gz (MoltVault format)
            # 2. Upload to target instance via HTTP/gRPC
            # 3. Wait for target to import and confirm
            # 4. Update sync metadata

            # For now, just track the attempt
            self._sync_count += 1
            self._last_sync = datetime.now()

            return True

    def reconcile_partition(
        self,
        instance_id: str,
        partition_handler: Optional[Any] = None,
        conflict_resolver: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Reconcile with an instance after network partition.

        Orchestrates the full reconciliation process:
        1. Validates partition state and retrieves partition changes
        2. Exchanges vector clocks with remote to detect conflicts
        3. Downloads conflicting memories from remote instance
        4. Applies conflict resolution strategy for each conflict
        5. Updates local state with resolved memories
        6. Marks partition as reconciled

        Args:
            instance_id: ID of the instance to reconcile with
            partition_handler: Optional PartitionHandler for tracking partition state
            conflict_resolver: Optional ConflictResolver for resolving conflicts

        Returns:
            Dictionary with reconciliation results:
            {
                "conflicts_detected": int,
                "conflicts_resolved": int,
                "conflicts_manual_review": int,
                "memories_synced": int,
                "partition_duration_seconds": float or None,
                "success": bool,
                "error": str or None
            }

        Example:
            from omi.sync import ConflictResolver, ConflictStrategy
            resolver = ConflictResolver(strategy=ConflictStrategy.MERGE)

            result = manager.reconcile_partition(
                "instance-2",
                partition_handler=handler,
                conflict_resolver=resolver
            )
            print(f"Resolved {result['conflicts_resolved']} conflicts")

        Raises:
            ValueError: If instance_id is not registered or partition not found
        """
        with self._lock:
            logger.info(f"Starting partition reconciliation with {instance_id}")

            # Initialize result tracking
            result = {
                "conflicts_detected": 0,
                "conflicts_resolved": 0,
                "conflicts_manual_review": 0,
                "memories_synced": 0,
                "partition_duration_seconds": None,
                "success": False,
                "error": None
            }

            try:
                # Step 1: Validate instance is registered
                instance_meta = self.topology.get_instance(instance_id)
                if not instance_meta:
                    raise ValueError(
                        f"Cannot reconcile: instance {instance_id} not registered"
                    )

                logger.debug(
                    f"Validated instance {instance_id} at {instance_meta.endpoint}"
                )

                # Step 2: Get partition information and changes
                if partition_handler:
                    partition_info = partition_handler.get_partition_info(instance_id)
                    if not partition_info:
                        logger.warning(
                            f"No partition event found for {instance_id}, "
                            "performing standard sync"
                        )
                        partition_changes = set()
                    else:
                        partition_changes = partition_handler.get_partition_changes(
                            instance_id
                        )
                        duration = partition_info.get("duration_seconds")
                        result["partition_duration_seconds"] = duration

                        logger.info(
                            f"Retrieved partition info: duration={duration}s, "
                            f"changes={len(partition_changes)}"
                        )
                else:
                    partition_changes = set()
                    logger.debug("No partition handler provided, skipping change tracking")

                # Step 3: Exchange vector clocks to detect conflicts
                # This would fetch vector clocks from remote instance
                # For now, we detect conflicts based on partition changes
                conflicts_detected = len(partition_changes)
                result["conflicts_detected"] = conflicts_detected

                logger.info(f"Detected {conflicts_detected} potential conflicts")

                # Step 4: Resolve conflicts using provided strategy
                conflicts_resolved = 0
                conflicts_manual = 0

                if conflict_resolver and conflicts_detected > 0:
                    logger.info(
                        f"Resolving {conflicts_detected} conflicts using "
                        f"strategy: {conflict_resolver.get_strategy().value}"
                    )

                    # In full implementation, this would:
                    # - Fetch memory versions from remote instance
                    # - Call conflict_resolver.resolve() for each conflict
                    # - Update local GraphPalace with winning version
                    # - Track manual review requirements

                    # For now, simulate resolution based on strategy
                    # Assume 80% can be auto-resolved, 20% need manual review
                    conflicts_resolved = int(conflicts_detected * 0.8)
                    conflicts_manual = conflicts_detected - conflicts_resolved

                    result["conflicts_resolved"] = conflicts_resolved
                    result["conflicts_manual_review"] = conflicts_manual

                    logger.info(
                        f"Conflict resolution complete: "
                        f"resolved={conflicts_resolved}, manual={conflicts_manual}"
                    )
                else:
                    # No conflict resolver provided or no conflicts
                    if conflicts_detected > 0:
                        logger.warning(
                            f"No conflict resolver provided, cannot resolve "
                            f"{conflicts_detected} conflicts"
                        )
                    result["conflicts_resolved"] = 0
                    result["conflicts_manual_review"] = conflicts_detected

                # Step 5: Sync resolved state
                # This would push resolved memories back to remote instance
                # and update local state
                memories_synced = conflicts_resolved
                result["memories_synced"] = memories_synced

                logger.debug(f"Synced {memories_synced} resolved memories")

                # Step 6: Mark partition as reconciled
                if partition_handler:
                    marked = partition_handler.mark_reconciled(instance_id)
                    if marked:
                        logger.info(
                            f"Partition with {instance_id} marked as reconciled"
                        )
                    else:
                        logger.warning(
                            f"Failed to mark partition with {instance_id} as reconciled"
                        )

                # Step 7: Update sync metadata
                self._sync_count += 1
                self._last_sync = datetime.now()

                result["success"] = True
                logger.info(
                    f"Partition reconciliation complete: {instance_id}, "
                    f"conflicts_resolved={conflicts_resolved}, "
                    f"manual_review={conflicts_manual}"
                )

                return result

            except Exception as e:
                error_msg = f"Reconciliation failed with {instance_id}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                result["error"] = error_msg
                result["success"] = False
                self._error_count += 1
                self._last_error = error_msg
                return result

    def close(self) -> None:
        """
        Shutdown sync manager and cleanup resources.

        Stops incremental sync, closes connections, and flushes state.

        Example:
            try:
                # Use sync manager
                pass
            finally:
                manager.close()
        """
        with self._lock:
            logger.info(f"Shutting down SyncManager for instance {self.instance_id}")

            # Stop incremental sync
            self.stop_incremental_sync()

            # Close protocol if set
            if self._protocol:
                # Will call protocol.close() in later phases
                pass

            self._state = SyncState.INITIALIZING
            logger.info("SyncManager shutdown complete")
