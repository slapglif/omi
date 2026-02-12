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

    def reconcile_partition(self, instance_id: str) -> Dict[str, Any]:
        """
        Reconcile with an instance after network partition.

        Compares local and remote state, resolves conflicts, and synchronizes.

        Args:
            instance_id: ID of the instance to reconcile with

        Returns:
            Dictionary with reconciliation results:
            {
                "conflicts_detected": int,
                "conflicts_resolved": int,
                "memories_synced": int,
                "success": bool
            }

        Example:
            result = manager.reconcile_partition("instance-2")
            print(f"Resolved {result['conflicts_resolved']} conflicts")

        Note:
            This will be fully implemented in Phase 4 (subtask-4-2).
            For now, it's a placeholder that logs the operation.
        """
        with self._lock:
            logger.info(
                f"Reconciling partition with {instance_id}. "
                f"Full implementation in Phase 4."
            )

            # Phase 4 will implement:
            # 1. Exchange vector clocks to detect conflicts
            # 2. Download conflicting memories from remote
            # 3. Apply conflict resolution strategy
            # 4. Sync resolved state back

            return {
                "conflicts_detected": 0,
                "conflicts_resolved": 0,
                "memories_synced": 0,
                "success": True
            }

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
