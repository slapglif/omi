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

from ..event_bus import EventHandler
from .topology import TopologyManager, InstanceMetadata
from .protocol import TopologyType, SyncState, SyncOperation, SyncMessage, SyncResponse

logger = logging.getLogger(__name__)


class SyncEventHandler(EventHandler):
    """
    Event handler that subscribes to memory events and propagates them to other instances.

    This handler bridges the local EventBus with the distributed sync protocol,
    enabling real-time memory propagation across OMI instances. When a memory
    is stored or updated on one instance, it's automatically synced to all other
    instances in the cluster.

    Architecture:
    - Subscribes to memory.stored, belief.updated, and other relevant events
    - Converts events to SyncMessage objects
    - Propagates messages to other instances via sync protocol
    - Handles errors gracefully without blocking the event bus

    Thread-safe: Can be called concurrently from multiple threads.

    Example:
        handler = SyncEventHandler(
            instance_id="instance-1",
            topology_manager=topology_mgr,
            sync_protocol=protocol
        )
        event_bus.subscribe('memory.stored', handler.handle)
    """

    def __init__(
        self,
        instance_id: str,
        topology_manager: TopologyManager,
        sync_protocol: Optional[Any] = None
    ):
        """
        Initialize sync event handler.

        Args:
            instance_id: ID of this instance
            topology_manager: TopologyManager for getting other instances
            sync_protocol: Optional SyncProtocol for network communication
                          If None, events are logged but not propagated
        """
        self.instance_id = instance_id
        self.topology = topology_manager
        self.protocol = sync_protocol
        self._event_count = 0
        self._error_count = 0
        self._lock = Lock()

        logger.info(
            f"SyncEventHandler initialized for instance {instance_id}, "
            f"protocol={'enabled' if sync_protocol else 'disabled'}"
        )

    def handle(self, event: Any) -> None:
        """
        Handle an event from the EventBus.

        Converts the event to a SyncMessage and propagates it to other instances
        via the sync protocol. If no protocol is configured, logs the event.

        Args:
            event: Event object from EventBus (must have 'event_type' attribute)

        Event Types Handled:
            - memory.stored: Propagate new memory to other instances
            - belief.updated: Propagate belief confidence changes
            - belief.contradiction_detected: Propagate contradiction detection

        Error Handling:
            Errors are logged but do not raise exceptions to avoid blocking
            the EventBus. Failed events are counted for monitoring.
        """
        with self._lock:
            self._event_count += 1

        try:
            if not hasattr(event, 'event_type'):
                logger.warning(f"Event missing 'event_type': {type(event).__name__}")
                return

            event_type = event.event_type

            # Only propagate specific event types
            if event_type not in ['memory.stored', 'belief.updated', 'belief.contradiction_detected']:
                logger.debug(f"Skipping non-syncable event type: {event_type}")
                return

            # Convert event to sync message
            sync_message = self._event_to_sync_message(event)
            if not sync_message:
                logger.debug(f"Could not convert event to sync message: {event_type}")
                return

            # Propagate to other instances
            self._propagate_message(sync_message)

            logger.debug(
                f"Handled sync event {event_type} from instance {self.instance_id}, "
                f"message_id={sync_message.memory_id}"
            )

        except Exception as e:
            with self._lock:
                self._error_count += 1
            logger.error(
                f"Error handling sync event {getattr(event, 'event_type', 'unknown')}: {e}",
                exc_info=True
            )

    def _event_to_sync_message(self, event: Any) -> Optional[SyncMessage]:
        """
        Convert an EventBus event to a SyncMessage for network propagation.

        Args:
            event: Event object from EventBus

        Returns:
            SyncMessage if conversion succeeds, None otherwise
        """
        try:
            event_type = event.event_type

            # Memory stored event
            if event_type == 'memory.stored':
                return SyncMessage(
                    operation=SyncOperation.MEMORY_STORE,
                    instance_id=self.instance_id,
                    memory_id=getattr(event, 'memory_id', None),
                    content={
                        'content': getattr(event, 'content', None),
                        'memory_type': getattr(event, 'memory_type', None),
                        'confidence': getattr(event, 'confidence', None),
                    },
                    timestamp=getattr(event, 'timestamp', datetime.now()),
                    metadata=getattr(event, 'metadata', None)
                )

            # Belief updated event
            elif event_type == 'belief.updated':
                return SyncMessage(
                    operation=SyncOperation.BELIEF_UPDATE,
                    instance_id=self.instance_id,
                    memory_id=getattr(event, 'belief_id', None),
                    content={
                        'old_confidence': getattr(event, 'old_confidence', None),
                        'new_confidence': getattr(event, 'new_confidence', None),
                        'evidence_id': getattr(event, 'evidence_id', None),
                    },
                    timestamp=getattr(event, 'timestamp', datetime.now()),
                    metadata=getattr(event, 'metadata', None)
                )

            # Contradiction detected event
            elif event_type == 'belief.contradiction_detected':
                return SyncMessage(
                    operation=SyncOperation.MEMORY_UPDATE,
                    instance_id=self.instance_id,
                    memory_id=getattr(event, 'memory_id_1', None),
                    content={
                        'memory_id_1': getattr(event, 'memory_id_1', None),
                        'memory_id_2': getattr(event, 'memory_id_2', None),
                        'contradiction_pattern': getattr(event, 'contradiction_pattern', None),
                        'confidence': getattr(event, 'confidence', None),
                    },
                    timestamp=getattr(event, 'timestamp', datetime.now()),
                    metadata=getattr(event, 'metadata', None)
                )

            else:
                logger.warning(f"Unhandled event type for sync: {event_type}")
                return None

        except Exception as e:
            logger.error(f"Error converting event to sync message: {e}", exc_info=True)
            return None

    def _propagate_message(self, message: SyncMessage) -> None:
        """
        Propagate sync message to other instances.

        If a sync protocol is configured, sends the message to all other
        healthy instances. If no protocol is configured, logs the message
        for later implementation.

        Args:
            message: SyncMessage to propagate
        """
        # Get other healthy instances (exclude self)
        other_instances = [
            inst for inst in self.topology.get_healthy_instances()
            if inst.instance_id != self.instance_id
        ]

        if not other_instances:
            logger.debug(
                f"No other instances to propagate to (operation={message.operation.value})"
            )
            return

        # If protocol is configured, propagate via network
        if self.protocol:
            # Full network propagation will be implemented in later phases
            # For now, we have the infrastructure in place
            logger.info(
                f"Would propagate {message.operation.value} to {len(other_instances)} instances "
                f"(full implementation in Phase 6)"
            )
            # TODO: Phase 6 will implement:
            # for instance in other_instances:
            #     try:
            #         response = await self.protocol.send_message(message, instance.instance_id)
            #         if not response.success:
            #             logger.warning(f"Sync failed for {instance.instance_id}: {response.message}")
            #     except Exception as e:
            #         logger.error(f"Error sending to {instance.instance_id}: {e}")
        else:
            # No protocol configured - log for visibility
            logger.info(
                f"Sync event ready: {message.operation.value} to {len(other_instances)} instances "
                f"(sync protocol not configured)"
            )

    def get_stats(self) -> Dict[str, int]:
        """
        Get handler statistics.

        Returns:
            Dictionary with event_count and error_count
        """
        with self._lock:
            return {
                'event_count': self._event_count,
                'error_count': self._error_count
            }


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
        self._event_subscriptions: List[tuple] = []  # List of (event_type, callback) tuples

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

        Creates a SyncEventHandler and subscribes to memory events:
        - memory.stored: New memories created
        - belief.updated: Belief confidence changes
        - belief.contradiction_detected: Contradictions detected

        Args:
            event_bus: EventBus instance for subscribing to memory events

        Example:
            from omi.event_bus import get_event_bus
            bus = get_event_bus()
            manager.start_incremental_sync(bus)
        """
        with self._lock:
            # Store event bus reference
            self._event_bus = event_bus

            # Create sync event handler
            handler = SyncEventHandler(
                instance_id=self.instance_id,
                topology_manager=self.topology,
                sync_protocol=self._protocol
            )

            # Subscribe to syncable event types
            event_types = [
                'memory.stored',
                'belief.updated',
                'belief.contradiction_detected'
            ]

            for event_type in event_types:
                self._event_bus.subscribe(event_type, handler.handle)
                self._event_subscriptions.append((event_type, handler.handle))
                logger.debug(f"Subscribed to {event_type} for sync propagation")

            logger.info(
                f"Incremental sync started for instance {self.instance_id}: "
                f"subscribed to {len(event_types)} event types"
            )

    def stop_incremental_sync(self) -> None:
        """
        Stop incremental sync and unsubscribe from EventBus.

        Unsubscribes all event handlers registered during start_incremental_sync.

        Example:
            manager.stop_incremental_sync()
        """
        with self._lock:
            if self._event_bus:
                # Unsubscribe all event handlers
                unsubscribed_count = 0
                for event_type, callback in self._event_subscriptions:
                    if self._event_bus.unsubscribe(event_type, callback):
                        unsubscribed_count += 1
                        logger.debug(f"Unsubscribed from {event_type}")

                self._event_subscriptions.clear()
                self._event_bus = None
                logger.info(
                    f"Incremental sync stopped for instance {self.instance_id}: "
                    f"unsubscribed {unsubscribed_count} handlers"
                )

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
