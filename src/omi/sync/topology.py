"""
Topology Management for Distributed Sync

Manages instance roles and relationships in leader-follower and multi-leader
configurations. Handles leader election, role assignment, and topology metadata.

Pattern: Explicit leadership assignment with health-based failover.
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .protocol import TopologyType, SyncState

logger = logging.getLogger(__name__)


@dataclass
class InstanceMetadata:
    """
    Metadata for a sync instance.

    Attributes:
        instance_id: Unique instance identifier
        is_leader: Whether this instance is a leader
        endpoint: Network endpoint (URL, address, etc.)
        state: Current sync state
        last_heartbeat: Timestamp of last heartbeat
        joined_at: When instance joined the cluster
    """
    instance_id: str
    is_leader: bool = False
    endpoint: Optional[str] = None
    state: SyncState = SyncState.INITIALIZING
    last_heartbeat: datetime = None
    joined_at: datetime = None

    def __post_init__(self):
        """Set default timestamps."""
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
        if self.joined_at is None:
            self.joined_at = datetime.now()

    def is_healthy(self, timeout_seconds: int = 30) -> bool:
        """
        Check if instance is healthy based on heartbeat.

        Args:
            timeout_seconds: Heartbeat timeout threshold

        Returns:
            True if instance has recent heartbeat, False otherwise
        """
        if self.last_heartbeat is None:
            return False

        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "is_leader": self.is_leader,
            "endpoint": self.endpoint,
            "state": self.state.value if self.state else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
        }


class TopologyManager:
    """
    Manages sync topology and instance roles.

    Responsibilities:
    - Track all instances in the sync cluster
    - Assign and manage leader/follower roles
    - Handle leader election and failover
    - Provide topology queries (is_leader, get_leaders, etc.)

    Leader Election Strategy:
    - LEADER_FOLLOWER: Single designated leader, explicit failover
    - MULTI_LEADER: All instances are leaders

    Pattern: Simple, deterministic leadership for predictable behavior.
    No complex consensus protocols - OMI values simplicity over Byzantine fault tolerance.

    Example:
        # Leader-follower topology
        tm = TopologyManager('instance-1', TopologyType.LEADER_FOLLOWER)
        assert tm.is_leader()  # First instance is leader by default

        # Multi-leader topology
        tm = TopologyManager('instance-2', TopologyType.MULTI_LEADER)
        assert tm.is_leader()  # All instances are leaders
    """

    def __init__(
        self,
        instance_id: str,
        topology: TopologyType,
        leader_instance_id: Optional[str] = None
    ):
        """
        Initialize topology manager.

        Args:
            instance_id: Unique identifier for this instance
            topology: Topology type (LEADER_FOLLOWER or MULTI_LEADER)
            leader_instance_id: Explicit leader designation (LEADER_FOLLOWER only).
                               If None, this instance becomes leader.
        """
        self.instance_id = instance_id
        self.topology = topology
        self._instances: Dict[str, InstanceMetadata] = {}

        # Determine leader based on topology
        if topology == TopologyType.LEADER_FOLLOWER:
            # Explicit leader or this instance
            self._leader_id = leader_instance_id or instance_id
            is_leader = (instance_id == self._leader_id)
        else:
            # Multi-leader: everyone is a leader
            self._leader_id = None
            is_leader = True

        # Register this instance
        self._instances[instance_id] = InstanceMetadata(
            instance_id=instance_id,
            is_leader=is_leader,
            state=SyncState.INITIALIZING
        )

        logger.info(
            f"TopologyManager initialized: instance={instance_id}, "
            f"topology={topology.value}, is_leader={is_leader}"
        )

    def is_leader(self) -> bool:
        """
        Check if this instance is a leader.

        Returns:
            True if this instance can accept writes, False otherwise
        """
        metadata = self._instances.get(self.instance_id)
        if not metadata:
            return False
        return metadata.is_leader

    def get_leader_id(self) -> Optional[str]:
        """
        Get the current leader instance ID.

        Returns:
            Leader instance ID for LEADER_FOLLOWER topology, None for MULTI_LEADER
        """
        return self._leader_id

    def get_leaders(self) -> List[str]:
        """
        Get all leader instance IDs.

        Returns:
            List of leader instance IDs
        """
        return [
            instance_id
            for instance_id, metadata in self._instances.items()
            if metadata.is_leader
        ]

    def get_followers(self) -> List[str]:
        """
        Get all follower instance IDs.

        Returns:
            List of follower instance IDs (empty for MULTI_LEADER)
        """
        return [
            instance_id
            for instance_id, metadata in self._instances.items()
            if not metadata.is_leader
        ]

    def register_instance(
        self,
        instance_id: str,
        endpoint: Optional[str] = None,
        is_leader: Optional[bool] = None
    ) -> None:
        """
        Register a new instance in the cluster.

        Args:
            instance_id: Unique identifier for the instance
            endpoint: Network endpoint (URL, address, etc.)
            is_leader: Explicit leader status. If None, determined by topology.
        """
        # Determine leader status based on topology
        if is_leader is None:
            if self.topology == TopologyType.LEADER_FOLLOWER:
                is_leader = (instance_id == self._leader_id)
            else:
                is_leader = True

        self._instances[instance_id] = InstanceMetadata(
            instance_id=instance_id,
            is_leader=is_leader,
            endpoint=endpoint,
            state=SyncState.INITIALIZING
        )

        logger.info(f"Registered instance: {instance_id} (leader={is_leader})")

    def unregister_instance(self, instance_id: str) -> bool:
        """
        Remove an instance from the cluster.

        Args:
            instance_id: ID of the instance to remove

        Returns:
            True if instance was removed, False if not found
        """
        if instance_id in self._instances:
            metadata = self._instances.pop(instance_id)
            logger.info(f"Unregistered instance: {instance_id}")

            # Handle leader removal in LEADER_FOLLOWER mode
            if self.topology == TopologyType.LEADER_FOLLOWER and metadata.is_leader:
                logger.warning(
                    f"Leader instance {instance_id} removed. "
                    "Manual failover required - call promote_to_leader()"
                )

            return True

        return False

    def promote_to_leader(self, instance_id: Optional[str] = None) -> bool:
        """
        Promote an instance to leader (LEADER_FOLLOWER only).

        This is used for manual failover when the current leader fails.

        Args:
            instance_id: Instance to promote. If None, promotes this instance.

        Returns:
            True if promotion succeeded, False otherwise
        """
        if self.topology != TopologyType.LEADER_FOLLOWER:
            logger.error("promote_to_leader only valid for LEADER_FOLLOWER topology")
            return False

        # Target instance
        target_id = instance_id or self.instance_id

        if target_id not in self._instances:
            logger.error(f"Cannot promote unknown instance: {target_id}")
            return False

        # Demote current leader
        if self._leader_id and self._leader_id in self._instances:
            self._instances[self._leader_id].is_leader = False

        # Promote new leader
        self._leader_id = target_id
        self._instances[target_id].is_leader = True

        logger.info(f"Promoted instance {target_id} to leader")
        return True

    def update_heartbeat(self, instance_id: str) -> None:
        """
        Update heartbeat timestamp for an instance.

        Args:
            instance_id: ID of the instance sending heartbeat
        """
        if instance_id in self._instances:
            self._instances[instance_id].last_heartbeat = datetime.now()

    def update_state(self, instance_id: str, state: SyncState) -> None:
        """
        Update sync state for an instance.

        Args:
            instance_id: ID of the instance
            state: New sync state
        """
        if instance_id in self._instances:
            self._instances[instance_id].state = state
            logger.debug(f"Instance {instance_id} state updated to {state.value}")

    def get_instance(self, instance_id: str) -> Optional[InstanceMetadata]:
        """
        Get metadata for a specific instance.

        Args:
            instance_id: ID of the instance

        Returns:
            InstanceMetadata if found, None otherwise
        """
        return self._instances.get(instance_id)

    def get_all_instances(self) -> List[InstanceMetadata]:
        """
        Get all registered instances.

        Returns:
            List of InstanceMetadata objects
        """
        return list(self._instances.values())

    def get_healthy_instances(self, timeout_seconds: int = 30) -> List[str]:
        """
        Get instance IDs with recent heartbeats.

        Args:
            timeout_seconds: Heartbeat timeout threshold

        Returns:
            List of healthy instance IDs
        """
        return [
            instance_id
            for instance_id, metadata in self._instances.items()
            if metadata.is_healthy(timeout_seconds)
        ]

    def get_unhealthy_instances(self, timeout_seconds: int = 30) -> List[str]:
        """
        Get instance IDs with stale heartbeats.

        Args:
            timeout_seconds: Heartbeat timeout threshold

        Returns:
            List of unhealthy instance IDs
        """
        return [
            instance_id
            for instance_id, metadata in self._instances.items()
            if not metadata.is_healthy(timeout_seconds)
        ]

    def check_leader_health(self, timeout_seconds: int = 30) -> bool:
        """
        Check if leader is healthy (LEADER_FOLLOWER only).

        Args:
            timeout_seconds: Heartbeat timeout threshold

        Returns:
            True if leader is healthy, False otherwise
        """
        if self.topology != TopologyType.LEADER_FOLLOWER:
            return True  # Not applicable for multi-leader

        if not self._leader_id or self._leader_id not in self._instances:
            return False

        return self._instances[self._leader_id].is_healthy(timeout_seconds)

    def get_topology_info(self) -> Dict[str, Any]:
        """
        Get complete topology information.

        Returns:
            Dictionary with topology details
        """
        return {
            "topology_type": self.topology.value,
            "instance_id": self.instance_id,
            "is_leader": self.is_leader(),
            "leader_id": self._leader_id,
            "total_instances": len(self._instances),
            "leaders": self.get_leaders(),
            "followers": self.get_followers(),
            "healthy_instances": self.get_healthy_instances(),
            "unhealthy_instances": self.get_unhealthy_instances(),
            "instances": [m.to_dict() for m in self._instances.values()]
        }
