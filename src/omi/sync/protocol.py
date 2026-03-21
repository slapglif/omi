"""
Sync Protocol Base Classes

Defines the abstract interface and data structures for distributed memory synchronization.
All sync implementations must inherit from SyncProtocol and implement its abstract methods.

This provides a pluggable architecture for different sync transports (HTTP, gRPC, WebSockets)
while maintaining consistent semantics for conflict resolution and topology management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """
    Sync topology configuration types.

    LEADER_FOLLOWER: One primary writer, multiple read replicas.
                     Followers sync from leader, leader handles all writes.
                     Provides strong consistency, simple conflict resolution.

    MULTI_LEADER: Multiple active writers with conflict resolution.
                  All instances can accept writes, conflicts resolved via
                  vector clocks and configurable strategies.
                  Provides high availability, eventual consistency.
    """
    LEADER_FOLLOWER = "leader_follower"
    MULTI_LEADER = "multi_leader"


class SyncState(Enum):
    """
    Instance sync state.

    INITIALIZING: Instance is starting up, not ready for sync
    SYNCING: Actively syncing with other instances
    ACTIVE: Fully synchronized and operational
    PARTITIONED: Network partition detected, operating independently
    RECONCILING: Reconnected after partition, resolving conflicts
    ERROR: Sync error occurred, manual intervention may be needed
    """
    INITIALIZING = "initializing"
    SYNCING = "syncing"
    ACTIVE = "active"
    PARTITIONED = "partitioned"
    RECONCILING = "reconciling"
    ERROR = "error"


class SyncOperation(Enum):
    """
    Types of sync operations.

    MEMORY_STORE: Store a new memory
    MEMORY_UPDATE: Update existing memory
    MEMORY_DELETE: Delete a memory
    EDGE_CREATE: Create a relationship edge
    EDGE_DELETE: Delete a relationship edge
    BELIEF_UPDATE: Update belief confidence
    CONSENSUS_VOTE: Record consensus vote
    HEARTBEAT: Keep-alive message
    BULK_SYNC_REQUEST: Request full database sync
    BULK_SYNC_RESPONSE: Response with full database dump
    """
    MEMORY_STORE = "memory_store"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    EDGE_CREATE = "edge_create"
    EDGE_DELETE = "edge_delete"
    BELIEF_UPDATE = "belief_update"
    CONSENSUS_VOTE = "consensus_vote"
    HEARTBEAT = "heartbeat"
    BULK_SYNC_REQUEST = "bulk_sync_request"
    BULK_SYNC_RESPONSE = "bulk_sync_response"


@dataclass
class SyncMessage:
    """
    Message structure for sync operations.

    All sync operations are encapsulated in SyncMessage objects for
    serialization and transport between instances.

    Attributes:
        operation: Type of sync operation
        instance_id: ID of the originating instance
        memory_id: ID of memory being synced (if applicable)
        content: Message payload (memory data, edge data, etc.)
        vector_clock: Vector clock for conflict resolution
        version: Version number for optimistic concurrency control
        timestamp: When the operation occurred
        metadata: Additional operation-specific data

    Example:
        message = SyncMessage(
            operation=SyncOperation.MEMORY_STORE,
            instance_id="instance-1",
            memory_id="abc123",
            content={"text": "Important fact", "type": "fact"},
            vector_clock={"instance-1": 5, "instance-2": 3},
            version=1
        )
    """
    operation: SyncOperation
    instance_id: str
    memory_id: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    vector_clock: Optional[Dict[str, int]] = None
    version: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation.value,
            "instance_id": self.instance_id,
            "memory_id": self.memory_id,
            "content": self.content,
            "vector_clock": self.vector_clock,
            "version": self.version,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncMessage':
        """Create SyncMessage from dictionary."""
        return cls(
            operation=SyncOperation(data["operation"]),
            instance_id=data["instance_id"],
            memory_id=data.get("memory_id"),
            content=data.get("content"),
            vector_clock=data.get("vector_clock"),
            version=data.get("version", 1),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            metadata=data.get("metadata")
        )


@dataclass
class SyncResponse:
    """
    Response to a sync operation.

    Attributes:
        success: Whether the operation succeeded
        message: Human-readable status message
        operation: The operation being responded to
        instance_id: ID of the responding instance
        conflicts: List of conflicts detected (if any)
        timestamp: When the response was generated
        metadata: Additional response data

    Example:
        response = SyncResponse(
            success=True,
            message="Memory stored successfully",
            operation=SyncOperation.MEMORY_STORE,
            instance_id="instance-2"
        )
    """
    success: bool
    message: str
    operation: SyncOperation
    instance_id: str
    conflicts: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "message": self.message,
            "operation": self.operation.value,
            "instance_id": self.instance_id,
            "conflicts": self.conflicts or [],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncResponse':
        """Create SyncResponse from dictionary."""
        return cls(
            success=data["success"],
            message=data["message"],
            operation=SyncOperation(data["operation"]),
            instance_id=data["instance_id"],
            conflicts=data.get("conflicts"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            metadata=data.get("metadata")
        )


class SyncProtocol(ABC):
    """
    Abstract base class for sync protocol implementations.

    This defines the interface that all sync implementations must provide,
    enabling pluggable transports (HTTP, gRPC, WebSockets, etc.) while
    maintaining consistent semantics.

    Implementations must handle:
    - Message serialization/deserialization
    - Network transport
    - Connection management
    - Error handling and retry logic
    - Heartbeat/keep-alive

    Example:
        class HTTPSyncProtocol(SyncProtocol):
            def __init__(self, instance_id: str, topology: TopologyType):
                super().__init__(instance_id, topology)
                self._session = requests.Session()

            async def send_message(self, message: SyncMessage, target_instance: str) -> SyncResponse:
                response = await self._session.post(
                    f"http://{target_instance}/sync",
                    json=message.to_dict()
                )
                return SyncResponse.from_dict(response.json())

            async def receive_message(self) -> Optional[SyncMessage]:
                # Poll or subscribe to incoming messages
                pass
    """

    def __init__(self, instance_id: str, topology: TopologyType):
        """
        Initialize sync protocol.

        Args:
            instance_id: Unique identifier for this instance
            topology: Topology type (LEADER_FOLLOWER or MULTI_LEADER)
        """
        self.instance_id = instance_id
        self.topology = topology
        self.state = SyncState.INITIALIZING
        logger.info(f"Initialized {self.__class__.__name__} for instance {instance_id} with topology {topology.value}")

    @abstractmethod
    async def send_message(self, message: SyncMessage, target_instance: str) -> SyncResponse:
        """
        Send a sync message to another instance.

        Args:
            message: The sync message to send
            target_instance: ID of the target instance

        Returns:
            SyncResponse from the target instance

        Raises:
            ConnectionError: If unable to reach target instance
            TimeoutError: If request times out
        """
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[SyncMessage]:
        """
        Receive the next sync message (non-blocking).

        Returns:
            SyncMessage if one is available, None otherwise

        Raises:
            ConnectionError: If connection is lost
        """
        pass

    @abstractmethod
    async def broadcast_message(self, message: SyncMessage) -> List[SyncResponse]:
        """
        Broadcast a sync message to all other instances.

        Args:
            message: The sync message to broadcast

        Returns:
            List of SyncResponse objects from all instances

        Raises:
            ConnectionError: If unable to reach any instances
        """
        pass

    @abstractmethod
    async def register_instance(self, instance_id: str, endpoint: str) -> bool:
        """
        Register a new instance in the sync cluster.

        Args:
            instance_id: ID of the instance to register
            endpoint: Network endpoint (URL, address, etc.)

        Returns:
            True if registration succeeded, False otherwise
        """
        pass

    @abstractmethod
    async def unregister_instance(self, instance_id: str) -> bool:
        """
        Remove an instance from the sync cluster.

        Args:
            instance_id: ID of the instance to unregister

        Returns:
            True if unregistration succeeded, False otherwise
        """
        pass

    @abstractmethod
    async def get_cluster_instances(self) -> List[Dict[str, Any]]:
        """
        Get list of all instances in the sync cluster.

        Returns:
            List of instance metadata dictionaries with keys:
            - instance_id: str
            - endpoint: str
            - state: SyncState
            - last_heartbeat: datetime
        """
        pass

    @abstractmethod
    async def send_heartbeat(self) -> None:
        """
        Send heartbeat to all instances to signal availability.

        Raises:
            ConnectionError: If unable to send heartbeat
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close all connections and cleanup resources.
        """
        pass

    def get_state(self) -> SyncState:
        """
        Get current sync state.

        Returns:
            Current SyncState
        """
        return self.state

    def set_state(self, state: SyncState) -> None:
        """
        Update sync state.

        Args:
            state: New SyncState
        """
        old_state = self.state
        self.state = state
        logger.info(f"Instance {self.instance_id} state changed: {old_state.value} -> {state.value}")
