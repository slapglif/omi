"""
OMI Sync Module - Distributed Memory Synchronization

This module provides distributed synchronization capabilities for OMI instances,
enabling leader-follower and multi-leader topologies with conflict resolution,
network partition handling, and eventual consistency guarantees.

Key Components:
    - SyncProtocol: Abstract base class for sync implementations
    - SyncMessage: Data structure for sync operations
    - TopologyType: Enum for sync topology configurations
    - SyncState: Enum for instance sync states
    - SyncOperation: Enum for sync operation types

Architecture:
    - Leader-Follower: One primary writer, multiple read replicas
    - Multi-Leader: Multiple active writers with conflict resolution
    - Incremental Sync: Real-time propagation via EventBus
    - Bulk Sync: Full database transfer via MoltVault

Usage:
    from omi.sync import SyncProtocol, TopologyType, SyncMessage

    # Create a sync protocol implementation
    protocol = MySyncProtocol(topology=TopologyType.LEADER_FOLLOWER)

    # Send sync message
    message = SyncMessage(
        operation=SyncOperation.MEMORY_STORE,
        memory_id="abc123",
        instance_id="instance-1"
    )
    protocol.send_message(message)
"""

from .protocol import (
    SyncProtocol,
    SyncMessage,
    SyncResponse,
    TopologyType,
    SyncState,
    SyncOperation,
)
from .topology import TopologyManager, InstanceMetadata
from .sync_manager import SyncManager

__all__ = [
    "SyncProtocol",
    "SyncMessage",
    "SyncResponse",
    "TopologyType",
    "SyncState",
    "SyncOperation",
    "TopologyManager",
    "InstanceMetadata",
    "SyncManager",
]
