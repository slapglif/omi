"""
FastAPI REST API with Server-Sent Events (SSE) endpoint for event streaming.

Provides real-time event streaming via /api/v1/events SSE endpoint.
Clients can connect and receive all memory operation events as they occur.

Usage:
    Start API server:
        uvicorn omi.rest_api:app --reload --host 0.0.0.0 --port 8000

    Connect to SSE endpoint:
        curl -N http://localhost:8000/api/v1/events

    Or use EventSource in JavaScript:
        const events = new EventSource('http://localhost:8000/api/v1/events');
        events.onmessage = (e) => console.log(JSON.parse(e.data));
"""

from fastapi import FastAPI, Query, HTTPException, status, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from typing import Optional, AsyncGenerator, Dict, Any, List
from pydantic import BaseModel, Field
from pathlib import Path
import json
import asyncio
import logging
import os
import yaml

from .event_bus import get_event_bus
from .events import (
    MemoryStoredEvent,
    MemoryRecalledEvent,
    BeliefUpdatedEvent,
    ContradictionDetectedEvent,
    SessionStartedEvent,
    SessionEndedEvent
)
from .dashboard_api import router as dashboard_router
from .api import MemoryTools, BeliefTools, SharedNamespaceTools
from .storage.graph_palace import GraphPalace
from .embeddings import OllamaEmbedder, EmbeddingCache
from .belief import BeliefNetwork, ContradictionDetector
from .auth import APIKeyManager, RateLimiter
from .user_manager import UserManager, User
from .rbac import RBACManager
import sqlite3
import uuid
from .shared_namespace import SharedNamespace
from .permissions import PermissionManager
from .subscriptions import SubscriptionManager
from .audit_log import AuditLogger

logger = logging.getLogger(__name__)


# Global rate limiter instance (60 second sliding window)
rate_limiter = RateLimiter(window_seconds=60)


# API Key Authentication
# API key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    api_key: Optional[str] = Query(None)
) -> User:
    """
    Verify API key from X-API-Key header or api_key query parameter and return associated user.

    Checks both header and query parameter for API key.
    Header takes precedence if both are provided.

    Args:
        x_api_key: API key from X-API-Key request header
        api_key: API key from api_key query parameter

    Returns:
        User: User object associated with the API key

    Raises:
        HTTPException: 401 Unauthorized if API key is missing or invalid
    """
    base_path = Path.home() / '.openclaw' / 'omi'
    db_path = base_path / 'palace.sqlite'
    config_path = base_path / 'config.yaml'

    # Load config to check auth_required flag
    auth_required = True  # Default to requiring auth
    if config_path.exists():
        try:
            config_data = yaml.safe_load(config_path.read_text()) or {}
            # Check security.auth_required (default: true)
            security_config = config_data.get('security', {})
            auth_required = security_config.get('auth_required', True)
        except Exception as e:
            logger.warning(f"Failed to load config.yaml: {e}. Defaulting to auth_required=True")
            auth_required = True

    # If auth is disabled in config, allow all requests (development mode)
    if not auth_required:
        logger.info("Authentication disabled via config (security.auth_required=false)")
        return User(id="development", username="development", email=None)

    # Check if database exists and has any API keys
    if not db_path.exists():
        # No database yet, allow requests (development mode)
        logger.warning("No database found - authentication disabled (development mode)")
        return User(id="development", username="development", email=None)

    # Initialize APIKeyManager to check if any keys are configured
    key_manager = APIKeyManager(db_path)

    # Check if any API keys exist
    existing_keys = key_manager.list_keys()
    if len(existing_keys) == 0:
        # No API keys configured, allow all requests (development mode)
        logger.warning("No API keys configured - authentication disabled (development mode)")
        return User(id="development", username="development", email=None)

    # Determine which key to validate (prefer header over query param)
    provided_key = x_api_key if x_api_key is not None else api_key

    # Check if API key was provided
    if provided_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header or api_key query parameter.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # First try to verify via UserManager (database-backed API keys)
    user_manager = get_user_manager()
    user = user_manager.verify_api_key(provided_key)

    if user is None:
        # API key not found in UserManager, check legacy OMI_API_KEY env var
        expected_key = os.environ.get("OMI_API_KEY")
        if expected_key and provided_key == expected_key:
            # Legacy mode: API key matches environment variable
            logger.warning("Using legacy OMI_API_KEY authentication - consider migrating to database-backed API keys")
            user = User(id="legacy", username="legacy", email=None)
        else:
            # Fall back to APIKeyManager validation
            validated_key = key_manager.validate_key(provided_key)
            if validated_key is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or revoked API key",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            # Check rate limit for this API key
            allowed, retry_after = rate_limiter.check_rate_limit(
                api_key=provided_key,
                limit=validated_key.rate_limit
            )
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )
            user = User(id=provided_key, username=provided_key, email=None)

    return user


# Pydantic models for request/response
class StoreMemoryRequest(BaseModel):
    """Request body for storing a memory."""
    content: str = Field(..., description="Memory content to store")
    memory_type: str = Field(default="experience", description="Type: fact|experience|belief|decision")
    related_to: Optional[List[str]] = Field(default=None, description="IDs of related memories")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")


class StoreMemoryResponse(BaseModel):
    """Response after storing a memory."""
    memory_id: str = Field(..., description="UUID of stored memory")
    message: str = Field(default="Memory stored successfully")


class RecallMemoryResponse(BaseModel):
    """Response containing recalled memories with pagination."""
    memories: List[dict] = Field(..., description="List of recalled memories")
    count: int = Field(..., description="Number of memories returned")
    next_cursor: str = Field(default="", description="Cursor for next page (empty if no more results)")
    has_more: bool = Field(..., description="Boolean indicating if more results exist")


class CreateBeliefRequest(BaseModel):
    """Request body for creating a belief."""
    content: str = Field(..., description="Belief statement")
    initial_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Starting confidence (0.0-1.0)")


class CreateBeliefResponse(BaseModel):
    """Response after creating a belief."""
    belief_id: str = Field(..., description="UUID of created belief")
    message: str = Field(default="Belief created successfully")


class UpdateBeliefRequest(BaseModel):
    """Request body for updating a belief with evidence."""
    evidence_memory_id: str = Field(..., description="ID of evidence memory")
    supports: bool = Field(..., description="True if evidence supports the belief, False if contradicts")
    strength: float = Field(..., ge=0.0, le=1.0, description="Evidence strength (0.0-1.0)")


class UpdateBeliefResponse(BaseModel):
    """Response after updating a belief."""
    new_confidence: float = Field(..., description="Updated confidence value")
    message: str = Field(default="Belief updated successfully")


class StartSessionRequest(BaseModel):
    """Request body for starting a session."""
    session_id: Optional[str] = Field(default=None, description="Optional session ID (auto-generated if not provided)")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata for the session")


class StartSessionResponse(BaseModel):
    """Response after starting a session."""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(default="Session started successfully")


class EndSessionRequest(BaseModel):
    """Request body for ending a session."""
    session_id: str = Field(..., description="Session ID to end")
    duration_seconds: Optional[float] = Field(default=None, description="Optional session duration in seconds")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata for session end")


class EndSessionResponse(BaseModel):
    """Response after ending a session."""
    session_id: str = Field(..., description="Ended session ID")
    message: str = Field(default="Session ended successfully")


class SyncStatusResponse(BaseModel):
    """Response containing sync status."""
    instance_id: str = Field(..., description="This instance ID")
    state: str = Field(..., description="Current sync state")
    topology: str = Field(..., description="Topology type (leader-follower or multi-leader)")
    is_leader: bool = Field(..., description="Whether this instance is a leader")
    last_sync: Optional[str] = Field(default=None, description="Timestamp of last sync")
    lag_seconds: Optional[float] = Field(default=None, description="Sync lag in seconds")
    sync_count: int = Field(..., description="Number of sync operations performed")
    error_count: int = Field(..., description="Number of sync errors")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    registered_instances: int = Field(..., description="Count of registered instances")
    healthy_instances: int = Field(..., description="Count of healthy instances")
    topology_info: dict = Field(..., description="Detailed topology information")


class BulkSyncRequest(BaseModel):
    """Request body for bulk sync operations."""
    instance_id: str = Field(..., description="Target/source instance ID")
    endpoint: str = Field(..., description="Network endpoint (URL)")


class BulkSyncResponse(BaseModel):
    """Response after bulk sync operation."""
    success: bool = Field(..., description="Whether sync completed successfully")
    instance_id: str = Field(..., description="Target/source instance ID")
    endpoint: str = Field(..., description="Network endpoint")
    message: str = Field(..., description="Operation result message")


class RegisterInstanceRequest(BaseModel):
    """Request body for registering an instance."""
    instance_id: str = Field(..., description="Unique identifier for instance")
    endpoint: Optional[str] = Field(default=None, description="Network endpoint (optional)")


class RegisterInstanceResponse(BaseModel):
    """Response after registering an instance."""
    status: str = Field(..., description="Registration status")
    instance_id: str = Field(..., description="Registered instance ID")
    endpoint: str = Field(..., description="Endpoint or 'not specified'")


class UnregisterInstanceResponse(BaseModel):
    """Response after unregistering an instance."""
    success: bool = Field(..., description="Whether instance was removed")
    instance_id: str = Field(..., description="Instance ID")
    message: str = Field(..., description="Operation result message")


class ReconcilePartitionRequest(BaseModel):
    """Request body for partition reconciliation."""
    instance_id: str = Field(..., description="ID of instance to reconcile with")


class IncrementalSyncResponse(BaseModel):
    """Response after starting/stopping incremental sync."""
    status: str = Field(..., description="Operation status (started/stopped)")
    message: str = Field(..., description="Status message")


# Admin endpoints Pydantic models
class CreateUserRequest(BaseModel):
    """Request body for creating a user."""
    username: str = Field(..., description="Unique username")
    email: Optional[str] = Field(default=None, description="User email address")
    role: Optional[str] = Field(default=None, description="Initial role to assign (admin, developer, reader, auditor)")


class CreateUserResponse(BaseModel):
    """Response after creating a user."""
    user_id: str = Field(..., description="UUID of created user")
    username: str = Field(..., description="Username")
    message: str = Field(default="User created successfully")


class UserResponse(BaseModel):
    """User information response."""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(default=None, description="Email address")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    roles: List[Dict[str, Any]] = Field(default_factory=list, description="Assigned roles")


class ListUsersResponse(BaseModel):
    """Response containing list of users."""
    users: List[UserResponse] = Field(..., description="List of users")
    count: int = Field(..., description="Number of users")


class AuditLogEntry(BaseModel):
    """Audit log entry."""
    id: str = Field(..., description="Audit log entry ID")
    user_id: str = Field(..., description="User who performed the action")
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource accessed")
    namespace: Optional[str] = Field(default=None, description="Namespace")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    timestamp: str = Field(..., description="Timestamp of the action")


class AuditLogResponse(BaseModel):
    """Response containing audit log entries."""
    entries: List[AuditLogEntry] = Field(..., description="List of audit log entries")
    count: int = Field(..., description="Number of entries")


class DeleteUserResponse(BaseModel):
    """Response after deleting a user."""
    user_id: str = Field(..., description="UUID of deleted user")
    message: str = Field(default="User deleted successfully")

class CreateNamespaceRequest(BaseModel):
    """Request body for creating a shared namespace."""
    namespace: str = Field(..., description="Namespace identifier (e.g., 'team-alpha/research')")
    created_by: str = Field(default="api-user", description="Agent ID creating the namespace")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata for the namespace")


class CreateNamespaceResponse(BaseModel):
    """Response after creating a shared namespace."""
    namespace: str = Field(..., description="Created namespace identifier")
    created_by: str = Field(..., description="Agent ID that created the namespace")
    created_at: str = Field(..., description="Creation timestamp")
    message: str = Field(default="Namespace created successfully")


class NamespaceInfoResponse(BaseModel):
    """Response containing namespace information."""
    namespace: str = Field(..., description="Namespace identifier")
    created_by: str = Field(..., description="Agent ID that created the namespace")
    created_at: str = Field(..., description="Creation timestamp")
    metadata: Optional[dict] = Field(default=None, description="Namespace metadata")


class ListNamespacesResponse(BaseModel):
    """Response containing list of namespaces."""
    namespaces: List[dict] = Field(..., description="List of namespace information")
    count: int = Field(..., description="Number of namespaces returned")


class DeleteNamespaceResponse(BaseModel):
    """Response after deleting a namespace."""
    success: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(default="Namespace deleted successfully")


class GrantPermissionRequest(BaseModel):
    """Request body for granting permissions."""
    agent_id: str = Field(..., description="Agent ID granting permission (must have ADMIN)")
    target_agent_id: str = Field(..., description="Agent ID to grant permission to")
    permission_level: str = Field(..., description="Permission level: read|write|admin")


class GrantPermissionResponse(BaseModel):
    """Response after granting permission."""
    namespace: str = Field(..., description="Namespace identifier")
    agent_id: str = Field(..., description="Agent ID that received permission")
    permission_level: str = Field(..., description="Granted permission level")
    message: str = Field(default="Permission granted successfully")


class RevokePermissionResponse(BaseModel):
    """Response after revoking permission."""
    success: bool = Field(..., description="Whether revocation was successful")
    message: str = Field(default="Permission revoked successfully")


class ListPermissionsResponse(BaseModel):
    """Response containing list of permissions."""
    permissions: List[dict] = Field(..., description="List of permission information")
    count: int = Field(..., description="Number of permissions returned")


class CheckPermissionResponse(BaseModel):
    """Response after checking permission."""
    namespace: str = Field(..., description="Namespace identifier")
    agent_id: str = Field(..., description="Agent ID checked")
    can_read: bool = Field(..., description="Whether agent has read permission")
    can_write: bool = Field(..., description="Whether agent has write permission")
    can_admin: bool = Field(..., description="Whether agent has admin permission")


class SubscribeRequest(BaseModel):
    """Request body for subscribing to events."""
    agent_id: str = Field(..., description="Agent ID subscribing")
    namespace: Optional[str] = Field(default=None, description="Namespace to subscribe to (global if omitted)")
    memory_id: Optional[str] = Field(default=None, description="Specific memory ID to subscribe to")
    event_types: Optional[List[str]] = Field(default=None, description="Event types to subscribe to")


class SubscribeResponse(BaseModel):
    """Response after subscribing."""
    message: str = Field(default="Subscription created successfully")


class UnsubscribeRequest(BaseModel):
    """Request body for unsubscribing from events."""
    agent_id: str = Field(..., description="Agent ID unsubscribing")
    subscription_id: str = Field(..., description="Subscription ID to remove")


class UnsubscribeResponse(BaseModel):
    """Response after unsubscribing."""
    message: str = Field(default="Subscription removed successfully")


class ListSubscriptionsResponse(BaseModel):
    """Response containing list of subscriptions."""
    subscriptions: List[dict] = Field(..., description="List of subscription information")
    count: int = Field(..., description="Number of subscriptions returned")


# Initialize components
_memory_tools_instance = None
_belief_tools_instance = None
_sync_tools_instance = None
_user_manager_instance = None
_rbac_manager_instance = None

_shared_namespace_tools_instance = None

def get_memory_tools() -> MemoryTools:
    """Initialize and return MemoryTools instance (lazy initialization)."""
    global _memory_tools_instance

    if _memory_tools_instance is None:
        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / 'palace.sqlite'

        # Initialize components
        palace = GraphPalace(db_path)
        # Try nomic-embed-text, fall back to available model
        try:
            embedder = OllamaEmbedder(model='nomic-embed-text')
            # Test if model is available
            embedder.embed("test")
        except Exception:
            # Use available embedding model as fallback
            embedder = OllamaEmbedder(model='nomic-embed-text-v2-moe')
        cache_path = base_path / 'embeddings'
        cache = EmbeddingCache(cache_path, embedder)

        _memory_tools_instance = MemoryTools(palace, embedder, cache)

    return _memory_tools_instance


def get_belief_tools() -> BeliefTools:
    """Initialize and return BeliefTools instance (lazy initialization)."""
    global _belief_tools_instance

    if _belief_tools_instance is None:
        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / 'palace.sqlite'

        # Initialize components
        palace = GraphPalace(db_path)
        belief_network = BeliefNetwork(palace)
        detector = ContradictionDetector()

        _belief_tools_instance = BeliefTools(belief_network, detector)

    return _belief_tools_instance


def get_sync_tools():
    """Initialize and return SyncTools instance (lazy initialization)."""
    global _sync_tools_instance

    if _sync_tools_instance is None:
        from .api import SyncTools
        from .sync.sync_manager import SyncManager

        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)

        # Get instance ID from config or use hostname
        import socket
        instance_id = os.environ.get('OMI_INSTANCE_ID', socket.gethostname())

        sync_manager = SyncManager(base_path, instance_id)
        _sync_tools_instance = SyncTools(sync_manager)

    return _sync_tools_instance


def get_user_manager() -> UserManager:
    """Initialize and return UserManager instance (lazy initialization)."""
    global _user_manager_instance

    if _user_manager_instance is None:
        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / 'palace.sqlite'

        # Initialize UserManager
        _user_manager_instance = UserManager(str(db_path))

    return _user_manager_instance


def get_rbac_manager() -> RBACManager:
    """Initialize and return RBACManager instance (lazy initialization)."""
    global _rbac_manager_instance

    if _rbac_manager_instance is None:
        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / 'palace.sqlite'

        # Initialize RBACManager
        _rbac_manager_instance = RBACManager(str(db_path))

    return _rbac_manager_instance


def log_audit(user_id: str, action: str, resource: str, metadata: Optional[Dict[str, Any]] = None, success: bool = True) -> None:
    """
    Log an audit event to the audit_log table.

    Args:
        user_id: User who performed the action
        action: Action performed (e.g., 'store_memory', 'recall_memory')
        resource: Resource accessed (e.g., 'memory/abc', 'belief/xyz')
        metadata: Optional additional metadata as JSON
        success: Whether the action succeeded (default: True)
    """
    try:
        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)
        db_path = base_path / 'palace.sqlite'

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        audit_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO audit_log (id, user_id, action, resource, namespace, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            audit_id,
            user_id,
            action,
            resource,
            None,  # namespace not used in API context
            json.dumps({"success": success, **(metadata or {})})
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}", exc_info=True)

def get_shared_namespace_tools() -> SharedNamespaceTools:
    """Initialize and return SharedNamespaceTools instance (lazy initialization)."""
    global _shared_namespace_tools_instance

    if _shared_namespace_tools_instance is None:
        base_path = Path.home() / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)

        db_path = base_path / 'palace.sqlite'

        # Initialize components
        shared_namespace = SharedNamespace(db_path)
        permissions = PermissionManager(db_path)
        subscriptions = SubscriptionManager(db_path)
        audit_logger = AuditLogger(db_path)

        _shared_namespace_tools_instance = SharedNamespaceTools(
            shared_namespace, permissions, subscriptions, audit_logger
        )

    return _shared_namespace_tools_instance


# Create FastAPI app
app = FastAPI(
    title="OMI REST API",
    description="""
OMI (Open Memory Interface) REST API provides comprehensive memory operations,
belief management, session lifecycle, real-time event streaming, and a web dashboard.

## Features

* **Memory Operations**: Store and recall memories with semantic search
* **Belief Management**: Create and update beliefs with evidence-based confidence
* **Session Lifecycle**: Track sessions with start/end events
* **Real-time Events**: Server-Sent Events (SSE) for live operation streaming
* **Web Dashboard**: Interactive memory graph exploration
* **Authentication**: API key authentication via X-API-Key header
* **CORS Support**: Configurable cross-origin resource sharing

## Authentication

Protected endpoints require an `X-API-Key` header. Set the `OMI_API_KEY`
environment variable to enable authentication.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "General",
            "description": "Root and health check endpoints"
        },
        {
            "name": "Memory Operations",
            "description": "Store and recall memories with semantic search and recency weighting"
        },
        {
            "name": "Belief Management",
            "description": "Create and update beliefs with evidence-based confidence tracking"
        },
        {
            "name": "Session Lifecycle",
            "description": "Manage session start and end with event tracking"
        },
        {
            "name": "Shared Namespaces",
            "description": "Multi-agent shared namespace management and permissions"
        },
        {
            "name": "Subscriptions",
            "description": "Event subscription management for multi-agent coordination"
        },
        {
            "name": "Events",
            "description": "Server-Sent Events (SSE) for real-time operation streaming"
        },
        {
            "name": "Distributed Sync",
            "description": "Multi-instance synchronization for leader-follower and multi-leader topologies"
        },
        {
            "name": "Admin",
            "description": "Admin-only user management and audit log endpoints (requires admin role)"
        }
    ]
)

# Mount dashboard router
app.include_router(dashboard_router)

# Configure static file serving for dashboard
dashboard_dist = Path(__file__).parent / "dashboard" / "dist"
if dashboard_dist.exists():
    # Mount static files at /dashboard
    app.mount(
        "/dashboard",
        StaticFiles(directory=str(dashboard_dist), html=True),
        name="dashboard"
    )
    logger.info(f"Dashboard static files mounted from {dashboard_dist}")
else:
    logger.warning(f"Dashboard dist directory not found at {dashboard_dist}")
    logger.warning("Run 'cd src/omi/dashboard && npm run build' to build the dashboard")


# Configure CORS
cors_origins_str = os.environ.get("OMI_CORS_ORIGINS", "*")
if cors_origins_str == "*":
    cors_origins = ["*"]
else:
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled with origins: {cors_origins}")


@app.get("/", tags=["General"], summary="API root endpoint")
async def root() -> Dict[str, Any]:
    """API root endpoint with service information and available endpoints."""
    return {
        "service": "OMI REST API",
        "version": "1.0.0",
        "endpoints": {
            "/dashboard": "Web dashboard for memory exploration (if built)",
            "/api/v1/store": "POST - Store a new memory",
            "/api/v1/recall": "GET - Recall memories by semantic search",
            "/api/v1/beliefs": "POST - Create a new belief",
            "/api/v1/beliefs/{id}": "PUT - Update a belief with evidence",
            "/api/v1/sessions/start": "POST - Start a new session",
            "/api/v1/sessions/end": "POST - End a session",
            "/api/v1/namespaces/shared": "POST/GET - Create or list shared namespaces",
            "/api/v1/namespaces/shared/{namespace}": "GET/DELETE - Get or delete a namespace",
            "/api/v1/namespaces/shared/{namespace}/permissions": "POST/GET - Grant or list permissions",
            "/api/v1/namespaces/shared/{namespace}/permissions/{agent_id}": "DELETE/GET - Revoke or check permissions",
            "/api/v1/subscriptions": "POST/GET/DELETE - Subscribe, list, or unsubscribe from events",
            "/api/v1/subscriptions/stream": "SSE endpoint for subscription notifications",
            "/api/v1/events": "SSE endpoint for real-time event streaming",
            "/api/v1/dashboard/memories": "Retrieve memories with filters and pagination",
            "/api/v1/dashboard/edges": "Retrieve relationship edges",
            "/api/v1/dashboard/graph": "Retrieve complete graph data (memories + edges)",
            "/api/v1/dashboard/beliefs": "Retrieve belief network data",
            "/api/v1/dashboard/stats": "Get database storage statistics",
            "/api/v1/dashboard/search": "Semantic search for memories",
            "/api/sync/status": "GET - Get distributed sync status",
            "/api/sync/incremental/start": "POST - Start incremental sync",
            "/api/sync/incremental/stop": "POST - Stop incremental sync",
            "/api/sync/bulk/from": "POST - Import memory snapshot from instance",
            "/api/sync/bulk/to": "POST - Export memory snapshot to instance",
            "/api/sync/instances/register": "POST - Register instance to cluster",
            "/api/sync/instances/{instance_id}": "DELETE - Unregister instance",
            "/api/sync/reconcile": "POST - Reconcile after network partition",
            "/api/v1/admin/users": "GET/POST - List or create users (admin only)",
            "/api/v1/admin/users/{id}": "DELETE - Delete user (admin only)",
            "/api/v1/admin/audit-log": "GET - View audit log (admin only)",
            "/health": "Health check endpoint"
        }
    }


@app.get("/health", tags=["General"], summary="Health check endpoint")
async def health() -> Dict[str, Any]:
    """Health check endpoint with version and detailed status."""
    return {
        "status": "healthy",
        "service": "omi-event-api",
        "version": "1.0.0"
    }


@app.post("/api/v1/store", response_model=StoreMemoryResponse, status_code=status.HTTP_201_CREATED, tags=["Memory Operations"], summary="Store a new memory")
async def store_memory(request: StoreMemoryRequest, user: User = Depends(verify_api_key)):
    """Store a new memory with semantic embedding."""
    # Check write permission on memory resource
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "write", "memory"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="store_memory",
            resource="memory",
            metadata={"reason": "permission_denied", "memory_type": request.memory_type},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have permission to write memories"
        )

    try:
        tools = get_memory_tools()
        memory_id = tools.store(
            content=request.content,
            memory_type=request.memory_type,
            related_to=request.related_to,
            confidence=request.confidence
        )

        # Log successful memory storage
        log_audit(
            user_id=user.id,
            action="store_memory",
            resource=f"memory/{memory_id}",
            metadata={"memory_type": request.memory_type, "memory_id": memory_id},
            success=True
        )

        return StoreMemoryResponse(memory_id=memory_id)
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="store_memory",
            resource="memory",
            metadata={"error": str(e), "memory_type": request.memory_type},
            success=False
        )
        logger.error(f"Error storing memory: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}"
        )


@app.get("/api/v1/recall", tags=["Memory Operations"], summary="Recall memories by semantic search")
async def recall_memory(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(10, ge=1, le=500, description="Maximum number of results per page"),
    min_relevance: float = Query(0.7, ge=0.0, le=1.0, description="Minimum relevance threshold"),
    memory_type: Optional[str] = Query(None, description="Filter by type: fact|experience|belief|decision"),
    cursor: Optional[str] = Query(None, description="Pagination cursor from previous response"),
    accept: Optional[str] = Header(None, alias="Accept"),
    current_user: User = Depends(verify_api_key)
):
    """
    Recall memories using semantic search with recency weighting and pagination.

    Supports both JSON and Server-Sent Events (SSE) streaming responses:
    - JSON: Default response format with full pagination metadata
    - SSE: Real-time streaming with Accept: text/event-stream header

    Query Parameters:
        query: Natural language search query
        limit: Maximum number of results per page (1-500, default 10)
        min_relevance: Minimum relevance threshold (0.0-1.0, default 0.7)
        memory_type: Filter by type (fact, experience, belief, decision)
        cursor: Pagination cursor from previous response (for next page)

    Headers:
        Accept: Set to 'text/event-stream' for SSE streaming mode

    Returns:
        - JSON mode: RecallMemoryResponse with memories, count, next_cursor, has_more
        - SSE mode: StreamingResponse with event stream

    Examples:
        JSON mode:
            curl http://localhost:8000/api/v1/recall?query=test&limit=10

        SSE streaming mode:
            curl -N -H "Accept: text/event-stream" http://localhost:8000/api/v1/recall?query=test
    """
    # Check read permission on memory resource
    rbac = get_rbac_manager()

    if not rbac.check_permission(current_user.id, "read", "memory"):
        log_audit(
            user_id=current_user.id,
            action="recall_memory",
            resource="memory",
            metadata={"reason": "permission_denied", "query": query[:100]},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{current_user.username}' does not have permission to read memories"
        )

    # Check if client requested SSE streaming
    if accept and "text/event-stream" in accept:
        # Return SSE streaming response
        return StreamingResponse(
            recall_stream(
                query=query,
                limit=limit,
                min_relevance=min_relevance,
                memory_type=memory_type,
                cursor=cursor
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering in nginx
            }
        )

    # Default JSON response
    try:
        tools = get_memory_tools()
        result = tools.recall(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
            memory_type=memory_type,
            cursor=cursor
        )

        # Log successful memory recall
        log_audit(
            user_id=current_user.id,
            action="recall_memory",
            resource="memory",
            metadata={"query": query[:100], "limit": limit, "results_count": len(result["memories"])},
            success=True
        )

        return RecallMemoryResponse(
            memories=result["memories"],
            count=len(result["memories"]),
            next_cursor=result.get("next_cursor", ""),
            has_more=result.get("has_more", False)
        )
    except Exception as e:
        # Log failure
        log_audit(
            user_id=current_user.id,
            action="recall_memory",
            resource="memory",
            metadata={"error": str(e), "query": query[:100]},
            success=False
        )
        logger.error(f"Error recalling memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recall memories: {str(e)}"
        )


@app.post("/api/v1/beliefs", response_model=CreateBeliefResponse, status_code=status.HTTP_201_CREATED, tags=["Belief Management"], summary="Create a new belief")
async def create_belief(request: CreateBeliefRequest, user: User = Depends(verify_api_key)):
    """Create a new belief with initial confidence."""
    # Check write permission on belief resource
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "write", "belief"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="create_belief",
            resource="belief",
            metadata={"reason": "permission_denied"},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have permission to write beliefs"
        )

    try:
        tools = get_belief_tools()
        belief_id = tools.create(
            content=request.content,
            initial_confidence=request.initial_confidence
        )

        # Log successful belief creation
        log_audit(
            user_id=user.id,
            action="create_belief",
            resource=f"belief/{belief_id}",
            metadata={"belief_id": belief_id, "initial_confidence": request.initial_confidence},
            success=True
        )

        return CreateBeliefResponse(belief_id=belief_id)
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="create_belief",
            resource="belief",
            metadata={"error": str(e)},
            success=False
        )
        logger.error(f"Error creating belief: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create belief: {str(e)}"
        )


@app.put("/api/v1/beliefs/{id}", response_model=UpdateBeliefResponse, tags=["Belief Management"], summary="Update belief with evidence")
async def update_belief(id: str, request: UpdateBeliefRequest, user: User = Depends(verify_api_key)):
    """Update a belief with new evidence using EMA confidence updates."""
    # Check write permission on belief resource
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "write", "belief"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="update_belief",
            resource=f"belief/{id}",
            metadata={"reason": "permission_denied", "belief_id": id},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have permission to write beliefs"
        )

    try:
        tools = get_belief_tools()
        new_confidence = tools.update(
            belief_id=id,
            evidence_memory_id=request.evidence_memory_id,
            supports=request.supports,
            strength=request.strength
        )

        # Log successful belief update
        log_audit(
            user_id=user.id,
            action="update_belief",
            resource=f"belief/{id}",
            metadata={
                "belief_id": id,
                "evidence_memory_id": request.evidence_memory_id,
                "supports": request.supports,
                "new_confidence": new_confidence
            },
            success=True
        )

        return UpdateBeliefResponse(new_confidence=new_confidence)
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="update_belief",
            resource=f"belief/{id}",
            metadata={"error": str(e), "belief_id": id},
            success=False
        )
        logger.error(f"Error updating belief: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update belief: {str(e)}"
        )


@app.post("/api/v1/sessions/start", response_model=StartSessionResponse, status_code=status.HTTP_200_OK, tags=["Session Lifecycle"], summary="Start a new session")
async def start_session(request: StartSessionRequest, user: User = Depends(verify_api_key)):
    """Start a new session."""
    # Check write permission on memory resource (sessions track memory operations)
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "write", "memory"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="start_session",
            resource="session",
            metadata={"reason": "permission_denied"},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have permission to start sessions"
        )

    try:
        import uuid
        session_id = request.session_id or str(uuid.uuid4())
        event = SessionStartedEvent(
            session_id=session_id,
            metadata=request.metadata
        )
        get_event_bus().publish(event)

        # Log successful session start
        log_audit(
            user_id=user.id,
            action="start_session",
            resource=f"session/{session_id}",
            metadata={"session_id": session_id},
            success=True
        )

        return StartSessionResponse(session_id=session_id)
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="start_session",
            resource="session",
            metadata={"error": str(e)},
            success=False
        )
        logger.error(f"Error starting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {str(e)}"
        )


@app.post("/api/v1/sessions/end", response_model=EndSessionResponse, status_code=status.HTTP_200_OK, tags=["Session Lifecycle"], summary="End a session")
async def end_session(request: EndSessionRequest, user: User = Depends(verify_api_key)):
    """End an existing session."""
    # Check write permission on memory resource (sessions track memory operations)
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "write", "memory"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="end_session",
            resource=f"session/{request.session_id}",
            metadata={"reason": "permission_denied", "session_id": request.session_id},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have permission to end sessions"
        )

    try:
        event = SessionEndedEvent(
            session_id=request.session_id,
            duration_seconds=request.duration_seconds,
            metadata=request.metadata
        )
        get_event_bus().publish(event)

        # Log successful session end
        log_audit(
            user_id=user.id,
            action="end_session",
            resource=f"session/{request.session_id}",
            metadata={"session_id": request.session_id, "duration_seconds": request.duration_seconds},
            success=True
        )

        return EndSessionResponse(session_id=request.session_id)
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="end_session",
            resource=f"session/{request.session_id}",
            metadata={"error": str(e), "session_id": request.session_id},
            success=False
        )
        logger.error(f"Error ending session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {str(e)}"
        )


async def recall_stream(
    query: str,
    limit: int = 10,
    min_relevance: float = 0.7,
    memory_type: Optional[str] = None,
    cursor: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream of recall results.

    Streams individual memories as they are recalled, followed by pagination metadata.

    Args:
        query: Natural language search query
        limit: Maximum number of results
        min_relevance: Minimum relevance threshold
        memory_type: Optional filter by memory type
        cursor: Pagination cursor

    Yields:
        SSE-formatted recall data
    """
    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'stream_start', 'message': 'Recall stream started'})}\n\n"

        # Perform recall
        tools = get_memory_tools()
        result = tools.recall(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
            memory_type=memory_type,
            cursor=cursor
        )

        # Stream each memory individually
        for idx, memory in enumerate(result["memories"]):
            memory_event = {
                'type': 'memory',
                'index': idx,
                'data': memory
            }
            sse_data = f"data: {json.dumps(memory_event)}\n\n"
            yield sse_data

            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)

        # Send pagination metadata
        metadata_event = {
            'type': 'metadata',
            'data': {
                'count': len(result["memories"]),
                'next_cursor': result.get('next_cursor', ''),
                'has_more': result.get('has_more', False)
            }
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"

        # Send stream completion message
        yield f"data: {json.dumps({'type': 'stream_end', 'message': 'Recall stream completed'})}\n\n"

    except asyncio.CancelledError:
        logger.info("Recall stream cancelled by client")
        raise
    except Exception as e:
        logger.error(f"Error in recall stream: {e}", exc_info=True)
        error_event = {
            'type': 'error',
            'message': str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        raise


@app.get("/api/sync/status", response_model=SyncStatusResponse, tags=["Distributed Sync"], summary="Get sync status")
async def get_sync_status(api_key: str = Depends(verify_api_key)):
    """Get comprehensive distributed sync status including topology, lag metrics, and instance list."""
    try:
        tools = get_sync_tools()
        status_data = tools.status()
        return SyncStatusResponse(**status_data)
    except Exception as e:
        logger.error(f"Error getting sync status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync status: {str(e)}"
        )


@app.post("/api/sync/incremental/start", response_model=IncrementalSyncResponse, tags=["Distributed Sync"], summary="Start incremental sync")
async def start_incremental_sync(api_key: str = Depends(verify_api_key)):
    """Start real-time event-based synchronization with other instances."""
    try:
        tools = get_sync_tools()
        result = tools.start_incremental()
        return IncrementalSyncResponse(**result)
    except Exception as e:
        logger.error(f"Error starting incremental sync: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start incremental sync: {str(e)}"
        )


@app.post("/api/sync/incremental/stop", response_model=IncrementalSyncResponse, tags=["Distributed Sync"], summary="Stop incremental sync")
async def stop_incremental_sync(api_key: str = Depends(verify_api_key)):
    """Stop real-time event-based synchronization."""
    try:
        tools = get_sync_tools()
        result = tools.stop_incremental()
        return IncrementalSyncResponse(**result)
    except Exception as e:
        logger.error(f"Error stopping incremental sync: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop incremental sync: {str(e)}"
        )


@app.post("/api/sync/bulk/from", response_model=BulkSyncResponse, tags=["Distributed Sync"], summary="Import memory snapshot")
async def bulk_sync_from(request: BulkSyncRequest, api_key: str = Depends(verify_api_key)):
    """Import full memory snapshot from another OMI instance."""
    try:
        tools = get_sync_tools()
        result = tools.bulk_from(request.instance_id, request.endpoint)
        return BulkSyncResponse(**result)
    except Exception as e:
        logger.error(f"Error performing bulk sync from: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform bulk sync from: {str(e)}"
        )


@app.post("/api/sync/bulk/to", response_model=BulkSyncResponse, tags=["Distributed Sync"], summary="Export memory snapshot")
async def bulk_sync_to(request: BulkSyncRequest, api_key: str = Depends(verify_api_key)):
    """Export full memory snapshot to another OMI instance."""
    try:
        tools = get_sync_tools()
        result = tools.bulk_to(request.instance_id, request.endpoint)
        return BulkSyncResponse(**result)
    except Exception as e:
        logger.error(f"Error performing bulk sync to: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform bulk sync to: {str(e)}"
        )


@app.post("/api/sync/instances/register", response_model=RegisterInstanceResponse, tags=["Distributed Sync"], summary="Register instance")
async def register_instance(request: RegisterInstanceRequest, api_key: str = Depends(verify_api_key)):
    """Register an OMI instance to the sync cluster."""
    try:
        tools = get_sync_tools()
        result = tools.register_instance(request.instance_id, request.endpoint)
        return RegisterInstanceResponse(**result)
    except Exception as e:
        logger.error(f"Error registering instance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register instance: {str(e)}"
        )


@app.delete("/api/sync/instances/{instance_id}", response_model=UnregisterInstanceResponse, tags=["Distributed Sync"], summary="Unregister instance")
async def unregister_instance(instance_id: str, api_key: str = Depends(verify_api_key)):
    """Remove an OMI instance from the sync cluster."""
    try:
        tools = get_sync_tools()
        result = tools.unregister_instance(instance_id)
        return UnregisterInstanceResponse(**result)
    except Exception as e:
        logger.error(f"Error unregistering instance: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister instance: {str(e)}"
        )


@app.post("/api/sync/reconcile", tags=["Distributed Sync"], summary="Reconcile after partition")
async def reconcile_partition(request: ReconcilePartitionRequest, api_key: str = Depends(verify_api_key)):
    """Reconcile memory stores after network partition with conflict resolution."""
    try:
        tools = get_sync_tools()
        result = tools.reconcile_partition(request.instance_id)
        return result
    except Exception as e:
        logger.error(f"Error reconciling partition: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reconcile partition: {str(e)}"
        )


# Admin-only endpoints
@app.get("/api/v1/admin/users", response_model=ListUsersResponse, tags=["Admin"], summary="List all users (admin only)")
async def admin_list_users(user: User = Depends(verify_api_key)):
    """
    List all users in the system.

    Requires admin role.

    Returns:
        ListUsersResponse with all users and their roles

    Raises:
        HTTPException: 403 if user is not an admin
    """
    # Check admin permission
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "admin", "user"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="admin_list_users",
            resource="user",
            metadata={"reason": "permission_denied"},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have admin permission"
        )

    try:
        user_manager = get_user_manager()
        users = user_manager.list_users()

        # Get roles for each user
        user_responses = []
        for u in users:
            roles = user_manager.get_user_roles(u.id)
            role_list = [{"role": role, "namespace": ns} for role, ns in roles]
            user_responses.append(UserResponse(
                id=u.id,
                username=u.username,
                email=u.email,
                created_at=u.created_at.isoformat() if u.created_at else None,
                roles=role_list
            ))

        # Log successful operation
        log_audit(
            user_id=user.id,
            action="admin_list_users",
            resource="user",
            metadata={"count": len(user_responses)},
            success=True
        )

        return ListUsersResponse(users=user_responses, count=len(user_responses))

    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="admin_list_users",
            resource="user",
            metadata={"error": str(e)},
            success=False
        )
        logger.error(f"Error listing users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )


@app.get("/api/v1/admin/audit-log", response_model=AuditLogResponse, tags=["Admin"], summary="View audit log (admin only)")
async def admin_get_audit_log(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of entries to return"),
    offset: int = Query(0, ge=0, description="Number of entries to skip"),
    user_id_filter: Optional[str] = Query(None, description="Filter by user ID"),
    action_filter: Optional[str] = Query(None, description="Filter by action"),
    user: User = Depends(verify_api_key)
):
    """
    Retrieve audit log entries.

    Requires admin role.

    Query Parameters:
        limit: Maximum number of entries to return (1-1000, default 100)
        offset: Number of entries to skip for pagination (default 0)
        user_id_filter: Optional filter by user ID
        action_filter: Optional filter by action

    Returns:
        AuditLogResponse with audit log entries

    Raises:
        HTTPException: 403 if user is not an admin
    """
    # Check audit permission
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "audit", "audit_log"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="admin_get_audit_log",
            resource="audit_log",
            metadata={"reason": "permission_denied"},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have audit permission"
        )

    try:
        base_path = Path.home() / '.openclaw' / 'omi'
        db_path = base_path / 'palace.sqlite'

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Build query with optional filters
        query = "SELECT id, user_id, action, resource, namespace, metadata, timestamp FROM audit_log WHERE 1=1"
        params = []

        if user_id_filter:
            query += " AND user_id = ?"
            params.append(user_id_filter)

        if action_filter:
            query += " AND action = ?"
            params.append(action_filter)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to AuditLogEntry objects
        entries = []
        for row in rows:
            metadata = json.loads(row[5]) if row[5] else None
            entries.append(AuditLogEntry(
                id=row[0],
                user_id=row[1],
                action=row[2],
                resource=row[3],
                namespace=row[4],
                metadata=metadata,
                timestamp=row[6]
            ))

        conn.close()

        # Log successful operation
        log_audit(
            user_id=user.id,
            action="admin_get_audit_log",
            resource="audit_log",
            metadata={"count": len(entries), "filters": {"user_id": user_id_filter, "action": action_filter}},
            success=True
        )

        return AuditLogResponse(entries=entries, count=len(entries))

    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="admin_get_audit_log",
            resource="audit_log",
            metadata={"error": str(e)},
            success=False
        )
        logger.error(f"Error retrieving audit log: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audit log: {str(e)}"
        )


@app.post("/api/v1/admin/users", response_model=CreateUserResponse, status_code=status.HTTP_201_CREATED, tags=["Admin"], summary="Create new user (admin only)")
async def admin_create_user(request: CreateUserRequest, user: User = Depends(verify_api_key)):
    """
    Create a new user.

    Requires admin role.

    Request Body:
        username: Unique username
        email: Optional email address
        role: Optional initial role to assign

    Returns:
        CreateUserResponse with new user ID

    Raises:
        HTTPException: 403 if user is not an admin, 400 if username exists
    """
    # Check admin permission
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "admin", "user"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="admin_create_user",
            resource="user",
            metadata={"reason": "permission_denied", "username": request.username},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have admin permission"
        )

    try:
        user_manager = get_user_manager()

        # Create user
        new_user_id = user_manager.create_user(request.username, request.email)

        # Assign role if provided
        if request.role:
            try:
                user_manager.assign_role(new_user_id, request.role)
            except ValueError as e:
                # User created but role assignment failed
                logger.warning(f"User created but role assignment failed: {e}")
                # Don't fail the request, just log it

        # Log successful operation
        log_audit(
            user_id=user.id,
            action="admin_create_user",
            resource=f"user/{new_user_id}",
            metadata={
                "new_user_id": new_user_id,
                "username": request.username,
                "role": request.role
            },
            success=True
        )

        return CreateUserResponse(user_id=new_user_id, username=request.username)

    except ValueError as e:
        # Log failure (likely duplicate username)
        log_audit(
            user_id=user.id,
            action="admin_create_user",
            resource="user",
            metadata={"error": str(e), "username": request.username},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="admin_create_user",
            resource="user",
            metadata={"error": str(e), "username": request.username},
            success=False
        )
        logger.error(f"Error creating user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )


@app.delete("/api/v1/admin/users/{user_id}", response_model=DeleteUserResponse, tags=["Admin"], summary="Delete user (admin only)")
async def admin_delete_user(user_id: str, user: User = Depends(verify_api_key)):
    """
    Delete a user and all associated data.

    Requires admin role.

    Path Parameters:
        user_id: UUID of user to delete

    Returns:
        DeleteUserResponse confirming deletion

    Raises:
        HTTPException: 403 if user is not an admin, 404 if user not found
    """
    # Check admin permission
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "admin", "user"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="admin_delete_user",
            resource=f"user/{user_id}",
            metadata={"reason": "permission_denied", "target_user_id": user_id},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have admin permission"
        )

    try:
        user_manager = get_user_manager()

        # Check if user exists
        target_user = user_manager.get_user(user_id)
        if not target_user:
            # Log failure
            log_audit(
                user_id=user.id,
                action="admin_delete_user",
                resource=f"user/{user_id}",
                metadata={"error": "User not found", "target_user_id": user_id},
                success=False
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID '{user_id}' not found"
            )

        # Delete user
        success = user_manager.delete_user(user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete user with ID '{user_id}'"
            )

        # Log successful operation
        log_audit(
            user_id=user.id,
            action="admin_delete_user",
            resource=f"user/{user_id}",
            metadata={
                "target_user_id": user_id,
                "target_username": target_user.username
            },
            success=True
        )

        return DeleteUserResponse(user_id=user_id)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
            action="admin_delete_user",
            resource=f"user/{user_id}",
            metadata={"error": str(e), "target_user_id": user_id},
            success=False
        )
        logger.error(f"Error deleting user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )

# Shared Namespace Endpoints

@app.post("/api/v1/namespaces/shared", response_model=CreateNamespaceResponse, status_code=status.HTTP_201_CREATED, tags=["Shared Namespaces"], summary="Create a shared namespace")
async def create_shared_namespace(request: CreateNamespaceRequest, api_key: str = Depends(verify_api_key)):
    """Create a new shared namespace for multi-agent coordination."""
    try:
        tools = get_shared_namespace_tools()
        result = tools.create_namespace(
            namespace=request.namespace,
            created_by=request.created_by,
            metadata=request.metadata
        )

        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )

        return CreateNamespaceResponse(
            namespace=result['namespace'],
            created_by=result['created_by'],
            created_at=result['created_at']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating shared namespace: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create shared namespace: {str(e)}"
        )


@app.get("/api/v1/namespaces/shared", response_model=ListNamespacesResponse, tags=["Shared Namespaces"], summary="List shared namespaces")
async def list_shared_namespaces(
    agent_id: Optional[str] = Query(None, description="Filter by creator agent ID"),
    api_key: str = Depends(verify_api_key)
):
    """List all shared namespaces, optionally filtered by creator."""
    try:
        tools = get_shared_namespace_tools()
        namespaces = tools.list_namespaces(agent_id=agent_id)
        return ListNamespacesResponse(namespaces=namespaces, count=len(namespaces))
    except Exception as e:
        logger.error(f"Error listing shared namespaces: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list shared namespaces: {str(e)}"
        )


@app.get("/api/v1/namespaces/shared/{namespace}", response_model=NamespaceInfoResponse, tags=["Shared Namespaces"], summary="Get namespace information")
async def get_shared_namespace(namespace: str, api_key: str = Depends(verify_api_key)):
    """Get information about a specific shared namespace."""
    try:
        tools = get_shared_namespace_tools()
        ns_info = tools.get_namespace(namespace)

        if ns_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Namespace '{namespace}' not found"
            )

        return NamespaceInfoResponse(**ns_info)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting shared namespace: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get shared namespace: {str(e)}"
        )


@app.delete("/api/v1/namespaces/shared/{namespace}", response_model=DeleteNamespaceResponse, tags=["Shared Namespaces"], summary="Delete a shared namespace")
async def delete_shared_namespace(
    namespace: str,
    agent_id: str = Query(..., description="Agent ID attempting deletion (must have ADMIN)"),
    api_key: str = Depends(verify_api_key)
):
    """Delete a shared namespace (requires ADMIN permission)."""
    try:
        tools = get_shared_namespace_tools()
        result = tools.delete_namespace(namespace=namespace, agent_id=agent_id)

        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=result['error']
            )

        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Namespace '{namespace}' not found"
            )

        return DeleteNamespaceResponse(success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting shared namespace: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete shared namespace: {str(e)}"
        )


@app.post("/api/v1/namespaces/shared/{namespace}/permissions", response_model=GrantPermissionResponse, status_code=status.HTTP_201_CREATED, tags=["Shared Namespaces"], summary="Grant permission")
async def grant_namespace_permission(
    namespace: str,
    request: GrantPermissionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Grant permission to an agent for a shared namespace (requires ADMIN)."""
    try:
        tools = get_shared_namespace_tools()
        result = tools.grant_permission(
            namespace=namespace,
            agent_id=request.agent_id,
            target_agent_id=request.target_agent_id,
            permission_level=request.permission_level
        )

        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=result['error']
            )

        return GrantPermissionResponse(
            namespace=result['namespace'],
            agent_id=result['agent_id'],
            permission_level=result['permission_level']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error granting permission: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to grant permission: {str(e)}"
        )


@app.delete("/api/v1/namespaces/shared/{namespace}/permissions/{target_agent_id}", response_model=RevokePermissionResponse, tags=["Shared Namespaces"], summary="Revoke permission")
async def revoke_namespace_permission(
    namespace: str,
    target_agent_id: str,
    agent_id: str = Query(..., description="Agent ID revoking permission (must have ADMIN)"),
    api_key: str = Depends(verify_api_key)
):
    """Revoke an agent's permission from a shared namespace (requires ADMIN)."""
    try:
        tools = get_shared_namespace_tools()
        result = tools.revoke_permission(
            namespace=namespace,
            agent_id=agent_id,
            target_agent_id=target_agent_id
        )

        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=result['error']
            )

        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Permission not found for agent '{target_agent_id}'"
            )

        return RevokePermissionResponse(success=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking permission: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke permission: {str(e)}"
        )


@app.get("/api/v1/namespaces/shared/{namespace}/permissions", response_model=ListPermissionsResponse, tags=["Shared Namespaces"], summary="List permissions")
async def list_namespace_permissions(
    namespace: str,
    api_key: str = Depends(verify_api_key)
):
    """List all permissions for a shared namespace."""
    try:
        tools = get_shared_namespace_tools()
        permissions = tools.list_permissions(namespace=namespace)
        return ListPermissionsResponse(permissions=permissions, count=len(permissions))
    except Exception as e:
        logger.error(f"Error listing permissions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list permissions: {str(e)}"
        )


@app.get("/api/v1/namespaces/shared/{namespace}/permissions/{agent_id}/check", response_model=CheckPermissionResponse, tags=["Shared Namespaces"], summary="Check permission")
async def check_namespace_permission(
    namespace: str,
    agent_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Check an agent's permissions for a shared namespace."""
    try:
        tools = get_shared_namespace_tools()
        # Check each permission level directly using PermissionManager methods
        can_read = tools.permissions.can_read(namespace, agent_id)
        can_write = tools.permissions.can_write(namespace, agent_id)
        can_admin = tools.permissions.can_admin(namespace, agent_id)

        return CheckPermissionResponse(
            namespace=namespace,
            agent_id=agent_id,
            can_read=can_read,
            can_write=can_write,
            can_admin=can_admin
        )
    except Exception as e:
        logger.error(f"Error checking permission: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check permission: {str(e)}"
        )


# Subscription Endpoints

@app.post("/api/v1/subscriptions", response_model=SubscribeResponse, status_code=status.HTTP_201_CREATED, tags=["Subscriptions"], summary="Subscribe to events")
async def subscribe_to_events(request: SubscribeRequest, api_key: str = Depends(verify_api_key)):
    """Subscribe to events for a namespace or memory."""
    try:
        tools = get_shared_namespace_tools()
        # Use default event_types if not provided
        event_types = request.event_types or []

        result = tools.subscribe(
            agent_id=request.agent_id,
            event_types=event_types,
            namespace=request.namespace,
            memory_id=request.memory_id
        )

        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )

        return SubscribeResponse()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error subscribing to events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to subscribe to events: {str(e)}"
        )


@app.delete("/api/v1/subscriptions", response_model=UnsubscribeResponse, tags=["Subscriptions"], summary="Unsubscribe from events")
async def unsubscribe_from_events(request: UnsubscribeRequest, api_key: str = Depends(verify_api_key)):
    """Unsubscribe from events by removing a specific subscription."""
    try:
        tools = get_shared_namespace_tools()
        result = tools.unsubscribe(
            agent_id=request.agent_id,
            subscription_id=request.subscription_id
        )

        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )

        if not result.get('success', False):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found or unauthorized"
            )

        return UnsubscribeResponse()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unsubscribing from events: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unsubscribe from events: {str(e)}"
        )


@app.get("/api/v1/subscriptions", response_model=ListSubscriptionsResponse, tags=["Subscriptions"], summary="List subscriptions")
async def list_event_subscriptions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    namespace: Optional[str] = Query(None, description="Filter by namespace"),
    api_key: str = Depends(verify_api_key)
):
    """List subscriptions, optionally filtered by agent or namespace."""
    try:
        tools = get_shared_namespace_tools()
        subscriptions = tools.list_subscriptions(agent_id=agent_id, namespace=namespace)
        return ListSubscriptionsResponse(subscriptions=subscriptions, count=len(subscriptions))
    except Exception as e:
        logger.error(f"Error listing subscriptions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list subscriptions: {str(e)}"
        )


async def event_stream(event_type_filter: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream of events from EventBus.

    Args:
        event_type_filter: Optional filter for specific event type

    Yields:
        SSE-formatted event data
    """
    # Queue to hold events from EventBus
    event_queue: asyncio.Queue[Any] = asyncio.Queue()

    def event_callback(event: Any) -> None:
        """Callback to receive events from EventBus and put them in queue."""
        try:
            # Put event in queue (non-blocking)
            asyncio.run_coroutine_threadsafe(event_queue.put(event), asyncio.get_event_loop())
        except Exception as e:
            logger.error(f"Error queuing event: {e}", exc_info=True)

    # Subscribe to EventBus
    bus = get_event_bus()
    subscription_type = event_type_filter if event_type_filter else '*'
    bus.subscribe(subscription_type, event_callback)

    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': 'SSE stream connected'})}\n\n"

        # Stream events as they arrive
        while True:
            try:
                # Wait for event with timeout to allow for graceful shutdown
                event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                # Serialize event to dict
                if hasattr(event, 'to_dict'):
                    event_data = event.to_dict()
                else:
                    # Fallback for events without to_dict method
                    event_data = {
                        'event_type': getattr(event, 'event_type', 'unknown'),
                        'timestamp': getattr(event, 'timestamp', None)
                    }

                # Format as SSE (Server-Sent Events)
                # SSE format: "data: {json}\n\n"
                sse_data = f"data: {json.dumps(event_data)}\n\n"
                yield sse_data

            except asyncio.TimeoutError:
                # Send keepalive ping every 30 seconds
                yield f": keepalive\n\n"

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by client")
        raise
    finally:
        # Unsubscribe when client disconnects
        bus.unsubscribe(subscription_type, event_callback)
        logger.info(f"Client disconnected from SSE stream (filter: {subscription_type})")


@app.get("/api/v1/events", tags=["Events"], summary="Real-time event stream (SSE)")
async def events_sse(
    event_type: Optional[str] = Query(
        None,
        description="Filter by event type (e.g., 'memory.stored', 'belief.updated'). Omit for all events."
    ),
    api_key: str = Depends(verify_api_key)
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint for real-time event streaming.

    Streams all memory operation events as they occur:
    - memory.stored: When a memory is stored
    - memory.recalled: When memories are recalled
    - belief.updated: When a belief's confidence is updated
    - belief.contradiction_detected: When a contradiction is detected
    - session.started: When a session starts
    - session.ended: When a session ends

    Query Parameters:
        event_type: Optional filter for specific event type

    Returns:
        StreamingResponse with text/event-stream content type

    Example:
        curl -N http://localhost:8000/api/v1/events
        curl -N "http://localhost:8000/api/v1/events?event_type=memory.stored"
    """
    return StreamingResponse(
        event_stream(event_type_filter=event_type),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


async def subscription_stream(agent_id: str) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream of subscription notifications for a specific agent.

    Args:
        agent_id: Agent ID to stream notifications for

    Yields:
        SSE-formatted notification data
    """
    # Queue to hold events from EventBus
    event_queue: asyncio.Queue[Any] = asyncio.Queue()

    def event_callback(event: Any) -> None:
        """Callback to receive events from EventBus and put them in queue."""
        try:
            # Put event in queue (non-blocking)
            asyncio.run_coroutine_threadsafe(event_queue.put(event), asyncio.get_event_loop())
        except Exception as e:
            logger.error(f"Error queuing event: {e}", exc_info=True)

    # Subscribe to EventBus for all events
    bus = get_event_bus()
    bus.subscribe('*', event_callback)

    # Get subscription manager to filter events
    tools = get_shared_namespace_tools()
    subscriptions_mgr = tools.subscriptions

    try:
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'message': f'Subscription stream connected for agent {agent_id}'})}\n\n"

        # Stream events as they arrive
        while True:
            try:
                # Wait for event with timeout to allow for graceful shutdown
                event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                # Serialize event to dict
                if hasattr(event, 'to_dict'):
                    event_data = event.to_dict()
                else:
                    # Fallback for events without to_dict method
                    event_data = {
                        'event_type': getattr(event, 'event_type', 'unknown'),
                        'timestamp': getattr(event, 'timestamp', None)
                    }

                # Check if this event matches any of the agent's subscriptions
                event_type = event_data.get('event_type', '')
                namespace = event_data.get('namespace')
                memory_id = event_data.get('memory_id')

                # Get agent's subscriptions
                agent_subscriptions = subscriptions_mgr.list_for_agent(agent_id)

                # Check if event matches any subscription
                should_send = False
                for sub in agent_subscriptions:
                    # Check event type filter
                    if sub.event_types and event_type not in sub.event_types and "*" not in sub.event_types:
                        continue

                    # Check namespace filter
                    if sub.namespace and sub.namespace != namespace:
                        continue

                    # Check memory_id filter
                    if sub.memory_id and sub.memory_id != memory_id:
                        continue

                    # If we got here, this subscription matches
                    should_send = True
                    break

                # Send event if it matches any subscription
                if should_send:
                    # Format as SSE (Server-Sent Events)
                    # SSE format: "data: {json}\n\n"
                    sse_data = f"data: {json.dumps(event_data)}\n\n"
                    yield sse_data

            except asyncio.TimeoutError:
                # Send keepalive ping every 30 seconds
                yield f": keepalive\n\n"

    except asyncio.CancelledError:
        logger.info(f"Subscription stream cancelled for agent {agent_id}")
        raise
    finally:
        # Unsubscribe when client disconnects
        bus.unsubscribe('*', event_callback)
        logger.info(f"Agent {agent_id} disconnected from subscription stream")


@app.get("/api/v1/subscriptions/stream", tags=["Subscriptions"], summary="Subscription notifications stream (SSE)")
async def subscriptions_sse(
    agent_id: str = Query(..., description="Agent ID to receive subscription notifications for")
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint for subscription notifications.

    Streams events that match the agent's active subscriptions. Only events
    matching the agent's subscription filters (namespace, memory_id, event_types)
    will be sent.

    Query Parameters:
        agent_id: Agent ID to stream notifications for (required)

    Returns:
        StreamingResponse with text/event-stream content type

    Example:
        curl -N "http://localhost:8000/api/v1/subscriptions/stream?agent_id=test-agent"
    """
    return StreamingResponse(
        subscription_stream(agent_id=agent_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Log startup message."""
    logger.info("OMI REST API started")
    logger.info("SSE endpoint available at /api/v1/events")
    logger.info("Dashboard API endpoints available at /api/v1/dashboard/*")
    if dashboard_dist.exists():
        logger.info("Dashboard UI available at /dashboard")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    logger.info("OMI REST API shutting down")


__all__ = ['app']
