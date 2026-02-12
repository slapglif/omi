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
from .api import MemoryTools, BeliefTools
from .storage.graph_palace import GraphPalace
from .embeddings import OllamaEmbedder, EmbeddingCache
from .belief import BeliefNetwork, ContradictionDetector
from .user_manager import UserManager, User
from .rbac import RBACManager
import sqlite3
import uuid

logger = logging.getLogger(__name__)


# API Key Authentication
# API key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> User:
    """
    Verify API key from X-API-Key header and return associated user.

    Args:
        x_api_key: API key from request header

    Returns:
        User: User object associated with the API key

    Raises:
        HTTPException: 401 Unauthorized if API key is missing or invalid
    """
    # Check if API key was provided
    if x_api_key is None:
        # Check for legacy environment variable for backward compatibility
        expected_key = os.environ.get("OMI_API_KEY")
        if expected_key is None:
            # Development mode: no authentication
            logger.warning("No API key provided and OMI_API_KEY not configured - authentication disabled (development mode)")
            # Return a synthetic development user
            return User(id="dev", username="development", email=None)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    # Verify API key against database
    user_manager = get_user_manager()
    user = user_manager.verify_api_key(x_api_key)

    if user is None:
        # API key not found in database, check legacy environment variable
        expected_key = os.environ.get("OMI_API_KEY")
        if expected_key and x_api_key == expected_key:
            # Legacy mode: API key matches environment variable
            logger.warning("Using legacy OMI_API_KEY authentication - consider migrating to database-backed API keys")
            # Return a synthetic legacy user
            return User(id="legacy", username="legacy", email=None)

        # Invalid API key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

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
    """Response containing recalled memories."""
    memories: List[dict] = Field(..., description="List of recalled memories")
    count: int = Field(..., description="Number of memories returned")


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


# Initialize components
_memory_tools_instance = None
_belief_tools_instance = None
_user_manager_instance = None
_rbac_manager_instance = None

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
            "name": "Events",
            "description": "Server-Sent Events (SSE) for real-time operation streaming"
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
            "/api/v1/events": "SSE endpoint for real-time event streaming",
            "/api/v1/dashboard/memories": "Retrieve memories with filters and pagination",
            "/api/v1/dashboard/edges": "Retrieve relationship edges",
            "/api/v1/dashboard/graph": "Retrieve complete graph data (memories + edges)",
            "/api/v1/dashboard/beliefs": "Retrieve belief network data",
            "/api/v1/dashboard/stats": "Get database storage statistics",
            "/api/v1/dashboard/search": "Semantic search for memories",
            "/api/v1/admin/users": "GET - List all users (admin only)",
            "/api/v1/admin/users": "POST - Create new user (admin only)",
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


@app.get("/api/v1/recall", response_model=RecallMemoryResponse, tags=["Memory Operations"], summary="Recall memories by semantic search")
async def recall_memory(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    min_relevance: float = Query(0.7, ge=0.0, le=1.0, description="Minimum relevance threshold"),
    memory_type: Optional[str] = Query(None, description="Filter by type: fact|experience|belief|decision"),
    user: User = Depends(verify_api_key)
):
    """Recall memories using semantic search with recency weighting."""
    # Check read permission on memory resource
    rbac = get_rbac_manager()

    if not rbac.check_permission(user.id, "read", "memory"):
        # Log permission denied
        log_audit(
            user_id=user.id,
            action="recall_memory",
            resource="memory",
            metadata={"reason": "permission_denied", "query": query[:100]},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user.username}' does not have permission to read memories"
        )

    try:
        tools = get_memory_tools()
        memories = tools.recall(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
            memory_type=memory_type
        )

        # Log successful memory recall
        log_audit(
            user_id=user.id,
            action="recall_memory",
            resource="memory",
            metadata={"query": query[:100], "limit": limit, "results_count": len(memories)},
            success=True
        )

        return RecallMemoryResponse(memories=memories, count=len(memories))
    except Exception as e:
        # Log failure
        log_audit(
            user_id=user.id,
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
    )
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
