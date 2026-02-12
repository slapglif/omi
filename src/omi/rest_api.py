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

logger = logging.getLogger(__name__)


# API Key Authentication
# API key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    Verify API key from X-API-Key header.

    Args:
        x_api_key: API key from request header

    Returns:
        str: Validated API key

    Raises:
        HTTPException: 401 Unauthorized if API key is missing or invalid
    """
    # Get expected API key from environment or config
    # For now, check environment variable; will be updated to use config.yaml
    expected_key = os.environ.get("OMI_API_KEY")

    # If no expected key is configured, allow all requests (development mode)
    if expected_key is None:
        logger.warning("No OMI_API_KEY configured - authentication disabled (development mode)")
        return "development"

    # Check if API key was provided
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


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


# Initialize components
_memory_tools_instance = None
_belief_tools_instance = None
_sync_tools_instance = None

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
            "name": "Distributed Sync",
            "description": "Multi-instance synchronization for leader-follower and multi-leader topologies"
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
            "/api/sync/status": "GET - Get distributed sync status",
            "/api/sync/incremental/start": "POST - Start incremental sync",
            "/api/sync/incremental/stop": "POST - Stop incremental sync",
            "/api/sync/bulk/from": "POST - Import memory snapshot from instance",
            "/api/sync/bulk/to": "POST - Export memory snapshot to instance",
            "/api/sync/instances/register": "POST - Register instance to cluster",
            "/api/sync/instances/{instance_id}": "DELETE - Unregister instance",
            "/api/sync/reconcile": "POST - Reconcile after network partition",
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
async def store_memory(request: StoreMemoryRequest, api_key: str = Depends(verify_api_key)):
    """Store a new memory with semantic embedding."""
    try:
        tools = get_memory_tools()
        memory_id = tools.store(
            content=request.content,
            memory_type=request.memory_type,
            related_to=request.related_to,
            confidence=request.confidence
        )
        return StoreMemoryResponse(memory_id=memory_id)
    except Exception as e:
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
    api_key: str = Depends(verify_api_key)
):
    """Recall memories using semantic search with recency weighting."""
    try:
        tools = get_memory_tools()
        memories = tools.recall(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
            memory_type=memory_type
        )
        return RecallMemoryResponse(memories=memories, count=len(memories))
    except Exception as e:
        logger.error(f"Error recalling memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recall memories: {str(e)}"
        )


@app.post("/api/v1/beliefs", response_model=CreateBeliefResponse, status_code=status.HTTP_201_CREATED, tags=["Belief Management"], summary="Create a new belief")
async def create_belief(request: CreateBeliefRequest, api_key: str = Depends(verify_api_key)):
    """Create a new belief with initial confidence."""
    try:
        tools = get_belief_tools()
        belief_id = tools.create(
            content=request.content,
            initial_confidence=request.initial_confidence
        )
        return CreateBeliefResponse(belief_id=belief_id)
    except Exception as e:
        logger.error(f"Error creating belief: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create belief: {str(e)}"
        )


@app.put("/api/v1/beliefs/{id}", response_model=UpdateBeliefResponse, tags=["Belief Management"], summary="Update belief with evidence")
async def update_belief(id: str, request: UpdateBeliefRequest, api_key: str = Depends(verify_api_key)):
    """Update a belief with new evidence using EMA confidence updates."""
    try:
        tools = get_belief_tools()
        new_confidence = tools.update(
            belief_id=id,
            evidence_memory_id=request.evidence_memory_id,
            supports=request.supports,
            strength=request.strength
        )
        return UpdateBeliefResponse(new_confidence=new_confidence)
    except Exception as e:
        logger.error(f"Error updating belief: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update belief: {str(e)}"
        )


@app.post("/api/v1/sessions/start", response_model=StartSessionResponse, status_code=status.HTTP_200_OK, tags=["Session Lifecycle"], summary="Start a new session")
async def start_session(request: StartSessionRequest, api_key: str = Depends(verify_api_key)):
    """Start a new session."""
    try:
        import uuid
        session_id = request.session_id or str(uuid.uuid4())
        event = SessionStartedEvent(
            session_id=session_id,
            metadata=request.metadata
        )
        get_event_bus().publish(event)
        return StartSessionResponse(session_id=session_id)
    except Exception as e:
        logger.error(f"Error starting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {str(e)}"
        )


@app.post("/api/v1/sessions/end", response_model=EndSessionResponse, status_code=status.HTTP_200_OK, tags=["Session Lifecycle"], summary="End a session")
async def end_session(request: EndSessionRequest, api_key: str = Depends(verify_api_key)):
    """End an existing session."""
    try:
        event = SessionEndedEvent(
            session_id=request.session_id,
            duration_seconds=request.duration_seconds,
            metadata=request.metadata
        )
        get_event_bus().publish(event)
        return EndSessionResponse(session_id=request.session_id)
    except Exception as e:
        logger.error(f"Error ending session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {str(e)}"
        )


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
