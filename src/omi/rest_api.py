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

from fastapi import FastAPI, Query, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator, List
from pydantic import BaseModel, Field
from pathlib import Path
import json
import asyncio
import logging

from .event_bus import get_event_bus
from .events import (
    MemoryStoredEvent,
    MemoryRecalledEvent,
    BeliefUpdatedEvent,
    ContradictionDetectedEvent,
    SessionStartedEvent,
    SessionEndedEvent
)
from .api import MemoryTools, BeliefTools
from .storage.graph_palace import GraphPalace
from .embeddings import OllamaEmbedder, EmbeddingCache
from .belief import BeliefNetwork, ContradictionDetector

logger = logging.getLogger(__name__)


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


# Initialize components
_memory_tools_instance = None
_belief_tools_instance = None

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


# Create FastAPI app
app = FastAPI(
    title="OMI Event Streaming API",
    description="Server-Sent Events (SSE) API for real-time memory operation events",
    version="1.0.0"
)


@app.get("/")
async def root():
    """API root endpoint with service information."""
    return {
        "service": "OMI Event Streaming API",
        "version": "1.0.0",
        "endpoints": {
            "/api/v1/store": "POST - Store a new memory",
            "/api/v1/recall": "GET - Recall memories by semantic search",
            "/api/v1/beliefs": "POST - Create a new belief",
            "/api/v1/beliefs/{id}": "PUT - Update a belief with evidence",
            "/api/v1/sessions/start": "POST - Start a new session",
            "/api/v1/sessions/end": "POST - End a session",
            "/api/v1/events": "SSE endpoint for real-time event streaming",
            "/health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint with version and detailed status."""
    return {
        "status": "healthy",
        "service": "omi-event-api",
        "version": "1.0.0"
    }


@app.post("/api/v1/store", response_model=StoreMemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(request: StoreMemoryRequest):
    """
    Store a new memory with semantic embedding.

    Args:
        request: Memory details (content, type, related_to, confidence)

    Returns:
        StoreMemoryResponse with memory_id

    Example:
        curl -X POST http://localhost:8420/api/v1/store \\
            -H "Content-Type: application/json" \\
            -d '{"content": "test memory", "memory_type": "fact"}'
    """
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


@app.get("/api/v1/recall", response_model=RecallMemoryResponse)
async def recall_memory(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    min_relevance: float = Query(0.7, ge=0.0, le=1.0, description="Minimum relevance threshold"),
    memory_type: Optional[str] = Query(None, description="Filter by type: fact|experience|belief|decision")
):
    """
    Recall memories using semantic search with recency weighting.

    Args:
        query: Natural language search query
        limit: Max results (1-100, default: 10)
        min_relevance: Minimum relevance threshold (0.0-1.0, default: 0.7)
        memory_type: Optional filter by type

    Returns:
        RecallMemoryResponse with list of memories

    Example:
        curl "http://localhost:8420/api/v1/recall?query=recent%20events&limit=5"
    """
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


@app.post("/api/v1/beliefs", response_model=CreateBeliefResponse, status_code=status.HTTP_201_CREATED)
async def create_belief(request: CreateBeliefRequest):
    """
    Create a new belief with initial confidence.

    Args:
        request: Belief details (content, initial_confidence)

    Returns:
        CreateBeliefResponse with belief_id

    Example:
        curl -X POST http://localhost:8420/api/v1/beliefs \\
            -H "Content-Type: application/json" \\
            -d '{"content": "test belief", "initial_confidence": 0.5}'
    """
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


@app.put("/api/v1/beliefs/{id}", response_model=UpdateBeliefResponse)
async def update_belief(id: str, request: UpdateBeliefRequest):
    """
    Update a belief with new evidence.

    Uses EMA (Exponential Moving Average) for confidence updates:
    - Supporting evidence: λ=0.15
    - Contradicting evidence: λ=0.30

    Args:
        id: Belief ID to update
        request: Evidence details (evidence_memory_id, supports, strength)

    Returns:
        UpdateBeliefResponse with new_confidence

    Example:
        curl -X PUT http://localhost:8420/api/v1/beliefs/{belief_id} \\
            -H "Content-Type: application/json" \\
            -d '{"evidence_memory_id": "mem_123", "supports": true, "strength": 0.8}'
    """
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


@app.post("/api/v1/sessions/start", response_model=StartSessionResponse, status_code=status.HTTP_200_OK)
async def start_session(request: StartSessionRequest):
    """
    Start a new session.

    Args:
        request: Session details (optional session_id and metadata)

    Returns:
        StartSessionResponse with session_id

    Example:
        curl -X POST http://localhost:8420/api/v1/sessions/start \\
            -H "Content-Type: application/json" \\
            -d '{"metadata": {"user": "test_user"}}'
    """
    try:
        import uuid

        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Emit SessionStartedEvent
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


@app.post("/api/v1/sessions/end", response_model=EndSessionResponse, status_code=status.HTTP_200_OK)
async def end_session(request: EndSessionRequest):
    """
    End an existing session.

    Args:
        request: Session end details (session_id, optional duration_seconds and metadata)

    Returns:
        EndSessionResponse with session_id

    Example:
        curl -X POST http://localhost:8420/api/v1/sessions/end \\
            -H "Content-Type: application/json" \\
            -d '{"session_id": "abc-123", "duration_seconds": 120.5}'
    """
    try:
        # Emit SessionEndedEvent
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


async def event_stream(event_type_filter: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream of events from EventBus.

    Args:
        event_type_filter: Optional filter for specific event type

    Yields:
        SSE-formatted event data
    """
    # Queue to hold events from EventBus
    event_queue = asyncio.Queue()

    def event_callback(event):
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


@app.get("/api/v1/events")
async def events_sse(
    event_type: Optional[str] = Query(
        None,
        description="Filter by event type (e.g., 'memory.stored', 'belief.updated'). Omit for all events."
    )
):
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
async def startup_event():
    """Log startup message."""
    logger.info("OMI Event Streaming API started")
    logger.info("SSE endpoint available at /api/v1/events")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("OMI Event Streaming API shutting down")


__all__ = ['app']
