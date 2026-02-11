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

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from typing import Optional, AsyncGenerator
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

logger = logging.getLogger(__name__)

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
            "/api/v1/events": "SSE endpoint for real-time event streaming",
            "/health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "omi-event-api"}


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
