"""
FastAPI router for dashboard API endpoints.

Provides read-only REST endpoints for dashboard visualization:
- Memory graph data (nodes and edges)
- Belief network data
- Storage statistics
- Semantic search

Usage:
    Mount in main app:
        from omi.dashboard_api import router
        app.include_router(router)
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import logging

from .storage.graph_palace import GraphPalace

logger = logging.getLogger(__name__)

# Create FastAPI router with dashboard prefix
router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["dashboard"],
    responses={404: {"description": "Not found"}}
)


def get_palace_instance() -> GraphPalace:
    """
    Get GraphPalace instance for data access.

    Uses the same path resolution as CLI:
    - OMI_BASE_PATH env var > default path

    Returns:
        GraphPalace instance connected to the database

    Raises:
        HTTPException: If database cannot be accessed
    """
    # Get base path from environment or use default
    env_path = os.getenv("OMI_BASE_PATH")
    if env_path:
        base_path = Path(env_path)
    else:
        base_path = Path.home() / ".openclaw" / "omi"

    db_path = base_path / "palace.sqlite"

    # Check if database exists
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at {db_path}. Run 'omi init' to initialize."
        )

    try:
        palace = GraphPalace(db_path)
        return palace
    except Exception as e:
        logger.error(f"Failed to initialize GraphPalace: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to database: {str(e)}"
        )


@router.get("/health")
async def dashboard_health() -> Dict[str, str]:
    """
    Dashboard API health check endpoint.

    Returns:
        Health status and service name
    """
    return {
        "status": "healthy",
        "service": "omi-dashboard-api"
    }


@router.get("/memories")
async def get_memories(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of memories to return"),
    offset: int = Query(default=0, ge=0, description="Number of memories to skip"),
    memory_type: Optional[str] = Query(default=None, description="Filter by memory type (fact, experience, belief, decision)"),
    order_by: str = Query(default="created_at", description="Field to order by (created_at, access_count, last_accessed)"),
    order_dir: str = Query(default="desc", description="Order direction (asc, desc)")
) -> Dict[str, Any]:
    """
    Retrieve memories with optional filters.

    Query Parameters:
        limit: Maximum number of memories to return (1-1000, default 100)
        offset: Number of memories to skip for pagination (default 0)
        memory_type: Filter by type (fact, experience, belief, decision)
        order_by: Field to order by (created_at, access_count, last_accessed)
        order_dir: Order direction (asc, desc)

    Returns:
        Dict containing:
            - memories: List of memory objects
            - total: Total count of memories matching filters
            - limit: Applied limit
            - offset: Applied offset

    Raises:
        HTTPException: If database access fails or invalid parameters provided
    """
    # Validate memory_type if provided
    if memory_type is not None:
        valid_types = {"fact", "experience", "belief", "decision"}
        if memory_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid memory_type: {memory_type}. Must be one of: {valid_types}"
            )

    # Validate order_by
    valid_order_fields = {"created_at", "access_count", "last_accessed"}
    if order_by not in valid_order_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid order_by: {order_by}. Must be one of: {valid_order_fields}"
        )

    # Validate order_dir
    order_dir_upper = order_dir.upper()
    if order_dir_upper not in {"ASC", "DESC"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid order_dir: {order_dir}. Must be 'asc' or 'desc'"
        )

    try:
        palace = get_palace_instance()

        # Build SQL query
        base_query = """
            SELECT id, content, embedding, memory_type, confidence,
                   created_at, last_accessed, access_count, instance_ids, content_hash
            FROM memories
        """
        count_query = "SELECT COUNT(*) FROM memories"
        params: List[Any] = []

        # Add WHERE clause if filtering by memory_type
        if memory_type is not None:
            where_clause = " WHERE memory_type = ?"
            base_query += where_clause
            count_query += where_clause
            params.append(memory_type)

        # Get total count
        with palace._db_lock:
            cursor = palace._conn.execute(count_query, params)
            total_count = cursor.fetchone()[0]

            # Add ORDER BY and pagination
            base_query += f" ORDER BY {order_by} {order_dir_upper}"
            base_query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            # Execute query
            cursor = palace._conn.execute(base_query, params)
            rows = cursor.fetchall()

        # Convert rows to Memory objects
        memories = []
        for row in rows:
            # Don't include embedding in response (too large)
            memory = {
                "id": row[0],
                "content": row[1],
                "memory_type": row[3],
                "confidence": row[4],
                "created_at": row[5],
                "last_accessed": row[6],
                "access_count": row[7],
                "instance_ids": row[8] if row[8] else "[]",
                "content_hash": row[9]
            }
            memories.append(memory)

        return {
            "memories": memories,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


__all__ = ['router', 'get_palace_instance']
