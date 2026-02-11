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


__all__ = ['router', 'get_palace_instance']
