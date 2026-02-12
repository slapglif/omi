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
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os
import json
import logging

from .storage.graph_palace import GraphPalace, Memory
from .embeddings import OllamaEmbedder, EmbeddingCache

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


# Global cache for embedder to avoid reinitializing
_embedder_cache: Optional[Tuple[OllamaEmbedder, EmbeddingCache]] = None


def get_embedder_and_cache() -> Tuple[OllamaEmbedder, EmbeddingCache]:
    """
    Get or create embedder and embedding cache instances.

    Uses lazy initialization - creates instances on first call and reuses them.

    Returns:
        Tuple of (OllamaEmbedder, EmbeddingCache)

    Raises:
        HTTPException: If embedder initialization fails
    """
    global _embedder_cache

    if _embedder_cache is not None:
        return _embedder_cache

    try:
        # Get base path for cache directory
        env_path = os.getenv("OMI_BASE_PATH")
        if env_path:
            base_path = Path(env_path)
        else:
            base_path = Path.home() / ".openclaw" / "omi"

        cache_dir = base_path / "embeddings"

        # Initialize embedder (Ollama as default - no API key needed)
        embedder = OllamaEmbedder()

        # Initialize cache
        cache = EmbeddingCache(cache_dir=cache_dir, embedder=embedder)

        _embedder_cache = (embedder, cache)
        return _embedder_cache

    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize embedder: {str(e)}. Make sure Ollama is running."
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


@router.get("/edges")
async def get_edges(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of edges to return"),
    offset: int = Query(default=0, ge=0, description="Number of edges to skip"),
    edge_type: Optional[str] = Query(default=None, description="Filter by edge type (SUPPORTS, CONTRADICTS, RELATED_TO, DEPENDS_ON, POSTED, DISCUSSED)"),
    order_by: str = Query(default="created_at", description="Field to order by (created_at, strength)"),
    order_dir: str = Query(default="desc", description="Order direction (asc, desc)")
) -> Dict[str, Any]:
    """
    Retrieve relationship edges with optional filters.

    Query Parameters:
        limit: Maximum number of edges to return (1-1000, default 100)
        offset: Number of edges to skip for pagination (default 0)
        edge_type: Filter by type (SUPPORTS, CONTRADICTS, RELATED_TO, DEPENDS_ON, POSTED, DISCUSSED)
        order_by: Field to order by (created_at, strength)
        order_dir: Order direction (asc, desc)

    Returns:
        Dict containing:
            - edges: List of edge objects
            - total_count: Total count of edges matching filters
            - limit: Applied limit
            - offset: Applied offset

    Raises:
        HTTPException: If database access fails or invalid parameters provided
    """
    # Validate edge_type if provided
    if edge_type is not None:
        valid_types = {"SUPPORTS", "CONTRADICTS", "RELATED_TO", "DEPENDS_ON", "POSTED", "DISCUSSED"}
        if edge_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid edge_type: {edge_type}. Must be one of: {valid_types}"
            )

    # Validate order_by
    valid_order_fields = {"created_at", "strength"}
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
            SELECT id, source_id, target_id, edge_type, strength, created_at
            FROM edges
        """
        count_query = "SELECT COUNT(*) FROM edges"
        params: List[Any] = []

        # Add WHERE clause if filtering by edge_type
        if edge_type is not None:
            where_clause = " WHERE edge_type = ?"
            base_query += where_clause
            count_query += where_clause
            params.append(edge_type)

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

        # Convert rows to edge objects
        edges = []
        for row in rows:
            edge = {
                "id": row[0],
                "source_id": row[1],
                "target_id": row[2],
                "edge_type": row[3],
                "strength": row[4],
                "created_at": row[5]
            }
            edges.append(edge)

        return {
            "edges": edges,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve edges: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve edges: {str(e)}"
        )


@router.get("/beliefs")
async def get_beliefs(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of beliefs to return"),
    offset: int = Query(default=0, ge=0, description="Number of beliefs to skip"),
    order_by: str = Query(default="last_updated", description="Field to order by (confidence, created_at, last_updated, evidence_count)"),
    order_dir: str = Query(default="desc", description="Order direction (asc, desc)")
) -> Dict[str, Any]:
    """
    Retrieve beliefs from the belief network.

    Query Parameters:
        limit: Maximum number of beliefs to return (1-1000, default 100)
        offset: Number of beliefs to skip for pagination (default 0)
        order_by: Field to order by (confidence, created_at, last_updated, evidence_count)
        order_dir: Order direction (asc, desc)

    Returns:
        Dict containing:
            - beliefs: List of belief objects
            - total_count: Total count of beliefs
            - limit: Applied limit
            - offset: Applied offset

    Raises:
        HTTPException: If database access fails or invalid parameters provided
    """
    # Validate order_by
    valid_order_fields = {"confidence", "created_at", "last_updated", "evidence_count"}
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
            SELECT id, content, confidence, created_at, last_updated, evidence_count
            FROM beliefs
        """
        count_query = "SELECT COUNT(*) FROM beliefs"
        params: List[Any] = []

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

        # Convert rows to belief objects
        beliefs = []
        for row in rows:
            belief = {
                "id": row[0],
                "content": row[1],
                "confidence": row[2],
                "created_at": row[3],
                "last_updated": row[4],
                "evidence_count": row[5]
            }
            beliefs.append(belief)

        return {
            "beliefs": beliefs,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve beliefs: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve beliefs: {str(e)}"
        )


@router.get("/graph")
async def get_graph(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of memories and edges to return")
) -> Dict[str, Any]:
    """
    Retrieve complete graph data (memories + edges) in one call.

    This endpoint is optimized for dashboard visualization, returning both
    memories and edges in a single request. Unlike /memories and /edges,
    this endpoint returns ALL edges (up to limit) and ALL memories (up to limit),
    without filtering.

    Query Parameters:
        limit: Maximum number of memories and edges to return (1-1000, default 100)

    Returns:
        Dict containing:
            - memories: List of memory objects (nodes)
            - edges: List of edge objects (relationships)
            - memory_count: Total number of memories in database
            - edge_count: Total number of edges in database
            - limit: Applied limit

    Raises:
        HTTPException: If database access fails
    """
    try:
        palace = get_palace_instance()

        with palace._db_lock:
            # Get total counts
            memory_count_cursor = palace._conn.execute("SELECT COUNT(*) FROM memories")
            total_memory_count = memory_count_cursor.fetchone()[0]

            edge_count_cursor = palace._conn.execute("SELECT COUNT(*) FROM edges")
            total_edge_count = edge_count_cursor.fetchone()[0]

            # Retrieve memories (ordered by most recently accessed)
            memory_query = """
                SELECT id, content, memory_type, confidence,
                       created_at, last_accessed, access_count, instance_ids, content_hash
                FROM memories
                ORDER BY last_accessed DESC
                LIMIT ?
            """
            memory_cursor = palace._conn.execute(memory_query, [limit])
            memory_rows = memory_cursor.fetchall()

            # Retrieve edges (ordered by most recent)
            edge_query = """
                SELECT id, source_id, target_id, edge_type, strength, created_at
                FROM edges
                ORDER BY created_at DESC
                LIMIT ?
            """
            edge_cursor = palace._conn.execute(edge_query, [limit])
            edge_rows = edge_cursor.fetchall()

        # Convert memory rows to dict objects
        memories = []
        for row in memory_rows:
            # Don't include embedding in response (too large for graph viz)
            memory = {
                "id": row[0],
                "content": row[1],
                "memory_type": row[2],
                "confidence": row[3],
                "created_at": row[4],
                "last_accessed": row[5],
                "access_count": row[6],
                "instance_ids": row[7] if row[7] else "[]",
                "content_hash": row[8]
            }
            memories.append(memory)

        # Convert edge rows to dict objects
        edges = []
        for row in edge_rows:
            edge = {
                "id": row[0],
                "source_id": row[1],
                "target_id": row[2],
                "edge_type": row[3],
                "strength": row[4],
                "created_at": row[5]
            }
            edges.append(edge)

        return {
            "memories": memories,
            "edges": edges,
            "memory_count": total_memory_count,
            "edge_count": total_edge_count,
            "limit": limit
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve graph data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve graph data: {str(e)}"
        )


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get database storage statistics.

    Returns aggregate counts and distributions for memories and edges,
    providing an overview of the knowledge graph structure.

    Returns:
        Dict containing:
            - memory_count: Total number of memories in database
            - edge_count: Total number of edges in database
            - type_distribution: Dict mapping memory_type to count
            - edge_distribution: Dict mapping edge_type to count

    Raises:
        HTTPException: If database access fails
    """
    try:
        palace = get_palace_instance()

        with palace._db_lock:
            # Get total memory count
            cursor = palace._conn.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]

            # Get total edge count
            cursor = palace._conn.execute("SELECT COUNT(*) FROM edges")
            edge_count = cursor.fetchone()[0]

            # Get memory type distribution
            cursor = palace._conn.execute("""
                SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type
            """)
            type_distribution = {row[0]: row[1] for row in cursor}

            # Get edge type distribution
            cursor = palace._conn.execute("""
                SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type
            """)
            edge_distribution = {row[0]: row[1] for row in cursor}

        return {
            "memory_count": memory_count,
            "edge_count": edge_count,
            "type_distribution": type_distribution,
            "edge_distribution": edge_distribution
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve database stats: {str(e)}"
        )


@router.get("/search")
async def search_memories(
    q: str = Query(..., description="Search query text", min_length=1),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results to return"),
    min_relevance: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum relevance threshold")
) -> Dict[str, Any]:
    """
    Semantic search for memories using embeddings.

    Performs semantic similarity search using embedding vectors and returns
    memories ranked by relevance with recency weighting.

    Query Parameters:
        q: Search query text (required)
        limit: Maximum number of results to return (1-100, default 10)
        min_relevance: Minimum similarity threshold (0.0-1.0, default 0.5)

    Returns:
        Dict containing:
            - results: List of matching memories with relevance scores
            - query: The search query that was used
            - limit: Applied limit
            - count: Number of results returned

    Raises:
        HTTPException: If search fails or embedder is unavailable
    """
    try:
        # Get palace instance
        palace = get_palace_instance()

        # Get embedder and cache
        embedder, cache = get_embedder_and_cache()

        # Generate embedding for query
        query_embedding = cache.get_or_compute(q)

        # Perform semantic recall
        results_tuples: List[Tuple[Memory, float]] = palace.recall(
            query_embedding=query_embedding,
            limit=limit,
            min_relevance=min_relevance
        )

        # Convert results to dict format
        results = []
        for memory, relevance_score in results_tuples:
            memory_dict = {
                "id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "confidence": memory.confidence,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
                "access_count": memory.access_count,
                "instance_ids": memory.instance_ids,
                "content_hash": memory.content_hash,
                "relevance_score": relevance_score
            }
            results.append(memory_dict)

        return {
            "results": results,
            "query": q,
            "limit": limit,
            "count": len(results)
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/versions/timeline")
async def get_version_timeline(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of versions to return"),
    offset: int = Query(default=0, ge=0, description="Number of versions to skip"),
    start_date: Optional[str] = Query(default=None, description="Start date filter (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date filter (YYYY-MM-DD)"),
    operation_type: Optional[str] = Query(default=None, description="Filter by operation type (CREATE, UPDATE, DELETE)"),
    memory_id: Optional[str] = Query(default=None, description="Filter by specific memory ID")
) -> Dict[str, Any]:
    """
    Retrieve version timeline with optional filters.

    Returns memory versions grouped by date for timeline visualization.
    Supports filtering by date range, operation type, and specific memory.

    Query Parameters:
        limit: Maximum number of versions to return (1-1000, default 100)
        offset: Number of versions to skip for pagination (default 0)
        start_date: Start date filter in YYYY-MM-DD format (inclusive)
        end_date: End date filter in YYYY-MM-DD format (inclusive)
        operation_type: Filter by type (CREATE, UPDATE, DELETE)
        memory_id: Filter by specific memory ID

    Returns:
        Dict containing:
            - versions: List of version objects with metadata
            - total: Total count of versions matching filters
            - limit: Applied limit
            - offset: Applied offset
            - grouped_by_date: Versions grouped by date (YYYY-MM-DD)

    Raises:
        HTTPException: If database access fails or invalid parameters provided
    """
    # Validate operation_type if provided
    if operation_type is not None:
        valid_ops = {"CREATE", "UPDATE", "DELETE"}
        if operation_type not in valid_ops:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation_type: {operation_type}. Must be one of: {valid_ops}"
            )

    # Validate date formats if provided
    if start_date is not None:
        try:
            from datetime import datetime
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid start_date format: {start_date}. Must be YYYY-MM-DD"
            )

    if end_date is not None:
        try:
            from datetime import datetime
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid end_date format: {end_date}. Must be YYYY-MM-DD"
            )

    try:
        palace = get_palace_instance()

        # Build SQL query
        base_query = """
            SELECT version_id, memory_id, content, version_number,
                   operation_type, created_at, previous_version_id
            FROM memory_versions
        """
        count_query = "SELECT COUNT(*) FROM memory_versions"
        params: List[Any] = []
        conditions: List[str] = []

        # Add filters
        if start_date is not None:
            conditions.append("DATE(created_at) >= ?")
            params.append(start_date)

        if end_date is not None:
            conditions.append("DATE(created_at) <= ?")
            params.append(end_date)

        if operation_type is not None:
            conditions.append("operation_type = ?")
            params.append(operation_type)

        if memory_id is not None:
            conditions.append("memory_id = ?")
            params.append(memory_id)

        # Build WHERE clause
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            base_query += where_clause
            count_query += where_clause

        # Add ordering and pagination
        base_query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute queries
        with palace._db_lock:
            # Get total count
            count_cursor = palace._conn.execute(count_query, params[:-2] if len(params) > 2 else [])
            total = count_cursor.fetchone()[0]

            # Get versions
            cursor = palace._conn.execute(base_query, params)
            rows = cursor.fetchall()

        # Format results
        versions = []
        grouped_by_date: Dict[str, List[Dict[str, Any]]] = {}

        for row in rows:
            version_obj = {
                "version_id": row[0],
                "memory_id": row[1],
                "content": row[2][:200] + "..." if len(row[2]) > 200 else row[2],  # Preview
                "content_full": row[2],  # Full content
                "version_number": row[3],
                "operation_type": row[4],
                "created_at": row[5],
                "previous_version_id": row[6]
            }
            versions.append(version_obj)

            # Group by date
            date_key = row[5].split("T")[0] if "T" in row[5] else row[5].split(" ")[0]
            if date_key not in grouped_by_date:
                grouped_by_date[date_key] = []
            grouped_by_date[date_key].append(version_obj)

        return {
            "versions": versions,
            "total": total,
            "limit": limit,
            "offset": offset,
            "grouped_by_date": grouped_by_date
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve version timeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve version timeline: {str(e)}"
        )


@router.get("/snapshots")
async def get_snapshots(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of snapshots to return"),
    offset: int = Query(default=0, ge=0, description="Number of snapshots to skip"),
    order_by: str = Query(default="created_at", description="Field to order by (created_at)"),
    order_dir: str = Query(default="desc", description="Order direction (asc, desc)")
) -> Dict[str, Any]:
    """
    Retrieve list of memory snapshots with metadata.

    Returns snapshots with memory counts and metadata for snapshot visualization.

    Query Parameters:
        limit: Maximum number of snapshots to return (1-500, default 50)
        offset: Number of snapshots to skip for pagination (default 0)
        order_by: Field to order by (created_at)
        order_dir: Order direction (asc, desc)

    Returns:
        Dict containing:
            - snapshots: List of snapshot objects with metadata
            - total: Total count of snapshots
            - limit: Applied limit
            - offset: Applied offset

    Raises:
        HTTPException: If database access fails or invalid parameters provided
    """
    # Validate order_by
    valid_order_fields = {"created_at"}
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
        base_query = f"""
            SELECT s.snapshot_id, s.created_at, s.description,
                   s.metadata_json, s.moltvault_backup_id,
                   COUNT(sm.memory_id) as memory_count
            FROM snapshots s
            LEFT JOIN snapshot_memories sm ON s.snapshot_id = sm.snapshot_id
            GROUP BY s.snapshot_id, s.created_at, s.description, s.metadata_json, s.moltvault_backup_id
            ORDER BY s.{order_by} {order_dir_upper}
            LIMIT ? OFFSET ?
        """

        count_query = "SELECT COUNT(*) FROM snapshots"

        # Execute queries
        with palace._db_lock:
            # Get total count
            count_cursor = palace._conn.execute(count_query)
            total = count_cursor.fetchone()[0]

            # Get snapshots
            cursor = palace._conn.execute(base_query, [limit, offset])
            rows = cursor.fetchall()

        # Format results
        snapshots = []
        for row in rows:
            # Parse metadata JSON if present
            metadata = {}
            if row[3]:
                try:
                    metadata = json.loads(row[3])
                except json.JSONDecodeError:
                    metadata = {}

            # Determine if snapshot is delta or full
            is_delta = metadata.get("is_delta", False)
            base_snapshot_id = metadata.get("base_snapshot_id")

            snapshot_obj = {
                "snapshot_id": row[0],
                "created_at": row[1],
                "description": row[2],
                "metadata": metadata,
                "moltvault_backup_id": row[4],
                "memory_count": row[5],
                "is_delta": is_delta,
                "base_snapshot_id": base_snapshot_id
            }
            snapshots.append(snapshot_obj)

        return {
            "snapshots": snapshots,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve snapshots: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve snapshots: {str(e)}"
        )


__all__ = ['router', 'get_palace_instance']
