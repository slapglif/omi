"""FastAPI server startup module for OMI REST API.

This module provides the start_server function used by the CLI serve command.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def start_server(host: str = "0.0.0.0", port: int = 8420, base_path: Optional[Path] = None) -> None:
    """Start the OMI FastAPI server using uvicorn.

    Args:
        host: Host address to bind to
        port: Port to bind to
        base_path: Base path for OMI data (optional, for future use)

    Raises:
        ImportError: If uvicorn is not installed
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install 'omi[server]' or pip install uvicorn"
        )

    logger.info(f"Starting OMI REST API server on {host}:{port}")

    # Import the FastAPI app from rest_api module
    # This must be done after environment variables are set by the CLI
    from omi.rest_api import app

    # Run uvicorn server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
