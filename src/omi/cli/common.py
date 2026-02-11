"""Shared utilities for OMI CLI commands."""
import os
from pathlib import Path
from typing import Optional

# Default paths
DEFAULT_BASE_PATH = Path.home() / ".openclaw" / "omi"
DEFAULT_CONFIG_PATH = DEFAULT_BASE_PATH / "config.yaml"


def get_base_path(ctx_data_dir: Optional[Path] = None) -> Path:
    """Get the base path for OMI data.

    Priority: --data-dir flag > OMI_BASE_PATH env var > default path.

    Args:
        ctx_data_dir: Value from --data-dir CLI option, if provided.
    """
    if ctx_data_dir:
        return Path(ctx_data_dir)
    env_path = os.getenv("OMI_BASE_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_BASE_PATH
