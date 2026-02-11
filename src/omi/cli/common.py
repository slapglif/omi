"""Shared utilities for OMI CLI commands."""
import os
import sys
from pathlib import Path
from typing import Optional

import click

# Default paths
DEFAULT_BASE_PATH = Path.home() / ".openclaw" / "omi"
DEFAULT_CONFIG_PATH = DEFAULT_BASE_PATH / "config.yaml"

# Verbosity levels
VERBOSITY_QUIET = 0
VERBOSITY_NORMAL = 1
VERBOSITY_VERBOSE = 2


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


def should_print(verbosity: int, message_level: int) -> bool:
    """Determine if a message should be printed based on verbosity settings.

    Args:
        verbosity: Current verbosity level (0=quiet, 1=normal, 2=verbose).
        message_level: Minimum verbosity level required for this message.

    Returns:
        True if the message should be printed, False otherwise.
    """
    return verbosity >= message_level


def echo_verbose(message: str, verbosity: int) -> None:
    """Print a message only in verbose mode.

    Args:
        message: The message to print.
        verbosity: Current verbosity level.
    """
    if should_print(verbosity, VERBOSITY_VERBOSE):
        click.echo(message, err=False)


def echo_normal(message: str, verbosity: int) -> None:
    """Print a message in normal and verbose modes.

    Args:
        message: The message to print.
        verbosity: Current verbosity level.
    """
    if should_print(verbosity, VERBOSITY_NORMAL):
        click.echo(message, err=False)


def echo_quiet(message: str, verbosity: int) -> None:
    """Print a critical message that should always be shown (even in quiet mode).

    Args:
        message: The critical message to print.
        verbosity: Current verbosity level (unused, but kept for consistency).
    """
    click.echo(message, err=False)
