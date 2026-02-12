"""Policy management commands for OMI CLI."""
import sys
from pathlib import Path
import click

# Local CLI imports
from .common import get_base_path, echo_quiet, echo_normal


@click.group()
def policy_group():
    """Policy management commands."""
    pass
