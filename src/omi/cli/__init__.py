"""OMI CLI - OpenClaw Memory Infrastructure Command Line Interface

The seeking is the continuity. The palace remembers what the river forgets.

This module provides a modular CLI structure for OMI commands.
Command groups will be organized into separate modules:
- session.py: init, session-start, session-end
- memory.py: store, recall, check
- monitoring.py: status, audit
- config.py: config set, get, show
- events.py: events list, subscribe
- common.py: shared utilities
"""
from pathlib import Path
from typing import Optional
import click

# Local imports
from .common import get_base_path
from .session import session_group
from .memory import memory_group
from .monitoring import monitoring_group
from .config import config_group
from .events import events_group
from .sync import sync_group
from .plugins import plugins_group

# CLI version - matches project version
__version__ = "0.1.0"


@click.group()
@click.version_option(version=__version__, prog_name="omi")
@click.option('--data-dir', type=click.Path(), default=None, envvar='OMI_BASE_PATH',
              help='Base directory for OMI data (default: ~/.openclaw/omi)')
@click.pass_context
def cli(ctx, data_dir):
    """OMI - OpenClaw Memory Infrastructure

    A unified memory system for AI agents.

    \b
    Key Commands:
        init              Initialize memory infrastructure
        session-start     Load context and start a session
        session-end       End session and backup
        store             Store a memory
        recall            Search memories
        check             Pre-compression checkpoint
        status            Show health and size
        audit             Security audit
        config            Configuration management

    \b
    Examples:
        omi init
        omi session-start
        omi store "Fixed the auth bug" --type experience
        omi recall "session checkpoint"
        omi check
        omi session-end
    """
    ctx.ensure_object(dict)
    if data_dir:
        ctx.obj['data_dir'] = Path(data_dir)
    else:
        ctx.obj['data_dir'] = None


# Register session commands (init, session-start, session-end)
cli.add_command(session_group.commands['init'])
cli.add_command(session_group.commands['session-start'])
cli.add_command(session_group.commands['session-end'])

# Register memory commands (store, recall, check)
cli.add_command(memory_group.commands['store'])
cli.add_command(memory_group.commands['recall'])
cli.add_command(memory_group.commands['check'])

# Register monitoring commands (status, audit)
cli.add_command(monitoring_group.commands['status'])
cli.add_command(monitoring_group.commands['audit'])

# Register config command group (config set, get, show)
cli.add_command(config_group, name='config')

# Register events command group (events list, subscribe)
cli.add_command(events_group, name='events')

# Register sync command group (sync status, push, pull)
cli.add_command(sync_group, name='sync')

# Register plugins command group (plugins list)
cli.add_command(plugins_group, name='plugins')


def main():
    """Entry point for the CLI."""
    cli()


__all__ = [
    '__version__',
    'cli',
    'main',
    'get_base_path',
]
