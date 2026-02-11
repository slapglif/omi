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
@click.option('--verbose', '-v', is_flag=True, default=False,
              help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, default=False,
              help='Suppress non-essential output')
@click.pass_context
def cli(ctx, data_dir, verbose, quiet):
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
    from .common import VERBOSITY_QUIET, VERBOSITY_NORMAL, VERBOSITY_VERBOSE

    ctx.ensure_object(dict)

    # Validate mutually exclusive flags
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive")

    # Set verbosity level
    if quiet:
        ctx.obj['verbosity'] = VERBOSITY_QUIET
    elif verbose:
        ctx.obj['verbosity'] = VERBOSITY_VERBOSE
    else:
        ctx.obj['verbosity'] = VERBOSITY_NORMAL

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


@cli.command()
@click.argument('shell', type=click.Choice(['bash', 'zsh']), required=True)
def completion(shell):
    """Generate shell completion script.

    Outputs a completion script for the specified shell that can be sourced
    to enable tab-completion for OMI commands.

    Args:
        shell: Shell type (bash or zsh)

    Examples:
        # Bash - add to ~/.bashrc:
        eval "$(omi completion bash)"

        # Zsh - add to ~/.zshrc:
        eval "$(omi completion zsh)"
    """
    # Click's shell completion works via environment variables
    # We simulate the completion script generation process
    prog_name = "omi"

    if shell == "bash":
        # Generate bash completion script
        script = f"""# omi bash completion
_{prog_name.upper()}_COMPLETE=bash_source {prog_name}() {{
    local IFS=$'\\n'
    local response

    response=$(env COMP_WORDS="${{COMP_WORDS[*]}}" COMP_CWORD=${{COMP_CWORD}} _{prog_name.upper()}_COMPLETE=bash_complete $1)

    for completion in $response; do
        IFS=',' read type value <<< "$completion"

        if [[ $type == 'dir' ]]; then
            COMPREPLY=()
            compopt -o dirnames
        elif [[ $type == 'file' ]]; then
            COMPREPLY=()
            compopt -o default
        elif [[ $type == 'plain' ]]; then
            COMPREPLY+=($value)
        fi
    done

    return 0
}}

complete -F {prog_name} -o nosort -o bashdefault -o default {prog_name}"""
    else:  # zsh
        # Generate zsh completion script
        script = f"""#compdef {prog_name}

_{prog_name}_completion() {{
    local -a completions
    local -a completions_with_descriptions
    local -a response
    (( ! $+commands[{prog_name}] )) && return 1

    response=("${{(@f)$(env COMP_WORDS="${{words[*]}}" COMP_CWORD=${{#words[@]}} _{prog_name.upper()}_COMPLETE=zsh_complete {prog_name})}}")

    for type value in ${{response}}; do
        if [[ $type == 'plain' ]]; then
            if [[ $value == *$'\\t'* ]]; then
                completions_with_descriptions+=("$value")
            else
                completions+=("$value")
            fi
        fi
    done

    if [ -n "$completions_with_descriptions" ]; then
        _describe -V unsorted completions_with_descriptions -U
    fi

    if [ -n "$completions" ]; then
        compadd -U -V unsorted -a completions
    fi
}}

if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    # autoload mode, define completion function
    _{prog_name}_completion
else
    # eval mode, call compdef
    compdef _{prog_name}_completion {prog_name}
fi"""

    click.echo(script)


def main():
    """Entry point for the CLI."""
    cli()


__all__ = [
    '__version__',
    'cli',
    'main',
    'get_base_path',
]
