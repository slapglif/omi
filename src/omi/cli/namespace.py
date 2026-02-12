"""Namespace management CLI commands"""
import click
from pathlib import Path
import os
import getpass

# OMI imports
from omi.shared_namespace import SharedNamespace

# Local CLI imports
from .common import (
    get_base_path,
    VERBOSITY_NORMAL,
    echo_verbose,
    echo_normal,
)


@click.group()
@click.pass_context
def namespace_group(ctx: click.Context) -> None:
    """Namespace management commands."""
    ctx.ensure_object(dict)


@namespace_group.command("create")
@click.argument("namespace")
@click.option(
    "--shared",
    is_flag=True,
    help="Create a shared namespace for multi-agent coordination"
)
@click.option(
    "--created-by",
    default=None,
    help="Agent ID creating the namespace (defaults to current user)"
)
@click.option(
    "--metadata",
    default=None,
    help="JSON metadata for the namespace"
)
@click.pass_context
def create(ctx: click.Context, namespace: str, shared: bool, created_by: str, metadata: str) -> None:
    """Create a shared namespace for multi-agent coordination.

    Creates a new shared namespace that can be used by multiple agents
    to coordinate memory operations. Shared namespaces support:

    - Per-agent read/write/admin permissions
    - Memory subscriptions and notifications
    - Conflict resolution for concurrent writes
    - Cross-agent belief propagation

    \b
    Examples:
        omi namespace create --shared team-alpha
        omi namespace create --shared research/project-x --created-by agent-001
        omi namespace create --shared collab --metadata '{"team": "research"}'
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)

    # Require --shared flag for now
    if not shared:
        click.echo(
            click.style(
                "Error: Only shared namespaces are currently supported. Use --shared flag.",
                fg="red"
            )
        )
        raise click.Abort()

    # Default created_by to current user if not specified
    if created_by is None:
        created_by = getpass.getuser()

    # Parse metadata if provided
    metadata_dict = None
    if metadata:
        import json
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError as e:
            click.echo(click.style(f"Error: Invalid JSON metadata: {e}", fg="red"))
            raise click.Abort()

    # Initialize SharedNamespace
    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(
            click.style(
                f"Error: Database not found at {db_path}. Run 'omi init' first.",
                fg="red"
            )
        )
        raise click.Abort()

    try:
        shared_ns = SharedNamespace(db_path)

        echo_verbose(
            f"Creating shared namespace '{namespace}' created by '{created_by}'...",
            verbosity
        )

        # Create the shared namespace
        ns_info = shared_ns.create(
            namespace=namespace,
            created_by=created_by,
            metadata=metadata_dict
        )

        echo_normal(
            click.style(f"✓ Created shared namespace: {ns_info.namespace}", fg="green"),
            verbosity
        )
        echo_verbose(f"  Created by: {ns_info.created_by}", verbosity)
        echo_verbose(f"  Created at: {ns_info.created_at}", verbosity)

        if ns_info.metadata:
            echo_verbose(f"  Metadata: {ns_info.metadata}", verbosity)

        echo_normal(
            "\nNext steps:",
            verbosity
        )
        echo_normal(
            f"  1. Grant permissions: omi namespace grant {namespace} <agent-id> <read|write|admin>",
            verbosity
        )
        echo_normal(
            f"  2. Subscribe to updates: omi subscribe {namespace}",
            verbosity
        )

    except ValueError as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        raise click.Abort()
    except Exception as e:
        click.echo(click.style(f"Unexpected error: {e}", fg="red"))
        raise click.Abort()
