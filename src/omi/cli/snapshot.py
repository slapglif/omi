"""Snapshot management commands for OMI CLI."""
import sys
import json
from pathlib import Path
from typing import Optional
import click

# OMI imports
from omi.storage.snapshots import SnapshotManager

# Local CLI imports
from .common import (
    get_base_path,
    echo_verbose,
    echo_normal,
    echo_quiet,
    VERBOSITY_NORMAL
)


@click.group()
def snapshot_group():
    """Snapshot management commands."""
    pass


@snapshot_group.command("create")
@click.option('--description', '-d', default=None,
              help='Optional description for the snapshot')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def create(ctx, description: Optional[str], json_output: bool) -> None:
    """Create a point-in-time snapshot of memory state.

    Creates a snapshot capturing the current state of all memories.
    Snapshots use delta encoding - only changes since the last snapshot
    are stored, making them efficient.

    Args:
        --description: Optional description for the snapshot
        --json: Output as JSON (for scripts)

    Examples:
        omi snapshot create --description "Before major refactor"
        omi snapshot create -d "Checkpoint after feature X"
        omi snapshot create --json
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        echo_quiet(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        manager = SnapshotManager(db_path)
        snapshot = manager.create_snapshot(description=description)

        if json_output:
            output = snapshot.to_dict()
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            echo_normal(click.style("✓ Snapshot created", fg="green", bold=True), verbosity)
            echo_normal(f"  ID: {click.style(snapshot.snapshot_id, fg='cyan')}", verbosity)
            echo_normal(f"  Type: {click.style('Delta' if snapshot.is_delta else 'Full', fg='yellow')}", verbosity)
            echo_normal(f"  Memories: {click.style(str(snapshot.memory_count), fg='cyan')}", verbosity)
            if snapshot.description:
                echo_normal(f"  Description: {snapshot.description}", verbosity)
            echo_normal(f"  Created: {snapshot.created_at.strftime('%Y-%m-%d %H:%M:%S')}", verbosity)

    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to create snapshot: {e}", fg="red"), verbosity)
        sys.exit(1)


@snapshot_group.command("list")
@click.option('--limit', '-l', default=None, type=int,
              help='Maximum number of snapshots to show')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def list_snapshots(ctx, limit: Optional[int], json_output: bool) -> None:
    """List all snapshots ordered by creation time.

    Shows all snapshots with their details, ordered from newest to oldest.
    Use --limit to restrict the number of results.

    Args:
        --limit: Maximum number of snapshots to show
        --json: Output as JSON (for scripts)

    Examples:
        omi snapshot list
        omi snapshot list --limit 10
        omi snapshot list --json
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        echo_quiet(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        manager = SnapshotManager(db_path)
        snapshots = manager.list_snapshots(limit=limit)

        if json_output:
            output = [s.to_dict() for s in snapshots]
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            if not snapshots:
                echo_normal(click.style("No snapshots found.", fg="yellow"), verbosity)
                echo_normal("Create your first snapshot with: omi snapshot create", verbosity)
                return

            echo_normal(click.style(f"Snapshots ({len(snapshots)} found)", fg="cyan", bold=True), verbosity)
            echo_normal("=" * 70, verbosity)

            for i, snapshot in enumerate(snapshots, 1):
                snapshot_type = "Delta" if snapshot.is_delta else "Full"
                type_color = "yellow" if snapshot.is_delta else "green"

                echo_normal(f"\n{i}. {click.style(snapshot.snapshot_id, fg='cyan', bold=True)}", verbosity)
                echo_normal(f"   Type: {click.style(snapshot_type, fg=type_color)} | Memories: {snapshot.memory_count}", verbosity)
                echo_normal(f"   Created: {snapshot.created_at.strftime('%Y-%m-%d %H:%M:%S')}", verbosity)
                if snapshot.description:
                    echo_normal(f"   Description: {snapshot.description}", verbosity)
                if snapshot.moltvault_backup_id:
                    echo_normal(f"   Backup: {click.style(snapshot.moltvault_backup_id, fg='blue')}", verbosity)

    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to list snapshots: {e}", fg="red"), verbosity)
        sys.exit(1)


@snapshot_group.command("diff")
@click.argument('snapshot1_id')
@click.argument('snapshot2_id')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def diff(ctx, snapshot1_id: str, snapshot2_id: str, json_output: bool) -> None:
    """Show differences between two snapshots.

    Compares two snapshots and displays what changed: memories added,
    modified, or deleted. Typically snapshot1 should be older than snapshot2.

    Args:
        snapshot1_id: First snapshot ID (typically older)
        snapshot2_id: Second snapshot ID (typically newer)
        --json: Output as JSON (for scripts)

    Examples:
        omi snapshot diff snap-abc123 snap-def456
        omi snapshot diff snap-abc123 snap-def456 --json
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        echo_quiet(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        manager = SnapshotManager(db_path)
        diff_result = manager.diff_snapshots(snapshot1_id, snapshot2_id)

        if json_output:
            output = diff_result.to_dict()
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            echo_normal(click.style(f"Snapshot Diff: {snapshot1_id} → {snapshot2_id}", fg="cyan", bold=True), verbosity)
            echo_normal("=" * 70, verbosity)

            if diff_result.total_changes == 0:
                echo_normal(click.style("\n✓ No changes between snapshots", fg="green"), verbosity)
                return

            echo_normal(f"\nTotal changes: {click.style(str(diff_result.total_changes), fg='cyan', bold=True)}", verbosity)

            if diff_result.added:
                echo_normal(f"\n{click.style('+ Added', fg='green', bold=True)} ({len(diff_result.added)} memories):", verbosity)
                for memory_id in diff_result.added[:10]:  # Show first 10
                    echo_normal(f"  + {click.style(memory_id, fg='green')}", verbosity)
                if len(diff_result.added) > 10:
                    echo_normal(f"  ... and {len(diff_result.added) - 10} more", verbosity)

            if diff_result.modified:
                echo_normal(f"\n{click.style('~ Modified', fg='yellow', bold=True)} ({len(diff_result.modified)} memories):", verbosity)
                for memory_id in diff_result.modified[:10]:  # Show first 10
                    echo_normal(f"  ~ {click.style(memory_id, fg='yellow')}", verbosity)
                if len(diff_result.modified) > 10:
                    echo_normal(f"  ... and {len(diff_result.modified) - 10} more", verbosity)

            if diff_result.deleted:
                echo_normal(f"\n{click.style('- Deleted', fg='red', bold=True)} ({len(diff_result.deleted)} memories):", verbosity)
                for memory_id in diff_result.deleted[:10]:  # Show first 10
                    echo_normal(f"  - {click.style(memory_id, fg='red')}", verbosity)
                if len(diff_result.deleted) > 10:
                    echo_normal(f"  ... and {len(diff_result.deleted) - 10} more", verbosity)

    except ValueError as e:
        echo_quiet(click.style(f"Error: {e}", fg="red"), verbosity)
        sys.exit(1)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to diff snapshots: {e}", fg="red"), verbosity)
        sys.exit(1)


@snapshot_group.command("rollback")
@click.argument('snapshot_id')
@click.option('--dry-run', is_flag=True,
              help='Show what would be changed without applying')
@click.option('--force', '-f', is_flag=True,
              help='Skip confirmation prompt')
@click.pass_context
def rollback(ctx, snapshot_id: str, dry_run: bool, force: bool) -> None:
    """Rollback memory state to a specific snapshot.

    WARNING: This is a destructive operation that will modify current memory
    state. It's recommended to create a backup snapshot before rolling back.

    Use --dry-run to preview changes without applying them.
    Use --force to skip the confirmation prompt.

    Args:
        snapshot_id: Snapshot ID to rollback to
        --dry-run: Preview changes without applying
        --force: Skip confirmation prompt

    Examples:
        omi snapshot rollback snap-abc123 --dry-run
        omi snapshot rollback snap-abc123
        omi snapshot rollback snap-abc123 --force
    """
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        echo_quiet(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        manager = SnapshotManager(db_path)

        # Verify snapshot exists
        snapshot = manager.get_snapshot(snapshot_id)
        if not snapshot:
            echo_quiet(click.style(f"Error: Snapshot not found: {snapshot_id}", fg="red"), verbosity)
            sys.exit(1)

        # Show snapshot info
        echo_normal(click.style("Rollback Target", fg="cyan", bold=True), verbosity)
        echo_normal(f"  Snapshot: {click.style(snapshot.snapshot_id, fg='cyan')}", verbosity)
        echo_normal(f"  Created: {snapshot.created_at.strftime('%Y-%m-%d %H:%M:%S')}", verbosity)
        if snapshot.description:
            echo_normal(f"  Description: {snapshot.description}", verbosity)

        # Create automatic backup before rollback (if not dry-run)
        if not dry_run:
            echo_normal(f"\n{click.style('Creating automatic backup...', fg='yellow')}", verbosity)
            backup_snapshot = manager.create_snapshot(
                description=f"Automatic backup before rollback to {snapshot_id}"
            )
            echo_normal(f"  Backup: {click.style(backup_snapshot.snapshot_id, fg='green')}", verbosity)

        # Get preview of changes
        echo_normal(f"\n{click.style('Analyzing changes...', fg='cyan')}", verbosity)

        # For dry-run, we need to show what would change
        # We can do this by comparing current state to target snapshot state
        current_snapshots = manager.list_snapshots(limit=1)
        if current_snapshots:
            latest = current_snapshots[0]
            diff_result = manager.diff_snapshots(snapshot_id, latest.snapshot_id)

            echo_normal(f"\nChanges that will be applied:", verbosity)
            echo_normal(f"  Memories to restore: {click.style(str(len(diff_result.deleted)), fg='green')}", verbosity)
            echo_normal(f"  Memories to revert: {click.style(str(len(diff_result.modified)), fg='yellow')}", verbosity)
            echo_normal(f"  Memories to remove: {click.style(str(len(diff_result.added)), fg='red')}", verbosity)
            echo_normal(f"  Total changes: {click.style(str(diff_result.total_changes), fg='cyan', bold=True)}", verbosity)

        if dry_run:
            echo_normal(f"\n{click.style('✓ Dry-run complete - no changes applied', fg='green')}", verbosity)
            echo_normal("Run without --dry-run to apply the rollback.", verbosity)
            return

        # Confirm with user (unless --force)
        if not force:
            echo_normal("", verbosity)  # Blank line
            echo_quiet(click.style("⚠ WARNING: This is a destructive operation!", fg="yellow", bold=True), verbosity)
            echo_quiet("Current memory state will be modified to match the snapshot.", verbosity)

            if not click.confirm(click.style("Do you want to proceed with the rollback?", fg="yellow")):
                echo_normal("Rollback cancelled.", verbosity)
                return

        # Perform rollback
        echo_normal(f"\n{click.style('Performing rollback...', fg='cyan', bold=True)}", verbosity)
        affected = manager.rollback_to_snapshot(snapshot_id)

        echo_normal(f"\n{click.style('✓ Rollback complete', fg='green', bold=True)}", verbosity)
        echo_normal(f"  Memories affected: {click.style(str(affected), fg='cyan')}", verbosity)
        if not force:
            echo_normal(f"  Backup saved as: {click.style(backup_snapshot.snapshot_id, fg='green')}", verbosity)

    except ValueError as e:
        echo_quiet(click.style(f"Error: {e}", fg="red"), verbosity)
        sys.exit(1)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to rollback: {e}", fg="red"), verbosity)
        sys.exit(1)
