"""Distributed multi-instance sync commands for OMI CLI."""
import os
import sys
from pathlib import Path
import click
import yaml

# Local imports
from .common import get_base_path, echo_verbose, echo_normal, echo_quiet


@click.group()
@click.pass_context
def sync_group(ctx):
    """Multi-instance synchronization commands.

    Sync memory stores across multiple OMI instances for high-availability
    and multi-region deployments.
    """
    pass


@sync_group.command('init')
@click.option('--instance-id', required=True, help='Unique identifier for this instance')
@click.option('--topology', type=click.Choice(['leader-follower', 'multi-leader']),
              default='leader-follower', help='Sync topology type')
@click.option('--role', type=click.Choice(['leader', 'follower']), default='leader',
              help='Instance role (for leader-follower topology)')
@click.pass_context
def sync_init(ctx, instance_id, topology, role):
    """Initialize distributed sync for this instance.

    Sets up the necessary configuration for multi-instance memory synchronization.
    Must be run before starting sync operations.

    Examples:
        omi sync init --instance-id=us-east-1 --topology=leader-follower --role=leader
        omi sync init --instance-id=us-west-2 --topology=multi-leader
    """
    verbosity = ctx.obj.get('verbosity', 1)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    # Check if OMI is initialized
    if not base_path.exists():
        click.echo("Error: OMI not initialized. Run 'omi init' first.", err=True)
        sys.exit(1)

    config_path = base_path / "config.yaml"
    if not config_path.exists():
        click.echo("Error: Configuration file not found.", err=True)
        sys.exit(1)

    echo_normal(f"Initializing distributed sync for instance '{instance_id}'...", verbosity)

    # Load existing config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    # Add sync configuration
    if 'sync' not in config_data:
        config_data['sync'] = {}

    config_data['sync']['enabled'] = True
    config_data['sync']['instance_id'] = instance_id
    config_data['sync']['topology'] = topology
    config_data['sync']['role'] = role
    config_data['sync']['conflict_resolution'] = 'last-writer-wins'  # Default strategy

    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    echo_normal(f"✓ Sync initialized", verbosity)
    echo_verbose(f"  Instance ID: {instance_id}", verbosity)
    echo_verbose(f"  Topology: {topology}", verbosity)
    if topology == 'leader-follower':
        echo_verbose(f"  Role: {role}", verbosity)
    echo_normal("\nNext steps:", verbosity)
    echo_normal("  1. Configure other instances with their instance IDs", verbosity)
    echo_normal("  2. Start sync daemon: omi sync start", verbosity)


@sync_group.command('status')
@click.pass_context
def sync_status(ctx):
    """Show distributed sync status.

    Displays sync configuration, connected instances, lag metrics,
    and partition status.
    """
    verbosity = ctx.obj.get('verbosity', 1)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    # Check if OMI is initialized
    if not base_path.exists():
        click.echo("Error: OMI not initialized. Run 'omi init' first.", err=True)
        sys.exit(1)

    config_path = base_path / "config.yaml"
    if not config_path.exists():
        click.echo("Error: Configuration file not found.", err=True)
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    echo_normal("=== Distributed Sync Status ===\n", verbosity)

    # Check if sync is configured
    if 'sync' not in config_data or not config_data['sync'].get('enabled', False):
        echo_normal("Status: Disabled", verbosity)
        echo_normal("Distributed sync is not configured.", verbosity)
        echo_normal("\nTo enable, run:", verbosity)
        echo_normal("  omi sync init --instance-id=<id> --topology=<type>", verbosity)
        return

    sync_config = config_data['sync']
    instance_id = sync_config.get('instance_id', 'not set')
    topology = sync_config.get('topology', 'not set')
    role = sync_config.get('role', 'N/A')

    echo_normal(f"Status: Configured", verbosity)
    echo_normal(f"Instance ID: {instance_id}", verbosity)
    echo_normal(f"Topology: {topology}", verbosity)

    if topology == 'leader-follower':
        echo_normal(f"Role: {role}", verbosity)

    # Try to get runtime status from SyncManager
    try:
        from omi.sync.sync_manager import SyncManager

        sm = SyncManager(base_path, instance_id)
        sync_state = sm.get_sync_state()

        echo_normal(f"\nRuntime Status: {sync_state.state.value}", verbosity)
        echo_verbose(f"Last sync: {sync_state.last_sync_time or 'Never'}", verbosity)

        # Show connected instances
        topology_mgr = sm.topology
        instances = topology_mgr.get_all_instances()

        if instances:
            echo_normal(f"\nConnected Instances ({len(instances)}):", verbosity)
            for inst in instances:
                status_icon = "✓" if inst.is_healthy else "✗"
                echo_normal(f"  {status_icon} {inst.instance_id} ({inst.role.value})", verbosity)
                echo_verbose(f"     Last seen: {inst.last_heartbeat}", verbosity)
        else:
            echo_verbose("\nNo other instances connected", verbosity)

        # Show partition status
        if hasattr(sm, 'partition_handler'):
            partition_status = sm.partition_handler.get_partition_status()
            if partition_status.get('in_partition', False):
                echo_normal(f"\n⚠ Network partition detected", verbosity)
                echo_normal(f"  Partition started: {partition_status.get('partition_start')}", verbosity)
                echo_verbose(f"  Affected instances: {len(partition_status.get('affected_instances', []))}", verbosity)

    except ImportError:
        echo_verbose("\nRuntime status: Unavailable (sync daemon not running)", verbosity)
    except Exception as e:
        echo_verbose(f"\nRuntime status: Error - {e}", verbosity)


@sync_group.command('start')
@click.option('--daemon', '-d', is_flag=True, help='Run sync in background as daemon')
@click.pass_context
def sync_start(ctx, daemon):
    """Start distributed sync daemon.

    Begins synchronizing memory stores with other configured instances.
    Runs in foreground by default; use --daemon to run in background.

    Examples:
        omi sync start
        omi sync start --daemon
    """
    verbosity = ctx.obj.get('verbosity', 1)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    # Check if OMI is initialized
    if not base_path.exists():
        click.echo("Error: OMI not initialized. Run 'omi init' first.", err=True)
        sys.exit(1)

    config_path = base_path / "config.yaml"
    if not config_path.exists():
        click.echo("Error: Configuration file not found.", err=True)
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    # Check if sync is configured
    if 'sync' not in config_data or not config_data['sync'].get('enabled', False):
        click.echo("Error: Distributed sync not configured.", err=True)
        click.echo("Run 'omi sync init' first.", err=True)
        sys.exit(1)

    sync_config = config_data['sync']
    instance_id = sync_config.get('instance_id')

    if not instance_id:
        click.echo("Error: Instance ID not configured.", err=True)
        sys.exit(1)

    echo_normal(f"Starting sync daemon for instance '{instance_id}'...", verbosity)

    try:
        from omi.sync.sync_manager import SyncManager
        from omi.event_bus import get_event_bus

        sm = SyncManager(base_path, instance_id)
        event_bus = get_event_bus()

        # Start incremental sync via event bus
        sm.start_incremental_sync(event_bus)

        echo_normal(f"✓ Sync daemon started", verbosity)
        echo_verbose(f"  Instance ID: {instance_id}", verbosity)
        echo_verbose(f"  Topology: {sync_config.get('topology')}", verbosity)

        if daemon:
            echo_normal("Running in background...", verbosity)
            # TODO: Implement daemonization
            echo_verbose("Note: Daemon mode not fully implemented yet", verbosity)
        else:
            echo_normal("\nPress Ctrl+C to stop sync", verbosity)
            try:
                # Keep running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                echo_normal("\nStopping sync daemon...", verbosity)
                sm.stop_incremental_sync(event_bus)
                echo_normal("✓ Sync daemon stopped", verbosity)

    except ImportError as e:
        click.echo(f"Error: Required sync module not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting sync daemon: {e}", err=True)
        sys.exit(1)


@sync_group.command('stop')
@click.pass_context
def sync_stop(ctx):
    """Stop distributed sync daemon.

    Stops the sync daemon and disconnects from other instances.
    """
    verbosity = ctx.obj.get('verbosity', 1)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    # Check if OMI is initialized
    if not base_path.exists():
        click.echo("Error: OMI not initialized. Run 'omi init' first.", err=True)
        sys.exit(1)

    config_path = base_path / "config.yaml"
    if not config_path.exists():
        click.echo("Error: Configuration file not found.", err=True)
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    # Check if sync is configured
    if 'sync' not in config_data or not config_data['sync'].get('enabled', False):
        click.echo("Error: Distributed sync not configured.", err=True)
        sys.exit(1)

    sync_config = config_data['sync']
    instance_id = sync_config.get('instance_id')

    echo_normal(f"Stopping sync daemon for instance '{instance_id}'...", verbosity)

    try:
        from omi.sync.sync_manager import SyncManager
        from omi.event_bus import get_event_bus

        sm = SyncManager(base_path, instance_id)
        event_bus = get_event_bus()

        # Stop incremental sync
        sm.stop_incremental_sync(event_bus)

        echo_normal(f"✓ Sync daemon stopped", verbosity)

    except ImportError as e:
        click.echo(f"Error: Required sync module not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error stopping sync daemon: {e}", err=True)
        sys.exit(1)


@sync_group.command('reconcile')
@click.option('--instance-id', help='Reconcile with specific instance (default: all)')
@click.option('--strategy', type=click.Choice(['last-writer-wins', 'merge', 'manual']),
              help='Conflict resolution strategy')
@click.pass_context
def sync_reconcile(ctx, instance_id, strategy):
    """Manually trigger reconciliation.

    Reconciles memory stores after a network partition or to resolve
    pending conflicts. By default, reconciles with all known instances.

    Examples:
        omi sync reconcile
        omi sync reconcile --instance-id=us-west-2
        omi sync reconcile --strategy=merge
    """
    verbosity = ctx.obj.get('verbosity', 1)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    # Check if OMI is initialized
    if not base_path.exists():
        click.echo("Error: OMI not initialized. Run 'omi init' first.", err=True)
        sys.exit(1)

    config_path = base_path / "config.yaml"
    if not config_path.exists():
        click.echo("Error: Configuration file not found.", err=True)
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f) or {}

    # Check if sync is configured
    if 'sync' not in config_data or not config_data['sync'].get('enabled', False):
        click.echo("Error: Distributed sync not configured.", err=True)
        click.echo("Run 'omi sync init' first.", err=True)
        sys.exit(1)

    sync_config = config_data['sync']
    local_instance_id = sync_config.get('instance_id')

    if instance_id:
        echo_normal(f"Reconciling with instance '{instance_id}'...", verbosity)
    else:
        echo_normal(f"Reconciling with all instances...", verbosity)

    try:
        from omi.sync.sync_manager import SyncManager
        from omi.sync.conflict_resolver import ConflictStrategy

        sm = SyncManager(base_path, local_instance_id)

        # Determine conflict resolution strategy
        if strategy:
            strategy_map = {
                'last-writer-wins': ConflictStrategy.LAST_WRITER_WINS,
                'merge': ConflictStrategy.MERGE,
                'manual': ConflictStrategy.MANUAL_QUEUE
            }
            conflict_strategy = strategy_map[strategy]
        else:
            # Use configured strategy or default
            default_strategy = sync_config.get('conflict_resolution', 'last-writer-wins')
            strategy_map = {
                'last-writer-wins': ConflictStrategy.LAST_WRITER_WINS,
                'merge': ConflictStrategy.MERGE,
                'manual': ConflictStrategy.MANUAL_QUEUE
            }
            conflict_strategy = strategy_map.get(default_strategy, ConflictStrategy.LAST_WRITER_WINS)

        echo_verbose(f"Using conflict resolution strategy: {conflict_strategy.value}", verbosity)

        # Get partitions that need reconciliation
        partition_handler = sm.partition_handler
        unreconciled = partition_handler.get_unreconciled_partitions()

        if not unreconciled:
            echo_normal("No partitions need reconciliation", verbosity)
            return

        echo_normal(f"Found {len(unreconciled)} partition(s) to reconcile", verbosity)

        # Reconcile each partition
        for partition_event in unreconciled:
            if instance_id and instance_id not in partition_event.affected_instances:
                continue

            echo_verbose(f"\nReconciling partition: {partition_event.partition_id}", verbosity)
            echo_verbose(f"  Affected instances: {', '.join(partition_event.affected_instances)}", verbosity)

            result = sm.reconcile_partition(
                partition_event.partition_id,
                conflict_strategy
            )

            if result.get('success', False):
                conflicts = result.get('conflicts_resolved', 0)
                synced = result.get('memories_synced', 0)
                echo_normal(f"✓ Reconciliation complete", verbosity)
                echo_verbose(f"  Memories synced: {synced}", verbosity)
                echo_verbose(f"  Conflicts resolved: {conflicts}", verbosity)
            else:
                error = result.get('error', 'Unknown error')
                click.echo(f"✗ Reconciliation failed: {error}", err=True)

        echo_normal("\n✓ All reconciliations complete", verbosity)

    except ImportError as e:
        click.echo(f"Error: Required sync module not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during reconciliation: {e}", err=True)
        sys.exit(1)
