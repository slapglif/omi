"""Cloud sync commands for OMI CLI."""
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
    """Synchronization commands.

    Cloud sync: Push/pull to cloud storage (S3, GCS, Azure)
    Distributed sync: Multi-instance memory synchronization
    """
    pass


@sync_group.command('cloud-status')
@click.pass_context
def sync_cloud_status(ctx):
    """Show cloud sync status and configuration."""
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
        config_data = yaml.safe_load(f)

    echo_normal("=== Cloud Sync Status ===\n", verbosity)

    # Check if backup is configured
    if 'backup' not in config_data or not config_data['backup']:
        echo_normal("Status: Disabled", verbosity)
        echo_normal("Cloud sync is not configured.", verbosity)
        echo_normal("\nTo enable, configure backup settings:", verbosity)
        echo_normal("  omi config set backup.backend s3", verbosity)
        echo_normal("  omi config set backup.bucket my-bucket", verbosity)
        return

    backup_config = config_data['backup']
    backend = backup_config.get('backend', 'not set')

    echo_normal(f"Status: Configured", verbosity)
    echo_normal(f"Backend: {backend}", verbosity)

    if backend == 's3':
        bucket = backup_config.get('bucket', 'not set')
        region = backup_config.get('region', 'not set')
        echo_normal(f"Bucket: {bucket}", verbosity)
        echo_verbose(f"Region: {region}", verbosity)
    elif backend == 'gcs':
        bucket = backup_config.get('bucket', 'not set')
        project = backup_config.get('project', 'not set')
        echo_normal(f"Bucket: {bucket}", verbosity)
        echo_verbose(f"Project: {project}", verbosity)
    elif backend == 'azure':
        container = backup_config.get('container', 'not set')
        account = backup_config.get('account_name', 'not set')
        echo_normal(f"Container: {container}", verbosity)
        echo_verbose(f"Account: {account}", verbosity)


@sync_group.command('push')
@click.option('--encrypt', is_flag=True, help='Encrypt backup before uploading')
@click.pass_context
def sync_push(ctx, encrypt):
    """Push local data to cloud storage."""
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
        config_data = yaml.safe_load(f)

    # Check if backup is configured
    if 'backup' not in config_data or not config_data['backup']:
        click.echo("Error: Cloud sync not configured.", err=True)
        click.echo("Run 'omi sync status' for configuration instructions.", err=True)
        sys.exit(1)

    backup_config = config_data['backup']
    backend = backup_config.get('backend', '').lower()
    bucket = backup_config.get('bucket', '')

    echo_normal("Pushing to cloud storage...", verbosity)
    echo_normal(f"Backend: {backend}", verbosity)
    echo_normal(f"Bucket: {bucket}", verbosity)

    if encrypt:
        echo_verbose("Encryption: enabled", verbosity)

    try:
        # Import moltvault for backup functionality
        from omi.moltvault import MoltVault

        # Create MoltVault instance
        vault = MoltVault(base_path=base_path, config=config_data)

        # Perform backup
        result = vault.backup()

        echo_normal(f"✓ Backup completed successfully", verbosity)
        echo_verbose(f"  Backup ID: {result.get('backup_id', 'N/A')}", verbosity)

    except ImportError as e:
        click.echo(f"Error: Required package not available: {e}", err=True)
        click.echo(f"Backend '{backend}' requires additional dependencies.", err=True)
        click.echo("Install with: pip install boto3 cryptography", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during backup: {e}", err=True)
        sys.exit(1)


@sync_group.command('pull')
@click.option('--backup-id', help='Specific backup ID to restore')
@click.pass_context
def sync_pull(ctx, backup_id):
    """Pull data from cloud storage."""
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
        config_data = yaml.safe_load(f)

    # Check if backup is configured
    if 'backup' not in config_data or not config_data['backup']:
        click.echo("Error: Cloud sync not configured.", err=True)
        click.echo("Run 'omi sync status' for configuration instructions.", err=True)
        sys.exit(1)

    backup_config = config_data['backup']
    backend = backup_config.get('backend', '').lower()

    echo_normal("Pulling from cloud storage...", verbosity)
    echo_normal(f"Backend: {backend}", verbosity)

    if backup_id:
        echo_verbose(f"Backup ID: {backup_id}", verbosity)
    else:
        echo_verbose("Restoring latest backup...", verbosity)

    try:
        # Import moltvault for restore functionality
        from omi.moltvault import MoltVault

        # Create MoltVault instance
        vault = MoltVault(base_path=base_path, config=config_data)

        # Perform restore
        if backup_id:
            result = vault.restore(backup_id=backup_id)
        else:
            result = vault.restore()

        echo_normal(f"✓ Restore completed successfully", verbosity)

    except ImportError as e:
        click.echo(f"Error: Required package not available: {e}", err=True)
        click.echo(f"Backend '{backend}' requires additional dependencies.", err=True)
        click.echo("Install with: pip install boto3 cryptography", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during restore: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Distributed Multi-Instance Sync Commands
# ============================================================================

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
    """Show synchronization status (cloud and distributed).

    Displays both cloud sync configuration and distributed sync status
    for a complete picture of memory synchronization across instances.
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

    # ========== CLOUD SYNC STATUS (ORIGINAL BEHAVIOR) ==========
    echo_normal("=== Cloud Sync Status ===\n", verbosity)

    # Check if backup is configured
    if 'backup' not in config_data or not config_data['backup']:
        echo_normal("Status: Disabled", verbosity)
        echo_normal("Cloud sync is not configured.", verbosity)
        echo_normal("\nTo enable, configure backup settings:", verbosity)
        echo_normal("  omi config set backup.backend s3", verbosity)
        echo_normal("  omi config set backup.bucket my-bucket", verbosity)
    else:
        backup_config = config_data['backup']
        backend = backup_config.get('backend', 'not set')
        echo_normal(f"Status: Configured", verbosity)
        echo_normal(f"Backend: {backend}", verbosity)

        if backend == 's3':
            bucket = backup_config.get('bucket', 'not set')
            region = backup_config.get('region', 'not set')
            echo_normal(f"Bucket: {bucket}", verbosity)
            echo_verbose(f"Region: {region}", verbosity)
        elif backend == 'gcs':
            bucket = backup_config.get('bucket', 'not set')
            project = backup_config.get('project', 'not set')
            echo_normal(f"Bucket: {bucket}", verbosity)
            echo_verbose(f"Project: {project}", verbosity)
        elif backend == 'azure':
            container = backup_config.get('container', 'not set')
            account = backup_config.get('account_name', 'not set')
            echo_normal(f"Container: {container}", verbosity)
            echo_verbose(f"Account: {account}", verbosity)

    # ========== DISTRIBUTED SYNC STATUS (NEW FEATURE) ==========
    echo_normal("\n=== Distributed Sync Status ===\n", verbosity)

    # Check if distributed sync is configured
    if 'sync' not in config_data or not config_data['sync'].get('enabled', False):
        echo_normal("Status: Disabled", verbosity)
        echo_normal("Distributed sync is not configured.", verbosity)
        echo_normal("\nTo enable, run:", verbosity)
        echo_normal("  omi sync init --instance-id=<id> --topology=<type>", verbosity)
    else:
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
