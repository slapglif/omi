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
    """Cloud synchronization commands.

    Sync your OMI data to and from cloud storage (S3, GCS, Azure).
    """
    pass


@sync_group.command('status')
@click.pass_context
def sync_status(ctx):
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
    echo_verbose(f"Backend: {backend}", verbosity)
    echo_verbose(f"Bucket: {bucket}", verbosity)

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
    echo_verbose(f"Backend: {backend}", verbosity)

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
        click.echo("Install with: pip install boto3 cryptography", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during restore: {e}", err=True)
        sys.exit(1)
