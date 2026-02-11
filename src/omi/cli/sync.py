"""Cloud sync commands for OMI CLI."""
import os
import sys
from pathlib import Path
import click
import yaml

# Local imports
from .common import get_base_path


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

    click.echo("=== Cloud Sync Status ===\n")

    # Check if backup is configured
    if 'backup' not in config_data or not config_data['backup']:
        click.echo("Status: Disabled")
        click.echo("Cloud sync is not configured.")
        click.echo("\nTo enable, configure backup settings:")
        click.echo("  omi config set backup.backend s3")
        click.echo("  omi config set backup.bucket my-bucket")
        return

    backup_config = config_data['backup']
    backend = backup_config.get('backend', 'not set')

    click.echo(f"Status: Configured")
    click.echo(f"Backend: {backend}")

    if backend == 's3':
        bucket = backup_config.get('bucket', 'not set')
        region = backup_config.get('region', 'not set')
        click.echo(f"Bucket: {bucket}")
        click.echo(f"Region: {region}")
    elif backend == 'gcs':
        bucket = backup_config.get('bucket', 'not set')
        project = backup_config.get('project', 'not set')
        click.echo(f"Bucket: {bucket}")
        click.echo(f"Project: {project}")
    elif backend == 'azure':
        container = backup_config.get('container', 'not set')
        account = backup_config.get('account_name', 'not set')
        click.echo(f"Container: {container}")
        click.echo(f"Account: {account}")


@sync_group.command('push')
@click.option('--encrypt', is_flag=True, help='Encrypt backup before uploading')
@click.pass_context
def sync_push(ctx, encrypt):
    """Push local data to cloud storage."""
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

    click.echo("Pushing to cloud storage...")
    click.echo(f"Backend: {backend}")
    click.echo(f"Bucket: {bucket}")

    if encrypt:
        click.echo("Encryption: enabled")

    try:
        # Import moltvault for backup functionality
        from omi.moltvault import MoltVault

        # Create MoltVault instance
        vault = MoltVault(base_path=base_path, config=config_data)

        # Perform backup
        result = vault.backup()

        click.echo(f"✓ Backup completed successfully")
        click.echo(f"  Backup ID: {result.get('backup_id', 'N/A')}")

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

    click.echo("Pulling from cloud storage...")
    click.echo(f"Backend: {backend}")

    if backup_id:
        click.echo(f"Backup ID: {backup_id}")
    else:
        click.echo("Restoring latest backup...")

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

        click.echo(f"✓ Restore completed successfully")

    except ImportError as e:
        click.echo(f"Error: Required package not available: {e}", err=True)
        click.echo("Install with: pip install boto3 cryptography", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during restore: {e}", err=True)
        sys.exit(1)
