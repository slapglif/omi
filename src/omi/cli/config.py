"""Configuration management commands for OMI CLI."""
import sys
from pathlib import Path
import click

# Local CLI imports
from .common import get_base_path


@click.group()
def config_group():
    """Configuration management commands."""
    pass


@config_group.command('set')
@click.argument('key')
@click.argument('value')
@click.pass_context
def config_set(ctx, key: str, value: str) -> None:
    """Set a configuration value.

    Args:
        key: Configuration key (e.g., 'embedding.provider')
        value: Value to set

    Examples:
        omi config set embedding.provider ollama
        omi config set embedding.model nomic-embed-text
        omi config set vault.enabled true
        omi config set events.webhook https://example.com/hook
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    config_path = base_path / "config.yaml"

    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    try:
        import yaml
        config_data = yaml.safe_load(config_path.read_text()) or {}

        # Parse nested keys (e.g., 'embedding.provider')
        keys = key.split('.')
        current = config_data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

        # Write back
        config_path.write_text(yaml.dump(config_data, default_flow_style=False))
        click.echo(click.style(f"✓ Set {key} = {value}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: Failed to set config: {e}", fg="red"))
        sys.exit(1)


@config_group.command('get')
@click.argument('key')
@click.pass_context
def config_get(ctx, key: str) -> None:
    """Get a configuration value.

    Args:
        key: Configuration key (e.g., 'embedding.provider')

    Examples:
        omi config get embedding.provider
        omi config get vault.enabled
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    config_path = base_path / "config.yaml"

    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    try:
        import yaml
        config_data = yaml.safe_load(config_path.read_text()) or {}

        # Parse nested keys
        keys = key.split('.')
        current = config_data
        for k in keys:
            if k not in current:
                click.echo(click.style(f"Key '{key}' not found", fg="yellow"))
                sys.exit(1)
            current = current[k]

        click.echo(current)
    except Exception as e:
        click.echo(click.style(f"Error: Failed to get config: {e}", fg="red"))
        sys.exit(1)


@config_group.command('show')
@click.pass_context
def config_show(ctx) -> None:
    """Display full configuration."""
    base_path = get_base_path(ctx.obj.get('data_dir'))
    config_path = base_path / "config.yaml"

    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    content = config_path.read_text()
    click.echo(click.style("Current configuration:", fg="cyan", bold=True))
    click.echo(content)
