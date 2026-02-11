"""Configuration management commands for OMI CLI."""
import sys
from pathlib import Path
import click

# Local CLI imports
from .common import get_base_path, echo_quiet, echo_normal


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
    verbosity = ctx.obj.get('verbosity', 1)

    if not config_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
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
        echo_normal(click.style(f"✓ Set {key} = {value}", fg="green"), verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to set config: {e}", fg="red"), verbosity)
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
    verbosity = ctx.obj.get('verbosity', 1)

    if not config_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    try:
        import yaml
        config_data = yaml.safe_load(config_path.read_text()) or {}

        # Parse nested keys
        keys = key.split('.')
        current = config_data
        for k in keys:
            if k not in current:
                echo_quiet(click.style(f"Key '{key}' not found", fg="yellow"), verbosity)
                sys.exit(1)
            current = current[k]

        echo_quiet(current, verbosity)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to get config: {e}", fg="red"), verbosity)
        sys.exit(1)


@config_group.command('show')
@click.pass_context
def config_show(ctx) -> None:
    """Display full configuration."""
    base_path = get_base_path(ctx.obj.get('data_dir'))
    config_path = base_path / "config.yaml"
    verbosity = ctx.obj.get('verbosity', 1)

    if not config_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    content = config_path.read_text()
    echo_normal(click.style("Current configuration:", fg="cyan", bold=True), verbosity)
    echo_quiet(content, verbosity)
