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


# API Key Management Commands
@config_group.group('api-key')
def api_key_group():
    """API key management commands."""
    pass


@api_key_group.command('generate')
@click.option('--name', required=True, help='Name for the API key (must be unique)')
@click.option('--rate-limit', type=int, default=None, help='Requests per minute (default: from config or 100)')
@click.pass_context
def api_key_generate(ctx, name: str, rate_limit: int) -> None:
    """Generate a new API key.

    Args:
        --name: Human-readable name for the key (must be unique)
        --rate-limit: Requests per minute allowed (default: from config or 100)

    Examples:
        omi config api-key generate --name production-agent
        omi config api-key generate --name dev-bot --rate-limit 50
    """
    from omi.auth import APIKeyManager
    import yaml

    base_path = get_base_path(ctx.obj.get('data_dir'))
    verbosity = ctx.obj.get('verbosity', 1)

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    # Read default rate limit from config if not specified
    if rate_limit is None:
        config_path = base_path / "config.yaml"
        default_rate_limit = 100  # Fallback default

        if config_path.exists():
            try:
                config_data = yaml.safe_load(config_path.read_text()) or {}
                # Read from security.default_rate_limit
                default_rate_limit = config_data.get('security', {}).get('default_rate_limit', 100)
            except Exception:
                # If config reading fails, use hardcoded default
                default_rate_limit = 100

        rate_limit = default_rate_limit

    db_path = base_path / "palace.sqlite"

    try:
        manager = APIKeyManager(db_path)
        api_key = manager.generate_key(name, rate_limit=rate_limit)

        echo_normal(click.style("✓ API key generated successfully", fg="green"), verbosity)
        echo_quiet("", verbosity)
        echo_quiet(click.style(f"Name: {name}", fg="cyan"), verbosity)
        echo_quiet(click.style(f"Rate Limit: {rate_limit} requests/minute", fg="cyan"), verbosity)
        echo_quiet("", verbosity)
        echo_quiet(click.style("API Key (save this - it won't be shown again):", fg="yellow", bold=True), verbosity)
        echo_quiet(click.style(api_key, fg="green", bold=True), verbosity)
        echo_quiet("", verbosity)
        echo_quiet(click.style("Use this key in requests:", fg="cyan"), verbosity)
        echo_quiet(f"  Header: X-API-Key: {api_key}", verbosity)
        echo_quiet(f"  Query param: ?api_key={api_key}", verbosity)

    except ValueError as e:
        echo_quiet(click.style(f"Error: {e}", fg="red"), verbosity)
        sys.exit(1)
    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to generate API key: {e}", fg="red"), verbosity)
        sys.exit(1)


@api_key_group.command('revoke')
@click.option('--name', help='Name of the API key to revoke')
@click.option('--key', help='The API key value to revoke')
@click.pass_context
def api_key_revoke(ctx, name: str, key: str) -> None:
    """Revoke an API key by name or key value.

    Args:
        --name: Name of the key to revoke
        --key: The API key value to revoke

    Examples:
        omi config api-key revoke --name production-agent
        omi config api-key revoke --key abc123...
    """
    from omi.auth import APIKeyManager

    base_path = get_base_path(ctx.obj.get('data_dir'))
    verbosity = ctx.obj.get('verbosity', 1)

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    if name is None and key is None:
        echo_quiet(click.style("Error: Must provide either --name or --key", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"

    try:
        manager = APIKeyManager(db_path)
        revoked = manager.revoke_key(name=name, api_key=key)

        if revoked:
            identifier = name if name else f"key {key[:16]}..."
            echo_normal(click.style(f"✓ Revoked API key: {identifier}", fg="green"), verbosity)
        else:
            identifier = name if name else "with provided value"
            echo_quiet(click.style(f"Warning: No active API key found {identifier}", fg="yellow"), verbosity)
            sys.exit(1)

    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to revoke API key: {e}", fg="red"), verbosity)
        sys.exit(1)


@api_key_group.command('list')
@click.option('--include-revoked', is_flag=True, help='Include revoked keys in the list')
@click.pass_context
def api_key_list(ctx, include_revoked: bool) -> None:
    """List all API keys with metadata.

    Args:
        --include-revoked: Include revoked keys in the list (default: False)

    Examples:
        omi config api-key list
        omi config api-key list --include-revoked
    """
    from omi.auth import APIKeyManager

    base_path = get_base_path(ctx.obj.get('data_dir'))
    verbosity = ctx.obj.get('verbosity', 1)

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    db_path = base_path / "palace.sqlite"

    try:
        manager = APIKeyManager(db_path)
        keys = manager.list_keys(include_revoked=include_revoked)

        if not keys:
            echo_normal(click.style("No API keys found.", fg="yellow"), verbosity)
            return

        # Display header
        status_text = " (including revoked)" if include_revoked else ""
        echo_normal(click.style(f"API Keys ({len(keys)} found{status_text})", fg="cyan", bold=True), verbosity)
        echo_quiet("", verbosity)

        # Display each key
        for api_key in keys:
            # Key name (with revoked status)
            status = click.style(" [REVOKED]", fg="red") if api_key.revoked else ""
            echo_normal(click.style(f"• {api_key.name}", fg="green", bold=True) + status, verbosity)

            # ID
            echo_quiet(f"  ID: {api_key.id}", verbosity)

            # Rate limit
            echo_quiet(f"  Rate Limit: {api_key.rate_limit} requests/minute", verbosity)

            # Created
            created_str = api_key.created_at.strftime('%Y-%m-%d %H:%M:%S') if api_key.created_at else 'N/A'
            echo_quiet(f"  Created: {created_str}", verbosity)

            # Last used
            if api_key.last_used:
                last_used_str = api_key.last_used.strftime('%Y-%m-%d %H:%M:%S')
                echo_quiet(f"  Last Used: {last_used_str}", verbosity)
            else:
                echo_quiet(f"  Last Used: Never", verbosity)

            # Key hash (first 16 chars for reference)
            echo_quiet(f"  Key Hash: {api_key.key_hash[:16]}...", verbosity)

            echo_quiet("", verbosity)

    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to list API keys: {e}", fg="red"), verbosity)
        sys.exit(1)
