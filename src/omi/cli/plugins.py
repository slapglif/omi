"""Plugin management CLI commands"""
import sys
import click
from ..plugins import PluginRegistry, PLUGIN_GROUPS


@click.group()
@click.pass_context
def plugins_group(ctx: click.Context) -> None:
    """Plugin management commands."""
    ctx.ensure_object(dict)


@plugins_group.command(name='list')
@click.pass_context
def list_plugins(ctx: click.Context) -> None:
    """List all installed plugins.

    Shows plugins grouped by type (Embedding Providers, Storage Backends, Event Handlers)
    with their name, version, status, and interface version.
    """
    try:
        # Initialize plugin registry and discover all plugins
        registry = PluginRegistry()
        registry.discover_all()

        # Get all plugins grouped by type
        all_plugins = registry.list_all()

        # Define friendly group names for display
        group_display_names = {
            'omi.embedding_providers': 'Embedding Providers',
            'omi.storage_backends': 'Storage Backends',
            'omi.event_handlers': 'Event Handlers',
        }

        # Track if any plugins were found
        total_plugins = sum(len(plugins) for plugins in all_plugins.values())

        # Display plugins grouped by type
        for group, display_name in group_display_names.items():
            plugins_in_group = all_plugins.get(group, [])

            # Always show the group header even if empty
            click.echo(click.style(f"\n{display_name}:", fg="cyan", bold=True))

            if not plugins_in_group:
                click.echo(f"  {click.style('(none)', fg='yellow')}")
            else:
                for plugin in plugins_in_group:
                    # Format plugin info
                    name = plugin.name
                    status = click.style("✓ loaded", fg="green") if plugin.loaded else click.style("✗ not loaded", fg="yellow")

                    # Build plugin line
                    plugin_line = f"  • {name}"

                    # Add version if available
                    if plugin.version:
                        plugin_line += f" (v{plugin.version})"

                    # Add interface version if available
                    if plugin.interface_version:
                        plugin_line += f" [interface: {plugin.interface_version}]"

                    plugin_line += f" - {status}"

                    click.echo(plugin_line)

                    # Show module info
                    click.echo(f"    Module: {plugin.module}")

                    # Show error if failed to load
                    if plugin.error:
                        click.echo(f"    {click.style('Error:', fg='red')} {plugin.error}")

        # Summary
        if total_plugins == 0:
            click.echo(click.style("\nNo plugins installed.", fg="yellow"))
            click.echo("To install plugins, use pip to install packages that provide OMI plugin entry points.")
        else:
            click.echo(click.style(f"\nTotal: {total_plugins} plugin(s) found", fg="cyan"))

    except Exception as e:
        click.echo(click.style(f"Error discovering plugins: {e}", fg="red"), err=True)
        sys.exit(1)
