"""OMI CLI - Serve Command

REST API server with optional dashboard UI.
"""
import sys
import click

# Local CLI imports
from .common import get_base_path

# Server imports
try:
    import uvicorn
except ImportError:
    uvicorn = None


@click.group()
@click.pass_context
def serve_group(ctx):
    """Server commands."""
    ctx.ensure_object(dict)


@serve_group.command()
@click.option('--dashboard/--no-dashboard', default=True,
              help='Enable dashboard UI (default: enabled)')
@click.option('--host', default='0.0.0.0',
              help='Host to bind to (default: 0.0.0.0)')
@click.option('--port', default=8420, type=int,
              help='Port to bind to (default: 8420)')
@click.option('--reload', is_flag=True, default=False,
              help='Enable auto-reload for development')
@click.pass_context
def serve(ctx, dashboard: bool, host: str, port: int, reload: bool) -> None:
    """Start the OMI REST API server with optional dashboard.

    Starts a FastAPI server that provides:
    - REST API endpoints at /api/v1/
    - SSE event streaming at /api/v1/events
    - Dashboard UI at /dashboard (if --dashboard enabled)

    Examples:
        omi serve
        omi serve --port 8421
        omi serve --no-dashboard
        omi serve --reload  # Development mode
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    # Check for uvicorn dependency
    if uvicorn is None:
        click.echo(click.style("Error: uvicorn is not installed.", fg="red"))
        click.echo("Install with: pip install 'omi[server]' or pip install uvicorn")
        sys.exit(1)

    # Print startup message
    click.echo(click.style("Starting OMI REST API Server...", fg="cyan", bold=True))
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Dashboard: {'enabled' if dashboard else 'disabled'}")
    click.echo(f"  Reload: {'enabled' if reload else 'disabled'}")
    click.echo()

    if dashboard:
        dashboard_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/dashboard"
        click.echo(click.style(f"Dashboard available at {dashboard_url}", fg="green", bold=True))
        click.echo()

    # Start uvicorn server
    try:
        uvicorn.run(
            "omi.rest_api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("Server stopped.", fg="yellow"))
    except Exception as e:
        click.echo()
        click.echo(click.style(f"Error starting server: {e}", fg="red"))
        sys.exit(1)
