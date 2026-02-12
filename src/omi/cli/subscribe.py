"""Subscribe command for OMI CLI - topic-based event subscription."""
import sys
import json
import signal
import time
from threading import Event as ThreadEvent
from typing import Any, Optional
import click

# Local CLI imports
from .common import (
    get_base_path,
    VERBOSITY_NORMAL,
    echo_verbose,
    echo_normal,
    echo_quiet,
)


@click.group()
def subscribe_group():
    """Subscribe commands."""
    pass


@subscribe_group.command('subscribe', context_settings=dict(ignore_unknown_options=True))
@click.argument('topic')
@click.option('--timeout', type=int, default=0, help='Timeout in seconds (0 = forever)')
@click.option('--format', 'output_format', type=click.Choice(['json', 'pretty']), default='pretty',
              help='Output format for events')
@click.pass_context
def subscribe_command(ctx: click.Context, topic: str, timeout: int, output_format: str) -> None:
    """Subscribe to events on a topic.

    Listen for events published to the specified topic and display them
    as they arrive. Use '*' to subscribe to all events.

    \b
    Available event types:
        memory.stored                   Memory stored in Graph Palace
        memory.recalled                 Memories recalled via search
        memory.shared_stored           Memory shared across agents
        belief.updated                  Belief confidence updated
        belief.contradiction_detected   Contradiction detected
        belief.propagated              Belief propagated between agents
        session.started                Session started
        session.ended                  Session ended
        *                              All events (wildcard)

    \b
    Examples:
        omi subscribe memory.stored
        omi subscribe "belief.*" --format json
        omi subscribe "*" --timeout 60
    """
    from omi.event_bus import get_event_bus

    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    # Get the global event bus
    event_bus = get_event_bus()

    # Event to signal shutdown
    shutdown_event = ThreadEvent()
    event_count = 0

    # Define event handler
    def handle_event(event: Any) -> None:
        nonlocal event_count
        event_count += 1

        if output_format == 'json':
            # JSON format
            if hasattr(event, 'to_dict'):
                event_data = event.to_dict()
            else:
                event_data = {'event_type': event.event_type}
            click.echo(json.dumps(event_data))
        else:
            # Pretty format
            timestamp = event.timestamp.strftime('%H:%M:%S') if hasattr(event, 'timestamp') else ''
            event_type = click.style(event.event_type, fg='cyan', bold=True)
            echo_normal(f"[{timestamp}] {event_type}", verbosity)

            # Display event-specific details
            if hasattr(event, 'to_dict'):
                event_dict = event.to_dict()
                for key, value in event_dict.items():
                    if key not in ['event_type', 'timestamp', 'metadata']:
                        echo_normal(f"  {key}: {value}", verbosity)
            echo_normal("", verbosity)  # Blank line for readability

    # Signal handler for graceful shutdown
    def signal_handler(sig: int, frame: Any) -> None:
        echo_normal(click.style("\n\nShutdown requested...", fg="yellow"), verbosity)
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Subscribe to topic
    event_bus.subscribe(topic, handle_event)

    echo_normal(click.style(f"Subscribing to topic: {topic}", fg="cyan", bold=True), verbosity)
    if timeout > 0:
        echo_normal(f"Timeout: {timeout} seconds", verbosity)
    echo_normal("Waiting for events... (Ctrl+C to stop)\n", verbosity)

    try:
        # Wait for timeout or shutdown signal
        if timeout > 0:
            shutdown_event.wait(timeout=timeout)
        else:
            # Wait indefinitely
            while not shutdown_event.is_set():
                time.sleep(0.1)
    finally:
        # Cleanup
        event_bus.unsubscribe(topic, handle_event)
        echo_normal(click.style(f"\nReceived {event_count} event(s)", fg="green"), verbosity)
        echo_normal(click.style("Unsubscribed from topic", fg="cyan"), verbosity)
