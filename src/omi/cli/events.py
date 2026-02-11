"""Event history commands for OMI CLI."""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
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
def events_group():
    """Event history commands."""
    pass


@events_group.command('list')
@click.option('--type', '-t', 'event_type', default=None, help='Filter by event type')
@click.option('--since', default=None, help='Filter events after this timestamp (ISO format)')
@click.option('--until', default=None, help='Filter events before this timestamp (ISO format)')
@click.option('--limit', '-l', default=100, help='Maximum number of results (default: 100)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def list_events(ctx, event_type: Optional[str], since: Optional[str], until: Optional[str],
                limit: int, json_output: bool) -> None:
    """List events from history with filters.

    Args:
        --type: Filter by event type (e.g., 'memory.stored', 'session.started')
        --since: Filter events after this timestamp (ISO format)
        --until: Filter events before this timestamp (ISO format)
        --limit: Maximum number of results (default: 100)
        --json-output: Output as JSON (for scripts)

    Examples:
        omi events list
        omi events list --type memory.stored --limit 10
        omi events list --since 2024-01-01T00:00:00 --json-output
    """
    from omi.event_history import EventHistory

    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    # Event history database path
    events_db_path = base_path / "events.sqlite"
    if not events_db_path.exists():
        if json_output:
            click.echo(json.dumps([], indent=2))
        else:
            echo_normal(click.style("No events found. Event history is empty.", fg="yellow"), verbosity)
        return

    # Parse timestamps if provided
    since_dt = None
    until_dt = None

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            echo_quiet(click.style(f"Error: Invalid --since timestamp format. Use ISO format (e.g., 2024-01-01T00:00:00)", fg="red"), verbosity)
            sys.exit(1)

    if until:
        try:
            until_dt = datetime.fromisoformat(until)
        except ValueError:
            echo_quiet(click.style(f"Error: Invalid --until timestamp format. Use ISO format (e.g., 2024-01-01T00:00:00)", fg="red"), verbosity)
            sys.exit(1)

    try:
        history = EventHistory(events_db_path)
        events_list = history.query_events(
            event_type=event_type,
            since=since_dt,
            until=until_dt,
            limit=limit
        )

        if json_output:
            output = [event.to_dict() for event in events_list]
            click.echo(json.dumps(output, indent=2))
        else:
            if not events_list:
                echo_normal(click.style("No events found matching filters.", fg="yellow"), verbosity)
                return

            # Display header
            filter_info = []
            if event_type:
                filter_info.append(f"type={event_type}")
            if since:
                filter_info.append(f"since={since}")
            if until:
                filter_info.append(f"until={until}")
            filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

            echo_normal(click.style(f"Event History ({len(events_list)} found{filter_str})", fg="cyan", bold=True), verbosity)
            echo_normal("", verbosity)

            # Display events
            for event in events_list:
                # Event header
                timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S') if event.timestamp else 'N/A'
                echo_normal(click.style(f"[{timestamp_str}] ", fg="blue") +
                          click.style(event.event_type, fg="green", bold=True), verbosity)

                # Event ID
                echo_normal(f"  ID: {click.style(event.id[:16] + '...', fg='cyan')}", verbosity)

                # Payload (truncated if too long)
                payload_str = json.dumps(event.payload, indent=2)
                if len(payload_str) > 200:
                    # Truncate long payloads
                    lines = payload_str.split('\n')
                    if len(lines) > 5:
                        payload_str = '\n'.join(lines[:5]) + '\n  ...'

                echo_normal(f"  Payload: {payload_str}", verbosity)

                # Metadata if present
                if event.metadata:
                    echo_verbose(f"  Metadata: {json.dumps(event.metadata)}", verbosity)

                echo_normal("", verbosity)  # Blank line between events

    except Exception as e:
        echo_quiet(click.style(f"Error: Failed to query events: {e}", fg="red"), verbosity)
        sys.exit(1)


@events_group.command('subscribe')
@click.option('--type', '-t', 'event_type', default=None, help='Filter by event type (default: all events)')
@click.pass_context
def subscribe_events(ctx, event_type: Optional[str]) -> None:
    """Subscribe to live event stream.

    Connects to the EventBus and prints events in real-time as they occur.
    Press Ctrl+C to exit.

    Args:
        --type: Filter by event type (e.g., 'memory.stored', 'session.started')
                If not specified, subscribes to all events

    Examples:
        omi events subscribe
        omi events subscribe --type memory.stored
        omi events subscribe -t belief.contradiction_detected
    """
    from omi.event_bus import get_event_bus

    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    # Get the global event bus
    bus = get_event_bus()

    # Determine subscription type
    subscription_type = event_type if event_type else '*'

    # Display subscription info
    if event_type:
        echo_normal(click.style(f"Subscribing to events: {event_type}", fg="cyan", bold=True), verbosity)
    else:
        echo_normal(click.style("Subscribing to all events", fg="cyan", bold=True), verbosity)
    echo_normal(click.style("Press Ctrl+C to exit", fg="yellow"), verbosity)
    echo_normal("", verbosity)

    # Event handler callback
    def print_event(event):
        """Print event to stdout when received."""
        try:
            # Format timestamp
            timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(event, 'timestamp') and event.timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Print event header
            echo_normal(click.style(f"[{timestamp_str}] ", fg="blue") +
                      click.style(event.event_type, fg="green", bold=True), verbosity)

            # Print event details (convert to dict for pretty printing)
            if hasattr(event, 'to_dict'):
                event_dict = event.to_dict()
                # Remove redundant fields for cleaner output
                event_dict.pop('event_type', None)
                event_dict.pop('timestamp', None)

                # Print each field
                for key, value in event_dict.items():
                    if value is not None:  # Skip None values
                        if isinstance(value, (dict, list)):
                            value_str = json.dumps(value, indent=2)
                        else:
                            value_str = str(value)
                        echo_normal(f"  {key}: {value_str}", verbosity)

            echo_normal("", verbosity)  # Blank line between events

        except Exception as e:
            echo_quiet(click.style(f"Error formatting event: {e}", fg="red"), verbosity)

    # Subscribe to event bus
    bus.subscribe(subscription_type, print_event)

    try:
        # Keep the process running and listening for events
        echo_verbose(click.style("Listening for events...", fg="green"), verbosity)
        while True:
            time.sleep(0.1)  # Sleep briefly to keep CPU usage low
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        echo_normal("", verbosity)
        echo_normal(click.style("Unsubscribing from events...", fg="yellow"), verbosity)
        bus.unsubscribe(subscription_type, print_event)
        echo_normal(click.style("Disconnected.", fg="cyan"), verbosity)
        sys.exit(0)
