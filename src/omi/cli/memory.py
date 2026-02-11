"""Memory management commands for OMI CLI."""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import click

# OMI imports
from omi import NOWStore, GraphPalace

# Local CLI imports
from .common import get_base_path


@click.group()
def memory_group():
    """Memory management commands."""
    pass


@memory_group.command("store")
@click.argument('content')
@click.option('--type', 'memory_type', default='experience',
              type=click.Choice(['fact', 'experience', 'belief', 'decision']),
              help='Type of memory to store')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence level (0.0-1.0, for beliefs only)')
@click.pass_context
def store(ctx, content: str, memory_type: str, confidence: Optional[float]) -> None:
    """Store a memory in the Graph Palace.

    Args:
        content: The memory content to store
        --type: Memory type (fact|experience|belief|decision)
        --confidence: Confidence level for beliefs (0.0-1.0)

    Examples:
        omi store "Fixed the auth bug" --type experience
        omi store "Python has GIL limitations" --type fact
        omi store "This approach works better" --type belief --confidence 0.85
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    # Validate confidence for beliefs
    if confidence is not None:
        if memory_type != 'belief':
            click.echo(click.style("Warning: --confidence is typically used with --type belief", fg="yellow"))
        confidence = max(0.0, min(1.0, confidence))

    try:
        palace = GraphPalace(db_path)
        memory_id = palace.store_memory(
            content=content,
            memory_type=memory_type,
            confidence=confidence
        )
        click.echo(click.style("✓ Memory stored", fg="green", bold=True))
        click.echo(f"  ID: {click.style(memory_id[:16] + '...', fg='cyan')}")
        click.echo(f"  Type: {click.style(memory_type, fg='cyan')}")
        if confidence is not None:
            conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
            click.echo(f"  Confidence: {click.style(f'{confidence:.2f}', fg=conf_color)}")
    except Exception as e:
        click.echo(click.style(f"Error: Failed to store memory: {e}", fg="red"))
        sys.exit(1)


@memory_group.command("recall")
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def recall(ctx, query: str, limit: int, json_output: bool) -> None:
    """Search memories using semantic recall.

    Args:
        query: Search query text
        --limit: Maximum number of results (default: 10)
        --json: Output as JSON (for scripts)

    Examples:
        omi recall "session checkpoint"
        omi recall "auth bug fix" --limit 5
        omi recall "recent decisions" --json
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    try:
        palace = GraphPalace(db_path)
        # Use full_text_search for query strings
        results = palace.full_text_search(query, limit=limit)

        if json_output:
            output = []
            for mem in results:
                output.append({
                    'id': mem.id,
                    'content': mem.content,
                    'memory_type': mem.memory_type,
                    'confidence': mem.confidence,
                    'created_at': mem.created_at.isoformat() if mem.created_at else None
                })
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            click.echo(click.style(f"Search Results ({len(results)} found)", fg="cyan", bold=True))
            click.echo("=" * 60)

            for i, mem in enumerate(results, 1):
                mem_type = mem.memory_type
                content = mem.content
                if len(content) > 80:
                    content = content[:77] + "..."

                type_color = {
                    'fact': 'blue',
                    'experience': 'green',
                    'belief': 'yellow',
                    'decision': 'magenta'
                }.get(mem_type, 'white')

                click.echo(f"\n{i}. [{click.style(mem_type.upper(), fg=type_color)}]")
                click.echo(f"   {content}")
                if mem.created_at:
                    click.echo(f"   {click.style('─', fg='bright_black') * 50}")

            if not results:
                click.echo(click.style("No memories found. Try a different query.", fg="yellow"))
    except Exception as e:
        click.echo(click.style(f"Error: Failed to search memories: {e}", fg="red"))
        sys.exit(1)


@memory_group.command("check")
@click.pass_context
def check(ctx) -> None:
    """Create a pre-compression checkpoint.

    Performs:
    - Updates NOW.md with current state
    - Creates state capsule
    - Reports memory system status
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("Creating checkpoint...", fg="cyan", bold=True))

    # CLI version - needed for capsule
    __version__ = "0.1.0"

    # Update NOW.md timestamp
    now_storage = NOWStore(base_path)
    if now_storage.now_file.exists():
        content = now_storage.read()
        if content and content != now_storage._default_content():
            # Parse existing content and re-write to update timestamp
            from ..persistence import NOWEntry
            now_entry = NOWEntry.from_markdown(content)
            now_storage.update(
                current_task=now_entry.current_task,
                recent_completions=now_entry.recent_completions,
                pending_decisions=now_entry.pending_decisions,
                key_files=now_entry.key_files
            )
            click.echo(f" ✓ Updated NOW.md")

    # Create state capsule
    capsule = {
        "timestamp": datetime.now().isoformat(),
        "version": __version__,
        "now_md_hash": None,
        "memory_summary": {
            "total_memories": 0,
            "types": {}
        }
    }

    db_path = base_path / "palace.sqlite"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        capsule["memory_summary"]["total_memories"] = cursor.fetchone()[0]
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        for row in cursor.fetchall():
            capsule["memory_summary"]["types"][row[0]] = row[1]
        conn.close()
        click.echo(f" ✓ Memory capsule created")

    # Report status
    click.echo(f"\n" + click.style("Checkpoint Status:", bold=True))
    click.echo(f"  Timestamp: {click.style(capsule['timestamp'], fg='cyan')}")
    click.echo(f"  Memories: {click.style(str(capsule['memory_summary']['total_memories']), fg='cyan')}")
    if capsule["memory_summary"]["types"]:
        click.echo(f"\n  Memory types:")
        for mem_type, count in capsule["memory_summary"]["types"].items():
            click.echo(f"    {mem_type}: {count}")

    click.echo(click.style("\n✓ Checkpoint complete", fg="green", bold=True))
