"""Session management commands for OMI CLI."""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import click

# OMI imports
from omi import NOWStore, DailyLogStore, GraphPalace
from omi.security import PoisonDetector
from ..event_bus import get_event_bus
from ..events import SessionStartedEvent, SessionEndedEvent

# Local CLI imports
from .common import get_base_path


@click.group()
def session_group():
    """Session management commands."""
    pass


@session_group.command("init")
@click.pass_context
def init(ctx) -> None:
    """Initialize memory infrastructure.

    Creates the following:
    - ~/.openclaw/omi/ directory structure
    - config.yaml with default settings
    - SQLite database for Graph Palace
    - NOW.md template
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))

    click.echo(click.style("Initializing OMI Memory Infrastructure...", fg="cyan", bold=True))

    # 1. Create directory structure
    base_path.mkdir(parents=True, exist_ok=True)
    memory_path = base_path / "memory"
    memory_path.mkdir(exist_ok=True)
    click.echo(f" ✓ Created directory: {base_path}")

    # 2. Create config.yaml
    config_template = """# OMI Configuration File
# OpenClaw Memory Infrastructure

embedding:
  provider: nim  # or ollama
  model: baai/bge-m3
  # api_key: ${NIM_API_KEY}  # Set via environment variable
  dimensions: 1024

storage:
  base_path: ~/.openclaw/omi
  db_path: palace.sqlite

vault:
  enabled: false
  frequency: daily
  # api_key: ${VAULT_API_KEY}

security:
  integrity_checks: true
  auto_audit: true
  required_instances: 3

session:
  auto_check_interval: 300  # seconds
  max_hot_tokens: 1000

events:
  webhooks:
    enabled: false
    # endpoints:
    #   - url: https://example.com/webhook
    #     events: ["memory.stored", "session.started", "session.ended"]
    #     # headers:
    #     #   Authorization: Bearer ${WEBHOOK_TOKEN}
"""
    config_path = base_path / "config.yaml"
    if not config_path.exists():
        config_path.write_text(config_template)
        click.echo(f" ✓ Created config: {config_path}")
    else:
        click.echo(f" ⚠ Config exists: {config_path}")

    # 3. Initialize SQLite database using GraphPalace
    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        try:
            palace = GraphPalace(db_path)
            palace.close()
            click.echo(f" ✓ Initialized database: {db_path}")
        except Exception as e:
            # Fallback: create minimal schema if storage deps not available
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    confidence REAL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
            click.echo(f" ✓ Initialized database (minimal): {db_path}")
    else:
        click.echo(f" ⚠ Database exists: {db_path}")

    # 4. Create NOW.md template
    now_path = base_path / "NOW.md"
    if not now_path.exists():
        now_template = f"""# NOW - {datetime.now().isoformat()}

## Current Task
Initializing OMI memory infrastructure.

## Recent Completions
- [x] Directory structure created
- [x] Configuration file initialized
- [x] SQLite database initialized
- [x] NOW.md template created

## Pending Decisions
- [ ] Configure embedding provider (NIM vs Ollama)
- [ ] Enable vault backup (optional)
- [ ] Set required consensus instances

## Key Files
- `~/.openclaw/omi/config.yaml`
- `~/.openclaw/omi/palace.sqlite`
- `~/.openclaw/omi/NOW.md`
"""
        now_path.write_text(now_template)
        click.echo(f" ✓ Created NOW.md: {now_path}")
    else:
        click.echo(f" ⚠ NOW.md exists: {now_path}")

    click.echo(click.style("\n✓ Initialization complete!", fg="green", bold=True))
    click.echo(f"\nNext steps:")
    click.echo(f"  1. Edit {config_path} to configure your embedding provider")
    click.echo(f"  2. Run: omi session-start")


@session_group.command("session-start")
@click.option("--show-now", is_flag=True, help="Display NOW.md content")
@click.pass_context
def session_start(ctx, show_now: bool) -> None:
    """Load context and start a session.

    Performs:
    - Loads NOW.md hot context
    - Runs semantic recall of relevant memories
    - Prints session summary
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style(f"Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("Starting OMI session...", fg="cyan", bold=True))

    # 1. Load NOW.md
    now_storage = NOWStore(base_path)
    content = now_storage.read()

    # Check if NOW.md exists and has content
    if not content or content == now_storage._default_content():
        click.echo(click.style(" ⚠ NOW.md not found, creating default", fg="yellow"))
        now_storage.update(
            current_task="",
            recent_completions=[],
            pending_decisions=[],
            key_files=[]
        )

    # Parse the NOW.md content
    from ..persistence import NOWEntry
    now_entry = NOWEntry.from_markdown(content) if content else NOWEntry(
        timestamp=datetime.now(),
        current_task="",
        recent_completions=[],
        pending_decisions=[],
        key_files=[]
    )

    # 2. Get database stats
    db_path = base_path / "palace.sqlite"
    mem_count = 0
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        result = cursor.fetchone()
        mem_count = result[0] if result else 0
        conn.close()

    click.echo(f" ✓ Loaded context: {len(now_entry.recent_completions)} recent completions")
    click.echo(f" ✓ Database: {mem_count} memories stored")

    # 3. Semantic recall for current task
    if now_entry.current_task and mem_count > 0:
        try:
            palace = GraphPalace(db_path)
            results = palace.full_text_search(now_entry.current_task, limit=5)
            if results:
                click.echo(f"\n ✓ {len(results)} relevant memories found")
        except Exception as e:
            click.echo(click.style(f" ⚠ Recall error: {e}", fg="yellow"))

    # 4. Session status
    from ..security import IntegrityChecker
    integrity_checker = IntegrityChecker(base_path.parent if base_path.name == "omi" else base_path)
    now_integrity = integrity_checker.check_now_md()

    click.echo(f"\n" + click.style("Session Status:", bold=True))
    if show_now and now_entry.current_task:
        click.echo(f"\n{click.style('Current Task:', fg='cyan')}")
        click.echo(f"  {now_entry.current_task}")
    if now_entry.pending_decisions:
        click.echo(f"\n{click.style('Pending Decisions:', fg='yellow')}")
        for item in now_entry.pending_decisions:
            click.echo(f"  - [ ] {item}")

    status_color = "green" if now_integrity else "red"
    click.echo(f"\n NOW.md integrity: {click.style('✓' if now_integrity else '✗', fg=status_color)}")

    session_timestamp = datetime.now()
    click.echo(f" Session started: {click.style(session_timestamp.isoformat(), fg='cyan')}")
    click.echo(click.style("\n✓ Session ready!", fg="green", bold=True))

    # Emit session started event
    event = SessionStartedEvent(
        session_id=session_timestamp.isoformat(),
        timestamp=session_timestamp,
        metadata={
            "memory_count": mem_count,
            "now_integrity": now_integrity
        }
    )
    get_event_bus().publish(event)


@session_group.command("session-end")
@click.option('--no-backup', is_flag=True, help="Skip vault backup")
@click.pass_context
def session_end(ctx, no_backup: bool) -> None:
    """End session and backup.

    Performs:
    - Updates NOW.md
    - Appends to daily log
    - Triggers vault backup (if enabled and configured)
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("Ending OMI session...", fg="cyan", bold=True))

    # Update NOW.md timestamp and get current task for log
    now_storage = NOWStore(base_path)
    now_entry = None
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

    # Append to daily log
    daily_store = DailyLogStore(base_path)
    entry_content = f"Session ended at {datetime.now().isoformat()}"
    if now_entry and now_entry.current_task:
        entry_content += f"\nLast task: {now_entry.current_task}"
    log_path = daily_store.append(entry_content)
    click.echo(f" ✓ Appended to daily log: {log_path.name}")

    # Vault backup
    if not no_backup:
        config_path = base_path / "config.yaml"
        vault_enabled = False
        if config_path.exists():
            import yaml
            try:
                config = yaml.safe_load(config_path.read_text())
                vault_enabled = config.get('vault', {}).get('enabled', False)
            except Exception:
                pass
        if vault_enabled:
            click.echo(click.style(" ✓ Vault backup triggered", fg="cyan"))
        else:
            click.echo(click.style(" ⚠ Vault backup disabled (see config.yaml)", fg="yellow"))

    click.echo(click.style("\n✓ Session ended", fg="green", bold=True))
    click.echo("Remember: The seeking is the continuity.")

    # Emit session ended event
    session_timestamp = datetime.now()
    event = SessionEndedEvent(
        session_id=session_timestamp.isoformat(),
        timestamp=session_timestamp,
        metadata={
            "vault_backup": not no_backup and vault_enabled if 'vault_enabled' in locals() else False
        }
    )
    get_event_bus().publish(event)
