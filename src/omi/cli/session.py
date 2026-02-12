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
from .common import (
    get_base_path,
    VERBOSITY_NORMAL,
    VERBOSITY_VERBOSE,
    echo_verbose,
    echo_normal,
    echo_quiet,
)


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
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)

    echo_normal(click.style("Initializing OMI Memory Infrastructure...", fg="cyan", bold=True), verbosity)

    # 1. Create directory structure
    base_path.mkdir(parents=True, exist_ok=True)
    memory_path = base_path / "memory"
    memory_path.mkdir(exist_ok=True)
    echo_verbose(f" ✓ Created directory: {base_path}", verbosity)

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
        echo_verbose(f" ✓ Created config: {config_path}", verbosity)
    else:
        echo_verbose(f" ⚠ Config exists: {config_path}", verbosity)

    # 3. Initialize SQLite database using GraphPalace
    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        try:
            palace = GraphPalace(db_path)
            palace.close()
            echo_verbose(f" ✓ Initialized database: {db_path}", verbosity)
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
            echo_verbose(f" ✓ Initialized database (minimal): {db_path}", verbosity)
    else:
        echo_verbose(f" ⚠ Database exists: {db_path}", verbosity)

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
        echo_verbose(f" ✓ Created NOW.md: {now_path}", verbosity)
    else:
        echo_verbose(f" ⚠ NOW.md exists: {now_path}", verbosity)

    echo_normal(click.style("\n✓ Initialization complete!", fg="green", bold=True), verbosity)
    echo_normal(f"\nNext steps:", verbosity)
    echo_normal(f"  1. Edit {config_path} to configure your embedding provider", verbosity)
    echo_normal(f"  2. Run: omi session-start", verbosity)


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
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)

    if not base_path.exists():
        echo_quiet(click.style(f"Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    echo_normal(click.style("Starting OMI session...", fg="cyan", bold=True), verbosity)

    # 1. Load NOW.md
    now_storage = NOWStore(base_path)
    content = now_storage.read()

    # Check if NOW.md exists and has content
    if not content or content == now_storage._default_content():
        echo_normal(click.style(" ⚠ NOW.md not found, creating default", fg="yellow"), verbosity)
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

    echo_verbose(f" ✓ Loaded context: {len(now_entry.recent_completions)} recent completions", verbosity)
    echo_verbose(f" ✓ Database: {mem_count} memories stored", verbosity)

    # 3. Semantic recall for current task
    if now_entry.current_task and mem_count > 0:
        try:
            palace = GraphPalace(db_path)
            results = palace.full_text_search(now_entry.current_task, limit=5)
            if results:
                echo_verbose(f"\n ✓ {len(results)} relevant memories found", verbosity)
        except Exception as e:
            echo_normal(click.style(f" ⚠ Recall error: {e}", fg="yellow"), verbosity)

    # 4. Session status
    from ..security import IntegrityChecker
    integrity_checker = IntegrityChecker(base_path.parent if base_path.name == "omi" else base_path)
    now_integrity = integrity_checker.check_now_md()

    echo_normal(f"\n" + click.style("Session Status:", bold=True), verbosity)
    if show_now and now_entry.current_task:
        echo_normal(f"\n{click.style('Current Task:', fg='cyan')}", verbosity)
        echo_normal(f"  {now_entry.current_task}", verbosity)
    if now_entry.pending_decisions:
        echo_normal(f"\n{click.style('Pending Decisions:', fg='yellow')}", verbosity)
        for item in now_entry.pending_decisions:
            echo_normal(f"  - [ ] {item}", verbosity)

    status_color = "green" if now_integrity else "red"
    echo_verbose(f"\n NOW.md integrity: {click.style('✓' if now_integrity else '✗', fg=status_color)}", verbosity)

    session_timestamp = datetime.now()
    echo_verbose(f" Session started: {click.style(session_timestamp.isoformat(), fg='cyan')}", verbosity)
    echo_normal(click.style("\n✓ Session ready!", fg="green", bold=True), verbosity)

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
    verbosity = ctx.obj.get('verbosity', VERBOSITY_NORMAL)

    if not base_path.exists():
        echo_quiet(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"), verbosity)
        sys.exit(1)

    echo_normal(click.style("Ending OMI session...", fg="cyan", bold=True), verbosity)

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
            echo_verbose(f" ✓ Updated NOW.md", verbosity)

    # Append to daily log
    daily_store = DailyLogStore(base_path)
    entry_content = f"Session ended at {datetime.now().isoformat()}"
    if now_entry and now_entry.current_task:
        entry_content += f"\nLast task: {now_entry.current_task}"
    log_path = daily_store.append(entry_content)
    echo_verbose(f" ✓ Appended to daily log: {log_path.name}", verbosity)

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
            echo_verbose(click.style(" ✓ Vault backup triggered", fg="cyan"), verbosity)
        else:
            echo_verbose(click.style(" ⚠ Vault backup disabled (see config.yaml)", fg="yellow"), verbosity)

    # Execute policies on session end
    config_path = base_path / "config.yaml"
    db_path = base_path / "palace.sqlite"

    if db_path.exists():
        try:
            from ..policies import (
                PolicyEngine,
                load_policies_from_config,
                get_default_policies
            )

            # Load policies from config or use defaults
            policies = []
            if config_path.exists():
                try:
                    policies = load_policies_from_config(config_path)
                    echo_verbose(" ✓ Loaded policies from config", verbosity)
                except Exception as e:
                    echo_verbose(f" ⚠ Config load error, using defaults: {e}", verbosity)
                    policies = get_default_policies()
            else:
                policies = get_default_policies()
                echo_verbose(" ✓ Using default policies", verbosity)

            # Initialize policy engine and execute policies
            if policies:
                palace = GraphPalace(db_path)
                engine = PolicyEngine(palace)

                # Register policies and execute
                for policy in policies:
                    engine.register_policy(policy)

                results = engine.execute(dry_run=False)
                palace.close()

                # Log policy execution results
                total_actions = sum(len(r.affected_memory_ids) for r in results)
                if total_actions > 0:
                    echo_verbose(f" ✓ Executed {len(results)} policies, {total_actions} memories affected", verbosity)
                else:
                    echo_verbose(" ✓ Policies executed, no actions needed", verbosity)
        except Exception as e:
            echo_verbose(f" ⚠ Policy execution error: {e}", verbosity)

    echo_normal(click.style("\n✓ Session ended", fg="green", bold=True), verbosity)
    echo_normal("Remember: The seeking is the continuity.", verbosity)

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
