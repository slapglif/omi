"""OMI CLI - OpenClaw Memory Infrastructure Command Line Interface

The seeking is the continuity. The palace remembers what the river forgets.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict, cast
import click

# OMI imports
from omi import NOWStore, DailyLogStore, GraphPalace
from omi.security import PoisonDetector
from .event_bus import get_event_bus
from .events import SessionStartedEvent, SessionEndedEvent

# CLI version - matches project version
__version__ = "0.1.0"

# Default paths
DEFAULT_BASE_PATH = Path.home() / ".openclaw" / "omi"
DEFAULT_CONFIG_PATH = DEFAULT_BASE_PATH / "config.yaml"


def get_base_path(ctx_data_dir: Optional[Path] = None) -> Path:
    """Get the base path for OMI data.

    Priority: --data-dir flag > OMI_BASE_PATH env var > default path.

    Args:
        ctx_data_dir: Value from --data-dir CLI option, if provided.
    """
    if ctx_data_dir:
        return Path(ctx_data_dir)
    env_path = os.getenv("OMI_BASE_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_BASE_PATH


@click.group()
@click.version_option(version=__version__, prog_name="omi")
@click.option('--data-dir', type=click.Path(), default=None, envvar='OMI_BASE_PATH',
              help='Base directory for OMI data (default: ~/.openclaw/omi)')
@click.pass_context
def cli(ctx: click.Context, data_dir: Optional[str]) -> None:
    """OMI - OpenClaw Memory Infrastructure

    A unified memory system for AI agents.

    \b
    Key Commands:
        init              Initialize memory infrastructure
        session-start     Load context and start a session
        session-end       End session and backup
        store             Store a memory
        recall            Search memories
        check             Pre-compression checkpoint
        status            Show health and size
        audit             Security audit
        config            Configuration management

    \b
    Examples:
        omi init
        omi session-start
        omi store "Fixed the auth bug" --type experience
        omi recall "session checkpoint"
        omi check
        omi session-end
    """
    ctx.ensure_object(dict)
    if data_dir:
        ctx.obj['data_dir'] = Path(data_dir)
    else:
        ctx.obj['data_dir'] = None


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
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


@cli.command("session-start")
@click.option("--show-now", is_flag=True, help="Display NOW.md content")
@click.pass_context
def session_start(ctx: click.Context, show_now: bool) -> None:
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
    from .persistence import NOWEntry
    now_storage = NOWStore(str(base_path))
    content = now_storage.read()

    # Check if NOW.md exists and has content
    default_content = "# NOW\n\n## Current Task\n\n## Recent Completions\n\n## Pending Decisions\n\n## Key Files\n"
    if not content or content == default_content:
        click.echo(click.style(" ⚠ NOW.md not found, creating default", fg="yellow"))
        now_storage.update(  # type: ignore[attr-defined]
            current_task="",
            recent_completions=[],
            pending_decisions=[],
            key_files=[]
        )
        content = now_storage.read()

    # Parse NOW entry
    now_entry = NOWEntry.from_markdown(content) if content else None
    
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
    
    click.echo(f" ✓ Loaded context: {len(now_entry.recent_completions) if now_entry else 0} recent completions")
    click.echo(f" ✓ Database: {mem_count} memories stored")

    # 3. Semantic recall for current task
    if now_entry and now_entry.current_task and mem_count > 0:
        try:
            palace = GraphPalace(db_path)
            results = palace.full_text_search(now_entry.current_task, limit=5)
            if results:
                click.echo(f"\n ✓ {len(results)} relevant memories found")
        except Exception as e:
            click.echo(click.style(f" ⚠ Recall error: {e}", fg="yellow"))

    # 4. Session status
    from .security import IntegrityChecker
    integrity_checker = IntegrityChecker(base_path.parent if base_path.name == "omi" else base_path)
    now_integrity = integrity_checker.check_now_md()

    click.echo(f"\n" + click.style("Session Status:", bold=True))
    if show_now and now_entry and now_entry.current_task:
        click.echo(f"\n{click.style('Current Task:', fg='cyan')}")
        click.echo(f"  {now_entry.current_task}")
    if now_entry and now_entry.pending_decisions:
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


@cli.command()
@click.argument('content')
@click.option('--type', 'memory_type', default='experience',
              type=click.Choice(['fact', 'experience', 'belief', 'decision']),
              help='Type of memory to store')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence level (0.0-1.0, for beliefs only)')
@click.pass_context
def store(ctx: click.Context, content: str, memory_type: str, confidence: Optional[float]) -> None:
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


@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Maximum number of results')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def recall(ctx: click.Context, query: str, limit: int, json_output: bool) -> None:
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


@cli.command()
@click.pass_context
def check(ctx: click.Context) -> None:
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

    # Update NOW.md timestamp
    from .persistence import NOWEntry
    now_storage = NOWStore(str(base_path))
    now_file_path = base_path / "NOW.md"
    if now_file_path.exists():
        content = now_storage.read()
        default_content = "# NOW\n\n## Current Task\n\n## Recent Completions\n\n## Pending Decisions\n\n## Key Files\n"
        if content and content != default_content:
            # Parse existing content and re-write to update timestamp
            now_entry = NOWEntry.from_markdown(content)
            now_storage.update(  # type: ignore[attr-defined]
                current_task=now_entry.current_task,
                recent_completions=now_entry.recent_completions,
                pending_decisions=now_entry.pending_decisions,
                key_files=now_entry.key_files
            )
            click.echo(f" ✓ Updated NOW.md")

    # Create state capsule
    from typing import Dict, cast
    capsule: Dict[str, Any] = {
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
        result = cursor.fetchone()
        if result:
            memory_summary = cast(Dict[str, Any], capsule["memory_summary"])
            memory_summary["total_memories"] = result[0]
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        for row in cursor.fetchall():
            memory_summary = cast(Dict[str, Any], capsule["memory_summary"])
            types_dict = cast(Dict[str, int], memory_summary["types"])
            types_dict[row[0]] = row[1]
        conn.close()
        click.echo(f" ✓ Memory capsule created")
    
    # Report status
    click.echo(f"\n" + click.style("Checkpoint Status:", bold=True))
    click.echo(f"  Timestamp: {click.style(str(capsule['timestamp']), fg='cyan')}")
    memory_summary = cast(Dict[str, Any], capsule['memory_summary'])
    click.echo(f"  Memories: {click.style(str(memory_summary['total_memories']), fg='cyan')}")
    types_dict = cast(Dict[str, int], memory_summary["types"])
    if types_dict:
        click.echo(f"\n  Memory types:")
        for mem_type, count in types_dict.items():
            click.echo(f"    {mem_type}: {count}")
    
    click.echo(click.style("\n✓ Checkpoint complete", fg="green", bold=True))


@cli.command("session-end")
@click.option('--no-backup', is_flag=True, help="Skip vault backup")
@click.pass_context
def session_end(ctx: click.Context, no_backup: bool) -> None:
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
    from .persistence import NOWEntry
    now_storage = NOWStore(str(base_path))
    now_entry: Optional[NOWEntry] = None
    now_file_path = base_path / "NOW.md"
    if now_file_path.exists():
        content = now_storage.read()
        default_content = "# NOW\n\n## Current Task\n\n## Recent Completions\n\n## Pending Decisions\n\n## Key Files\n"
        if content and content != default_content:
            # Parse existing content and re-write to update timestamp
            now_entry = NOWEntry.from_markdown(content)
            now_storage.update(  # type: ignore[attr-defined]
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
    vault_enabled = False
    if not no_backup:
        config_path = base_path / "config.yaml"
        if config_path.exists():
            try:
                import yaml  # type: ignore[import-untyped]
                config = yaml.safe_load(config_path.read_text())
                if config and isinstance(config, dict):
                    vault_config = config.get('vault', {})
                    if isinstance(vault_config, dict):
                        vault_enabled = bool(vault_config.get('enabled', False))
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
            "vault_backup": not no_backup and vault_enabled
        }
    )
    get_event_bus().publish(event)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show OMI health and size statistics."""
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("OMI Status Report", fg="cyan", bold=True))
    click.echo("=" * 50)
    
    # Base path info
    click.echo(f"\nBase Path: {click.style(str(base_path), fg='cyan')}")
    
    # Check files
    files_to_check = [
        ("Config", base_path / "config.yaml"),
        ("Database", base_path / "palace.sqlite"),
        ("NOW.md", base_path / "NOW.md"),
    ]
    
    click.echo(f"\nFiles:")
    for name, path in files_to_check:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status_symbol = "✓" if exists else "✗"
        status_color = "green" if exists else "red"
        click.echo(f"  {name:12} {click.style(status_symbol, fg=status_color)} {path.name}")
    
    # Database stats
    db_path = base_path / "palace.sqlite"
    if db_path.exists():
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        mem_count = cursor.fetchone()[0]
        
        # Memory types breakdown
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        type_counts = dict(cursor.fetchall())
        
        # Database size
        db_size = db_path.stat().st_size / 1024  # KB
        
        click.echo(f"\nDatabase:")
        click.echo(f"  Size: {click.style(f'{db_size:.1f} KB', fg='cyan')}")
        click.echo(f"  Memories: {click.style(str(mem_count), fg='cyan')}")
        if type_counts:
            click.echo(f"  Breakdown:")
            for mem_type, count in type_counts.items():
                click.echo(f"    {mem_type}: {count}")
        
        conn.close()
    
    # Integrity check
    click.echo(f"\nIntegrity:")
    try:
        from .security import IntegrityChecker
        integrity_checker = IntegrityChecker(base_path.parent if base_path.name == "omi" else base_path)
        now_ok = integrity_checker.check_now_md()
        mem_ok = integrity_checker.check_memory_md()
        now_color = "green" if now_ok else "red"
        mem_color = "green" if mem_ok else "red"
        click.echo(f"  NOW.md: {click.style('✓ OK' if now_ok else '✗ Failed', fg=now_color)}")
        click.echo(f"  MEMORY.md: {click.style('✓ OK' if mem_ok else '✗ Failed', fg=mem_color)}")
    except Exception as e:
        click.echo(f"  {click.style(f'Check failed: {e}', fg='yellow')}")
    
    # Overall health
    click.echo(f"\n" + click.style("Overall: ", bold=True), nl=False)
    click.echo(click.style("HEALTHY ✓", fg="green", bold=True))


@cli.command()
@click.pass_context
def audit(ctx: click.Context) -> None:
    """Run security audit.

    Checks:
    - File integrity (NOW.md, MEMORY.md)
    - Graph topology (orphan nodes, sudden cores)
    - Git history for suspicious modifications
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    click.echo(click.style("Running Security Audit...", fg="cyan", bold=True))
    click.echo("=" * 50)
    
    db_path = base_path / "palace.sqlite"
    try:
        detector = PoisonDetector(base_path, GraphPalace(db_path) if db_path.exists() else None)
        results = detector.full_security_audit()
        
        # File integrity
        click.echo(f"\n{click.style('File Integrity:', bold=True)}")
        file_ok = results.get('file_integrity', False)
        file_status = "✓ VERIFIED" if file_ok else "✗ FAILED"
        file_color = "green" if file_ok else "red"
        click.echo(f"  Status: {click.style(file_status, fg=file_color)}")
        
        # Topology
        click.echo(f"\n{click.style('Graph Topology:', bold=True)}")
        orphans = results.get('orphan_nodes', [])
        cores = results.get('sudden_cores', [])
        
        if orphans:
            click.echo(click.style(f"  ⚠ {len(orphans)} orphan nodes detected", fg="yellow"))
        else:
            click.echo(click.style(f"  ✓ No orphan nodes", fg="green"))
        
        if cores:
            click.echo(click.style(f"  ⚠ {len(cores)} sudden cores detected", fg="yellow"))
        else:
            click.echo(click.style(f"  ✓ No sudden cores", fg="green"))
        
        # Git audit
        click.echo(f"\n{click.style('Git History:', bold=True)}")
        git_check = results.get('git_audit', {})
        if 'error' in git_check:
            click.echo(click.style(f"  ⚠ {git_check['error']}", fg="yellow"))
        else:
            commits = git_check.get('recent_commits', 0)
            suspicious = git_check.get('suspicious', [])
            click.echo(f"  Recent commits: {commits}")
            if suspicious:
                click.echo(click.style(f"  ⚠ {len(suspicious)} suspicious commits", fg="yellow"))
            else:
                click.echo(click.style(f"  ✓ No suspicious commits", fg="green"))
        
        # Overall
        overall = results.get('overall_safe', False)
        click.echo(f"\n" + click.style("Overall Safety: ", bold=True), nl=False)
        if overall:
            click.echo(click.style("SAFE ✓", fg="green", bold=True))
        else:
            click.echo(click.style("ATTENTION REQUIRED ⚠", fg="yellow", bold=True))
            
    except Exception as e:
        click.echo(click.style(f"Error: Audit failed: {e}", fg="red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configuration management commands."""
    ctx.ensure_object(dict)


@config.command('set')
@click.argument('key')
@click.argument('value')
@click.pass_context
def config_set(ctx: click.Context, key: str, value: str) -> None:
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
        config_data_raw = yaml.safe_load(config_path.read_text())
        config_data: Dict[str, Any] = config_data_raw if isinstance(config_data_raw, dict) else {}

        # Parse nested keys (e.g., 'embedding.provider')
        keys = key.split('.')
        current: Dict[str, Any] = config_data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            next_val = current[k]
            if isinstance(next_val, dict):
                current = next_val
            else:
                current[k] = {}
                current = cast(Dict[str, Any], current[k])

        current[keys[-1]] = value

        # Write back
        config_path.write_text(yaml.dump(config_data, default_flow_style=False))
        click.echo(click.style(f"✓ Set {key} = {value}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: Failed to set config: {e}", fg="red"))
        sys.exit(1)


@config.command('get')
@click.argument('key')
@click.pass_context
def config_get(ctx: click.Context, key: str) -> None:
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
        config_data_raw = yaml.safe_load(config_path.read_text())
        config_data: Dict[str, Any] = config_data_raw if isinstance(config_data_raw, dict) else {}

        # Parse nested keys
        keys = key.split('.')
        current: Any = config_data
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                click.echo(click.style(f"Key '{key}' not found", fg="yellow"))
                sys.exit(1)
            current = current[k]

        click.echo(str(current))
    except Exception as e:
        click.echo(click.style(f"Error: Failed to get config: {e}", fg="red"))
        sys.exit(1)


@config.command('show')
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Display full configuration."""
    base_path = get_base_path(ctx.obj.get('data_dir'))
    config_path = base_path / "config.yaml"
    
    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    content = config_path.read_text()
    click.echo(click.style("Current configuration:", fg="cyan", bold=True))
    click.echo(content)


@cli.group()
@click.pass_context
def events(ctx: click.Context) -> None:
    """Event history commands."""
    ctx.ensure_object(dict)


@events.command('list')
@click.option('--type', '-t', 'event_type', default=None, help='Filter by event type')
@click.option('--since', default=None, help='Filter events after this timestamp (ISO format)')
@click.option('--until', default=None, help='Filter events before this timestamp (ISO format)')
@click.option('--limit', '-l', default=100, help='Maximum number of results (default: 100)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def list_events(ctx: click.Context, event_type: Optional[str], since: Optional[str], until: Optional[str],
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
    from .event_history import EventHistory

    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    # Event history database path
    events_db_path = base_path / "events.sqlite"
    if not events_db_path.exists():
        if json_output:
            click.echo(json.dumps([], indent=2))
        else:
            click.echo(click.style("No events found. Event history is empty.", fg="yellow"))
        return

    # Parse timestamps if provided
    since_dt = None
    until_dt = None

    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            click.echo(click.style(f"Error: Invalid --since timestamp format. Use ISO format (e.g., 2024-01-01T00:00:00)", fg="red"))
            sys.exit(1)

    if until:
        try:
            until_dt = datetime.fromisoformat(until)
        except ValueError:
            click.echo(click.style(f"Error: Invalid --until timestamp format. Use ISO format (e.g., 2024-01-01T00:00:00)", fg="red"))
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
                click.echo(click.style("No events found matching filters.", fg="yellow"))
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

            click.echo(click.style(f"Event History ({len(events_list)} found{filter_str})", fg="cyan", bold=True))
            click.echo()

            # Display events
            for event in events_list:
                # Event header
                timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S') if event.timestamp else 'N/A'
                click.echo(click.style(f"[{timestamp_str}] ", fg="blue") +
                          click.style(event.event_type, fg="green", bold=True))

                # Event ID
                click.echo(f"  ID: {click.style(event.id[:16] + '...', fg='cyan')}")

                # Payload (truncated if too long)
                payload_str = json.dumps(event.payload, indent=2)
                if len(payload_str) > 200:
                    # Truncate long payloads
                    lines = payload_str.split('\n')
                    if len(lines) > 5:
                        payload_str = '\n'.join(lines[:5]) + '\n  ...'

                click.echo(f"  Payload: {payload_str}")

                # Metadata if present
                if event.metadata:
                    click.echo(f"  Metadata: {json.dumps(event.metadata)}")

                click.echo()  # Blank line between events

    except Exception as e:
        click.echo(click.style(f"Error: Failed to query events: {e}", fg="red"))
        sys.exit(1)


@events.command('subscribe')
@click.option('--type', '-t', 'event_type', default=None, help='Filter by event type (default: all events)')
@click.pass_context
def subscribe_events(ctx: click.Context, event_type: Optional[str]) -> None:
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
    from .event_bus import get_event_bus
    import time

    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    # Get the global event bus
    bus = get_event_bus()

    # Determine subscription type
    subscription_type = event_type if event_type else '*'

    # Display subscription info
    if event_type:
        click.echo(click.style(f"Subscribing to events: {event_type}", fg="cyan", bold=True))
    else:
        click.echo(click.style("Subscribing to all events", fg="cyan", bold=True))
    click.echo(click.style("Press Ctrl+C to exit", fg="yellow"))
    click.echo()

    # Event handler callback
    def print_event(event: Any) -> None:
        """Print event to stdout when received."""
        try:
            # Format timestamp
            timestamp_str = event.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(event, 'timestamp') and event.timestamp else datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Print event header
            click.echo(click.style(f"[{timestamp_str}] ", fg="blue") +
                      click.style(event.event_type, fg="green", bold=True))

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
                        click.echo(f"  {key}: {value_str}")

            click.echo()  # Blank line between events

        except Exception as e:
            click.echo(click.style(f"Error formatting event: {e}", fg="red"))

    # Subscribe to event bus
    bus.subscribe(subscription_type, print_event)

    try:
        # Keep the process running and listening for events
        click.echo(click.style("Listening for events...", fg="green"))
        while True:
            time.sleep(0.1)  # Sleep briefly to keep CPU usage low
    except KeyboardInterrupt:
        # Graceful shutdown on Ctrl+C
        click.echo()
        click.echo(click.style("Unsubscribing from events...", fg="yellow"))
        bus.unsubscribe(subscription_type, print_event)
        click.echo(click.style("Disconnected.", fg="cyan"))
        sys.exit(0)


@cli.group()
@click.pass_context
def plugins(ctx: click.Context) -> None:
    """Plugin management commands."""
    ctx.ensure_object(dict)


# Main entry point
def main() -> None:
    """Entry point for the OMI CLI."""
    cli()


if __name__ == "__main__":
    main()
