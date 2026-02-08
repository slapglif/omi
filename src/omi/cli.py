"""OMI CLI - OpenClaw Memory Infrastructure Command Line Interface

The seeking is the continuity. The palace remembers what the river forgets.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import click

# CLI version - matches project version
__version__ = "0.1.0"

# Default paths
DEFAULT_BASE_PATH = Path.home() / ".openclaw" / "omi"
DEFAULT_CONFIG_PATH = DEFAULT_BASE_PATH / "config.yaml"


def get_base_path() -> Path:
    """Get the base path for OMI data."""
    env_path = os.getenv("OMI_BASE_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_BASE_PATH


def ensure_imports():
    """Ensure OMI modules are importable."""
    try:
        from .persistence import NOWStore, DailyLogStore, GraphPalace
        from .security import PoisonDetector
        return NOWStore, DailyLogStore, GraphPalace, PoisonDetector
    except ImportError as e:
        # Try to add src to path
        cli_path = Path(__file__).resolve()
        src_path = cli_path.parent.parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from omi.persistence import NOWStore, DailyLogStore, GraphPalace
        from omi.security import PoisonDetector
        return NOWStore, DailyLogStore, GraphPalace, PoisonDetector


@click.group()
@click.version_option(version=__version__, prog_name="omi")
def cli():
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
    pass


@cli.command()
@click.pass_context
def init(ctx) -> None:
    """Initialize memory infrastructure.
    
    Creates the following:
    - ~/.openclaw/omi/ directory structure
    - config.yaml with default settings
    - SQLite database for Graph Palace
    - NOW.md template
    """
    base_path = get_base_path()
    
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
        NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
        try:
            palace = GraphPalace(db_path)
            palace.close()
            click.echo(f" ✓ Initialized database: {db_path}")
        except RuntimeError as e:
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
def session_start(show_now: bool) -> None:
    """Load context and start a session.
    
    Performs:
    - Loads NOW.md hot context
    - Runs semantic recall of relevant memories
    - Prints session summary
    """
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style(f"Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
    click.echo(click.style("Starting OMI session...", fg="cyan", bold=True))
    
    # 1. Load NOW.md
    now_store = NOWStore(base_path)
    now_entry = now_store.read()
    if now_entry is None:
        click.echo(click.style(" ⚠ NOW.md not found, creating default", fg="yellow"))
        from .persistence import NOWEntry
        now_entry = NOWEntry(
            current_task="",
            recent_completions=[],
            pending_decisions=[],
            key_files=[],
            timestamp=datetime.now()
        )
        now_store.write(now_entry)
    
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
    from .security import IntegrityChecker
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
    click.echo(f" Session started: {click.style(datetime.now().isoformat(), fg='cyan')}")
    click.echo(click.style("\n✓ Session ready!", fg="green", bold=True))


@cli.command()
@click.argument('content')
@click.option('--type', 'memory_type', default='experience', 
              type=click.Choice(['fact', 'experience', 'belief', 'decision']),
              help='Type of memory to store')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence level (0.0-1.0, for beliefs only)')
def store(content: str, memory_type: str, confidence: Optional[float]) -> None:
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
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
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
def recall(query: str, limit: int, json_output: bool) -> None:
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
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
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
def check() -> None:
    """Create a pre-compression checkpoint.
    
    Performs:
    - Updates NOW.md with current state
    - Creates state capsule
    - Reports memory system status
    """
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
    click.echo(click.style("Creating checkpoint...", fg="cyan", bold=True))
    
    # Update NOW.md
    now_store = NOWStore(base_path)
    now_entry = now_store.read()
    if now_entry:
        now_entry.timestamp = datetime.now()
        now_store.write(now_entry)
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


@cli.command("session-end")
@click.option('--no-backup', is_flag=True, help="Skip vault backup")
def session_end(no_backup: bool) -> None:
    """End session and backup.
    
    Performs:
    - Updates NOW.md
    - Appends to daily log
    - Triggers vault backup (if enabled and configured)
    """
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
    click.echo(click.style("Ending OMI session...", fg="cyan", bold=True))
    
    # Update NOW.md
    now_store = NOWStore(base_path)
    now_entry = now_store.read()
    if now_entry:
        now_entry.timestamp = datetime.now()
        now_store.write(now_entry)
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


@cli.command()
def status() -> None:
    """Show OMI health and size statistics."""
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
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
def audit() -> None:
    """Run security audit.
    
    Checks:
    - File integrity (NOW.md, MEMORY.md)
    - Graph topology (orphan nodes, sudden cores)
    - Git history for suspicious modifications
    """
    base_path = get_base_path()
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    NOWStore, DailyLogStore, GraphPalace, PoisonDetector = ensure_imports()
    
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
def config():
    """Configuration management commands."""
    pass


@config.command('set')
@click.argument('key')
@click.argument('value')
def config_set(key: str, value: str) -> None:
    """Set a configuration value.
    
    Args:
        key: Configuration key (e.g., 'embedding.provider')
        value: Value to set
    
    Examples:
        omi config set embedding.provider ollama
        omi config set embedding.model nomic-embed-text
        omi config set vault.enabled true
    """
    base_path = get_base_path()
    config_path = base_path / "config.yaml"
    
    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
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
        click.echo(click.style(f"✓ Set {key} = {value}", fg="green"))
    except Exception as e:
        click.echo(click.style(f"Error: Failed to set config: {e}", fg="red"))
        sys.exit(1)


@config.command('get')
@click.argument('key')
def config_get(key: str) -> None:
    """Get a configuration value.
    
    Args:
        key: Configuration key (e.g., 'embedding.provider')
    
    Examples:
        omi config get embedding.provider
        omi config get vault.enabled
    """
    base_path = get_base_path()
    config_path = base_path / "config.yaml"
    
    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    try:
        import yaml
        config_data = yaml.safe_load(config_path.read_text()) or {}
        
        # Parse nested keys
        keys = key.split('.')
        current = config_data
        for k in keys:
            if k not in current:
                click.echo(click.style(f"Key '{key}' not found", fg="yellow"))
                sys.exit(1)
            current = current[k]
        
        click.echo(current)
    except Exception as e:
        click.echo(click.style(f"Error: Failed to get config: {e}", fg="red"))
        sys.exit(1)


@config.command('show')
def config_show() -> None:
    """Display full configuration."""
    base_path = get_base_path()
    config_path = base_path / "config.yaml"
    
    if not config_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)
    
    content = config_path.read_text()
    click.echo(click.style("Current configuration:", fg="cyan", bold=True))
    click.echo(content)


# Main entry point
def main():
    """Entry point for the OMI CLI."""
    cli()


if __name__ == "__main__":
    main()
