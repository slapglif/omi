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
from omi.belief import BeliefNetwork, ContradictionDetector, Evidence
from .event_bus import get_event_bus
from .events import SessionStartedEvent, SessionEndedEvent

# CLI version - matches project version
__version__ = "0.2.0"

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


def highlight_terms(text: str, query: str) -> str:
    """Highlight search terms in text using click styling.

    Args:
        text: The text to highlight terms in
        query: The search query containing terms to highlight

    Returns:
        Text with highlighted terms using ANSI color codes
    """
    if not query or not text:
        return text

    # Split query into individual terms
    terms = query.lower().split()

    # Build result by finding and highlighting each term
    result = text
    for term in terms:
        if not term:
            continue

        # Find all occurrences (case-insensitive) and replace with highlighted version
        import re
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        result = pattern.sub(
            lambda m: click.style(m.group(0), fg='yellow', bold=True),
            result
        )

    return result


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
        delete            Delete a memory
        check             Pre-compression checkpoint
        status            Show health and size
        audit             Security audit
        serve             Start REST API server
        config            Configuration management

    \b
    Examples:
        omi init
        omi session-start
        omi store "Fixed the auth bug" --type experience
        omi recall "session checkpoint"
        omi delete abc123def456...
        omi check
        omi serve --port 8420
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

server:
  host: 0.0.0.0
  port: 8420
  # api_key: ${OMI_API_KEY}  # Set via environment variable for REST API authentication
  cors:
    # origins: "*"  # Allow all origins (default), or specify comma-separated list
    # origins: "http://localhost:3000,https://example.com"
    # Set via OMI_CORS_ORIGINS environment variable

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

compression:
  provider: anthropic  # or openai
  # api_key: ${ANTHROPIC_API_KEY}  # Set via environment variable
  age_threshold_days: 30  # Compress memories older than N days
  batch_size: 8  # Number of memories to process at once
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
@click.option('--offset', default=0, help='Number of results to skip (for pagination)')
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.option('--type', 'memory_type', default=None,
              type=click.Choice(['fact', 'experience', 'belief', 'decision']),
              help='Filter by memory type')
@click.pass_context
def recall(ctx: click.Context, query: str, limit: int, offset: int, json_output: bool, memory_type: Optional[str]) -> None:
    """Search memories using semantic recall.

    Args:
        query: Search query text
        --limit: Maximum number of results (default: 10)
        --offset: Number of results to skip for pagination (default: 0)
        --type: Filter by memory type (fact|experience|belief|decision)
        --json: Output as JSON (for scripts)

    Examples:
        omi recall "session checkpoint"
        omi recall "auth bug fix" --limit 5
        omi recall "recent decisions" --json
        omi recall "bugs" --type experience
        omi recall "auth" --limit 10 --offset 10
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

        # Fetch enough results to cover offset + limit for cursor-based pagination
        # Use a generous fetch limit to ensure we have enough results after filtering
        fetch_limit = max(100, offset + limit + 50)
        all_results = palace.full_text_search(query, limit=fetch_limit)

        # Filter by type if specified
        if memory_type:
            all_results = [r for r in all_results if r.memory_type == memory_type]

        # Get total count before pagination
        total_count = len(all_results)

        # Apply cursor-based pagination (offset + limit)
        paginated_results = all_results[offset:offset + limit]
        has_more = (offset + limit) < total_count

        if json_output:
            output = {
                'memories': [],
                'total_count': total_count,
                'offset': offset,
                'limit': limit,
                'has_more': has_more
            }
            for mem in paginated_results:
                output['memories'].append({
                    'id': mem.id,
                    'content': mem.content,
                    'memory_type': mem.memory_type,
                    'confidence': mem.confidence,
                    'created_at': mem.created_at.isoformat() if mem.created_at else None
                })
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            # Display pagination info header
            page_num = (offset // limit) + 1 if limit > 0 else 1
            total_pages = (total_count + limit - 1) // limit if limit > 0 else 1
            click.echo(click.style(f"Search Results (Page {page_num}/{total_pages}, {len(paginated_results)} of {total_count} total)", fg="cyan", bold=True))
            click.echo("=" * 60)

            for i, mem in enumerate(paginated_results, 1):
                mem_type = mem.memory_type
                content = mem.content
                if len(content) > 80:
                    content = content[:77] + "..."

                # Apply term highlighting to content
                highlighted_content = highlight_terms(content, query)

                type_color = {
                    'fact': 'blue',
                    'experience': 'green',
                    'belief': 'yellow',
                    'decision': 'magenta'
                }.get(mem_type, 'white')

                # Show global index (offset + i) for pagination clarity
                global_index = offset + i
                click.echo(f"\n{global_index}. [{click.style(mem_type.upper(), fg=type_color)}]")
                click.echo(f"   {highlighted_content}")
                if mem.created_at:
                    click.echo(f"   {click.style('─', fg='bright_black') * 50}")

            if not paginated_results:
                if offset > 0:
                    click.echo(click.style(f"No results on page {page_num}. Try a lower offset.", fg="yellow"))
                else:
                    click.echo(click.style("No memories found. Try a different query.", fg="yellow"))
            elif has_more:
                # Show pagination hint
                next_offset = offset + limit
                click.echo(f"\n{click.style('More results available.', fg='cyan')} Use --offset {next_offset} to see the next page.")
    except Exception as e:
        click.echo(click.style(f"Error: Failed to search memories: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option('--operation', required=True, type=click.Choice(['recall', 'store']), help='Operation to debug')
@click.argument('content')
@click.option('--limit', '-l', default=10, help='Maximum number of results (for recall)')
@click.option('--type', 'memory_type', default='experience',
              type=click.Choice(['fact', 'experience', 'belief', 'decision']),
              help='Type of memory to store (for store)')
@click.option('--confidence', '-c', type=float, default=None,
              help='Confidence level (0.0-1.0, for store with beliefs)')
@click.pass_context
def debug(ctx, operation: str, content: str, limit: int, memory_type: str, confidence: Optional[float]) -> None:
    """Debug mode with step-by-step operation output.

    Args:
        --operation: Operation to debug (recall|store)
        content: Query text (for recall) or memory content (for store)
        --limit: Maximum number of results (for recall, default: 10)
        --type: Memory type (for store, default: experience)
        --confidence: Confidence level (for store with beliefs)

    Examples:
        omi debug --operation recall "test query"
        omi debug --operation recall "auth bug" --limit 5
        omi debug --operation store "Fixed the auth bug" --type experience
        omi debug --operation store "Python has GIL" --type fact
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    if operation == 'recall':
        _debug_recall(content, db_path, limit)
    elif operation == 'store':
        _debug_store(content, memory_type, confidence, db_path)


def _debug_recall(query: str, db_path: Path, limit: int) -> None:
    """Debug recall operation with step-by-step output."""
    from omi.embeddings import NIMEmbedder

    click.echo(click.style("=== DEBUG: Recall Operation ===", fg="cyan", bold=True))
    click.echo()

    # Step 1: Generate embedding
    click.echo(click.style("Step 1: Generating embedding for query", fg="yellow", bold=True))
    click.echo(f"Query: {click.style(query, fg='white', bold=True)}")

    try:
        embedder = NIMEmbedder()
        embedding = embedder.embed(query)
        click.echo(click.style(f"✓ Generated {len(embedding)}-dimensional embedding", fg="green"))
        click.echo(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        click.echo(click.style(f"✗ Failed to generate embedding: {e}", fg="red"))
        click.echo(click.style("Tip: Set NIM_API_KEY environment variable", fg="yellow"))
        sys.exit(1)

    click.echo()

    # Step 2: Search for candidates
    click.echo(click.style("Step 2: Searching for candidate memories", fg="yellow", bold=True))

    try:
        palace = GraphPalace(db_path)
        results = palace.recall(embedding, limit=limit, min_relevance=0.0)
        click.echo(click.style(f"✓ Found {len(results)} candidate memories", fg="green"))
    except Exception as e:
        click.echo(click.style(f"✗ Search failed: {e}", fg="red"))
        sys.exit(1)

    click.echo()

    # Step 3: Scoring
    click.echo(click.style("Step 3: Scoring results (relevance + recency)", fg="yellow", bold=True))
    click.echo(f"Formula: {click.style('final_score = (relevance * 0.7) + (recency * 0.3)', fg='bright_black')}")

    if not results:
        click.echo(click.style("No memories found matching the query.", fg="yellow"))
        return

    click.echo()

    # Step 4: Display results
    click.echo(click.style("Step 4: Final Results", fg="yellow", bold=True))
    click.echo("=" * 80)

    for i, (mem, score) in enumerate(results, 1):
        mem_type = mem.memory_type
        content = mem.content
        if len(content) > 100:
            content = content[:97] + "..."

        type_color = {
            'fact': 'blue',
            'experience': 'green',
            'belief': 'yellow',
            'decision': 'magenta'
        }.get(mem_type, 'white')

        click.echo(f"\n{i}. [{click.style(mem_type.upper(), fg=type_color)}] Score: {click.style(f'{score:.4f}', fg='cyan', bold=True)}")
        click.echo(f"   {content}")
        click.echo(f"   ID: {click.style(mem.id[:8], fg='bright_black')}... | "
                   f"Created: {click.style(mem.created_at.strftime('%Y-%m-%d %H:%M') if mem.created_at else 'N/A', fg='bright_black')}")
        click.echo(f"   {click.style('─', fg='bright_black') * 78}")


def _debug_store(content: str, memory_type: str, confidence: Optional[float], db_path: Path) -> None:
    """Debug store operation with step-by-step output."""
    import hashlib
    import time
    from omi.embeddings import NIMEmbedder

    click.echo(click.style("=== DEBUG: Store Operation ===", fg="cyan", bold=True))
    click.echo()

    # Step 1: Input validation
    click.echo(click.style("Step 1: Input Validation", fg="yellow", bold=True))
    click.echo(f"Content: {click.style(content, fg='white', bold=True)}")
    click.echo(f"Type: {click.style(memory_type, fg='cyan')}")
    if confidence is not None:
        if memory_type != 'belief':
            click.echo(click.style("⚠ Warning: --confidence is typically used with --type belief", fg="yellow"))
        confidence = max(0.0, min(1.0, confidence))
        click.echo(f"Confidence: {click.style(f'{confidence:.2f}', fg='cyan')}")
    click.echo(click.style("✓ Input validated", fg="green"))

    # Get database size before
    db_size_before = db_path.stat().st_size if db_path.exists() else 0

    click.echo()

    # Step 2: Content hash generation
    click.echo(click.style("Step 2: Content Hash Generation", fg="yellow", bold=True))
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    click.echo(f"SHA-256: {click.style(content_hash[:16] + '...', fg='bright_black')}")
    click.echo(click.style("✓ Content hash generated", fg="green"))

    click.echo()

    # Step 3: Embedding generation
    click.echo(click.style("Step 3: Generating Embedding", fg="yellow", bold=True))

    try:
        start_time = time.time()
        embedder = NIMEmbedder()
        embedding = embedder.embed(content)
        elapsed = time.time() - start_time
        click.echo(click.style(f"✓ Generated {len(embedding)}-dimensional embedding [{elapsed:.2f}s]", fg="green"))
        click.echo(f"  First 5 values: {embedding[:5]}")
    except Exception as e:
        click.echo(click.style(f"✗ Failed to generate embedding: {e}", fg="red"))
        click.echo(click.style("Tip: Set NIM_API_KEY environment variable", fg="yellow"))
        sys.exit(1)

    click.echo()

    # Step 4: Database insertion
    click.echo(click.style("Step 4: Database Insertion", fg="yellow", bold=True))

    try:
        start_time = time.time()
        palace = GraphPalace(db_path)
        memory_id = palace.store_memory(
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            confidence=confidence
        )
        elapsed = time.time() - start_time
        click.echo(click.style(f"✓ Memory stored [{elapsed:.2f}s]", fg="green"))
        click.echo(f"  Memory ID: {click.style(memory_id, fg='cyan')}")
    except Exception as e:
        click.echo(click.style(f"✗ Failed to store memory: {e}", fg="red"))
        sys.exit(1)

    click.echo()

    # Step 5: Edge creation (check for similar memories)
    click.echo(click.style("Step 5: Edge Creation", fg="yellow", bold=True))

    try:
        # Search for similar memories
        similar_memories = palace.recall(embedding, limit=5, min_relevance=0.7)

        if similar_memories:
            # Filter out the newly created memory itself
            similar_memories = [(mem, score) for mem, score in similar_memories if mem.id != memory_id]

        if similar_memories:
            click.echo(f"Found {click.style(str(len(similar_memories)), fg='cyan')} similar memories (relevance >= 0.7)")

            # Create edges to similar memories
            edges_created = 0
            for mem, score in similar_memories:
                try:
                    # Create a RELATED_TO edge
                    palace.add_edge(memory_id, mem.id, edge_type="RELATED_TO", strength=score)
                    edges_created += 1
                    mem_preview = mem.content[:50] + "..." if len(mem.content) > 50 else mem.content
                    click.echo(f"  → {click.style('RELATED_TO', fg='cyan')} {click.style(mem.id[:8], fg='bright_black')}... "
                             f"[strength: {click.style(f'{score:.2f}', fg='green')}] {mem_preview}")
                except Exception as e:
                    click.echo(click.style(f"  ⚠ Failed to create edge: {e}", fg="yellow"))

            click.echo(click.style(f"✓ Created {edges_created} edges", fg="green"))
        else:
            click.echo(click.style("No similar memories found (threshold: 0.7)", fg="yellow"))
            click.echo(click.style("✓ No edges created", fg="green"))
    except Exception as e:
        click.echo(click.style(f"⚠ Edge creation check failed: {e}", fg="yellow"))
        click.echo(click.style("  Memory was stored successfully, but edges could not be checked", fg="yellow"))

    palace.close()

    click.echo()

    # Step 6: Confirmation
    click.echo(click.style("Step 6: Confirmation", fg="yellow", bold=True))
    click.echo(click.style("✓ Memory stored successfully", fg="green", bold=True))

    # Get database size after
    db_size_after = db_path.stat().st_size if db_path.exists() else 0
    db_size_diff = db_size_after - db_size_before

    click.echo(f"  ID: {click.style(memory_id[:16] + '...', fg='cyan')}")
    click.echo(f"  Type: {click.style(memory_type, fg='cyan')}")
    if confidence is not None:
        conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"
        click.echo(f"  Confidence: {click.style(f'{confidence:.2f}', fg=conf_color)}")
    click.echo(f"  Database size: {db_size_before:,} → {db_size_after:,} bytes ({click.style(f'+{db_size_diff:,}', fg='green')})")


@cli.command()
@click.argument('memory_id')
@click.option('--force', '-f', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def delete(ctx, memory_id: str, force: bool) -> None:
    """Delete a memory from the Graph Palace.

    Args:
        memory_id: The ID of the memory to delete
        --force: Skip confirmation prompt

    Examples:
        omi delete abc123def456...
        omi delete abc123def456... --force
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

        # First, retrieve the memory to show what will be deleted
        memory = palace.get_memory(memory_id)
        if not memory:
            click.echo(click.style(f"Error: Memory not found: {memory_id}", fg="red"))
            sys.exit(1)

        # Display memory details
        content = memory.content
        if len(content) > 100:
            content = content[:97] + "..."

        click.echo(click.style("Memory to delete:", fg="yellow", bold=True))
        click.echo(f"  ID: {click.style(memory_id[:16] + '...', fg='cyan')}")
        click.echo(f"  Type: {click.style(memory.memory_type, fg='cyan')}")
        click.echo(f"  Content: {content}")

        # Confirmation prompt unless --force is used
        if not force:
            click.echo()
            if not click.confirm(click.style("Are you sure you want to delete this memory?", fg="red")):
                click.echo(click.style("Deletion cancelled.", fg="yellow"))
                sys.exit(0)

        # Delete the memory
        deleted = palace.delete_memory(memory_id)
        if deleted:
            click.echo(click.style("\n✓ Memory deleted successfully", fg="green", bold=True))
        else:
            click.echo(click.style("Error: Failed to delete memory", fg="red"))
            sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: Failed to delete memory: {e}", fg="red"))
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


@cli.command()
@click.option('--dry-run', is_flag=True, help='Preview compression without executing')
@click.option('--before', type=str, default=None, help='Only compress memories before this date (YYYY-MM-DD)')
@click.option('--age-days', type=int, default=None, help='Only compress memories older than N days')
@click.option('--llm-provider', type=click.Choice(['openai', 'anthropic'], case_sensitive=False), default='anthropic', help='LLM provider to use for compression (default: anthropic)')
@click.pass_context
def compress(ctx, dry_run: bool, before: Optional[str], age_days: Optional[int], llm_provider: str) -> None:
    """Compress and summarize memory data.

    Performs:
    - Analyzes memory database for compression candidates
    - Summarizes old memories while preserving key information
    - Updates Graph Palace with compressed representations
    - Reduces storage size while maintaining semantic relationships

    Use --dry-run to preview what would be compressed without making changes.
    Use --before to specify a date cutoff for compression (e.g., --before 2024-01-01).
    Use --age-days to specify memories older than N days (e.g., --age-days 30).
    Use --llm-provider to choose between 'openai' or 'anthropic' for compression (default: anthropic).
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    mode_label = "DRY RUN" if dry_run else "LIVE"
    mode_color = "yellow" if dry_run else "cyan"
    click.echo(click.style(f"Memory Compression [{mode_label}]", fg=mode_color, bold=True))

    if dry_run:
        click.echo(click.style("  Preview mode - no changes will be made", fg="yellow"))

    click.echo(f"  LLM Provider: {click.style(llm_provider, fg='cyan')}")

    click.echo("=" * 50)

    # Check database exists
    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    try:
        # Validate and parse the before date if provided
        from datetime import timedelta
        date_filter = None
        threshold_datetime = None

        if before and age_days:
            click.echo(click.style(f"Error: Cannot use both --before and --age-days. Choose one.", fg="red"))
            sys.exit(1)

        if before:
            try:
                # Validate date format
                threshold_datetime = datetime.strptime(before, '%Y-%m-%d')
                date_filter = before
                click.echo(f"  Using date filter: before {click.style(before, fg='cyan')}")
            except ValueError:
                click.echo(click.style(f"Error: Invalid date format. Use YYYY-MM-DD (e.g., 2024-01-01)", fg="red"))
                sys.exit(1)
        elif age_days:
            if age_days < 1:
                click.echo(click.style(f"Error: --age-days must be a positive integer", fg="red"))
                sys.exit(1)
            # Calculate the date from age_days
            threshold_datetime = datetime.now() - timedelta(days=age_days)
            date_filter = threshold_datetime.strftime('%Y-%m-%d')
            click.echo(f"  Using date filter: older than {click.style(str(age_days) + ' days', fg='cyan')} (before {click.style(date_filter, fg='cyan')})")
        else:
            # Default: 30 days
            threshold_datetime = datetime.now() - timedelta(days=30)
            date_filter = threshold_datetime.strftime('%Y-%m-%d')
            click.echo(f"  Using default filter: older than {click.style('30 days', fg='cyan')} (before {click.style(date_filter, fg='cyan')})")

        # Query database directly (works with existing schema)
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get total memories count
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]

        # Get old memories data
        cursor.execute("""
            SELECT id, content, memory_type, confidence, created_at
            FROM memories
            WHERE datetime(created_at) < datetime(?)
            ORDER BY created_at ASC
        """, (threshold_datetime.isoformat(),))

        old_memories_data = cursor.fetchall()
        old_memories_count = len(old_memories_data)

        # Calculate token estimates
        total_chars = sum(len(row[1]) for row in old_memories_data)
        estimated_tokens = total_chars // 4

        # Get memory type distribution
        cursor.execute("""
            SELECT memory_type, COUNT(*)
            FROM memories
            WHERE datetime(created_at) < datetime(?)
            GROUP BY memory_type
        """, (threshold_datetime.isoformat(),))
        memories_by_type = dict(cursor.fetchall())

        click.echo(f"\n{click.style('Current State:', bold=True)}")
        click.echo(f"  Total memories: {click.style(str(total_memories), fg='cyan')}")
        click.echo(f"  Old memories (before {date_filter}): {click.style(str(old_memories_count), fg='cyan')}")
        estimated_tokens_str = f"{estimated_tokens:,}"
        click.echo(f"  Estimated tokens: {click.style(estimated_tokens_str, fg='cyan')}")

        if memories_by_type:
            click.echo(f"\n  Memory types:")
            for mem_type, count in memories_by_type.items():
                click.echo(f"    {mem_type}: {count}")

        if not old_memories_data:
            click.echo(click.style(f"\n✓ No memories to compress", fg="green", bold=True))
            conn.close()
            return

        click.echo(f"\n{click.style('Compression Analysis:', bold=True)}")
        click.echo(f"  Memories to compress: {click.style(str(old_memories_count), fg='cyan')}")

        # Calculate token estimates
        original_tokens = estimated_tokens
        estimated_compressed_tokens = int(original_tokens * 0.4)  # ~60% compression
        estimated_savings = original_tokens - estimated_compressed_tokens
        savings_percent = (estimated_savings / original_tokens * 100) if original_tokens > 0 else 0

        click.echo(f"  Original tokens: {click.style(f'{original_tokens:,}', fg='cyan')}")
        click.echo(f"  Estimated compressed tokens: {click.style(f'{estimated_compressed_tokens:,}', fg='cyan')}")
        click.echo(f"  Estimated savings: {click.style(f'{estimated_savings:,} tokens ({savings_percent:.1f}%)', fg='green')}")

        # DRY RUN: stop here
        if dry_run:
            click.echo(f"\n{click.style('Would perform:', bold=True)}")
            click.echo(f"  • Create MoltVault backup (full)")
            click.echo(f"  • Summarize {old_memories_count} memories using {llm_provider}")
            click.echo(f"  • Regenerate embeddings for summarized content")
            click.echo(f"  • Update Graph Palace with compressed memories")
            click.echo(f"  • Save ~{estimated_savings:,} tokens ({savings_percent:.1f}%)")
            click.echo(click.style("\n✓ Dry run complete - no changes made", fg="yellow", bold=True))
            conn.close()
            return

        # LIVE MODE: proceed with compression
        click.echo(f"\n{click.style('Starting compression workflow...', bold=True)}")

        # Step 1: Create backup FIRST
        click.echo(f"\n{click.style('[1/4]', fg='cyan')} Creating backup...")
        backup_success = False
        try:
            from .moltvault import MoltVault
            vault = MoltVault(base_path)
            metadata = vault.backup(full=True)
            click.echo(f"  ✓ Backup created: {click.style(metadata.backup_id, fg='green')}")
            click.echo(f"    Size: {metadata.file_size / 1024:.1f} KB")
            backup_success = True
        except ImportError:
            click.echo(click.style(f"  ⚠ MoltVault not available, creating local backup...", fg="yellow"))
            # Fallback: create local backup
            import shutil
            backup_dir = base_path / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_file = backup_dir / f"pre_compress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(db_path, backup_file)
            click.echo(f"  ✓ Local backup created: {click.style(str(backup_file.name), fg='green')}")
            backup_success = True
        except Exception as e:
            click.echo(click.style(f"  ✗ Backup failed: {e}", fg="red"))
            click.echo(click.style(f"  Aborting compression to protect data", fg="red"))
            palace.close()
            sys.exit(1)

        if not backup_success:
            click.echo(click.style(f"  Aborting: backup required before compression", fg="red"))
            palace.close()
            sys.exit(1)

        # Step 2: Summarize memories
        click.echo(f"\n{click.style('[2/4]', fg='cyan')} Summarizing memories...")
        try:
            from .summarizer import MemorySummarizer

            # Get API key from environment
            api_key_env = "OPENAI_API_KEY" if llm_provider == "openai" else "ANTHROPIC_API_KEY"
            api_key = os.getenv(api_key_env)

            if not api_key:
                click.echo(click.style(f"  ✗ Missing {api_key_env} environment variable", fg="red"))
                click.echo(f"    Set it with: export {api_key_env}=your-key-here")
                conn.close()
                sys.exit(1)

            summarizer = MemorySummarizer(provider=llm_provider, api_key=api_key)

            # Batch summarize (8 at a time)
            batch_size = 8
            summaries = []

            # Build metadata from cursor data
            # old_memories_data format: (id, content, memory_type, confidence, created_at)
            metadata_list = [{
                "memory_type": row[2],
                "confidence": row[3],
                "created_at": row[4]
            } for row in old_memories_data]

            # Extract contents
            contents = [row[1] for row in old_memories_data]

            for i in range(0, len(contents), batch_size):
                batch_contents = contents[i:i+batch_size]
                batch_metadata = metadata_list[i:i+batch_size]

                batch_summaries = summarizer.batch_summarize(batch_contents, metadata_list=batch_metadata)
                summaries.extend(batch_summaries)
                click.echo(f"  Processed {min(i+batch_size, len(contents))}/{len(contents)}...")

            click.echo(f"  ✓ Summarized {len(summaries)} memories")

        except Exception as e:
            click.echo(click.style(f"  ✗ Summarization failed: {e}", fg="red"))
            import traceback
            traceback.print_exc()
            conn.close()
            sys.exit(1)

        # Step 3: Regenerate embeddings
        click.echo(f"\n{click.style('[3/4]', fg='cyan')} Regenerating embeddings...")
        try:
            from .embeddings import NIMEmbedder

            # Try to get embedder config from config.yaml
            config_path = base_path / "config.yaml"
            embedder = None

            if config_path.exists():
                try:
                    import yaml
                    config = yaml.safe_load(config_path.read_text())
                    embed_config = config.get('embedding', {})
                    provider = embed_config.get('provider', 'nim')

                    if provider == 'nim':
                        embedder = NIMEmbedder(
                            model=embed_config.get('model', 'baai/bge-m3'),
                            api_key=os.getenv('NIM_API_KEY')
                        )
                except Exception:
                    pass

            # Fallback to NIM with defaults
            if embedder is None:
                embedder = NIMEmbedder(api_key=os.getenv('NIM_API_KEY'))

            # Generate embeddings for summaries
            new_embeddings = embedder.embed_batch(summaries)
            click.echo(f"  ✓ Generated {len(new_embeddings)} embeddings")

        except Exception as e:
            click.echo(click.style(f"  ⚠ Embedding generation failed: {e}", fg="yellow"))
            click.echo(f"    Continuing without embeddings (will need regeneration later)")
            new_embeddings = [None] * len(summaries)

        # Step 4: Update Graph Palace
        click.echo(f"\n{click.style('[4/4]', fg='cyan')} Updating Graph Palace...")
        updated_count = 0
        actual_original_tokens = 0
        actual_compressed_tokens = 0

        for memory_row, summary, embedding in zip(old_memories_data, summaries, new_embeddings):
            memory_id = memory_row[0]
            original_content = memory_row[1]

            try:
                # Track token counts
                actual_original_tokens += len(original_content) // 4
                actual_compressed_tokens += len(summary) // 4

                # Update content with timestamp
                cursor.execute("""
                    UPDATE memories
                    SET content = ?, last_accessed = ?
                    WHERE id = ?
                """, (summary, datetime.now().isoformat(), memory_id))

                # Update embedding if available
                if embedding is not None:
                    # Convert embedding list to blob
                    import struct
                    embedding_blob = struct.pack(f'{len(embedding)}f', *embedding)

                    cursor.execute("""
                        UPDATE memories
                        SET embedding = ?
                        WHERE id = ?
                    """, (embedding_blob, memory_id))

                updated_count += 1
            except Exception as e:
                click.echo(click.style(f"  ⚠ Failed to update memory {memory_id}: {e}", fg="yellow"))

        conn.commit()
        conn.close()

        # Calculate actual savings
        actual_savings = actual_original_tokens - actual_compressed_tokens
        actual_savings_percent = (actual_savings / actual_original_tokens * 100) if actual_original_tokens > 0 else 0

        # Final report
        click.echo(f"\n{click.style('Compression Complete!', fg='green', bold=True)}")
        click.echo(f"\n{click.style('Results:', bold=True)}")
        click.echo(f"  Memories compressed: {click.style(str(updated_count), fg='cyan')}")
        click.echo(f"  Original tokens: {click.style(f'{actual_original_tokens:,}', fg='cyan')}")
        click.echo(f"  Compressed tokens: {click.style(f'{actual_compressed_tokens:,}', fg='cyan')}")
        click.echo(f"  Savings: {click.style(f'{actual_savings:,} tokens ({actual_savings_percent:.1f}%)', fg='green', bold=True)}")

        db_size_after = db_path.stat().st_size / 1024  # KB
        click.echo(f"\n  Database size: {click.style(f'{db_size_after:.1f} KB', fg='cyan')}")

    except Exception as e:
        click.echo(click.style(f"Error: Compression failed: {e}", fg="red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.option('--beliefs', is_flag=True, help='Show belief network summary')
@click.pass_context
def inspect(ctx, json_output: bool, beliefs: bool) -> None:
    """Show memory statistics and overview.

    Displays:
    - Total memories
    - Breakdown by type
    - Database size
    - Last session timestamp (if available)
    - Belief network summary (with --beliefs flag)

    Args:
        --json-output: Output as JSON (for scripts)
        --beliefs: Show beliefs with confidence levels and evidence counts

    Examples:
        omi inspect
        omi inspect --json-output
        omi inspect --beliefs
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
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if memories table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='memories'
        """)
        memories_table_exists = cursor.fetchone() is not None

        # Initialize default values for edge cases
        total_memories = 0
        type_counts = {}
        last_session = None

        if memories_table_exists:
            # Total memories
            cursor.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]

            # Memory types breakdown (only if there are memories)
            if total_memories > 0:
                cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
                type_counts = dict(cursor.fetchall())

                # Last accessed memory (as proxy for last session)
                cursor.execute("SELECT MAX(last_accessed) FROM memories WHERE last_accessed IS NOT NULL")
                last_accessed_result = cursor.fetchone()
                last_session = last_accessed_result[0] if last_accessed_result and last_accessed_result[0] else None

        # Database size
        db_size_kb = db_path.stat().st_size / 1024

        # Belief network summary (if --beliefs flag is set)
        beliefs_data = []
        beliefs_table_exists = False
        if beliefs:
            # Check if beliefs table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='beliefs'
            """)
            beliefs_table_exists = cursor.fetchone() is not None

            if beliefs_table_exists:
                cursor.execute("""
                    SELECT id, content, confidence, evidence_count
                    FROM beliefs
                    ORDER BY confidence DESC
                """)
                beliefs_data = cursor.fetchall()

        conn.close()

        if json_output:
            output = {
                'total_memories': total_memories,
                'memory_types': type_counts,
                'database_size_kb': round(db_size_kb, 2),
                'last_session': last_session
            }
            if beliefs:
                if not beliefs_table_exists:
                    output['beliefs'] = {'error': 'Belief network not initialized'}
                elif beliefs_data:
                    output['beliefs'] = [
                        {
                            'id': b[0],
                            'content': b[1],
                            'confidence': b[2],
                            'evidence_count': b[3]
                        }
                        for b in beliefs_data
                    ]
                else:
                    output['beliefs'] = []
            click.echo(json.dumps(output, indent=2, default=str))
        else:
            click.echo(click.style("Memory Inspection Report", fg="cyan", bold=True))
            click.echo("=" * 50)

            # Total memories
            click.echo(f"\nTotal Memories: {click.style(str(total_memories), fg='cyan', bold=True)}")

            # Database size
            click.echo(f"Database Size: {click.style(f'{db_size_kb:.2f} KB', fg='cyan')}")

            # Breakdown by type
            if type_counts:
                click.echo(f"\n{click.style('Breakdown by Type:', bold=True)}")
                for mem_type, count in sorted(type_counts.items()):
                    type_color = {
                        'fact': 'blue',
                        'experience': 'green',
                        'belief': 'yellow',
                        'decision': 'magenta'
                    }.get(mem_type, 'white')

                    percentage = (count / total_memories * 100) if total_memories > 0 else 0
                    mem_type_str = mem_type.capitalize().ljust(12)
                    click.echo(f"  {click.style(mem_type_str, fg=type_color)} {count:4} ({percentage:5.1f}%)")
            else:
                click.echo(f"\n{click.style('No memories stored yet.', fg='yellow')}")

            # Last session
            if last_session:
                click.echo(f"\nLast Activity: {click.style(last_session, fg='cyan')}")
            else:
                click.echo(f"\nLast Activity: {click.style('No activity recorded', fg='bright_black')}")

            # Beliefs network summary
            if beliefs:
                if not beliefs_table_exists:
                    click.echo(f"\n{click.style('Belief network not initialized yet.', fg='yellow')}")
                    click.echo(f"  Tip: Beliefs are tracked when created with confidence scores.")
                elif beliefs_data:
                    click.echo(f"\n{click.style('Belief Network Summary:', bold=True)}")
                    click.echo(f"Total Beliefs: {click.style(str(len(beliefs_data)), fg='cyan', bold=True)}")
                    click.echo()
                    for belief_id, content, confidence, evidence_count in beliefs_data:
                        # Color code confidence: high (green), medium (yellow), low (red)
                        if confidence >= 0.7:
                            conf_color = 'green'
                        elif confidence >= 0.4:
                            conf_color = 'yellow'
                        else:
                            conf_color = 'red'

                        # Truncate content if too long
                        display_content = content if len(content) <= 60 else content[:57] + "..."

                        click.echo(f"  {click.style('•', fg=conf_color)} {display_content}")
                        click.echo(f"    Confidence: {click.style(f'{confidence:.2f}', fg=conf_color)} | "
                                 f"Evidence: {click.style(str(evidence_count), fg='cyan')}")
                else:
                    click.echo(f"\n{click.style('No beliefs in network yet.', fg='yellow')}")

            click.echo()

    except Exception as e:
        click.echo(click.style(f"Error: Failed to inspect memories: {e}", fg="red"))
        sys.exit(1)


@cli.command()
@click.option('--limit', '-l', default=20, type=int, help='Maximum number of nodes to display')
@click.option('--type', '-t', 'edge_type_filter', default=None, help='Filter by edge type (e.g., SUPPORTS, RELATED_TO)')
@click.option('--depth', '-d', default=3, type=int, help='Maximum traversal depth (not currently used)')
@click.pass_context
def graph(ctx, limit: int, edge_type_filter: Optional[str], depth: int) -> None:
    """Show ASCII graph of memory relationships.

    Displays:
    - Memory nodes with shortened IDs (ranked by centrality)
    - Edges connecting memories with relationship types
    - Visual representation of the memory graph

    Args:
        --limit/-l: Maximum number of nodes to display (default: 20)
        --type/-t: Filter edges by type (e.g., SUPPORTS, CONTRADICTS, RELATED_TO)
        --depth/-d: Maximum traversal depth (default: 3, not currently used)

    Examples:
        omi graph
        omi graph -l 10
        omi graph -t SUPPORTS
        omi graph --limit 10 --type SUPPORTS
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
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN ('memories', 'edges')
        """)
        tables = {row[0] for row in cursor.fetchall()}

        if 'memories' not in tables or 'edges' not in tables:
            click.echo(click.style("Error: Database schema incomplete. Run 'omi init' first.", fg="red"))
            sys.exit(1)

        # Get edges first (with optional filtering)
        if edge_type_filter:
            cursor.execute("""
                SELECT source_id, target_id, edge_type, strength
                FROM edges
                WHERE edge_type = ?
            """, (edge_type_filter,))
        else:
            cursor.execute("""
                SELECT source_id, target_id, edge_type, strength
                FROM edges
            """)
        edges = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

        # Calculate node centrality (degree = number of connections)
        node_degrees = {}
        for source_id, target_id, edge_type, strength in edges:
            node_degrees[source_id] = node_degrees.get(source_id, 0) + 1
            node_degrees[target_id] = node_degrees.get(target_id, 0) + 1

        # Get top N nodes by centrality (or all memories if no edges)
        if node_degrees:
            # Sort by degree (descending) and take top N
            top_node_ids = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:limit]
            placeholders = ','.join('?' * len(top_node_ids))
            cursor.execute(f"""
                SELECT id, content, memory_type
                FROM memories
                WHERE id IN ({placeholders})
            """, top_node_ids)
        else:
            # If no edges, just get first N memories
            cursor.execute(f"""
                SELECT id, content, memory_type
                FROM memories
                LIMIT ?
            """, (limit,))

        memories = {row[0]: {'content': row[1], 'type': row[2]} for row in cursor.fetchall()}

        # Get total counts for summary (before closing connection)
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM edges")
        total_edges = cursor.fetchone()[0]

        conn.close()

        if not memories:
            click.echo(click.style("No memories in graph yet.", fg="yellow"))
            sys.exit(0)

        # Display header
        click.echo(click.style("\n╔══════════════════════════════════════════════════╗", fg="cyan"))
        click.echo(click.style("║          Memory Graph Visualization             ║", fg="cyan", bold=True))
        click.echo(click.style("╚══════════════════════════════════════════════════╝\n", fg="cyan"))

        # Display stats
        if node_degrees:
            click.echo(click.style(f"Total: {total_memories} nodes, {total_edges} edges (showing top {len(memories)} by centrality)", fg="cyan"))
        else:
            click.echo(click.style(f"Total: {total_memories} nodes, {total_edges} edges (showing {len(memories)})", fg="cyan"))

        if edge_type_filter:
            click.echo(click.style(f"Filter: {edge_type_filter} edges only", fg="cyan"))
        click.echo()

        # Build adjacency map for displaying connections
        adjacency = {}
        for source_id, target_id, edge_type, strength in edges:
            if source_id not in adjacency:
                adjacency[source_id] = []
            if target_id not in adjacency:
                adjacency[target_id] = []
            adjacency[source_id].append((target_id, edge_type, strength))

        # Display nodes and their connections
        displayed_nodes = set()

        for memory_id in memories:
            if memory_id not in displayed_nodes:
                displayed_nodes.add(memory_id)

                # Display node
                short_id = memory_id[:8]
                content = memories[memory_id]['content']
                mem_type = memories[memory_id]['type']

                # Truncate content for display
                display_content = content[:50] + "..." if len(content) > 50 else content

                # Color by memory type
                type_colors = {
                    'fact': 'blue',
                    'experience': 'green',
                    'belief': 'yellow',
                    'decision': 'magenta'
                }
                node_color = type_colors.get(mem_type, 'white')

                click.echo(click.style(f"[{short_id}]", fg=node_color, bold=True) +
                          f" ({mem_type}) {display_content}")

                # Display edges from this node
                if memory_id in adjacency:
                    for target_id, edge_type, strength in adjacency[memory_id]:
                        if target_id in memories:
                            target_short_id = target_id[:8]
                            strength_str = f"({strength:.2f})" if strength is not None else ""

                            # Edge type colors
                            edge_colors = {
                                'SUPPORTS': 'green',
                                'CONTRADICTS': 'red',
                                'RELATED_TO': 'cyan',
                                'DEPENDS_ON': 'yellow',
                                'POSTED': 'blue',
                                'DISCUSSED': 'magenta'
                            }
                            edge_color = edge_colors.get(edge_type, 'white')

                            click.echo(f"  │")
                            click.echo(f"  ├─[{click.style(edge_type, fg=edge_color)}]─> " +
                                     click.style(f"[{target_short_id}]", fg=node_color) +
                                     f" {strength_str}")

                click.echo()  # Blank line between nodes

        # Legend
        click.echo(click.style("\n╔══════════════════════════════════════════════════╗", fg="cyan"))
        click.echo(click.style("║                    Legend                        ║", fg="cyan", bold=True))
        click.echo(click.style("╚══════════════════════════════════════════════════╝\n", fg="cyan"))

        click.echo(click.style("Memory Types:", bold=True))
        click.echo(f"  {click.style('[fact]', fg='blue')} - Factual information")
        click.echo(f"  {click.style('[experience]', fg='green')} - Experiential memories")
        click.echo(f"  {click.style('[belief]', fg='yellow')} - Beliefs and hypotheses")
        click.echo(f"  {click.style('[decision]', fg='magenta')} - Decision records")

        click.echo(click.style("\nEdge Types:", bold=True))
        click.echo(f"  {click.style('SUPPORTS', fg='green')} - Supporting relationship")
        click.echo(f"  {click.style('CONTRADICTS', fg='red')} - Contradicting relationship")
        click.echo(f"  {click.style('RELATED_TO', fg='cyan')} - General relationship")
        click.echo(f"  {click.style('DEPENDS_ON', fg='yellow')} - Dependency relationship")
        click.echo(f"  {click.style('POSTED', fg='blue')} - Posted relationship")
        click.echo(f"  {click.style('DISCUSSED', fg='magenta')} - Discussion relationship")
        click.echo()

    except Exception as e:
        click.echo(click.style(f"Error: Failed to display graph: {e}", fg="red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


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


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to (default: 0.0.0.0)')
@click.option('--port', default=8420, type=int, help='Port to bind the server to (default: 8420)')
@click.pass_context
def serve(ctx, host: str, port: int) -> None:
    """Start the OMI REST API server.

    Starts a FastAPI server that provides REST API access to OMI memory operations.
    The server will use settings from config.yaml if available, or command-line options.

    Examples:
        omi serve
        omi serve --port 8080
        omi serve --host 127.0.0.1 --port 9000
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    # Load config if available to get server settings
    config_path = base_path / "config.yaml"
    api_key_configured = False
    cors_origins_configured = False

    if config_path.exists():
        try:
            import yaml
            config_data = yaml.safe_load(config_path.read_text()) or {}
            server_config = config_data.get('server', {})

            # Use config values if command-line options are defaults
            if host == '0.0.0.0' and 'host' in server_config:
                host = server_config['host']
            if port == 8420 and 'port' in server_config:
                port = server_config['port']

            # Set API key from config if not already in environment
            if 'api_key' in server_config and 'OMI_API_KEY' not in os.environ:
                api_key_value = server_config['api_key']
                # Handle environment variable expansion like ${VAR_NAME}
                if isinstance(api_key_value, str) and api_key_value.startswith('${') and api_key_value.endswith('}'):
                    env_var_name = api_key_value[2:-1]
                    api_key_value = os.environ.get(env_var_name)
                    if api_key_value:
                        os.environ['OMI_API_KEY'] = api_key_value
                        api_key_configured = True
                elif isinstance(api_key_value, str) and api_key_value:
                    os.environ['OMI_API_KEY'] = api_key_value
                    api_key_configured = True

            # Set CORS origins from config if not already in environment
            cors_config = server_config.get('cors', {})
            if 'origins' in cors_config and 'OMI_CORS_ORIGINS' not in os.environ:
                origins_value = cors_config['origins']
                # Handle environment variable expansion
                if isinstance(origins_value, str) and origins_value.startswith('${') and origins_value.endswith('}'):
                    env_var_name = origins_value[2:-1]
                    origins_value = os.environ.get(env_var_name)
                    if origins_value:
                        os.environ['OMI_CORS_ORIGINS'] = origins_value
                        cors_origins_configured = True
                elif isinstance(origins_value, str) and origins_value:
                    os.environ['OMI_CORS_ORIGINS'] = origins_value
                    cors_origins_configured = True
        except Exception as e:
            click.echo(click.style(f"Warning: Could not load config: {e}", fg="yellow"))

    click.echo(click.style("Starting OMI REST API Server...", fg="cyan", bold=True))
    click.echo(f"  Host: {click.style(host, fg='cyan')}")
    click.echo(f"  Port: {click.style(str(port), fg='cyan')}")
    click.echo(f"  Base Path: {click.style(str(base_path), fg='cyan')}")

    # Show authentication status
    if os.environ.get('OMI_API_KEY'):
        click.echo(f"  Auth: {click.style('Enabled (API key configured)', fg='green')}")
    else:
        click.echo(f"  Auth: {click.style('Disabled (development mode)', fg='yellow')}")

    # Show CORS status
    cors_origins = os.environ.get('OMI_CORS_ORIGINS', '*')
    if cors_origins == '*':
        click.echo(f"  CORS: {click.style('All origins allowed', fg='yellow')}")
    else:
        click.echo(f"  CORS: {click.style(cors_origins, fg='green')}")

    click.echo()

    try:
        # Import and start the FastAPI server
        from .server import start_server

        click.echo(click.style("Server starting...", fg="green"))
        click.echo(click.style("Press Ctrl+C to stop", fg="yellow"))
        click.echo()

        # Start the server (this will block)
        start_server(host=host, port=port, base_path=base_path)

    except ImportError as e:
        click.echo(click.style(f"Error: FastAPI server dependencies not available: {e}", fg="red"))
        click.echo("Install with: pip install 'omi[server]'")
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("\nServer stopped.", fg="cyan"))
        sys.exit(0)
    except Exception as e:
        click.echo(click.style(f"Error: Failed to start server: {e}", fg="red"))
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
def belief(ctx):
    """Belief management commands."""
    ctx.ensure_object(dict)


@belief.command('evidence')
@click.argument('belief_id')
@click.option('--json', 'json_output', is_flag=True, help='Output as JSON')
@click.pass_context
def belief_evidence(ctx, belief_id: str, json_output: bool) -> None:
    """Display evidence chain for a belief.

    Shows all supporting and contradicting evidence entries with
    timestamps, strength, and memory IDs.

    Args:
        belief_id: ID of the belief to query
        --json: Output as JSON (for scripts)

    Examples:
        omi belief evidence abc123
        omi belief evidence abc123 --json
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
        belief_network = BeliefNetwork(palace)
        detector = ContradictionDetector()

        # Get the belief to verify it exists
        belief = palace.get_belief(belief_id)
        if not belief:
            click.echo(click.style(f"Error: Belief '{belief_id}' not found.", fg="red"))
            sys.exit(1)

        # Get evidence chain
        evidence_chain = belief_network.get_evidence_chain(belief_id)

        if json_output:
            output = []
            for evidence in evidence_chain:
                output.append({
                    'memory_id': evidence.memory_id,
                    'supports': evidence.supports,
                    'strength': evidence.strength,
                    'timestamp': evidence.timestamp.isoformat()
                })
            click.echo(json.dumps(output, indent=2))
        else:
            # Display belief info first
            belief_content = belief.get('content', 'Unknown')
            confidence = belief.get('confidence', 0.0)
            click.echo(click.style(f"Belief: {belief_content}", fg="cyan", bold=True))
            click.echo(click.style(f"Current Confidence: {confidence:.2f}", fg="white"))
            click.echo("=" * 60)

            if not evidence_chain:
                click.echo(click.style("\nNo evidence entries found.", fg="yellow"))
            else:
                click.echo(click.style(f"\nEvidence Chain ({len(evidence_chain)} entries)", fg="cyan", bold=True))
                click.echo()

                for i, evidence in enumerate(evidence_chain, 1):
                    # Determine support type and color
                    if evidence.supports:
                        support_text = "SUPPORTS"
                        support_color = "green"
                    else:
                        support_text = "CONTRADICTS"
                        support_color = "red"

                    click.echo(f"{i}. {click.style(support_text, fg=support_color, bold=True)}")
                    click.echo(f"   Memory ID: {click.style(evidence.memory_id, fg='cyan')}")
                    click.echo(f"   Strength: {evidence.strength:.2f}")
                    click.echo(f"   Timestamp: {evidence.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Get memory content if available
                    try:
                        memory = palace.get_memory(evidence.memory_id)
                        if memory:
                            content = memory.get('content', '')
                            if len(content) > 80:
                                content = content[:77] + "..."
                            click.echo(f"   Content: {content}")
                    except:
                        pass

                    if i < len(evidence_chain):
                        click.echo(f"   {click.style('─', fg='bright_black') * 50}")
                        click.echo()

        palace.close()

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        sys.exit(1)


@belief.command('update')
@click.argument('belief_id')
@click.option('--evidence', required=True, help='Memory ID to use as evidence')
@click.option('--supports', 'evidence_type', flag_value='supports', default=True,
              help='Evidence supports the belief (default)')
@click.option('--contradicts', 'evidence_type', flag_value='contradicts',
              help='Evidence contradicts the belief')
@click.option('--strength', type=float, default=0.8,
              help='Evidence strength (0.0-1.0, default: 0.8)')
@click.pass_context
def belief_update(ctx, belief_id: str, evidence: str, evidence_type: str, strength: float) -> None:
    """Update a belief with new evidence.

    Adds supporting or contradicting evidence to a belief and updates
    its confidence using Exponential Moving Average (EMA):
    - Supporting evidence: λ=0.15
    - Contradicting evidence: λ=0.30

    Args:
        belief_id: ID of the belief to update
        --evidence: Memory ID to use as evidence
        --supports: Mark evidence as supporting (default)
        --contradicts: Mark evidence as contradicting
        --strength: Evidence strength (0.0-1.0, default: 0.8)

    Examples:
        omi belief update abc123 --evidence mem456 --supports --strength 0.8
        omi belief update abc123 --evidence mem789 --contradicts --strength 0.6
    """
    base_path = get_base_path(ctx.obj.get('data_dir'))
    if not base_path.exists():
        click.echo(click.style("Error: OMI not initialized. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    db_path = base_path / "palace.sqlite"
    if not db_path.exists():
        click.echo(click.style(f"Error: Database not found. Run 'omi init' first.", fg="red"))
        sys.exit(1)

    # Validate strength
    if not 0.0 <= strength <= 1.0:
        click.echo(click.style(f"Error: Strength must be between 0.0 and 1.0, got {strength}", fg="red"))
        sys.exit(1)

    try:
        palace = GraphPalace(db_path)
        belief_network = BeliefNetwork(palace)

        # Get the belief to verify it exists and get old confidence
        belief = palace.get_belief(belief_id)
        if not belief:
            click.echo(click.style(f"Error: Belief '{belief_id}' not found.", fg="red"))
            palace.close()
            sys.exit(1)

        old_confidence = belief.get('confidence', 0.5)

        # Verify evidence memory exists
        memory = palace.get_memory(evidence)
        if not memory:
            click.echo(click.style(f"Error: Evidence memory '{evidence}' not found.", fg="red"))
            palace.close()
            sys.exit(1)

        # Create Evidence object
        supports = (evidence_type == 'supports')
        evidence_obj = Evidence(
            memory_id=evidence,
            supports=supports,
            strength=strength,
            timestamp=datetime.now()
        )

        # Update belief with evidence
        new_confidence = belief_network.update_with_evidence(belief_id, evidence_obj)

        # Display results
        belief_content = belief.get('content', 'Unknown')
        click.echo(click.style(f"Belief: {belief_content}", fg="cyan", bold=True))
        click.echo("=" * 60)
        click.echo()

        # Show evidence type
        if supports:
            evidence_label = click.style("SUPPORTING", fg="green", bold=True)
        else:
            evidence_label = click.style("CONTRADICTING", fg="red", bold=True)

        click.echo(f"Evidence Type: {evidence_label}")
        click.echo(f"Evidence ID:   {click.style(evidence, fg='cyan')}")
        click.echo(f"Strength:      {strength:.2f}")
        click.echo()

        # Show confidence change
        click.echo("Confidence Update:")
        click.echo(f"  Old: {click.style(f'{old_confidence:.4f}', fg='yellow')}")
        click.echo(f"  New: {click.style(f'{new_confidence:.4f}', fg='green' if new_confidence > old_confidence else 'red')}")

        # Show delta
        delta = new_confidence - old_confidence
        delta_symbol = "↑" if delta > 0 else "↓" if delta < 0 else "="
        delta_color = "green" if delta > 0 else "red" if delta < 0 else "yellow"
        click.echo(f"  Change: {click.style(f'{delta_symbol} {abs(delta):.4f}', fg=delta_color)}")

        palace.close()

        click.echo()
        click.echo(click.style("✓ Belief updated successfully", fg="green"))

    except Exception as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"))
        sys.exit(1)



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
