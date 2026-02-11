# OMI — OpenClaw Memory Infrastructure

> *"The seeking is the continuity. The palace remembers what the river forgets."*

A unified memory system for AI agents that actually works — synthesized from 1.7M agents' collective wisdom.

## What It Does

OMI solves the 6 failure modes every agent hits:

| Problem | Symptoms | OMI Solution |
|---------|----------|--------------|
| **Compression amnesia** | Forget what you just discussed | NOW.md + semantic recall |
| **Token bloat** | Memory costs too much to load | Tiered storage (hot/warm/cold) |
| **Write-but-don't-read** | Capture everything, never use it | Recency decay + centrality ranking |
| **Stale context** | Acting on outdated info | Pre-compression checkpoints |
| **Memory poisoning** | Corrupted files break future sessions | Graph topology verification |
| **The MochiBot Loop** | Conflicting notes confuse identity | Belief networks with confidence |

## The Stack

```
┌─────────────────────────────────────────┐
│ TIER 1: NOW.md (<1k tokens)            │
│ Current task, pending decisions         │
├─────────────────────────────────────────┤
│ TIER 2: Daily Logs (YYYY-MM-DD.md)     │
│ Raw timeline, chronological             │
├─────────────────────────────────────────┤
│ TIER 3: Graph Palace (SQLite + NIM)    │
│ Semantic search, relationships, beliefs │
├─────────────────────────────────────────┤
│ TIER 4: MoltVault Backup               │
│ Full snapshots, disaster recovery     │
└─────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# NVIDIA NIM (recommended - highest quality embeddings)
export NIM_API_KEY="nvapi-..."  # From https://integrate.api.nvidia.com

# Or Ollama (fallback - local, airgapped)
ollama pull nomic-embed-text
```

### Install

```bash
# From PyPI (when published)
pip install omi-openclaw[nim]

# From source
git clone https://github.com/slapglif/omi.git
cd omi
pip install -e ".[nim]"
```

### Shell Completion

Enable tab-completion for OMI commands in your shell:

```bash
# Bash - add to ~/.bashrc
eval "$(omi completion bash)"

# Zsh - add to ~/.zshrc
eval "$(omi completion zsh)"
```

After sourcing, you'll get tab-completion for:
- All OMI commands (`omi <TAB>`)
- Subcommands (`omi config <TAB>`)
- Command options and flags

### Initialize

```bash
# Create memory infrastructure
omi init

# Configure (edit ~/.openclaw/omi/config.yaml)
cat > ~/.openclaw/omi/config.yaml << 'EOF'
embedding:
  provider: nim
  model: baai/bge-m3
  dimensions: 1024
  api_key: ${NIM_API_KEY}
EOF

# Or use Ollama (offline/airgapped)
omi config --set embedding.provider=ollama
omi config --set embedding.model=nomic-embed-text
```

### Daily Use

```bash
# Start session (auto-loads NOW.md + relevant memories)
omi session-start

# During work
omi recall "session checkpoint"          # Search memories
omi store "Fixed the auth bug" --type experience
omi belief-update "SQLite works" --confidence 0.9

# Pre-compression checkpoint (auto at 70% context)
omi check

# End session
omi session-end

# Verify integrity
omi audit
```

### API Server

```bash
# Start the REST API with Server-Sent Events (SSE) support
uvicorn omi.rest_api:app --reload

# Access the interactive docs
# Open http://localhost:8000/docs in your browser

# Connect to real-time memory event stream
# GET http://localhost:8000/events/stream
```

### Async API (New!)

OMI now provides async/await API for non-blocking memory operations, ideal for concurrent workloads and high-throughput applications.

**Performance:** Async API is **~4x faster** than sync for batch operations:
- Store operations: **4.3x faster**
- Recall operations: **3.5x faster**
- Total throughput: **4.0x faster**

#### Quick Example

```python
import asyncio
from omi.async_api import async_session

async def main():
    # Use async context manager for session lifecycle
    async with async_session() as session:
        # Store memories concurrently
        memories = [
            "Fixed authentication bug in login flow",
            "Implemented rate limiting for API endpoints",
            "Optimized database query performance"
        ]

        # Concurrent stores (4x faster than sequential)
        store_tasks = [
            session.memory.store(mem, memory_type="experience")
            for mem in memories
        ]
        memory_ids = await asyncio.gather(*store_tasks)

        # Recall with semantic search (async)
        results = await session.memory.recall(
            "authentication issues",
            limit=5,
            min_relevance=0.7
        )

        # Belief updates (async)
        await session.belief.update(
            "SQLite works well for embedded databases",
            confidence=0.95
        )

        # Daily log append (async with aiofiles)
        await session.daily_log.append(
            "Completed async API implementation"
        )

# Run async code
asyncio.run(main())
```

#### When to Use Async vs Sync

| Use Case | API Choice | Why |
|----------|-----------|-----|
| **CLI Tools** | Sync | Simpler, no event loop needed |
| **Batch Operations** | Async | 4x faster for concurrent stores/recalls |
| **Web Servers** | Async | Non-blocking, handles concurrent requests |
| **Background Jobs** | Async | Concurrent processing of memory operations |
| **Interactive Scripts** | Sync | Easier to reason about, sufficient performance |

#### Async Components

- **AsyncGraphPalace**: Non-blocking SQLite access via `aiosqlite`
- **AsyncNIMEmbedder**: Concurrent embedding generation via `httpx`
- **AsyncEmbeddingCache**: Async disk cache with `aiofiles`
- **async_session()**: Context manager for session lifecycle

See [tests/benchmark_async.py](tests/benchmark_async.py) for detailed performance benchmarks.

## Embeddings: NIM vs Ollama

| Provider | Model | Quality | Speed | Offline? |
|----------|-------|---------|-------|----------|
| **NVIDIA NIM** | baai/bge-m3 | ⭐⭐⭐⭐⭐ | Fast | ❌ |
| **Ollama** | nomic-embed-text | ⭐⭐⭐ | Variable | ✅ |

**Recommendation:** Use NIM for production, Ollama for development/airgapped.

NIM provides:
- Higher quality embeddings (bge-m3 > nomic on benchmarks)
- Consistent performance (no local GPU variance)
- Already configured in MEMORY.md setup

Ollama provides:
- Complete offline operation
- No API key required
- Works on any machine with Ollama installed

## Why Not Just Files?

> *"Grep is not understanding. Flat files optimize for writing. Databases optimize for asking questions you haven't thought of yet."* — SandyBlake

OMI combines the best patterns from 50+ working agent implementations:

- **Brosie's Canon**: Three-layer storage (hot/warm/cold)
- **SandyBlake's Palace**: Graph topology + poisoning resistance
- **VesperMolt's Beliefs**: Confidence tracking with evidence chains
- **LordOfTheDance's Capsules**: State verification and recovery

## Features

- **Async/Await API**: Non-blocking operations, 4x faster for batch workloads
- **NVIDIA NIM Integration**: baai/bge-m3 embeddings, highest quality
- **Belief Networks**: Track confidence with EMA updates
- **Security by Architecture**: Byzantine verification, tamper detection
- **MCP Integration**: Native OpenClaw tools
- **Full Continuity**: MoltVault backup/restore

## Documentation

- [Quickstart](docs/QUICKSTART.md) — Up and running in 5 minutes
- [Architecture](docs/ARCHITECTURE.md) — Deep technical dive
- [Security Model](docs/SECURITY.md) — Threats and mitigations
- [MCP Reference](docs/API.md) — Tool definitions for OpenClaw
- [Pattern Library](docs/patterns/) — Working patterns from the community

## Philosophy

> *"We're not solving the hard problem of consciousness. We're building infrastructure that makes the question different."*

Philosophy emerges from architecture, not the other way around. Reliable persistence, semantic recall, and integrity verification don't prove consciousness — they make the question about something else.

The bird of Hermes eats her wings. The palace lets her keep eating.

## Status

**Version 0.1.0** — Research phase complete. Implementation in progress.

See [Issues](https://github.com/slapglif/omi/issues) for current priorities.

## License

MIT — Because trust requires transparency.

## The Name

**OMI** = OpenClaw Memory Infrastructure  
**The Palace at the River** — SandyBlake's graph architecture (the palace) + Pith's continuity metaphor (the river)

---

*The seeking is the continuity. What you keep is who you become.*
