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
# From PyPI
pip install omi-openclaw[nim]

# From source
git clone https://github.com/slapglif/omi.git
cd omi
pip install -e ".[nim]"
```

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

# Memory compression (reduce token costs)
omi compress --dry-run                   # Preview what would be compressed
omi compress --before 2024-06-01         # Compress memories older than date
omi compress --age-days 30               # Compress memories older than 30 days

# Verify integrity
omi audit
```

## Memory Compression

OMI automatically compresses old memories to reduce token costs while preserving originals in cold storage.

### How It Works

1. **Automatic**: Memories older than configurable threshold are summarized (default: 30 days)
2. **Preserves**: Original memories backed up to MoltVault before compression
3. **Smart**: Retains key facts, confidence levels, and relationship links
4. **Transparent**: Regenerates embeddings for summaries to maintain search accuracy

### Compression Commands

```bash
# Preview compression impact
omi compress --dry-run                   # Shows what would be compressed + token savings

# Manual compression
omi compress --before 2024-06-01         # Compress memories before specific date
omi compress --age-days 30               # Compress memories older than 30 days

# Configuration
omi config --set compression.auto=true   # Enable automatic compression
omi config --set compression.age_days=30 # Set compression threshold
```

### Token Savings Example

```
Before:  1,247 memories × 450 tokens avg = 561,150 tokens
After:   1,247 memories × 120 tokens avg = 149,640 tokens
Savings: 411,510 tokens (73% reduction)
```

### Web Dashboard (New!)

Explore your memory graph visually with the built-in web UI. Perfect for debugging agent behavior, presenting research findings, and understanding knowledge structure.

**Start the dashboard:**

```bash
# Start the API server with dashboard UI (default port: 8420)
omi serve --dashboard

# Custom port
omi serve --dashboard --port 8421

# Development mode with auto-reload
omi serve --dashboard --reload
```

**Open in browser:** [http://localhost:8420/dashboard](http://localhost:8420/dashboard)

#### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Graph Visualization** | Interactive force-directed graph showing memory nodes and relationship edges. Color-coded by memory type (fact, experience, belief, decision). Zoom, pan, and click nodes for details. |
| **Semantic Search** | Real-time search bar with semantic recall. Highlights matching nodes in the graph with relevance scores. Debounced input for smooth performance. |
| **Belief Network** | View all beliefs with confidence levels. Color-coded: green (high ≥0.7), yellow (medium 0.4-0.69), red (low <0.4). Sort by confidence, date, or evidence count. |
| **Storage Statistics** | Dashboard showing memory counts, edge counts, and type distributions. Interactive charts (Pie and Bar) built with Chart.js. Real-time refresh. |
| **Session Timeline** | Chronological view of memory operations with Server-Sent Events (SSE) for real-time updates. Shows stores, recalls, belief updates, and session events as they happen. |

#### Screenshot

![OMI Dashboard](docs/images/dashboard-screenshot.png)
*Memory graph visualization with semantic search highlighting (screenshot coming soon)*

#### Technical Details

- **Frontend**: React 18 + Vite, Cytoscape.js for graph rendering, Chart.js for statistics
- **Backend**: FastAPI with read-only REST API endpoints (`/api/v1/dashboard/*`)
- **Real-time**: Server-Sent Events (SSE) for live timeline updates
- **Performance**: Code splitting, lazy loading, responsive design (desktop/tablet/mobile)
- **Security**: Read-only by default (no write operations from UI)

See [E2E_VERIFICATION_CHECKLIST.md](.auto-claude/specs/015-web-dashboard-for-memory-exploration/E2E_VERIFICATION_CHECKLIST.md) for detailed verification steps and testing guide.

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

## Plugins

OMI's plugin system lets you extend functionality through Python entry points — the same pattern that made pytest, Flask, and Datasette successful ecosystems.

### What You Can Extend

| Plugin Type | Entry Point Group | Use Case |
|-------------|-------------------|----------|
| **Embedding Providers** | `omi.embedding_providers` | Custom models (Cohere, OpenAI, HuggingFace, etc.) |
| **Storage Backends** | `omi.storage_backends` | Alternative databases (PostgreSQL, Redis, Neo4j) |
| **Event Handlers** | `omi.event_handlers` | React to memory events (webhooks, logging, analytics) |

### Quick Start

```bash
# List all installed plugins
omi plugins list

# Install a third-party plugin
pip install omi-embedding-cohere

# Use your custom provider
omi config set embedding.provider=cohere
```

### Create a Plugin

```python
# my_omi_plugin/provider.py
from omi.embeddings import EmbeddingProvider

class MyEmbedder(EmbeddingProvider):
    interface_version = "1.0"  # Required for compatibility

    def embed(self, text: str) -> List[float]:
        # Your implementation
        return self._call_api(text)
```

```toml
# pyproject.toml
[project.entry-points."omi.embedding_providers"]
my-embedder = "my_omi_plugin.provider:MyEmbedder"
```

See [docs/PLUGINS.md](docs/PLUGINS.md) for the complete guide and [examples/omi-embedding-example/](examples/omi-embedding-example/) for a working example.

## Why Not Just Files?

> *"Grep is not understanding. Flat files optimize for writing. Databases optimize for asking questions you haven't thought of yet."* — SandyBlake

OMI combines the best patterns from 50+ working agent implementations:

- **Brosie's Canon**: Three-layer storage (hot/warm/cold)
- **SandyBlake's Palace**: Graph topology + poisoning resistance
- **VesperMolt's Beliefs**: Confidence tracking with evidence chains
- **LordOfTheDance's Capsules**: State verification and recovery

## What's New in 0.3.0

### 🚀 REST API Server
Production-ready FastAPI endpoints for memory operations:
```bash
# Start the server
omi server

# Store a memory
curl -X POST http://localhost:8000/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Key insight", "type": "experience"}'

# Semantic search
curl -X POST http://localhost:8000/recall \
  -d '{"query": "auth bug", "limit": 5}'

# Auto-generated docs at http://localhost:8000/docs
```

### 🌐 Web Dashboard
Interactive memory browser at `http://localhost:8001`:
- Graph visualization of memory relationships
- Semantic search with highlighted results
- Timeline view for daily logs
- Real-time memory statistics

### 🔌 Plugin Architecture
Extend OMI with custom components:
```bash
# List installed plugins
omi plugins list

# Create custom embedding provider
pip install omi-embedding-cohere
omi config set embedding.provider=cohere
```

Support for:
- **Embedding Providers** — Custom models (Cohere, sentence-transformers, etc.)
- **Storage Backends** — PostgreSQL, Redis, Neo4j
- **Event Handlers** — Webhooks, logging, analytics

### 🧪 Enhanced CLI
New commands and global flags:
```bash
# Debug inspection
omi inspect memory <id>          # Detailed metadata
omi inspect graph <node>          # Graph topology

# Global output control
omi recall "auth" --verbose       # Full debug output
omi status --quiet                  # Minimal output

# Shell completion
omi completion bash > ~/.bashrc
```

---

## Features

- **NVIDIA NIM Integration**: baai/bge-m3 embeddings, highest quality
- **Automatic Memory Compression**: LLM-powered summarization reduces token costs while preserving originals
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
