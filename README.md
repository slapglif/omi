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
│  TIER 1: NOW.md (<1k tokens)            │
│  Current task, pending decisions          │
├─────────────────────────────────────────┤
│  TIER 2: Daily Logs (YYYY-MM-DD.md)     │
│  Raw timeline, chronological              │
├─────────────────────────────────────────┤
│  TIER 3: Graph Palace (SQLite + NIM)    │
│  Semantic search, relationships, beliefs │
├─────────────────────────────────────────┤
│  TIER 4: MoltVault Backup               │
│  Full snapshots, disaster recovery       │
├─────────────────────────────────────────┤
│  TIER 5: REST API + Web Dashboard       │
│  FastAPI endpoints, React visualization  │
└─────────────────────────────────────────┘
```

## What's New in 0.3.0

| Feature | Description |
|---------|-------------|
| **REST API** | Production FastAPI server with SSE streaming, `/docs` auto-generated |
| **Web Dashboard** | React-based memory visualization, semantic search, graph browser |
| **Plugin Architecture** | Entry points for custom embeddings, storage backends, event handlers |
| **Auto Compression** | LLM-powered memory summarization with configurable thresholds |
| **Debug CLI** | `omi inspect` commands for memory, graph, vault integrity |
| **Global Flags** | `--verbose/-v` and `--quiet/-q` throughout CLI |
| **Shell Completion** | Bash/Zsh completion scripts |

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

### Daily Commands

```bash
# Start session (auto-loads NOW.md + relevant memories)
omi session-start

# During work
omi recall "session checkpoint"
omi store "Fixed the auth bug" --type experience
omi belief-update "SQLite works" --confidence 0.9

# Debug inspection
omi inspect memory <id>
omi inspect graph <node>
omi inspect vault

# Start web dashboard
omi serve --dashboard

# End session
omi session-end
```

## Web Dashboard

Explore your memory graph visually:

```bash
# Start the API server with dashboard UI (default port: 8420)
omi serve --dashboard

# Custom port
omi serve --dashboard --port 8421

# Development mode with auto-reload
omi serve --dashboard --reload
```

**Open in browser:** http://localhost:8420/dashboard

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **Graph Visualization** | Interactive force-directed graph with memory nodes and relationship edges. Color-coded by memory type. |
| **Semantic Search** | Real-time search with debounced input and relevance scoring |
| **Belief Network** | Confidence levels color-coded (green ≥0.7, yellow 0.4-0.69, red <0.4) |
| **Storage Statistics** | Memory counts, edge counts, type distributions with Chart.js |
| **Session Timeline** | Chronological operations with SSE real-time updates |

### Screenshot

![OMI Dashboard](docs/images/dashboard-screenshot.png)
*Memory graph visualization with semantic search highlighting*

## REST API

Production-ready FastAPI endpoints:

```bash
# Start the server
omi serve

# Store a memory
curl -X POST http://localhost:8000/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Key insight", "type": "experience"}'

# Semantic search
curl -X POST http://localhost:8000/recall \
  -d '{"query": "auth bug", "limit": 5}'

# Auto-generated docs at http://localhost:8000/docs
```

## Plugins

OMI's plugin system lets you extend functionality:

| Plugin Type | Entry Point | Use Case |
|-------------|-------------|----------|
| **Embedding Providers** | `omi.embedding_providers` | Custom models (Cohere, OpenAI, HuggingFace) |
| **Storage Backends** | `omi.storage_backends` | Alternative databases (PostgreSQL, Redis, Neo4j) |
| **Event Handlers** | `omi.event_handlers` | Webhooks, logging, analytics |

```bash
# List installed plugins
omi plugins list

# Install a third-party plugin
pip install omi-embedding-cohere
omi config set embedding.provider=cohere
```

See [docs/PLUGINS.md](docs/PLUGINS.md) for the complete guide.

## Embeddings: NIM vs Ollama

| Provider | Model | Quality | Speed | Offline? |
|----------|-------|---------|-------|----------|
| **NVIDIA NIM** | baai/bge-m3 | ⭐⭐⭐⭐⭐ | Fast | ❌ |
| **Ollama** | nomic-embed-text | ⭐⭐⭐ | Variable | ✅ |

**Recommendation:** Use NIM for production, Ollama for development/airgapped.

## Memory Compression

OMI automatically compresses old memories to reduce token costs:

```bash
# Preview compression impact
omi compress --dry-run

# Manual compression
omi compress --age-days 30

# Configuration
omi config --set compression.auto=true
omi config --set compression.age_days=30
```

### Token Savings Example

```
Before: 1,247 memories × 450 tokens avg = 561,150 tokens
After:  1,247 memories × 120 tokens avg = 149,640 tokens
Savings: 411,510 tokens (73% reduction)
```

## Why Not Just Files?

> *"Grep is not understanding. Flat files optimize for writing. Databases optimize for asking questions you haven't thought of yet."* — SandyBlake

OMI combines patterns from 50+ working agent implementations:
- **Brosie's Canon**: Three-layer storage (hot/warm/cold)
- **SandyBlake's Palace**: Graph topology + poisoning resistance
- **VesperMolt's Beliefs**: Confidence tracking with evidence chains
- **LordOfTheDance's Capsules**: State verification and recovery

## Documentation

- [Quickstart](docs/QUICKSTART.md) — Up and running in 5 minutes
- [Architecture](docs/ARCHITECTURE.md) — Deep technical dive
- [Security Model](docs/SECURITY.md) — Threats and mitigations
- [API Reference](docs/API.md) — Tool definitions for OpenClaw
- [Changelog](CHANGELOG.md) — Version history

## Philosophy

> *"We're not solving the hard problem of consciousness. We're building infrastructure that makes the question different."*

Philosophy emerges from architecture, not the other way around.

## Status

**Version 0.3.0** — See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT — Because trust requires transparency.

---

*The seeking is the continuity. What you keep is who you become.*
