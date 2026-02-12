# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OMI (OpenClaw Memory Infrastructure) is a Python library providing persistent, tiered memory for AI agents. It stores memories in a SQLite-backed graph with semantic search, belief networks with confidence tracking, and backup/restore via MoltVault. It exposes tools via MCP (Model Context Protocol) for OpenClaw integration and a CLI (`omi`) for direct use.

## Build & Test Commands

```bash
# Install from source (editable)
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run a single test file
pytest tests/test_graph_palace.py

# Run a specific test
pytest tests/test_graph_palace.py::TestGraphPalace::test_store_memory -v

# Skip tests requiring NIM API key or slow tests
pytest -m "not nim" -m "not slow"

# Type checking
mypy src/omi

# Formatting
black src/ tests/
```

Coverage is configured to fail under 60% (`pyproject.toml [tool.coverage.report]`). Test markers: `nim`, `slow`, `integration`.

## Architecture

### 4-Tier Storage Model

```
Tier 1: NOW.md    — Hot context (<1k tokens), loaded first on session start
Tier 2: Daily Logs — Append-only markdown files (YYYY-MM-DD.md) in memory/ dir
Tier 3: Graph Palace — SQLite with FTS5, vector embeddings, edges, centrality
Tier 4: MoltVault  — tar.gz backup to local filesystem or R2/S3
```

### Two Parallel Implementations

There are **two GraphPalace classes** — this is the most important architectural detail:

| Module | Location | Purpose |
|--------|----------|---------|
| `persistence.GraphPalace` | `src/omi/persistence.py` | Minimal stub with basic LIKE search. Used by `api.py` and `cli.py` imports. |
| `storage.GraphPalace` | `src/omi/storage/graph_palace.py` | Full implementation with FTS5, vector search, cosine similarity, BFS traversal, centrality scoring. |

The `BeliefNetwork` class is in `src/omi/belief.py` and is used by `api.py` MCP tools. It takes a `GraphPalace` instance.

The `__init__.py` re-exports from `storage.graph_palace` as the primary implementation, but `api.py` and `cli.py` import from `persistence.py`. Be aware of which class you're working with.

### Key Modules

- **`api.py`** — MCP tool classes: `MemoryTools`, `BeliefTools`, `CheckpointTools`, `SecurityTools`, `DailyLogTools`. Entry point: `get_all_mcp_tools(config)` returns a dict of callable tools for OpenClaw registration.
- **`cli.py`** — Click-based CLI. Commands: `init`, `session-start`, `session-end`, `store`, `recall`, `check`, `status`, `audit`, `config {set,get,show}`.
- **`embeddings.py`** — `NIMEmbedder` (NVIDIA NIM baai/bge-m3, 1024-dim) with `OllamaEmbedder` fallback (nomic-embed-text, 768-dim). `EmbeddingCache` persists to `.npy` files.
- **`security.py`** — `IntegrityChecker` (SHA-256 file hashes), `TopologyVerifier` (orphan nodes, sudden cores, hash mismatches), `PoisonDetector` (unified audit), `ConsensusManager` (multi-instance voting).
- **`moltvault.py`** — `MoltVault` class with R2/S3 backup via boto3, optional Fernet encryption via `MOLTVAULT_KEY`, retention policy cleanup. Also `VaultBackup` in `persistence.py` for local-only tar.gz backup.

### Data Flow

1. **Session start**: `omi session-start` → loads NOW.md → queries Graph Palace for relevant memories
2. **During work**: `omi store` writes to Graph Palace (SQLite). `omi recall` does FTS5 search or vector cosine similarity.
3. **Belief updates**: EMA with asymmetric lambdas — supporting evidence λ=0.15, contradicting λ=0.30 (contradictions hit 2x harder).
4. **Pre-compression**: `omi check` creates a state capsule checkpoint before context window fills.
5. **Session end**: `omi session-end` → appends to daily log → triggers vault backup if enabled.

### Recency Decay

Score formula: `final = (relevance * 0.7) + (recency * 0.3)` where `recency = exp(-days_ago / 30)`.

### Data Paths

Default base: `~/.openclaw/omi/`. Override with `--data-dir` flag or `OMI_BASE_PATH` env var.

- `palace.sqlite` — Graph Palace database (WAL mode)
- `NOW.md` — Hot context
- `config.yaml` — Configuration
- `memory/` — Daily log files
- `embeddings/` — Cached `.npy` embedding files
- `vault/` — Local backup archives
- `.now.hash` / `.memory.hash` — Integrity checksums

## Plugin Configuration

`plugin.json` defines MCP tools for OpenClaw: `memory_recall`, `memory_store`, `belief_update`, `now_read`, `now_update`, `integrity_check`. Entry point: `omi.api:get_all_mcp_tools`.

## Environment Variables

- `NIM_API_KEY` — NVIDIA NIM API key (for production embeddings)
- `MOLTVAULT_KEY` — Encryption passphrase for encrypted backups
- `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` — Cloudflare R2 credentials for cloud backup
- `OMI_BASE_PATH` — Override default data directory

## Working Protocol

### Hermeneutic Circle Thinking

Apply iterative understanding at every level of work. Read the whole before the parts, then re-read the parts to revise understanding of the whole. Specifically:

1. **Before starting work**: Reread the user's input prompt carefully. State back the objective in your own words to confirm understanding. Do not paraphrase loosely — capture the actual intent.
2. **During work**: Continuously revisit assumptions as new details emerge. If implementing a feature reveals the original understanding was incomplete, stop and re-interpret before continuing.
3. **After completing work**: State the objective again and confirm what was delivered against it. Identify any drift between what was asked and what was produced.

### Challenge Wishful Thinking

Actively resist optimistic assumptions:

- If something "should work" but hasn't been tested, it doesn't work yet.
- If a fix "probably handles" an edge case, verify it explicitly.
- If an implementation "looks complete", re-read the requirements and diff against them.
- When estimating scope, assume the pessimistic case. The first approach rarely works.
- Name what you're uncertain about rather than glossing over it.

### Decompose and Parallelize

Always look for opportunities to break work into independent units that can run concurrently:

- Before implementing, decompose the task into subtasks. Identify which are independent (parallelizable) vs. which have dependencies (sequential).
- Use parallel subagents for independent research, analysis, or implementation tasks.
- Prefer many small focused agents over one monolithic pass.
- When facing 3+ independent tasks, dispatch them in parallel rather than sequentially.

### Commit Everything to Memory

Persist all meaningful context — decisions, patterns, failures, and lessons:

- After completing work, save key findings, architectural decisions, and gotchas to auto-memory files.
- When a non-obvious bug is found and fixed, record the root cause and solution.
- When a pattern is confirmed across multiple interactions, write it down.
- Err on the side of recording too much rather than too little. Memory is cheap; re-discovery is expensive.

## Conventions

- Python 3.10+. Black formatting at 100 char line length.
- Memory types: `fact`, `experience`, `belief`, `decision` (enforced by CHECK constraints in SQLite).
- Edge types: `SUPPORTS`, `CONTRADICTS`, `RELATED_TO`, `DEPENDS_ON`, `POSTED`, `DISCUSSED`.
- Embeddings stored as binary blobs (float32 packed via `struct`).
- FTS5 uses a standalone virtual table (`memories_fts`) because memory IDs are TEXT UUIDs, not INTEGER rowids.
