<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-10 | Updated: 2026-02-10 -->

# storage

## Purpose
Full implementations of Tier 1 (NOW.md) and Tier 3 (Graph Palace) storage. These are the production-quality versions with FTS5 full-text search, vector similarity, graph traversal, and centrality scoring.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Exports `NowStorage`, `GraphPalace`, `Memory`, `Edge` |
| `graph_palace.py` | **Core of OMI's intelligence.** SQLite with WAL mode, FTS5, binary embeddings (float32 via struct), cosine similarity, BFS traversal, centrality scoring (degree 40% + access 35% + recency 25%). ~838 lines. |
| `now.py` | `NowStorage` — NOW.md lifecycle: read, update, checkpoint. Manages heartbeat state JSON. |

## For AI Agents

### Working In This Directory
- `graph_palace.py` is the most complex module. It manages its own SQLite schema with CHECK constraints, indexes, and a standalone FTS5 virtual table.
- FTS5 uses a separate `memories_fts` table (not content-synced) because memory IDs are TEXT UUIDs, not INTEGER rowids. When storing/deleting memories, both `memories` and `memories_fts` must be updated together.
- Embeddings are packed as binary blobs: `struct.pack(f'{n}f', *embedding)` and unpacked with `struct.unpack`.
- `GraphPalace` caches embeddings in-memory (`_embedding_cache` dict) for fast recall.
- `now.py` writes heartbeat state to `heartbeat-state.json` alongside NOW.md.

### Testing Requirements
- Primary tests in `tests/test_graph_palace.py`.
- All tests use `tmp_path` fixtures — never touches real databases.
- Test vector operations with known embeddings to verify cosine similarity math.

### Common Patterns
- Context manager support: `with GraphPalace(path) as palace:`.
- Centrality formula: `0.4 * degree + 0.35 * log_access + 0.25 * recency_decay`.
- Recall scoring: `final = min(relevance * 0.7 + recency * 0.3, 1.0)`.
- Valid memory types: `fact`, `experience`, `belief`, `decision` (SQL CHECK).
- Valid edge types: `SUPPORTS`, `CONTRADICTS`, `RELATED_TO`, `DEPENDS_ON`, `POSTED`, `DISCUSSED` (SQL CHECK).

## Dependencies

### Internal
- Used by `omi.api` (MCP tools) and `omi.cli` (CLI commands)
- `omi.__init__` re-exports `GraphPalace` from here as the primary implementation

### External
- `numpy` — Cosine similarity calculations
- `sqlite3` — Standard library, WAL mode

<!-- MANUAL: -->
