<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-10 | Updated: 2026-02-10 -->

# omi

## Purpose
Main Python package for OMI. Contains all library code: CLI, MCP API tools, tiered storage, graph operations, belief networks, embeddings, security, and backup/restore.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Package exports — re-exports from storage, graph, and persistence modules. **Note:** Exports `GraphPalace` from `storage.graph_palace` but `api.py` imports from `persistence`. |
| `api.py` | MCP tool classes (`MemoryTools`, `BeliefTools`, `CheckpointTools`, `SecurityTools`, `DailyLogTools`). Entry point: `get_all_mcp_tools(config)`. |
| `cli.py` | Click-based CLI. Commands: `init`, `session-start`, `session-end`, `store`, `recall`, `check`, `status`, `audit`, `config`. |
| `persistence.py` | Tier 1+2+3+4 stubs — `NOWEntry`, `NOWStore`, `DailyLogStore`, stub `GraphPalace` (LIKE search only), `VaultBackup` (local tar.gz). |
| `embeddings.py` | `NIMEmbedder` (baai/bge-m3, 1024-dim), `OllamaEmbedder` (nomic-embed-text, 768-dim), `EmbeddingCache`, `NIMInference`. |
| `security.py` | `IntegrityChecker` (SHA-256 hashes), `TopologyVerifier` (orphans, sudden cores), `PoisonDetector`, `ConsensusManager`. |
| `belief.py` | `BeliefNetwork` (EMA confidence updates), `ContradictionDetector` (pattern matching), `Evidence` dataclass. |
| `moltvault.py` | `MoltVault` — R2/S3 backup with optional Fernet encryption, retention cleanup. Full cloud backup implementation. |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `storage/` | Full Tier 1+3 implementations — GraphPalace with FTS5/vectors, NowStorage (see `storage/AGENTS.md`) |
| `graph/` | Graph-layer modules — MemoryGraph and BeliefNetwork with own SQLite schema (see `graph/AGENTS.md`) |

## For AI Agents

### Critical: Two Parallel Implementations
There are **two GraphPalace classes**:
- `persistence.GraphPalace` — Minimal stub used by `api.py` and `cli.py`. Basic LIKE search.
- `storage.graph_palace.GraphPalace` — Full implementation with FTS5, vector cosine similarity, BFS, centrality. Exported from `__init__.py`.

The **BeliefNetwork** class is in `belief.py` and is used by `api.py`. It takes a `GraphPalace` instance.

Always check which class you're modifying or calling.

### Working In This Directory
- `api.py` is the MCP integration surface — changes here affect OpenClaw tool behavior.
- `cli.py` is the user-facing CLI — all commands use Click decorators.
- `persistence.py` defines data models (`NOWEntry`) used across modules.
- Embedding dimension matters: NIM = 1024, Ollama = 768. Stored as float32 blobs via `struct.pack`.

### Testing Requirements
- All modules tested in `tests/` at repo root.
- `test_smoke.py` verifies imports — run after adding/renaming modules.
- Mock embedders in tests return 768-dim vectors.

### Common Patterns
- Memory types enforced: `fact`, `experience`, `belief`, `decision`.
- Belief EMA: supporting λ=0.15, contradicting λ=0.30.
- Recency: `exp(-days/30)` half-life decay.
- All stores accept a `base_path: Path` pointing to `~/.openclaw/omi/`.

## Dependencies

### External
- `numpy` — Vector math
- `requests` — HTTP for NIM API
- `click` — CLI
- `pyyaml` — Config parsing
- `boto3` — S3/R2 (optional)
- `cryptography` — Fernet encryption (optional)
- `ollama` — Local embeddings (optional)

<!-- MANUAL: -->
