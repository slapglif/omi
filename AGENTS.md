<!-- Generated: 2026-02-10 | Updated: 2026-02-10 -->

# omi

## Purpose
OpenClaw Memory Infrastructure — a Python library providing persistent, tiered memory for AI agents. Stores memories in a SQLite-backed graph with semantic search, belief networks with confidence tracking, and backup/restore. Exposes tools via MCP (Model Context Protocol) for OpenClaw integration and a CLI (`omi`) for direct use.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Build config, dependencies, pytest/coverage/mypy/black settings |
| `plugin.json` | MCP plugin manifest — defines tools, hooks, and config for OpenClaw |
| `CLAUDE.md` | Claude Code guidance — architecture, commands, working protocol |
| `README.md` | User-facing docs — quick start, embedding providers, philosophy |
| `CONTRIBUTING.md` | Contribution guidelines |
| `ROADMAP.md` | Feature roadmap and milestones |
| `CHANGELOG.md` | Version history |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | All source code (see `src/AGENTS.md`) |
| `tests/` | Test suite — pytest with coverage (see `tests/AGENTS.md`) |
| `.github/` | CI/CD and issue templates (see `.github/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Install with `pip install -e ".[dev]"` for editable development.
- The CLI entry point is `omi = "omi.cli:main"` (defined in pyproject.toml).
- Python 3.10+ required. Black at 100 chars, mypy strict.
- Two parallel implementations exist for GraphPalace and BeliefNetwork — see CLAUDE.md for details.

### Testing Requirements
- `pytest` runs all tests with coverage. Must pass >60% (pyproject.toml).
- CI runs on Python 3.10, 3.11, 3.12 via `.github/workflows/test.yml`.
- Use markers to skip: `-m "not nim"` (no API key), `-m "not slow"`, `-m "not integration"`.

### Common Patterns
- 4-tier storage: NOW.md (hot) → Daily Logs → Graph Palace (SQLite) → MoltVault (backup).
- Memory types: `fact`, `experience`, `belief`, `decision`.
- Edge types: `SUPPORTS`, `CONTRADICTS`, `RELATED_TO`, `DEPENDS_ON`, `POSTED`, `DISCUSSED`.
- Recency decay: `score = relevance * 0.7 + exp(-days/30) * 0.3`.

## Dependencies

### External
- `numpy` — Vector operations, cosine similarity
- `requests` — HTTP client for NIM API
- `click` — CLI framework
- `pyyaml` — Config file parsing
- `pytest`, `pytest-asyncio`, `pytest-cov` — Testing (dev)
- `black`, `mypy` — Formatting and type checking (dev)
- `boto3` — S3/R2 backup (optional, for MoltVault cloud)
- `cryptography` — Backup encryption (optional)
- `ollama` — Local embeddings fallback (optional)

<!-- MANUAL: -->
