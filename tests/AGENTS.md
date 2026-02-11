<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-10 | Updated: 2026-02-10 -->

# tests

## Purpose
Pytest test suite for the OMI package. Covers all tiers of the storage model, CLI commands, MCP integration, and security verification.

## Key Files

| File | Description |
|------|-------------|
| `conftest.py` | Shared fixtures — `temp_omi_setup` (temp dirs), `mock_embedder`, `persistence_stores`, `belief_network_setup`, `security_setup`, sample data |
| `test_smoke.py` | Smoke tests — verifies all modules import and basic instantiation works |
| `test_graph_palace.py` | Graph Palace (Tier 3) — store, recall, edges, centrality, BFS traversal, FTS5 search |
| `test_moltvault.py` | MoltVault (Tier 4) — backup creation, restore with integrity verification |
| `test_cli.py` | CLI commands — init, session-start, session-end, store, recall, status, audit |
| `test_nim_integration.py` | NIM embedding API integration (requires `NIM_API_KEY`, marker: `nim`) |
| `test_mcp_integration.py` | MCP tool wiring — memory_recall, memory_store, belief_update, now_read |
| `test_unit_persistence.py` | Persistence layer unit tests — NOWStore, DailyLogStore, stub GraphPalace |

## For AI Agents

### Working In This Directory
- All tests use `tmp_path` or `temp_omi_setup` fixture for isolation — never write to real `~/.openclaw/omi/`.
- `conftest.py` provides `mock_embedder` returning 768-dim vectors and `mock_vault` for backup tests.
- Test markers: `nim` (needs API key), `slow`, `integration`. CI runs with `-m "not nim and not integration"`.
- Add new test files as `test_*.py` — pytest auto-discovers via `python_files = ["test_*.py"]`.

### Testing Requirements
- Run `pytest` from repo root. Coverage must stay above 60%.
- Run `pytest tests/test_smoke.py` as a quick sanity check after changes.
- For NIM tests: `pytest -m nim` (requires `NIM_API_KEY` env var).

### Common Patterns
- Fixtures create isolated temp directories with SQLite databases.
- Use `mock_embedder` instead of real embedding calls for unit tests.
- Test classes follow `Test*` naming, test functions `test_*`.

## Dependencies

### Internal
- `src/omi/` — All modules under test

### External
- `pytest` — Test framework
- `pytest-asyncio` — Async test support
- `pytest-cov` — Coverage reporting
- `responses` — HTTP mocking for NIM/Ollama tests

<!-- MANUAL: -->
