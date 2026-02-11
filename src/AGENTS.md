<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-10 | Updated: 2026-02-10 -->

# src

## Purpose
Source root for the OMI Python package. Contains the single `omi/` package directory which holds all library code.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `omi/` | Main Python package — CLI, MCP API, storage, graph, security, embeddings (see `omi/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- This is a `setuptools` source layout configured via `[tool.setuptools.packages.find] where = ["src"]` in pyproject.toml.
- All imports are from the `omi` package: `from omi.storage.graph_palace import GraphPalace`.
- Do not add files directly in `src/` — all code goes inside `src/omi/`.

### Testing Requirements
- All code under `src/omi/` is covered by tests in the `tests/` directory at the repo root.
- Coverage source is configured as `src/omi` in pyproject.toml.

<!-- MANUAL: -->
