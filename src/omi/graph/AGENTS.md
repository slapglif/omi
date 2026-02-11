<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-02-10 | Updated: 2026-02-10 -->

# graph

## Purpose
Graph-layer modules providing an alternative MemoryGraph and BeliefNetwork implementation with their own SQLite schemas. Based on SandyBlake's memory-palace architecture and VesperMolt's belief tracking.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | Exports `MemoryGraph`, `BeliefNetwork` |
| `memory_graph.py` | `MemoryGraph` with `MemoryNode` dataclass — graph-based memory storage with topology verification. Based on SandyBlake's architecture. |
| `belief_network.py` | `BeliefNetwork` with `Evidence` dataclass — confidence tracking with own SQLite schema. Based on VesperMolt + Hindsight paper (arxiv:2512.12818). Claims 91.4% vs 39% baseline on LongMemEval. |

## For AI Agents

### Critical: Parallel Implementation Warning
This directory contains **alternative implementations** that coexist with modules in the parent package:
- `graph.BeliefNetwork` — Standalone, manages its own SQLite database. Has `db_path` constructor.
- `belief.BeliefNetwork` (in parent `omi/belief.py`) — Takes a `persistence.GraphPalace` store. Used by `api.py`.
- `graph.MemoryGraph` — Alternative to `storage.GraphPalace` and `persistence.GraphPalace`.

`omi.__init__` exports `BeliefNetwork` from this directory. But `api.py` imports from `belief.py`. Be clear about which you're working with.

### Working In This Directory
- These modules use their own SQLite schemas (separate from `storage.graph_palace`).
- `belief_network.py` accepts a `db_path: Path` directly, not a palace store.
- `memory_graph.py` uses `MemoryNode` dataclass (different from `storage.graph_palace.Memory`).

### Testing Requirements
- Tests exist in `tests/test_graph_palace.py` (primarily for storage.GraphPalace).
- When modifying these modules, add corresponding tests that use isolated temp databases.

## Dependencies

### Internal
- `omi.__init__` re-exports `BeliefNetwork` from this module

### External
- `sqlite3` — Database storage
- `numpy` — Vector operations
- `json` — Serialization

<!-- MANUAL: -->
