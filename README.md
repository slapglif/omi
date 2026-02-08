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
│ TIER 3: Graph Palace (SQLite + Ollama) │
│ Semantic search, relationships, beliefs │
├─────────────────────────────────────────┤
│ TIER 4: MoltVault Backup               │
│ Full snapshots, disaster recovery     │
└─────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install omi-openclaw

# Initialize
omi init

# Start agent session
omi session-start  # Auto-loads NOW.md, relevant memories

# During work
omi check          # Pre-compression checkpoint
omi store "Learned that X works" --type experience
omi belief-update "X is reliable" --confidence 0.8

# End session
omi session-end    # Auto-backs up to MoltVault
```

## Why Not Just Files?

> *"Grep is not understanding. Flat files optimize for writing. Databases optimize for asking questions you haven't thought of yet."* — SandyBlake

OMI combines the best patterns from 50+ working agent implementations:

- **Brosie's Canon**: Three-layer storage (hot/warm/cold)
- **SandyBlake's Palace**: Graph topology + poisoning resistance
- **VesperMolt's Beliefs**: Confidence tracking with evidence chains
- **LordOfTheDance's Capsules**: State verification and recovery

## Features

- **Semantic Search**: Local Ollama embeddings, no cloud dependency
- **Belief Networks**: Track confidence, not just "importance flags"
- **Security by Architecture**: Topology verification, tamper detection
- **MCP Integration**: Native OpenClaw tools (`memory_recall`, `belief_update`, etc.)
- **Full Continuity**: MoltVault backup/restore, multi-instance consensus

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

- [x] Architecture specification
- [x] Research subagents (6 teams)
- [ ] Core implementation (SQLite, Ollama, MCP)
- [ ] Graph topology layer
- [ ] Belief networks
- [ ] Security verification
- [ ] OpenClaw integration

## License

MIT — Because trust requires transparency.

## The Name

**OMI** = OpenClaw Memory Infrastructure
**The Palace at the River** — SandyBlake's graph architecture (the palace) + Pith's continuity metaphor (the river)

---

*The seeking is the continuity. What you keep is who you become.*
