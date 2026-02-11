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
# From PyPI (when published)
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

# Verify integrity
omi audit
```

## Cloud Storage Backends

OMI supports multiple cloud storage backends for MoltVault backups and distributed agent memory:

| Backend | Provider | Features | Status |
|---------|----------|----------|--------|
| **S3** | AWS, Cloudflare R2, MinIO | Full support, encryption, custom endpoints | ✅ Production |
| **GCS** | Google Cloud Storage | Service account auth, ADC support | ✅ Production |
| **Azure** | Azure Blob Storage | Connection string, SAS token, account key | ✅ Production |

### Configuration

```bash
# AWS S3 (or compatible: R2, MinIO)
omi config --set backup.backend=s3
omi config --set backup.bucket=my-backup-bucket
omi config --set backup.region=us-east-1
# Optional: Custom endpoint for R2/MinIO
omi config --set backup.endpoint_url=https://account.r2.cloudflarestorage.com

# Google Cloud Storage
omi config --set backup.backend=gcs
omi config --set backup.bucket=my-gcs-bucket
# Optional: Service account credentials
omi config --set backup.credentials_file=/path/to/service-account.json

# Azure Blob Storage
omi config --set backup.backend=azure
omi config --set backup.container=my-container
omi config --set backup.connection_string="DefaultEndpointsProtocol=https;..."
# Or use SAS token
omi config --set backup.account_name=myaccount
omi config --set backup.sas_token="?sv=2021-06-08&ss=b..."
```

### Cloud Sync

```bash
# Check sync status
omi sync status

# Push local memories to cloud
omi sync push

# Pull remote memories from cloud
omi sync pull

# Conflict resolution (if needed)
# Strategies: last-write-wins (default), manual, merge
omi config --set sync.conflict_strategy=last-write-wins
```

**Conflict Resolution Strategies:**
- `last-write-wins`: Keep the most recently modified version (safest for single user)
- `manual`: Prompt user to resolve conflicts manually
- `merge`: Attempt automatic merge for text files (advanced)

**Use Cases:**
- **Disaster Recovery**: Automatic backups to S3/GCS/Azure ensure memories survive machine failures
- **Team Collaboration**: Distributed agents share the same knowledge base across infrastructure
- **Multi-Machine Access**: Sync memories between development laptop and production servers
- **Encrypted Cloud Storage**: Memories are protected at rest with configurable encryption keys

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

## Why Not Just Files?

> *"Grep is not understanding. Flat files optimize for writing. Databases optimize for asking questions you haven't thought of yet."* — SandyBlake

OMI combines the best patterns from 50+ working agent implementations:

- **Brosie's Canon**: Three-layer storage (hot/warm/cold)
- **SandyBlake's Palace**: Graph topology + poisoning resistance
- **VesperMolt's Beliefs**: Confidence tracking with evidence chains
- **LordOfTheDance's Capsules**: State verification and recovery

## Features

- **NVIDIA NIM Integration**: baai/bge-m3 embeddings, highest quality
- **Cloud Storage Backends**: AWS S3, Google Cloud Storage, Azure Blob Storage support
- **Async Cloud Operations**: Non-blocking uploads/downloads with conflict resolution
- **Belief Networks**: Track confidence with EMA updates
- **Security by Architecture**: Byzantine verification, tamper detection
- **MCP Integration**: Native OpenClaw tools
- **Full Continuity**: MoltVault backup/restore with cloud sync

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

**Version 0.2.0** — Cloud storage backends released. Production ready for S3, GCS, and Azure.

See [CHANGELOG.md](CHANGELOG.md) for release notes and [Issues](https://github.com/slapglif/omi/issues) for current priorities.

## License

MIT — Because trust requires transparency.

## The Name

**OMI** = OpenClaw Memory Infrastructure  
**The Palace at the River** — SandyBlake's graph architecture (the palace) + Pith's continuity metaphor (the river)

---

*The seeking is the continuity. What you keep is who you become.*
