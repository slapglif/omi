# Quick Start Guide

Get OMI up and running in 5 minutes.

## Prerequisites

Choose your embedding provider:

### Option 1: NVIDIA NIM (Recommended)

Best for production — highest quality embeddings with consistent performance.

```bash
# Get your API key from https://integrate.api.nvidia.com
export NIM_API_KEY="nvapi-..."

# Add to your shell profile for persistence
echo 'export NIM_API_KEY="nvapi-..."' >> ~/.bashrc
```

### Option 2: Ollama (Local/Airgapped)

Best for development or offline environments.

```bash
# Install Ollama first: https://ollama.ai
ollama pull nomic-embed-text
```

## Installation

### From PyPI (Recommended)

```bash
# With NIM support
pip install omi-openclaw[nim]

# Or minimal install (Ollama only)
pip install omi-openclaw
```

### From Source

```bash
git clone https://github.com/slapglif/omi.git
cd omi
pip install -e ".[nim]"

# Run tests to verify
pytest
```

## Initialize Your Memory Palace

```bash
# Create the memory infrastructure
omi init

# Output:
# ✓ Created ~/.openclaw/omi/
# ✓ Initialized palace.sqlite
# ✓ Created NOW.md
# ✓ Created memory/ directory
# ✓ Generated config.yaml
```

## Configure Embeddings

### For NIM (Recommended)

```bash
# Edit the config file
cat > ~/.openclaw/omi/config.yaml << 'EOF'
embedding:
  provider: nim
  model: baai/bge-m3
  dimensions: 1024
  api_key: ${NIM_API_KEY}

storage:
  max_now_tokens: 1000
  recency_half_life_days: 30

vault:
  enabled: true
  provider: local
  retention_days: 90
EOF
```

### For Ollama (Offline)

```bash
omi config set embedding.provider ollama
omi config set embedding.model nomic-embed-text
omi config set embedding.dimensions 768
```

## Daily Workflow

### 1. Start Your Session

```bash
omi session-start

# This automatically:
# - Loads NOW.md into context
# - Retrieves relevant memories from Graph Palace
# - Checks system integrity
# - Creates today's daily log file
```

### 2. Store Memories During Work

```bash
# Store a fact
omi store "SQLite supports full-text search via FTS5" --type fact

# Store an experience
omi store "Fixed the auth bug by validating tokens earlier" --type experience

# Store a decision
omi store "Chose pytest over unittest for better fixtures" --type decision

# With metadata
omi store "Migrated to async/await API" --type experience --tags "async,refactor"
```

### 3. Recall Information

```bash
# Semantic search with NIM embeddings
omi recall "authentication issues"

# Fuzzy text search (faster, works without embeddings)
omi recall "pytest" --mode fts

# Limit results and set relevance threshold
omi recall "database performance" --limit 10 --min-relevance 0.7

# Get specific memory by ID
omi recall --id "mem_abc123"
```

### 4. Update Beliefs

```bash
# Update confidence based on evidence
omi belief-update "Async API is faster than sync" --confidence 0.95

# Record contradicting evidence (decreases confidence)
omi belief-update "SQLite is bad for concurrency" --confidence 0.3

# View all beliefs
omi belief-list --min-confidence 0.5
```

### 5. Pre-Compression Checkpoints

```bash
# Manually create a state capsule (auto-triggers at 70% context)
omi check

# This creates:
# - Snapshot of NOW.md
# - Current belief states
# - Recent memory summary
# - Hash verification
```

### 6. End Your Session

```bash
omi session-end

# This automatically:
# - Appends session summary to daily log
# - Triggers vault backup (if enabled)
# - Updates memory statistics
# - Verifies integrity
```

## Verify System Health

```bash
# Quick status check
omi status

# Full integrity audit
omi audit

# Output:
# ✓ NOW.md hash verified
# ✓ Daily logs hash verified
# ✓ Graph Palace topology verified
# ✓ No orphan nodes detected
# ✓ Belief network consistent
```

## Using the Python API

### Sync API (Simple)

```python
from omi import MemorySession

# Initialize session
session = MemorySession()

# Store a memory
memory_id = session.store(
    content="Discovered async API performance benefits",
    memory_type="experience"
)

# Recall memories
results = session.recall(
    query="async performance",
    limit=5,
    min_relevance=0.7
)

for memory in results:
    print(f"{memory.content} (relevance: {memory.relevance_score})")

# Update beliefs
session.belief_update(
    statement="Async operations improve throughput",
    confidence=0.9
)

# Clean up
session.close()
```

### Async API (High Performance)

For batch operations and concurrent workloads — **4x faster** than sync.

```python
import asyncio
from omi.async_api import async_session

async def main():
    async with async_session() as session:
        # Store multiple memories concurrently
        memories = [
            "Fixed authentication bug in login flow",
            "Implemented rate limiting for API endpoints",
            "Optimized database query performance"
        ]

        store_tasks = [
            session.memory.store(mem, memory_type="experience")
            for mem in memories
        ]
        memory_ids = await asyncio.gather(*store_tasks)

        # Concurrent recall
        results = await session.memory.recall(
            "authentication issues",
            limit=5,
            min_relevance=0.7
        )

        print(f"Found {len(results)} relevant memories")

asyncio.run(main())
```

## Using the REST API

Start the server:

```bash
uvicorn omi.rest_api:app --reload
```

Access interactive docs at `http://localhost:8000/docs`

### HTTP Examples

```bash
# Store a memory
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Implemented new feature",
    "memory_type": "experience"
  }'

# Recall memories
curl -X POST http://localhost:8000/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "feature implementation",
    "limit": 5
  }'

# Real-time event stream (SSE)
curl -N http://localhost:8000/events/stream
```

## Using with OpenClaw (MCP)

OMI provides Model Context Protocol tools for agent integration.

### Available Tools

- `memory_recall` — Semantic search across Graph Palace
- `memory_store` — Persist facts, experiences, beliefs, decisions
- `belief_update` — Update confidence with evidence
- `now_read` — Read current hot context
- `now_update` — Update hot context
- `integrity_check` — Verify memory consistency

### Configuration

Edit your OpenClaw config to include OMI tools:

```yaml
# ~/.openclaw/config.yaml
plugins:
  - name: omi
    enabled: true
    config:
      data_dir: ~/.openclaw/omi
      embedding_provider: nim
```

## Troubleshooting

### "NIM API key not found"

```bash
# Verify the environment variable is set
echo $NIM_API_KEY

# If empty, export it
export NIM_API_KEY="nvapi-..."

# Make it permanent
echo 'export NIM_API_KEY="nvapi-..."' >> ~/.bashrc
source ~/.bashrc
```

### "Ollama model not found"

```bash
# Pull the model
ollama pull nomic-embed-text

# Verify it's available
ollama list | grep nomic
```

### "Permission denied on config.yaml"

```bash
# Fix permissions
chmod 600 ~/.openclaw/omi/config.yaml
```

### "Database is locked"

```bash
# Check for stale sessions
ps aux | grep omi

# Kill stale processes
pkill -f "omi session"

# If that fails, rebuild the database
omi init --force
```

### Slow recall queries

```bash
# Rebuild FTS5 index
sqlite3 ~/.openclaw/omi/palace.sqlite "INSERT INTO memories_fts(memories_fts) VALUES('rebuild');"

# Or use FTS mode instead of vector search
omi recall "query" --mode fts
```

## Next Steps

- **Architecture Deep Dive**: Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the 4-tier storage model
- **Security Model**: See [SECURITY.md](SECURITY.md) for threat analysis and mitigations
- **API Reference**: Check [API.md](API.md) for complete MCP tool definitions
- **Pattern Library**: Browse [patterns/](patterns/) for community-tested workflows

## Common Patterns

### Pre-Session Memory Priming

```bash
# Before starting work, warm up relevant context
omi recall "previous session checkpoints" --limit 5
omi recall "unresolved issues" --limit 3
omi session-start
```

### End-of-Day Synthesis

```bash
# Summarize the day before ending
omi store "Today: Fixed auth, added tests, deployed to staging" --type experience
omi check  # Create checkpoint
omi session-end
```

### Periodic Integrity Checks

```bash
# Add to cron (daily at 2am)
0 2 * * * cd ~/.openclaw/omi && omi audit
```

### Backup to Cloud

```bash
# Configure R2 backup
omi config set vault.provider r2
omi config set vault.r2_bucket my-omi-backups
omi config set vault.r2_endpoint https://[account-id].r2.cloudflarestorage.com

# Trigger manual backup
omi vault backup

# Restore from backup
omi vault restore --archive vault/backup-2024-01-15.tar.gz
```

---

**You're ready.** The palace remembers what the river forgets.
