# OMI Architecture

This document describes the architectural design of OMI (OpenClaw Memory Infrastructure), focusing on the tiered storage model, data flow, and key implementation details.

## Overview

OMI implements a **4-tier storage hierarchy** that balances immediate access, semantic retrieval, and long-term persistence. Each tier serves a specific purpose in the memory lifecycle, optimized for different access patterns and retention requirements.

The system is designed to solve six critical failure modes that plague AI agent memory systems:

1. **Compression amnesia** — Losing context during session transitions
2. **Token bloat** — Memory that's too expensive to load
3. **Write-but-don't-read** — Captured but unused information
4. **Stale context** — Acting on outdated information
5. **Memory poisoning** — Corrupted data breaking future sessions
6. **The MochiBot Loop** — Conflicting information causing identity confusion

## 4-Tier Storage Model

```
┌──────────────────────────────────────────────────────────────┐
│ Tier 1: NOW.md                                               │
│ - Hot context (<1k tokens)                                   │
│ - Loaded first on every session start                        │
│ - Current task, pending decisions, immediate context         │
│ - Access: Direct file read (no search)                       │
│ - Latency: ~1ms                                              │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Tier 2: Daily Logs                                           │
│ - Append-only markdown files (YYYY-MM-DD.md)                 │
│ - Raw chronological timeline                                 │
│ - Stored in memory/ directory                                │
│ - Access: Direct read, grep search                           │
│ - Latency: ~10ms per file                                    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Tier 3: Graph Palace                                         │
│ - SQLite database with FTS5 full-text search                 │
│ - Vector embeddings (1024-dim NIM or 768-dim Ollama)         │
│ - Graph edges: SUPPORTS, CONTRADICTS, RELATED_TO, etc.       │
│ - Centrality scoring and BFS traversal                       │
│ - Access: Semantic search, SQL queries                       │
│ - Latency: ~50-200ms depending on search complexity          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Tier 4: MoltVault Backup                                     │
│ - Compressed tar.gz archives                                 │
│ - Optional Fernet encryption (MOLTVAULT_KEY)                 │
│ - Local filesystem or R2/S3 storage                          │
│ - Retention policy with automatic cleanup                    │
│ - Access: Restore only (disaster recovery)                   │
│ - Latency: seconds to minutes                                │
└──────────────────────────────────────────────────────────────┘
```

### Tier 1: NOW.md (Hot Context)

**Purpose:** Immediate working memory for the current session.

**Characteristics:**
- **Size limit:** <1,000 tokens (~4KB)
- **Format:** Markdown
- **Location:** `~/.openclaw/omi/NOW.md`
- **Update frequency:** Throughout session
- **Loaded:** First thing on `omi session-start`

**Content:**
- Current task description
- Pending decisions
- Session-specific context
- Recent findings that need immediate attention

**Implementation:**
- Plain file I/O (no database)
- Integrity checked with `.now.hash` (SHA-256)
- Atomically updated to prevent corruption

**Access Pattern:**
```python
# Session start: Load entire file
context = path.read_text()

# During session: Update sections
update_now_section("Current Task", new_task)

# Pre-compression: Save critical state
checkpoint_now()
```

### Tier 2: Daily Logs (Warm Storage)

**Purpose:** Chronological record of all session activity.

**Characteristics:**
- **File naming:** `YYYY-MM-DD.md` (e.g., `2024-01-15.md`)
- **Format:** Append-only markdown
- **Location:** `~/.openclaw/omi/memory/`
- **Retention:** Indefinite (pruned only by MoltVault policies)

**Content Structure:**
```markdown
# 2024-01-15

## 09:30 - Session Start
- Loaded context from NOW.md
- Retrieved 5 relevant memories

## 10:15 - Development
- Fixed authentication bug in login flow
- Added rate limiting to API endpoints

## 12:00 - Session End
- Stored 3 new experiences
- Created checkpoint before compression
```

**Implementation:**
- Append-only writes (no in-place edits)
- Integrity checked with `.memory.hash`
- Indexed by date for fast retrieval
- Written on `omi session-end`

**Access Pattern:**
```python
# Append during session
daily_log.append(f"## {timestamp} - {event}\n{details}")

# Search across days
grep_logs(start_date, end_date, pattern)

# Replay session history
replay_day("2024-01-15")
```

### Tier 3: Graph Palace (Semantic Storage)

**Purpose:** Structured, searchable memory graph with semantic relationships.

**Characteristics:**
- **Database:** SQLite with WAL mode
- **Location:** `~/.openclaw/omi/palace.sqlite`
- **Full-text search:** FTS5 virtual table
- **Vector search:** Cosine similarity on embeddings
- **Graph structure:** Nodes (memories) + Edges (relationships)

**Schema Overview:**

```sql
-- Core memories table
CREATE TABLE memories (
    id TEXT PRIMARY KEY,              -- UUID
    content TEXT NOT NULL,
    memory_type TEXT CHECK(memory_type IN ('fact', 'experience', 'belief', 'decision')),
    created_at REAL,
    last_accessed REAL,
    access_count INTEGER DEFAULT 0,
    embedding BLOB,                   -- float32 array
    tags TEXT,                        -- JSON array
    context TEXT,                     -- JSON metadata
    centrality_score REAL DEFAULT 0.0
);

-- Full-text search index
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    tags,
    content='memories',
    content_rowid='id'
);

-- Relationship edges
CREATE TABLE edges (
    from_id TEXT,
    to_id TEXT,
    edge_type TEXT CHECK(edge_type IN ('SUPPORTS', 'CONTRADICTS', 'RELATED_TO', 'DEPENDS_ON')),
    weight REAL DEFAULT 1.0,
    created_at REAL,
    PRIMARY KEY (from_id, to_id, edge_type)
);
```

**Search Methods:**

1. **FTS5 Full-Text Search**
   ```python
   # Fast keyword matching
   results = graph.search_fts("authentication bug", limit=10)
   ```

2. **Vector Semantic Search**
   ```python
   # Cosine similarity on embeddings
   query_embedding = embedder.embed("login issues")
   results = graph.vector_search(query_embedding, limit=10)
   ```

3. **Graph Traversal**
   ```python
   # BFS from a memory node
   related = graph.traverse_from(memory_id, max_depth=2)
   ```

**Recency Decay:**

Relevance scores are adjusted by recency:

```python
def compute_final_score(relevance: float, created_at: float) -> float:
    days_ago = (time.time() - created_at) / 86400
    recency = exp(-days_ago / 30)  # Half-life of 30 days
    return (relevance * 0.7) + (recency * 0.3)
```

**Centrality Scoring:**

Centrality measures how "important" a memory is based on graph topology:

```python
def update_centrality():
    # PageRank-style centrality
    for node in graph.nodes:
        incoming = sum(edge.weight for edge in node.incoming_edges)
        outgoing = sum(edge.weight for edge in node.outgoing_edges)
        node.centrality = (incoming + outgoing) / 2
```

**Two Parallel Implementations:**

⚠️ **Critical:** OMI has **two separate GraphPalace classes**:

| Module | Location | Purpose |
|--------|----------|---------|
| `persistence.GraphPalace` | `src/omi/persistence.py` | Minimal stub with LIKE search. Used by `api.py` and `cli.py`. |
| `storage.GraphPalace` | `src/omi/storage/graph_palace.py` | Full implementation with FTS5, vector search, BFS. |

The `__init__.py` re-exports `storage.GraphPalace` as the primary implementation, but some modules still import from `persistence.py` directly. When modifying GraphPalace code, verify which implementation you're working with.

The `BeliefNetwork` class is in `src/omi/belief.py` and is used by MCP tools. It takes a `GraphPalace` instance.

### Tier 4: MoltVault Backup (Cold Storage)

**Purpose:** Disaster recovery and long-term archival.

**Characteristics:**
- **Format:** Compressed tar.gz archives
- **Encryption:** Optional Fernet (symmetric, password-based)
- **Destinations:** Local filesystem, Cloudflare R2, AWS S3
- **Retention:** Configurable policies (e.g., keep daily for 7 days, weekly for 4 weeks)

**Backup Contents:**
- `palace.sqlite` (Graph Palace database)
- `NOW.md` (current hot context)
- `config.yaml` (configuration)
- `memory/*.md` (all daily logs)
- `embeddings/*.npy` (cached embeddings)

**Implementation:**

```python
class MoltVault:
    def backup(self, include_embeddings=True):
        # Create tar.gz archive
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add("palace.sqlite")
            tar.add("NOW.md")
            tar.add("config.yaml")
            tar.add("memory/")
            if include_embeddings:
                tar.add("embeddings/")

        # Encrypt if key provided
        if self.encryption_key:
            encrypted = fernet.encrypt(backup_path.read_bytes())
            backup_path.write_bytes(encrypted)

        # Upload to R2/S3 if configured
        if self.remote_storage:
            self.upload_to_remote(backup_path)

        # Apply retention policy
        self.cleanup_old_backups()
```

**Restore Process:**

```bash
# Restore from latest backup
omi restore --latest

# Restore from specific backup
omi restore --backup vault/backup-2024-01-15-1200.tar.gz

# Restore from R2
omi restore --remote --backup backup-2024-01-15-1200.tar.gz
```

## Data Flow

### Session Start Flow

```
1. omi session-start
   ↓
2. Load NOW.md (Tier 1)
   ↓
3. Extract key terms from NOW.md
   ↓
4. Query Graph Palace (Tier 3) for relevant memories
   ↓
5. Rank by relevance + recency + centrality
   ↓
6. Display top 5-10 memories to agent
   ↓
7. Agent begins work with full context
```

### During Session Flow

```
Agent discovers new information
   ↓
omi store "content" --type experience
   ↓
1. Generate embedding (NIM or Ollama)
   ↓
2. Store in Graph Palace (Tier 3)
   ↓
3. Create edges to related memories
   ↓
4. Update centrality scores
   ↓
5. Cache embedding to disk
```

### Pre-Compression Checkpoint

```
Context window approaching limit (70%)
   ↓
omi check (or auto-triggered)
   ↓
1. Extract critical state
   ↓
2. Update NOW.md with compressed summary
   ↓
3. Create vault snapshot
   ↓
4. Mark checkpoint timestamp
   ↓
5. Agent continues with fresh context
```

### Session End Flow

```
omi session-end
   ↓
1. Append session summary to daily log (Tier 2)
   ↓
2. Update memory access counts in Graph Palace
   ↓
3. Trigger vault backup (Tier 4) if enabled
   ↓
4. Update NOW.md with next session context
   ↓
5. Compute integrity hashes
```

## Belief Network

OMI includes a **Belief Network** for tracking confidence in statements over time, with asymmetric updates that weight contradictions more heavily than supporting evidence.

**Asymmetric EMA Update:**

```python
# Supporting evidence: λ = 0.15 (slow increase)
if evidence_type == "support":
    new_confidence = (1 - 0.15) * old_confidence + 0.15 * new_evidence

# Contradicting evidence: λ = 0.30 (fast decrease, 2x impact)
if evidence_type == "contradict":
    new_confidence = (1 - 0.30) * old_confidence + 0.30 * (1 - new_evidence)
```

**Rationale:** Contradictions should erode confidence faster than confirmations build it (epistemic humility).

**Schema:**

```sql
CREATE TABLE beliefs (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    confidence REAL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    supporting_memories TEXT,  -- JSON array of memory IDs
    contradicting_memories TEXT,
    last_updated REAL
);
```

## Security & Integrity

### Integrity Verification

**SHA-256 Hashing:**
- `NOW.md` → `.now.hash`
- `memory/` directory → `.memory.hash`
- Checked on session start and before critical operations

**Topology Verification:**

```python
class TopologyVerifier:
    def verify(self):
        issues = []

        # Check for orphan nodes
        orphans = graph.find_orphans()
        if orphans:
            issues.append(f"Found {len(orphans)} orphan memories")

        # Check for sudden centrality changes
        high_centrality = graph.find_high_centrality(threshold=10.0)
        if high_centrality:
            issues.append(f"Suspicious centrality spikes")

        # Check for hash mismatches
        for memory in graph.all_memories():
            if not verify_memory_hash(memory):
                issues.append(f"Hash mismatch: {memory.id}")

        return issues
```

### Poison Detection

**PoisonDetector** runs a unified audit combining:
- Integrity checks (file hashes)
- Topology verification (graph anomalies)
- Consistency checks (belief contradictions)

```bash
# Run full audit
omi audit

# Output format
✓ Integrity: PASS (all hashes valid)
✓ Topology: PASS (no orphans, centrality normal)
⚠ Consistency: 2 warnings (conflicting beliefs detected)
```

### Consensus Management

For multi-instance deployments, **ConsensusManager** implements voting-based agreement:

```python
class ConsensusManager:
    def vote_on_memory(self, memory_id: str, instances: List[str]):
        votes = []
        for instance in instances:
            vote = instance.verify_memory(memory_id)
            votes.append(vote)

        # Majority consensus
        return sum(votes) > len(votes) / 2
```

## Module Reference

### Core Modules

- **`api.py`** — MCP tool classes for OpenClaw integration
  - `MemoryTools`: store, recall
  - `BeliefTools`: update, query
  - `CheckpointTools`: create, restore
  - `SecurityTools`: audit, verify
  - `DailyLogTools`: append, search

- **`cli.py`** — Command-line interface (Click-based)
  - Commands: `init`, `session-start`, `session-end`, `store`, `recall`, `check`, `status`, `audit`
  - Config management: `config set`, `config get`, `config show`

- **`embeddings.py`** — Vector embedding generation
  - `NIMEmbedder`: NVIDIA NIM API (baai/bge-m3, 1024-dim)
  - `OllamaEmbedder`: Local Ollama (nomic-embed-text, 768-dim)
  - `EmbeddingCache`: Disk cache using `.npy` files

- **`security.py`** — Integrity and verification
  - `IntegrityChecker`: SHA-256 file hashing
  - `TopologyVerifier`: Graph anomaly detection
  - `PoisonDetector`: Unified audit
  - `ConsensusManager`: Multi-instance voting

- **`moltvault.py`** — Backup and restore
  - `MoltVault`: R2/S3 backup with encryption
  - Retention policies
  - Automatic cleanup

### Storage Modules

- **`persistence.py`** — Minimal GraphPalace stub
  - Used by `api.py` and `cli.py`
  - Basic LIKE search
  - Simple edge management

- **`storage/graph_palace.py`** — Full GraphPalace implementation
  - FTS5 full-text search
  - Vector cosine similarity
  - BFS traversal
  - Centrality scoring

### Graph Modules

- **`belief.py`** — BeliefNetwork for MCP tools
  - Takes a `GraphPalace` instance
  - Asymmetric EMA updates

## Configuration

Default configuration location: `~/.openclaw/omi/config.yaml`

```yaml
embedding:
  provider: nim  # or ollama
  model: baai/bge-m3
  dimensions: 1024
  api_key: ${NIM_API_KEY}

vault:
  enabled: true
  local_path: vault/
  remote_storage: r2  # or s3
  encryption: true
  retention:
    daily: 7
    weekly: 4
    monthly: 6

graph:
  recency_weight: 0.3
  relevance_weight: 0.7
  recency_halflife: 30  # days

belief:
  support_lambda: 0.15
  contradict_lambda: 0.30
```

Override with:
- `--data-dir` CLI flag
- `OMI_BASE_PATH` environment variable

## Performance Characteristics

### Latency by Tier

| Tier | Operation | Typical Latency |
|------|-----------|-----------------|
| Tier 1 (NOW.md) | Read | ~1ms |
| Tier 1 (NOW.md) | Write | ~2ms |
| Tier 2 (Daily Log) | Append | ~5ms |
| Tier 2 (Daily Log) | Grep search | ~10-50ms |
| Tier 3 (FTS5) | Full-text search | ~20-100ms |
| Tier 3 (Vector) | Semantic search (10k memories) | ~100-200ms |
| Tier 3 (Graph) | BFS traversal (depth 2) | ~50ms |
| Tier 4 (Vault) | Backup create | ~1-5s |
| Tier 4 (Vault) | Restore | ~2-10s |

### Scalability

- **Memories:** Tested up to 100,000 memories without degradation
- **Embeddings:** Cached on disk, loaded lazily
- **SQLite:** WAL mode handles concurrent reads
- **FTS5:** Scales logarithmically with memory count

## Design Rationale

### Why Tiered Storage?

1. **Different access patterns:** Hot context needs instant access; old memories are rarely retrieved
2. **Cost optimization:** Embeddings are expensive to generate; cache aggressively
3. **Reliability:** Multiple tiers provide redundancy
4. **Performance:** Fast path for common operations (NOW.md), slow path for deep search

### Why SQLite?

1. **Zero-copy:** Embeddings stored as BLOBs, memory-mapped for speed
2. **ACID guarantees:** No risk of corrupted state
3. **FTS5 integration:** Best-in-class full-text search
4. **Single file:** Easy to backup and transfer
5. **No server:** Embedded, no deployment complexity

### Why Asymmetric Belief Updates?

1. **Epistemic humility:** Easier to disprove than prove
2. **Prevents overconfidence:** Slow to believe, quick to doubt
3. **Aligns with scientific method:** Falsification > confirmation
4. **Practical:** Prevents runaway confidence from repeated similar evidence

## Future Considerations

### Potential Enhancements

1. **Distributed consensus:** Multi-agent memory sharing with CRDTs
2. **Compression:** LZ4 compression for daily logs and old memories
3. **Partitioning:** Shard Graph Palace by time or memory type
4. **Read replicas:** SQLite read-only mirrors for scale-out
5. **Event sourcing:** Append-only event log for full auditability

### Migration Path

If schema changes are needed:

```python
# Version-aware migrations
def migrate_v1_to_v2(db):
    db.execute("ALTER TABLE memories ADD COLUMN new_field TEXT")
    db.execute("UPDATE schema_version SET version = 2")
```

Migrations tracked in `schema_version` table.

## References

- [SQLite FTS5 Documentation](https://www.sqlite.org/fts5.html)
- [NVIDIA NIM Embedding API](https://docs.api.nvidia.com/nim/)
- [Ollama Embedding Models](https://ollama.com/library/nomic-embed-text)
- [Fernet Encryption Spec](https://github.com/fernet/spec/blob/master/Spec.md)
