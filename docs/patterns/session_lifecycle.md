# Session Lifecycle Pattern

> *"The palace is not a database. It is a conversation partner that remembers."*

This pattern defines the canonical workflow for managing agent memory across work sessions. Following this lifecycle ensures optimal memory persistence, retrieval, and integrity.

## The Four-Phase Model

Every OMI session follows a strict lifecycle:

```
┌─────────────┐
│ 1. START    │  Load context, prime memory
├─────────────┤
│ 2. WORK     │  Store, recall, update beliefs
├─────────────┤
│ 3. CHECK    │  Pre-compression checkpoint
├─────────────┤
│ 4. END      │  Persist, backup, verify
└─────────────┘
```

### Phase 1: Session Start

**Purpose:** Load hot context and relevant historical memory to bootstrap agent awareness.

**Command:**
```bash
omi session-start [--show-now]
```

**What Happens:**
1. Loads `NOW.md` into working context (<1k tokens)
2. Queries Graph Palace for semantically relevant memories
3. Checks system integrity (hashes, topology)
4. Creates today's daily log file (`memory/YYYY-MM-DD.md`)
5. Outputs session summary with memory count

**Best Practice:**
```bash
# Start every session with this command
omi session-start

# Review NOW.md if uncertain about current state
omi session-start --show-now
```

**Anti-Pattern:**
```bash
# ❌ Don't start working without session-start
omi store "Some memory"  # NOW.md not loaded, no context priming

# ✓ Always start session first
omi session-start
omi store "Some memory"
```

### Phase 2: Work (Store, Recall, Believe)

**Purpose:** Interact with memory during active work. Store new memories, recall relevant context, update beliefs based on evidence.

#### Storing Memories

**Command:**
```bash
omi store <content> --type <type> [--tags <tags>]
```

**Memory Types:**

| Type | When to Use | Example |
|------|-------------|---------|
| `fact` | Objective, verifiable knowledge | "SQLite supports FTS5 full-text search" |
| `experience` | What happened, actions taken | "Fixed auth bug by validating tokens earlier" |
| `belief` | Subjective assessment with confidence | "Async API is faster for batch operations" |
| `decision` | Choice made with rationale | "Chose pytest over unittest for better fixtures" |

**Examples:**
```bash
# Store a technical fact
omi store "Python 3.10+ required for match/case syntax" --type fact

# Store an experience with metadata
omi store "Deployed v2.1.0 to staging" --type experience --tags "deploy,staging"

# Store a decision with context
omi store "Using NIM for production embeddings (higher quality than Ollama)" --type decision
```

#### Recalling Memories

**Command:**
```bash
omi recall <query> [--mode <mode>] [--limit <n>] [--min-relevance <score>]
```

**Search Modes:**

| Mode | When to Use | Speed | Quality |
|------|-------------|-------|---------|
| `semantic` (default) | Finding conceptually related memories | Slower | Higher |
| `fts` | Exact keyword matching | Faster | Lower |

**Examples:**
```bash
# Semantic search (uses embeddings)
omi recall "authentication problems"

# Full-text search (faster, keyword-based)
omi recall "pytest" --mode fts

# Limit results and filter by relevance
omi recall "database performance" --limit 5 --min-relevance 0.7

# Retrieve specific memory by ID
omi recall --id "mem_abc123"
```

**Recency Scoring:**
Recall results are ranked by: `final_score = (relevance × 0.7) + (recency × 0.3)`

Where recency decays exponentially: `recency = exp(-days_ago / 30)`

This balances semantic relevance with temporal freshness — recent memories boost slightly even if less relevant.

#### Updating Beliefs

**Command:**
```bash
omi belief-update <statement> --confidence <0.0-1.0>
```

**Confidence Semantics:**

| Range | Interpretation |
|-------|----------------|
| 0.9 - 1.0 | High confidence (strong evidence) |
| 0.7 - 0.9 | Moderate confidence (good evidence) |
| 0.5 - 0.7 | Low confidence (weak evidence) |
| 0.0 - 0.5 | Contradicting evidence (doubt) |

**Belief Update Formula (EMA):**

OMI uses **asymmetric exponential moving average** with:
- **Supporting evidence:** λ = 0.15 (gradual increase)
- **Contradicting evidence:** λ = 0.30 (rapid decrease)

**Contradictions impact confidence 2x faster than support.**

**Examples:**
```bash
# Strong supporting evidence
omi belief-update "Async API is 4x faster than sync for batch ops" --confidence 0.95

# Weak supporting evidence
omi belief-update "Redis helps with caching" --confidence 0.6

# Contradicting evidence (lowers confidence)
omi belief-update "SQLite is bad for concurrency" --confidence 0.3

# List all beliefs above threshold
omi belief-list --min-confidence 0.5
```

### Phase 3: Pre-Compression Checkpoint

**Purpose:** Create a state capsule **before** the context window fills, enabling graceful continuation after compression.

**Command:**
```bash
omi check
```

**When to Use:**
- **Automatic trigger:** At ~70% context window capacity
- **Manual trigger:** Before long-running tasks, end of major work phase

**What Gets Captured:**
1. Snapshot of `NOW.md` at checkpoint time
2. Current belief states with confidence scores
3. Summary of recent high-centrality memories
4. SHA-256 hash verification of all files

**Output Example:**
```
✓ Checkpoint created at 2024-01-15T14:23:11Z
✓ NOW.md: 847 tokens (84% capacity)
✓ Beliefs: 12 active (avg confidence: 0.82)
✓ Recent memories: 8 high-centrality nodes
✓ Hash verification: PASSED
```

**Best Practice:**
```bash
# Before ending a major task
omi check
omi store "Completed feature X" --type experience
omi session-end

# Before context compression
# (Auto-triggered by most agent frameworks at 70% capacity)
```

**Why This Matters:**
Without checkpoints, agent context compression causes **compression amnesia** — losing working context that isn't yet persisted to Graph Palace. Pre-compression checkpoints solve this.

### Phase 4: Session End

**Purpose:** Persist session summary, trigger backup, verify integrity.

**Command:**
```bash
omi session-end [--no-backup]
```

**What Happens:**
1. Appends session summary to today's daily log (`memory/YYYY-MM-DD.md`)
2. Triggers vault backup (if enabled in `config.yaml`)
3. Updates memory statistics (total nodes, edges, belief count)
4. Runs integrity verification (hashes, topology)

**Output Example:**
```
✓ Session summary appended to memory/2024-01-15.md
✓ Vault backup triggered (local provider)
✓ Memory stats updated: 1,247 nodes, 3,891 edges, 12 beliefs
✓ Integrity check: PASSED
```

**Best Practice:**
```bash
# Always end sessions explicitly
omi session-end

# Skip backup for quick iterations (development only)
omi session-end --no-backup
```

**Anti-Pattern:**
```bash
# ❌ Don't skip session-end
# (No daily log append, no backup, no verification)

# ✓ Always close sessions
omi session-end
```

## Complete Daily Workflow Example

```bash
# Morning: Start session
omi session-start

# Work: Store and recall as needed
omi store "Investigating slow queries in user table" --type experience
omi recall "database performance" --limit 5

# Mid-day: Update beliefs based on findings
omi belief-update "Adding indexes significantly improves read performance" --confidence 0.9

# Afternoon: Checkpoint before major refactor
omi check
omi store "Decided to use composite index on (user_id, created_at)" --type decision

# Evening: End session
omi store "Deployed indexing changes to staging" --type experience
omi session-end
```

## Integration with 4-Tier Storage

The session lifecycle orchestrates data flow across all four tiers:

| Phase | Tier 1 (NOW.md) | Tier 2 (Daily Logs) | Tier 3 (Graph Palace) | Tier 4 (Vault) |
|-------|----------------|---------------------|----------------------|----------------|
| **START** | ✓ Load | ✓ Create today's file | ✓ Query relevant memories | — |
| **WORK** | ✓ Reference | — | ✓ Store/recall | — |
| **CHECK** | ✓ Snapshot | — | ✓ Summarize high-centrality | — |
| **END** | — | ✓ Append summary | ✓ Update stats | ✓ Backup all tiers |

## Common Patterns

### Pre-Session Memory Priming

Warm up context with targeted recalls before starting:

```bash
# Recall relevant context first
omi recall "previous session checkpoints" --limit 5
omi recall "unresolved issues" --limit 3

# Then start session
omi session-start
```

### End-of-Day Synthesis

Summarize the day before ending:

```bash
# Store high-level summary
omi store "Today: Fixed auth, added tests, deployed to staging" --type experience

# Create checkpoint
omi check

# End session with backup
omi session-end
```

### Periodic Integrity Checks

Run automated audits to catch corruption early:

```bash
# Add to cron (daily at 2am)
0 2 * * * cd ~/.openclaw/omi && omi audit
```

### Multi-Agent Coordination

When multiple agents share a memory palace:

```bash
# Agent 1: Store with metadata
omi store "API rate limit is 100 req/min" --type fact --tags "agent-1,api"

# Agent 2: Recall agent-1's findings
omi recall "API rate limit" --mode fts

# Agent 2: Update belief based on Agent 1's fact
omi belief-update "API rate limiting is enforced" --confidence 0.95
```

## Anti-Patterns to Avoid

### 1. **Skipping Session Start**

**Problem:** Agent operates without loaded context, missing relevant memories.

```bash
# ❌ Bad
omi recall "authentication"  # Runs without NOW.md context

# ✓ Good
omi session-start
omi recall "authentication"
```

### 2. **Storing Everything**

**Problem:** Graph Palace fills with noise, diluting signal.

```bash
# ❌ Bad
omi store "Ran ls command"  # Trivial, not useful
omi store "Opened vim"       # Low-value noise

# ✓ Good
omi store "Fixed critical auth bypass (CVE-2024-1234)" --type experience
omi store "Chose bcrypt over scrypt for password hashing" --type decision
```

### 3. **Never Creating Checkpoints**

**Problem:** Compression amnesia — lose working context when LLM context window fills.

```bash
# ❌ Bad
# (Work for hours without checkpoints, context compresses, lose NOW.md state)

# ✓ Good
omi check  # Every 1-2 hours or before major milestones
```

### 4. **Ignoring Recency Decay**

**Problem:** Relying on stale memories without checking freshness.

```bash
# ❌ Bad
omi recall "project status" --min-relevance 0.5
# (May return outdated info from weeks ago)

# ✓ Good
omi recall "project status" --min-relevance 0.7
# (Higher threshold ensures recent, relevant results)
```

### 5. **Not Ending Sessions**

**Problem:** No daily log append, no backup, no verification.

```bash
# ❌ Bad
# (Just stop working, no session-end)

# ✓ Good
omi session-end
```

## Session Lifecycle in Code (Python API)

### Sync API

```python
from omi import MemorySession

# Phase 1: Start
session = MemorySession()

# Phase 2: Work
session.store(
    content="Implemented rate limiting",
    memory_type="experience"
)

results = session.recall(
    query="rate limiting patterns",
    limit=5,
    min_relevance=0.7
)

session.belief_update(
    statement="Rate limiting prevents abuse",
    confidence=0.95
)

# Phase 3: Checkpoint (manual)
session.create_checkpoint()

# Phase 4: End
session.close()
```

### Async API

```python
import asyncio
from omi.async_api import async_session

async def main():
    # Phase 1: Start (context manager)
    async with async_session() as session:
        # Phase 2: Work (concurrent operations)
        await session.memory.store(
            "Implemented rate limiting",
            memory_type="experience"
        )

        results = await session.memory.recall(
            "rate limiting patterns",
            limit=5,
            min_relevance=0.7
        )

        await session.belief.update(
            "Rate limiting prevents abuse",
            confidence=0.95
        )

        # Phase 3: Checkpoint
        await session.checkpoint.create()

    # Phase 4: End (automatic on context manager exit)

asyncio.run(main())
```

## Verification

After implementing this pattern, verify correct behavior:

```bash
# Check session-start works
omi session-start
# Expected: Loads NOW.md, queries Graph Palace, creates daily log

# Check storage and recall
omi store "Test memory" --type fact
omi recall "Test memory"
# Expected: Returns stored memory with relevance score

# Check checkpoint
omi check
# Expected: Creates snapshot, verifies hashes

# Check session-end
omi session-end
# Expected: Appends to daily log, triggers backup, verifies integrity
```

## Related Patterns

- **[Memory Types](memory_types.md)** — When to use fact vs experience vs belief vs decision
- **[Search Strategies](search_strategies.md)** — Semantic vs FTS, relevance tuning
- **[Backup & Recovery](backup_recovery.md)** — MoltVault configuration, restore procedures
- **[Integrity Verification](integrity_verification.md)** — Topology checks, poison detection

---

**Remember:** The session lifecycle is not optional ceremony — it's the protocol that prevents the six failure modes (compression amnesia, token bloat, write-but-don't-read, stale context, memory poisoning, conflicting identity).

Follow it strictly. The palace remembers what the river forgets.
