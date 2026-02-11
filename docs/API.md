# API Reference

Complete reference for OMI's Model Context Protocol (MCP) tools for OpenClaw integration.

## Overview

OMI exposes 6 MCP tools for agent integration:

| Tool | Purpose | Use Case |
|------|---------|----------|
| `memory_recall` | Semantic search with recency weighting | Retrieve relevant memories from Graph Palace |
| `memory_store` | Persist memory with embedding | Store facts, experiences, beliefs, decisions |
| `belief_update` | Update belief confidence using EMA | Add supporting/contradicting evidence |
| `now_read` | Load current operational context | **Call FIRST on session start** |
| `now_update` | Update operational state | Update at 70% context or task completion |
| `integrity_check` | Verify memory file integrity | Check for corruption, anomalies |

## Tool Definitions

### memory_recall

Semantic search across Graph Palace with recency weighting.

**Scoring Formula:**
```
final_score = (relevance * 0.7) + (recency * 0.3)
recency = exp(-days_ago / 30)
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✓ | — | Natural language search query |
| `limit` | integer | | 10 | Maximum number of results to return |
| `min_relevance` | number | | 0.7 | Cosine similarity threshold (0.0-1.0) |
| `memory_type` | string | | — | Filter by type: `fact`, `experience`, `belief`, `decision` |

#### Returns

```json
[
  {
    "memory_id": "mem_abc123",
    "content": "Discovered async API performance benefits",
    "memory_type": "experience",
    "created_at": "2024-01-15T10:30:00",
    "relevance": 0.85,
    "final_score": 0.72,
    "related_memories": ["mem_def456"]
  }
]
```

#### Example

```python
from omi.api import get_all_mcp_tools

config = {'base_path': '~/.openclaw/omi'}
tools = get_all_mcp_tools(config)

# Recall memories about authentication
results = tools['memory'].recall(
    query="authentication issues",
    limit=5,
    min_relevance=0.75,
    memory_type="experience"
)

for mem in results:
    print(f"{mem['content']} (score: {mem['final_score']:.2f})")
```

#### Implementation Notes

- Uses vector embeddings for semantic similarity (NIM baai/bge-m3 or Ollama nomic-embed-text)
- Applies recency decay with 30-day half-life
- Results sorted by final weighted score
- Emits `MemoryRecalledEvent` to event bus

---

### memory_store

Persist memory with automatic embedding generation.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | ✓ | — | Memory text to store |
| `memory_type` | string | | `experience` | Type: `fact`, `experience`, `belief`, `decision` |
| `related_to` | array[string] | | — | IDs of related memories for graph edges |
| `confidence` | number | | — | For beliefs: confidence level 0.0-1.0 |

#### Returns

```json
{
  "memory_id": "mem_xyz789"
}
```

#### Example

```python
# Store a fact
memory_id = tools['memory'].store(
    content="SQLite supports full-text search via FTS5",
    memory_type="fact"
)

# Store an experience with relationships
memory_id = tools['memory'].store(
    content="Fixed auth bug by validating tokens earlier in the middleware chain",
    memory_type="experience",
    related_to=["mem_auth_design_001", "mem_middleware_refactor_002"]
)

# Store a belief with confidence
memory_id = tools['memory'].store(
    content="Async operations improve throughput by 40% in our API",
    memory_type="belief",
    confidence=0.85
)
```

#### Implementation Notes

- Generates embedding using configured provider (NIM or Ollama)
- Embeddings cached to `.npy` files for deduplication
- Creates `RELATED_TO` edges with weight 0.5 for related memories
- Emits `MemoryStoredEvent` to event bus
- Memory types enforced by SQLite CHECK constraint

---

### belief_update

Update belief confidence using exponential moving average (EMA) with asymmetric learning rates.

**EMA Formula:**
```
Supporting evidence:   λ = 0.15
Contradicting evidence: λ = 0.30  (contradictions hit 2x harder)

new_confidence = old_confidence + λ * strength * (target - old_confidence)
  where target = 1.0 (supporting) or 0.0 (contradicting)
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `belief_id` | string | ✓ | — | ID of belief to update |
| `evidence_memory_id` | string | ✓ | — | Memory serving as evidence |
| `supports` | boolean | ✓ | — | `true` = supports belief, `false` = contradicts |
| `strength` | number | ✓ | — | Evidence strength 0.0-1.0 |

#### Returns

```json
{
  "new_confidence": 0.72
}
```

#### Example

```python
# Create a belief first (via memory_store)
belief_id = tools['memory'].store(
    content="TypeScript prevents runtime errors",
    memory_type="belief",
    confidence=0.5
)

# Add supporting evidence
evidence_id = tools['memory'].store(
    content="Caught 15 type errors at compile time that would have been runtime bugs",
    memory_type="experience"
)

new_conf = tools['belief'].update(
    belief_id=belief_id,
    evidence_memory_id=evidence_id,
    supports=True,
    strength=0.9
)
# new_conf ≈ 0.57 (increased from 0.5)

# Add contradicting evidence
counter_id = tools['memory'].store(
    content="Still got a runtime type error from JSON parsing",
    memory_type="experience"
)

new_conf = tools['belief'].update(
    belief_id=belief_id,
    evidence_memory_id=counter_id,
    supports=False,
    strength=0.6
)
# new_conf ≈ 0.47 (decreased more due to higher λ)
```

#### Implementation Notes

- Creates `SUPPORTS` or `CONTRADICTS` edge in Graph Palace
- Evidence chain preserved for audit via `get_evidence_chain()`
- Emits `BeliefUpdatedEvent` with old and new confidence
- `ContradictionDetector` can flag conflicting evidence patterns

---

### now_read

Load current operational context from NOW.md.

**Call this FIRST on session start** to warm the agent's context with hot operational state.

#### Parameters

None.

#### Returns

```json
{
  "current_task": "Implementing authentication middleware",
  "recent_completions": [
    "Fixed database migration scripts",
    "Added rate limiting to API endpoints"
  ],
  "pending_decisions": [
    "Choose between JWT and session-based auth",
    "Decide on password hashing algorithm"
  ],
  "key_files": [
    "src/auth/middleware.py",
    "tests/test_auth.py",
    "config/security.yaml"
  ],
  "timestamp": "2024-01-15T14:23:00"
}
```

If NOW.md is empty or contains default content, returns `{}`.

#### Example

```python
# Session start workflow
now_context = tools['checkpoint'].now_read()

if now_context:
    print(f"Resuming task: {now_context['current_task']}")
    print(f"Recently completed: {len(now_context['recent_completions'])} items")

    # Load relevant memories based on current task
    memories = tools['memory'].recall(
        query=now_context['current_task'],
        limit=10
    )
else:
    print("Starting fresh session")
```

#### Implementation Notes

- Reads from `~/.openclaw/omi/NOW.md`
- Parses markdown structure into structured data
- Returns empty dict if file missing or default content
- Should be called **before** any other operations in a session

---

### now_update

Update operational state at context threshold or task completion.

**Trigger this:**
- At 70% context window usage (pre-compression)
- On task completion
- Before major context shifts

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `current_task` | string | | — | Current focus (replaces previous) |
| `recent_completions` | array[string] | | — | Just-completed items (appends to list) |
| `pending_decisions` | array[string] | | — | Unresolved choices (replaces list) |
| `key_files` | array[string] | | — | Critical file paths (replaces list) |

All parameters are optional. Omitted fields are not updated.

#### Returns

None (void).

#### Example

```python
# Update after completing a subtask
tools['checkpoint'].now_update(
    current_task="Writing integration tests for auth middleware",
    recent_completions=[
        "Implemented JWT token validation",
        "Added refresh token rotation"
    ],
    pending_decisions=[
        "Should we support OAuth2 providers?"
    ],
    key_files=[
        "src/auth/jwt.py",
        "tests/integration/test_auth_flow.py"
    ]
)

# Minimal update (just mark current task)
tools['checkpoint'].now_update(
    current_task="Code review PR #123"
)
```

#### Implementation Notes

- Updates NOW.md with new structured content
- Preserves fields not included in update
- Timestamp automatically set to current time
- Triggers state capsule creation in pre-compression flow

---

### integrity_check

Verify memory file integrity and graph topology.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `scope` | string | | `all` | Scope: `now`, `daily`, `graph`, `all` |

#### Returns

```json
{
  "now_md": true,
  "memory_md": true,
  "topology": {
    "orphan_nodes": 2,
    "sudden_cores": 0,
    "warnings": ["mem_orphan_001", "mem_orphan_002"]
  },
  "overall_safe": true
}
```

#### Example

```python
# Full integrity check
result = tools['security'].integrity_check(scope='all')

if not result['overall_safe']:
    print("⚠️  Integrity issues detected!")
    if result['topology']['orphan_nodes'] > 5:
        print(f"  - {result['topology']['orphan_nodes']} orphan nodes")
    if result['topology']['sudden_cores'] > 0:
        print(f"  - {result['topology']['sudden_cores']} sudden core nodes")
else:
    print("✓ All integrity checks passed")

# Quick check (just NOW.md and daily logs)
result = tools['security'].integrity_check(scope='now')
```

#### Implementation Notes

- Verifies SHA-256 checksums for NOW.md and daily logs
- Detects orphan nodes (no edges) in Graph Palace
- Flags sudden cores (nodes gaining edges too quickly)
- Returns `overall_safe: false` if >5 orphans or any sudden cores
- Does NOT auto-repair — reports issues for manual review

---

## Entry Point

```python
def get_all_mcp_tools(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all MCP tools with configuration

    Args:
        config: Configuration dict with keys:
            - base_path: Data directory (default: ~/.openclaw/omi)
            - embedding_model: Model name (default: nomic-embed-text)
            - embedding_provider: 'nim' or 'ollama'

    Returns:
        {
            'memory': MemoryTools instance,
            'belief': BeliefTools instance,
            'checkpoint': CheckpointTools instance,
            'security': SecurityTools instance,
            'daily': DailyLogTools instance
        }
    """
```

## Integration Patterns

### Session Lifecycle

```python
config = {'base_path': '~/.openclaw/omi'}
tools = get_all_mcp_tools(config)

# 1. Session start
context = tools['checkpoint'].now_read()
relevant_memories = tools['memory'].recall(
    query=context.get('current_task', 'recent work'),
    limit=10
)

# 2. During work
memory_id = tools['memory'].store(
    content="Implemented rate limiting with sliding window algorithm",
    memory_type="experience"
)

# 3. Update beliefs
tools['belief'].update(
    belief_id="belief_rate_limit_effective",
    evidence_memory_id=memory_id,
    supports=True,
    strength=0.85
)

# 4. Pre-compression (at 70% context)
tools['checkpoint'].now_update(
    current_task="Deploying rate limiter to production",
    recent_completions=["Implemented rate limiter", "Added tests"],
    key_files=["src/middleware/rate_limit.py"]
)

# 5. Session end
tools['daily'].append(f"Session summary: {context.get('current_task')}")
tools['checkpoint'].vault_backup()
```

### Belief Evolution Tracking

```python
# Initial belief
belief_id = tools['memory'].store(
    content="GraphQL is better than REST for our use case",
    memory_type="belief",
    confidence=0.5
)

# Add evidence over time
evidence_items = [
    ("Reduced API calls by 60% using GraphQL batching", True, 0.8),
    ("Hit N+1 query problem, had to add DataLoader", False, 0.6),
    ("Client teams love the flexibility of query selection", True, 0.7),
    ("Struggling with caching compared to REST", False, 0.5),
]

for content, supports, strength in evidence_items:
    evidence_id = tools['memory'].store(content, memory_type="experience")
    new_conf = tools['belief'].update(
        belief_id=belief_id,
        evidence_memory_id=evidence_id,
        supports=supports,
        strength=strength
    )
    print(f"Confidence now: {new_conf:.2f}")

# View full evidence chain
chain = tools['belief'].get_evidence_chain(belief_id)
for e in chain:
    print(f"  {'✓' if e['supports'] else '✗'} {e['memory_id']} (strength: {e['strength']})")
```

### Periodic Integrity Audits

```python
import schedule
import time

def daily_audit():
    result = tools['security'].integrity_check(scope='all')

    if not result['overall_safe']:
        # Alert via logging, webhook, etc.
        print(f"⚠️  Daily audit failed at {datetime.now()}")
        print(f"Orphans: {result['topology']['orphan_nodes']}")
        print(f"Sudden cores: {result['topology']['sudden_cores']}")
    else:
        print(f"✓ Daily audit passed at {datetime.now()}")

# Run every day at 2am
schedule.every().day.at("02:00").do(daily_audit)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Event System

All MCP tools emit events to the global event bus:

| Event | Triggered By | Payload |
|-------|--------------|---------|
| `MemoryStoredEvent` | `memory_store` | `memory_id`, `content`, `memory_type`, `confidence` |
| `MemoryRecalledEvent` | `memory_recall` | `query`, `result_count`, `top_results` |
| `BeliefUpdatedEvent` | `belief_update` | `belief_id`, `old_confidence`, `new_confidence`, `evidence_id` |
| `ContradictionDetectedEvent` | `belief_check_contradiction` | `memory_id_1`, `memory_id_2`, `contradiction_pattern` |

### Subscribe to Events

```python
from omi.event_bus import get_event_bus
from omi.events import MemoryStoredEvent

def on_memory_stored(event: MemoryStoredEvent):
    print(f"New memory: {event.memory_id} ({event.memory_type})")

bus = get_event_bus()
bus.subscribe(MemoryStoredEvent, on_memory_stored)
```

## Configuration

Default configuration (`~/.openclaw/omi/config.yaml`):

```yaml
embedding:
  provider: nim  # or 'ollama'
  model: baai/bge-m3  # or 'nomic-embed-text' for ollama
  dimensions: 1024  # or 768 for ollama
  api_key: ${NIM_API_KEY}

storage:
  max_now_tokens: 1000
  recency_half_life_days: 30

vault:
  enabled: true
  provider: local  # or 'r2'
  retention_days: 90

security:
  enable_consensus: false
  integrity_check_on_start: true
```

## OpenClaw Plugin Integration

Edit `~/.openclaw/config.yaml`:

```yaml
plugins:
  - name: omi
    enabled: true
    config:
      data_dir: ~/.openclaw/omi
      embedding_provider: nim
```

OMI will register all 6 MCP tools automatically via `plugin.json` entry point.

## Error Handling

All tools raise exceptions on errors:

```python
from omi.exceptions import MemoryNotFoundError, IntegrityError

try:
    result = tools['memory'].recall(query="test")
except MemoryNotFoundError as e:
    print(f"Memory not found: {e}")
except IntegrityError as e:
    print(f"Integrity check failed: {e}")
```

## Performance Notes

- **Embedding caching**: Identical content reuses cached embeddings (`.npy` files)
- **FTS5 fallback**: For text-only search without embeddings, use CLI `omi recall --mode fts`
- **Batch operations**: Use async API for concurrent stores/recalls (4x faster)
- **Database**: SQLite in WAL mode for concurrent reads
- **Vector search**: Linear scan with numpy optimizations (acceptable for <100k memories)

## Type Definitions

```python
from typing import TypedDict, Literal, Optional, List

MemoryType = Literal['fact', 'experience', 'belief', 'decision']
ScopeType = Literal['now', 'daily', 'graph', 'all']

class MemoryResult(TypedDict):
    memory_id: str
    content: str
    memory_type: MemoryType
    created_at: str
    relevance: float
    final_score: float
    related_memories: List[str]

class NOWContext(TypedDict):
    current_task: str
    recent_completions: List[str]
    pending_decisions: List[str]
    key_files: List[str]
    timestamp: str

class IntegrityResult(TypedDict):
    now_md: bool
    memory_md: bool
    topology: dict
    overall_safe: bool
```

---

**Next Steps:**
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for the 4-tier storage model
- See [QUICKSTART.md](QUICKSTART.md) for setup and daily workflow
- Check [SECURITY.md](SECURITY.md) for threat model and mitigations
