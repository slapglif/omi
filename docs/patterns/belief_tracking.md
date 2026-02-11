# Belief Tracking Pattern

> *"Facts are what happened. Beliefs are what we think will happen again."*

This pattern defines how to manage confidence-weighted beliefs with evidence tracking in OMI. Unlike facts (objective, verifiable), beliefs are subjective assessments that evolve as contradicting or supporting evidence accumulates. The belief network uses **asymmetric EMA updates** to make contradictions impact confidence twice as hard as confirmations.

## Why Beliefs Are Special

Beliefs differ from facts in three critical ways:

| Dimension | Facts | Beliefs |
|-----------|-------|---------|
| **Half-life** | 30 days | 60 days (2x longer) |
| **Confidence** | Always 1.0 | 0.0 - 1.0 (tracked) |
| **Update rule** | Immutable | EMA with asymmetric lambdas |

**Key Insight:** Beliefs decay slower because they represent accumulated wisdom, not point-in-time observations. But when contradictions arise, they erode confidence quickly.

## The EMA Update Formula

OMI uses **Exponential Moving Average (EMA)** with asymmetric learning rates:

```
new_confidence = old_confidence + λ × (target - old_confidence)
```

Where:
- **Supporting evidence:** λ = 0.15 (gentle nudge upward)
- **Contradicting evidence:** λ = 0.30 (rapid erosion)

**Contradictions hit 2x harder than support.**

### Example: Confidence Evolution

Start with belief: *"SQLite is fast for reads"* at confidence `0.5`

```
Evidence 1 (supporting, strength 0.8):
  target = min(1.0, 0.5 + 0.8) = 1.0
  new = 0.5 + 0.15 × (1.0 - 0.5) = 0.575

Evidence 2 (supporting, strength 0.9):
  target = min(1.0, 0.575 + 0.9) = 1.0
  new = 0.575 + 0.15 × (1.0 - 0.575) = 0.639

Evidence 3 (contradicting, strength 0.6):
  target = max(0.0, 0.639 - 0.6) = 0.039
  new = 0.639 + 0.30 × (0.039 - 0.639) = 0.459

Final confidence: 0.459 (dropped from 0.639 → one contradiction undid two confirmations)
```

This asymmetry prevents belief calcification — contradictions force re-evaluation.

## Creating Beliefs

### Command-Line API

```bash
# Create new belief with initial confidence
omi store "Async API is faster for batch operations" \
  --type belief \
  --confidence 0.5

# Create belief with high initial confidence (strong prior)
omi store "Rate limiting prevents abuse" \
  --type belief \
  --confidence 0.8
```

### Python API

```python
from omi.belief import BeliefNetwork
from omi.persistence import GraphPalace

palace = GraphPalace(db_path="palace.sqlite")
belief_net = BeliefNetwork(palace)

# Create belief with default confidence (0.5)
belief_id = belief_net.create_belief(
    content="Async API is faster for batch operations",
    initial_confidence=0.5
)
```

**Best Practice:** Start at `0.5` (neutral) unless you have strong prior evidence.

## Updating with Evidence

### Adding Supporting Evidence

When you observe something that confirms a belief:

```bash
# CLI: Update belief with supporting evidence
omi belief-update <belief_id> \
  --evidence-memory-id <memory_id> \
  --supports true \
  --strength 0.8
```

```python
# Python: Create evidence and update
from omi.belief import Evidence
from datetime import datetime

evidence = Evidence(
    memory_id="mem_abc123",
    supports=True,
    strength=0.8,
    timestamp=datetime.now()
)

new_confidence = belief_net.update_with_evidence(belief_id, evidence)
print(f"Updated confidence: {new_confidence:.3f}")
```

**Evidence strength scale:**

| Strength | Interpretation |
|----------|----------------|
| 0.9 - 1.0 | Conclusive evidence (benchmark, controlled test) |
| 0.7 - 0.9 | Strong evidence (multiple observations) |
| 0.5 - 0.7 | Moderate evidence (single observation) |
| 0.3 - 0.5 | Weak evidence (anecdotal) |

### Adding Contradicting Evidence

When you observe something that contradicts a belief:

```python
# Create contradicting evidence
evidence = Evidence(
    memory_id="mem_xyz789",
    supports=False,  # This is contradicting
    strength=0.7,
    timestamp=datetime.now()
)

new_confidence = belief_net.update_with_evidence(belief_id, evidence)
# Confidence drops rapidly (λ=0.30 vs λ=0.15 for support)
```

**Anti-Pattern:**
```python
# ❌ Don't manually set confidence directly
palace.update_belief_confidence(belief_id, 0.9)  # Bypasses EMA

# ✓ Always use evidence-based updates
belief_net.update_with_evidence(belief_id, evidence)
```

## Retrieving Beliefs with Confidence Weighting

Standard recall treats all memories equally. **Confidence-weighted retrieval** exponentially boosts high-confidence beliefs.

### The Weighting Formula

```
weighted_score = relevance × (1.5 ^ confidence)
```

**Impact:**
- Belief at `confidence=0.9` → weight = 1.5^0.9 = **1.39x**
- Belief at `confidence=0.5` → weight = 1.5^0.5 = **1.22x**
- Belief at `confidence=0.3` → weight = 1.5^0.3 = **1.14x**

High-confidence beliefs rank significantly higher in results.

### Using Confidence-Weighted Retrieval

```python
# Retrieve beliefs with confidence weighting
results = belief_net.retrieve_with_confidence_weighting(
    query="performance optimization strategies",
    min_confidence=0.6  # Filter low-confidence beliefs
)

for belief in results:
    print(f"{belief['content']}")
    print(f"  Confidence: {belief['confidence']:.2f}")
    print(f"  Weighted score: {belief['weighted_score']:.3f}")
```

**Best Practice:**
```python
# Use confidence filtering for critical decisions
high_confidence_beliefs = belief_net.retrieve_with_confidence_weighting(
    query="deployment safety",
    min_confidence=0.8  # Only highly confident beliefs
)

# Use lower threshold for exploratory research
all_beliefs = belief_net.retrieve_with_confidence_weighting(
    query="deployment safety",
    min_confidence=0.3  # Include uncertain beliefs
)
```

## Evidence Chains

Every belief maintains a **complete evidence chain** showing what supports it and what contradicts it.

### Viewing Evidence Chain

```python
# Get full evidence chain for a belief
evidence_chain = belief_net.get_evidence_chain(belief_id)

for evidence in evidence_chain:
    memory = palace.get_memory(evidence.memory_id)
    support_str = "✓ SUPPORTS" if evidence.supports else "✗ CONTRADICTS"

    print(f"[{evidence.timestamp}] {support_str} (strength {evidence.strength:.2f})")
    print(f"  {memory['content']}")
```

**Output Example:**
```
[2024-01-10 09:15:23] ✓ SUPPORTS (strength 0.85)
  Benchmark shows async is 3.2x faster for 100+ concurrent requests

[2024-01-11 14:22:01] ✓ SUPPORTS (strength 0.75)
  Deployed async version to staging, saw 40% latency reduction

[2024-01-12 11:05:44] ✗ CONTRADICTS (strength 0.60)
  Async version caused timeouts under load spikes (>500 req/s)
```

### Evidence Chain Audit

Use evidence chains to understand **why** a belief has its current confidence:

```python
def audit_belief(belief_id: str):
    """Audit belief confidence by analyzing evidence chain"""
    belief = palace.get_belief(belief_id)
    evidence_chain = belief_net.get_evidence_chain(belief_id)

    supporting = [e for e in evidence_chain if e.supports]
    contradicting = [e for e in evidence_chain if not e.supports]

    print(f"Belief: {belief['content']}")
    print(f"Current confidence: {belief['confidence']:.3f}")
    print(f"Evidence: {len(supporting)} supporting, {len(contradicting)} contradicting")

    if len(contradicting) > len(supporting):
        print("⚠️  More contradictions than support — consider retiring this belief")
```

## Automatic Contradiction Detection

The `ContradictionDetector` automatically flags conflicting evidence using linguistic patterns.

### Opposition Patterns

```python
from omi.belief import ContradictionDetector

detector = ContradictionDetector()

memory1 = "SQLite should always use WAL mode for concurrency"
memory2 = "SQLite should never use WAL mode on network filesystems"

is_contradiction, pattern = detector.detect_contradiction_with_pattern(memory1, memory2)
# Returns: (True, "should always vs should never")
```

**Detected Patterns:**

| Pattern Type | Example |
|--------------|---------|
| `should always` vs `should never` | "Should always validate" vs "Should never validate untrusted input" |
| `works well` vs `doesn't work` | "Works well under load" vs "Doesn't work at scale" |
| `causes` vs `prevents` | "Causes memory leaks" vs "Prevents memory leaks" |
| `increases` vs `decreases` | "Increases throughput" vs "Decreases throughput" |
| `enables` vs `blocks` | "Enables caching" vs "Blocks caching" |

### Using Contradiction Detection

```python
# Before adding evidence, check for contradictions
def smart_evidence_update(belief_id: str, new_memory_id: str, strength: float):
    """Add evidence with automatic contradiction detection"""
    detector = ContradictionDetector()

    # Get belief content
    belief = palace.get_belief(belief_id)
    new_memory = palace.get_memory(new_memory_id)

    # Detect contradiction
    is_contradiction, pattern = detector.detect_contradiction_with_pattern(
        belief['content'],
        new_memory['content']
    )

    # Create evidence
    evidence = Evidence(
        memory_id=new_memory_id,
        supports=not is_contradiction,  # Flip if contradiction detected
        strength=strength,
        timestamp=datetime.now()
    )

    if is_contradiction:
        print(f"⚠️  Contradiction detected: {pattern}")

    # Update belief
    new_confidence = belief_net.update_with_evidence(belief_id, evidence)
    return new_confidence
```

## Common Patterns

### Pattern 1: Gradual Belief Formation

Start neutral, accumulate evidence over time:

```python
# Day 1: Neutral hypothesis
belief_id = belief_net.create_belief(
    "Redis caching improves API response time",
    initial_confidence=0.5
)

# Day 2: First observation (moderate support)
belief_net.update_with_evidence(belief_id, Evidence(
    memory_id="mem_001",
    supports=True,
    strength=0.6,
    timestamp=datetime.now()
))
# Confidence: 0.5 → 0.56

# Day 3: Benchmark confirms (strong support)
belief_net.update_with_evidence(belief_id, Evidence(
    memory_id="mem_002",
    supports=True,
    strength=0.9,
    timestamp=datetime.now()
))
# Confidence: 0.56 → 0.69

# Day 7: Production deployment confirms (conclusive support)
belief_net.update_with_evidence(belief_id, Evidence(
    memory_id="mem_003",
    supports=True,
    strength=1.0,
    timestamp=datetime.now()
))
# Confidence: 0.69 → 0.78 (high confidence achieved)
```

### Pattern 2: Belief Revision After Contradiction

When contradictions emerge, confidence erodes rapidly:

```python
# Existing belief at high confidence
belief_id = "belief_xyz"  # Confidence: 0.85

# Unexpected failure in production (strong contradiction)
belief_net.update_with_evidence(belief_id, Evidence(
    memory_id="mem_failure",
    supports=False,
    strength=0.8,
    timestamp=datetime.now()
))
# Confidence: 0.85 → 0.36 (massive drop due to λ=0.30)

# Investigate and find edge case
# Create new, more nuanced belief:
new_belief_id = belief_net.create_belief(
    "Redis caching improves response time EXCEPT under cache stampede conditions",
    initial_confidence=0.7
)
```

### Pattern 3: Confidence Thresholding for Decisions

Use confidence to gate critical actions:

```python
def should_deploy_feature(feature_name: str) -> bool:
    """Only deploy if high-confidence beliefs support it"""
    beliefs = belief_net.retrieve_with_confidence_weighting(
        query=f"{feature_name} deployment safety",
        min_confidence=0.8  # High bar for deployment
    )

    if not beliefs:
        print("⚠️  No high-confidence beliefs about safety — aborting deployment")
        return False

    # Check if top belief is positive
    top_belief = beliefs[0]
    if "safe" in top_belief['content'].lower():
        print(f"✓ Deploying based on belief: {top_belief['content']}")
        return True
    else:
        print(f"✗ Top belief is negative: {top_belief['content']}")
        return False
```

## Anti-Patterns to Avoid

### 1. **Treating Beliefs as Facts**

**Problem:** Beliefs are uncertain — don't treat them as ground truth.

```python
# ❌ Bad: Assume belief is true
belief = palace.get_belief(belief_id)
if belief:  # Wrong! Check confidence, not just existence
    apply_optimization()

# ✓ Good: Check confidence threshold
belief = palace.get_belief(belief_id)
if belief and belief['confidence'] >= 0.8:
    apply_optimization()
else:
    print(f"⚠️  Belief confidence too low ({belief['confidence']:.2f})")
```

### 2. **Ignoring Contradictions**

**Problem:** Dismissing contradicting evidence leads to false confidence.

```python
# ❌ Bad: Only add supporting evidence
if observation_confirms_belief:
    belief_net.update_with_evidence(belief_id, supporting_evidence)
# (Ignores contradictions)

# ✓ Good: Add ALL evidence, supporting or contradicting
evidence = Evidence(
    memory_id=observation_id,
    supports=observation_confirms_belief,  # Honest assessment
    strength=observation_strength,
    timestamp=datetime.now()
)
belief_net.update_with_evidence(belief_id, evidence)
```

### 3. **Creating Beliefs Without Evidence**

**Problem:** Beliefs should be evidence-based, not speculative.

```python
# ❌ Bad: Create belief without any supporting memories
belief_id = belief_net.create_belief(
    "Microservices are always better than monoliths",
    initial_confidence=0.9  # Unjustified high confidence
)

# ✓ Good: Start neutral, build evidence
belief_id = belief_net.create_belief(
    "Microservices are better for this project",
    initial_confidence=0.5  # Neutral starting point
)

# Then add evidence as you gather it
belief_net.update_with_evidence(belief_id, Evidence(
    memory_id="mem_research",
    supports=True,
    strength=0.6,
    timestamp=datetime.now()
))
```

### 4. **Never Retiring Low-Confidence Beliefs**

**Problem:** Accumulating uncertain beliefs creates noise.

```python
# ✓ Good: Periodically prune low-confidence beliefs
def prune_uncertain_beliefs(min_confidence: float = 0.3):
    """Remove beliefs that have fallen below confidence threshold"""
    all_beliefs = palace.get_all_beliefs()

    for belief in all_beliefs:
        if belief['confidence'] < min_confidence:
            evidence_chain = belief_net.get_evidence_chain(belief['id'])
            contradicting = [e for e in evidence_chain if not e.supports]

            if len(contradicting) >= 3:  # Multiple contradictions
                print(f"Retiring belief: {belief['content']} (conf={belief['confidence']:.2f})")
                palace.delete_memory(belief['id'])
```

### 5. **Using Equal Lambda for Support/Contradiction**

**Problem:** Symmetric updates don't reflect asymmetric risk of false positives.

```python
# ❌ Bad: Custom implementation with symmetric lambda
LAMBDA = 0.20  # Same for both
new_conf = old_conf + LAMBDA * (target - old_conf)

# ✓ Good: Use asymmetric lambdas (built into BeliefNetwork)
# Supporting: λ=0.15
# Contradicting: λ=0.30
belief_net.update_with_evidence(belief_id, evidence)  # Uses correct lambdas
```

## Integration with Session Lifecycle

Beliefs integrate into the [Session Lifecycle Pattern](session_lifecycle.md):

```bash
# Phase 1: Start session
omi session-start

# Phase 2: Work — create and update beliefs
omi store "Async improves throughput" --type belief --confidence 0.5
omi belief-update <belief_id> --evidence <memory_id> --supports true --strength 0.8

# Retrieve high-confidence beliefs for decision-making
omi recall "deployment strategies" --type belief --min-confidence 0.8

# Phase 3: Checkpoint (beliefs included in snapshot)
omi check

# Phase 4: End session
omi session-end
```

## Verification

Verify belief tracking works correctly:

```bash
# Create test belief
BELIEF_ID=$(omi store "Test belief about EMA" --type belief --confidence 0.5 | grep -oP 'belief_\w+')

# Add supporting evidence
omi belief-update $BELIEF_ID --evidence mem_001 --supports true --strength 0.8

# Check confidence increased
omi recall --id $BELIEF_ID
# Expected: Confidence > 0.5

# Add contradicting evidence
omi belief-update $BELIEF_ID --evidence mem_002 --supports false --strength 0.7

# Check confidence decreased
omi recall --id $BELIEF_ID
# Expected: Confidence dropped (λ=0.30 applied)

# View evidence chain
omi belief-chain $BELIEF_ID
# Expected: Shows 2 evidence entries (1 supporting, 1 contradicting)
```

## Related Patterns

- **[Session Lifecycle](session_lifecycle.md)** — When to create/update beliefs in the work phase
- **[Memory Types](memory_types.md)** — Fact vs belief vs experience vs decision
- **[Search Strategies](search_strategies.md)** — Confidence-weighted retrieval vs semantic search

---

**Remember:** Beliefs are living hypotheses, not static facts. The EMA update rule with asymmetric lambdas ensures confidence tracks reality — accumulating gradually with support, eroding rapidly with contradiction.

Trust beliefs proportional to their confidence. Question beliefs when contradictions emerge. The palace learns what the river cannot teach.
