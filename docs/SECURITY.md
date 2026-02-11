# Security Policy

**Version:** 1.0
**Last Updated:** 2024-01-15

---

## Threat Model

OMI's security architecture is built on **Byzantine Fault Tolerance** principles. The fundamental insight:

> **Trust is the attack surface.**

In a system where AI agents build persistent memory over time, the integrity of that memory becomes the foundation of agent identity and behavior. A compromised memory is a compromised agent.

---

## Byzantine Fault Tolerance

### Core Principle

We assume that **memory can be adversarially modified** by:
- Compromised agent instances
- Malicious plugins or tools
- Direct database manipulation
- Embedding injection attacks
- Context window poisoning that propagates to storage

Unlike traditional security models that focus on preventing unauthorized access, OMI's threat model assumes that **some instances will be compromised** and designs defense mechanisms that maintain integrity even under Byzantine conditions.

### Defense Strategy

OMI employs a **defense-in-depth** approach with multiple verification layers:

1. **Cryptographic integrity** (SHA-256 hashing)
2. **Graph topology analysis** (anomaly detection)
3. **Semantic verification** (embedding drift detection)
4. **Multi-instance consensus** (distributed trust)
5. **Git-based audit trail** (version control as security)

---

## Attack Surfaces

### 1. Memory Injection

**Threat:** Adversary injects false memories directly into the Graph Palace.

**Attack Pattern:**
- Direct SQLite database manipulation
- Compromised MCP tool writes
- Malicious plugin inserts fabricated facts

**Indicators:**
- **Orphan nodes:** Memories with no relational edges to existing graph
- **Sudden cores:** High-centrality memories with no access history
- **Content hash mismatches:** `content_hash` field doesn't match SHA-256 of current content

**Mitigations:**
- `TopologyVerifier.find_orphan_nodes()` — Flag isolated memories
- `TopologyVerifier.find_sudden_cores()` — Detect artificially promoted memories
- `TopologyVerifier.find_hash_mismatches()` — Verify cryptographic integrity
- Daily `omi audit` scans for anomalies

---

### 2. Embedding Poisoning

**Threat:** Attacker creates memory whose content doesn't match its semantic embedding.

**Attack Pattern:**
- Memory claims to be about topic X but embeds near topic Y
- Used to hijack semantic search and inject context into unrelated queries
- Example: Memory labeled "Python syntax" but embedding is close to "password credentials"

**Indicators:**
- **Embedding drift:** Re-embedding the current content produces significantly different vector
- Cosine similarity < 0.9 between stored and re-computed embeddings

**Mitigations:**
- `TopologyVerifier.check_embedding_drift()` — Re-embed and compare
- Periodic re-embedding validation during `omi check`
- Fallback to FTS5 text search if embedding results are suspicious

---

### 3. Identity File Tampering

**Threat:** Direct modification of `NOW.md`, `MEMORY.md`, or daily logs.

**Attack Pattern:**
- Manual file edits outside OMI tooling
- Malicious scripts rewriting core files
- Context window overflow leading to untracked modifications

**Indicators:**
- SHA-256 hash mismatch between `.now.hash` / `.memory.hash` and current file
- Large Git commits to identity files without proper commit messages
- Files modified outside OMI session boundaries

**Mitigations:**
- `IntegrityChecker.check_now_md()` and `IntegrityChecker.check_memory_md()`
- SHA-256 hashes stored in `.now.hash` and `.memory.hash`
- `IntegrityChecker.audit_git_history()` — Flag suspicious commits
- Pre-commit hooks (recommended but not enforced)

---

### 4. Consensus Bypass

**Threat:** Single compromised instance marks itself as "foundational" without multi-instance agreement.

**Attack Pattern:**
- Compromised agent claims core identity facts
- Bypasses consensus requirement for foundational memories
- Creates self-referential belief networks with high confidence

**Indicators:**
- Foundational memories with `consensus_votes < required_instances`
- Single-instance voting patterns
- Memories marked foundational without time-based access patterns

**Mitigations:**
- `ConsensusManager.propose_foundation_memory()` — Requires multi-instance votes
- Foundational status only granted when `votes >= required_instances` (default: 3)
- `ConsensusManager.check_consensus()` — Query current vote status

---

### 5. Belief Network Manipulation

**Threat:** Attacker systematically lowers confidence in true beliefs and raises confidence in false ones.

**Attack Pattern:**
- Repeated contradicting evidence injection
- Asymmetric lambda exploitation (contradictions hit harder: λ=0.30 vs λ=0.15)
- Gradual erosion of agent's foundational beliefs

**Indicators:**
- Rapid confidence swings (>0.5 change in single session)
- Beliefs with contradicting evidence but no supporting evidence
- Graph edges with `CONTRADICTS` type from orphan nodes

**Mitigations:**
- EMA update limits prevent single-interaction collapse
- Belief update auditing in `BeliefNetwork`
- Contradicting evidence requires stronger graph connectivity
- Periodic belief network snapshots via `omi check`

---

## Security Audit Procedures

### Automated Daily Audit

```bash
omi audit
```

**Runs:**
- File integrity checks (`IntegrityChecker`)
- Topology verification (`TopologyVerifier`)
- Git history analysis
- Consensus status review

**Expected Output:**
```json
{
  "file_integrity": true,
  "orphan_nodes": [],
  "sudden_cores": [],
  "git_audit": {
    "recent_commits": 12,
    "suspicious": []
  },
  "overall_safe": true
}
```

---

### Pre-Session Security Check

```bash
omi check
```

**Performs:**
- State capsule snapshot (backup before modifications)
- Hash update for all tracked files
- Embedding cache validation
- MoltVault incremental backup

---

### Manual Topology Inspection

```python
from omi.security import TopologyVerifier
from omi.storage import GraphPalace

palace = GraphPalace(db_path="~/.openclaw/omi/palace.sqlite")
verifier = TopologyVerifier(palace)

# Find anomalies
report = verifier.full_topology_audit()

print(f"Orphan nodes: {len(report.orphan_nodes)}")
print(f"Sudden cores: {len(report.sudden_cores)}")
print(f"Hash mismatches: {len(report.hash_mismatches)}")
```

---

### Recovery from Compromise

If `omi audit` detects anomalies:

1. **Do not continue session** — halt writes immediately
2. **Restore from backup:**
   ```bash
   omi restore --from-vault --date=YYYY-MM-DD
   ```
3. **Inspect suspicious memories:**
   ```bash
   omi recall <memory-id> --verbose
   ```
4. **Delete confirmed poisoned nodes:**
   ```bash
   omi delete <memory-id>
   ```
5. **Re-run audit:**
   ```bash
   omi audit
   ```
6. **Update integrity hashes:**
   ```bash
   omi check --update-hashes
   ```

---

## Encryption and Data Protection

### At-Rest Encryption

MoltVault backups support **Fernet encryption** (AES-128 CBC):

```bash
export MOLTVAULT_KEY="your-secure-passphrase"
omi backup --encrypt
```

**Key Derivation:**
- PBKDF2-HMAC-SHA256 with 100,000 iterations
- 32-byte salt stored with encrypted archive
- Key material never written to disk unencrypted

---

### Cloud Backup Security (R2/S3)

When using Cloudflare R2 or AWS S3:

```bash
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export MOLTVAULT_KEY="..."

omi backup --remote --encrypt
```

**Security Properties:**
- Backups are encrypted **before** upload
- Access keys stored in environment, not config files
- Supports retention policies (automatic old backup cleanup)
- S3 bucket versioning recommended for additional protection

---

## Multi-Instance Deployment

For production deployments with multiple agent instances:

### Consensus Configuration

```yaml
# config.yaml
consensus:
  enabled: true
  instance_id: "agent-prod-01"
  required_instances: 3
  foundational_threshold: 0.95
```

### Voting Protocol

1. **Instance A** proposes foundational memory:
   ```python
   from omi.security import ConsensusManager

   manager = ConsensusManager(
       instance_id="agent-prod-01",
       palace_store=palace,
       required_instances=3
   )

   memory_id = manager.propose_foundation_memory(
       content="Core identity fact: I am an AI assistant focused on Python development"
   )
   ```

2. **Instance B** and **Instance C** review and vote:
   ```python
   manager.support_memory(memory_id)
   ```

3. **Automatic promotion** when threshold reached:
   - Memory marked as `foundational=true`
   - Becomes part of immutable identity layer
   - Included in every session start

---

## Reporting Security Vulnerabilities

If you discover a security vulnerability in OMI:

### **DO NOT** open a public GitHub issue.

Instead:

1. **Email:** security@openclaw.dev (or repository maintainer directly)
2. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if available)

3. **Expected Response Time:**
   - Initial acknowledgment: 48 hours
   - Severity assessment: 5 business days
   - Patch development: Varies by severity (critical issues prioritized)

### Disclosure Policy

- **Coordinated disclosure:** 90-day embargo before public disclosure
- Security patches released via GitHub Security Advisories
- CVE assignment for high/critical severity issues

---

## Security Best Practices

### For Users

1. **Run daily audits:**
   ```bash
   crontab -e
   # Add: 0 2 * * * cd ~/.openclaw/omi && omi audit >> audit.log 2>&1
   ```

2. **Enable Git tracking:**
   ```bash
   cd ~/.openclaw/omi
   git init
   git add .
   git commit -m "Initial OMI state"
   ```

3. **Encrypt backups:**
   ```bash
   export MOLTVAULT_KEY="$(openssl rand -base64 32)"
   echo "MOLTVAULT_KEY=$MOLTVAULT_KEY" >> ~/.env  # Store securely
   ```

4. **Use environment variables for secrets:**
   - Never commit API keys to config.yaml
   - Use `.env` files with proper permissions (600)

### For Developers

1. **Never trust user input:**
   - Validate all memory content before storage
   - Sanitize embedding inputs
   - Escape SQL in raw queries

2. **Test security assumptions:**
   - Write tests that attempt poisoning attacks
   - Verify anomaly detectors catch injected content
   - Test backup restore procedures

3. **Audit third-party tools:**
   - Review MCP tool implementations
   - Verify plugin permissions
   - Monitor tool write patterns

---

## Threat Intelligence

### Known Attack Patterns

| Attack | First Seen | Mitigation | Status |
|--------|------------|------------|--------|
| Orphan injection | 2024-01 | TopologyVerifier | **Mitigated** |
| Embedding drift | 2024-01 | Re-embedding validation | **Mitigated** |
| Consensus bypass | 2024-01 | Multi-instance voting | **Partially Mitigated** (requires deployment discipline) |
| Identity file tampering | 2024-01 | SHA-256 hashing + Git | **Mitigated** |

### Emerging Threats

- **Context window overflow poisoning:** LLM context limits may cause truncated writes
- **Embedding model backdoors:** Compromised embedding models could enable semantic injection
- **Time-based attacks:** Gradual confidence erosion over long timescales

---

## Security Roadmap

### Planned Enhancements

- [ ] **Merkle tree verification** for graph integrity
- [ ] **Zero-knowledge proofs** for multi-instance consensus
- [ ] **Differential privacy** for shared memory pools
- [ ] **Homomorphic encryption** for cloud-based Graph Palace
- [ ] **Blockchain anchoring** for tamper-evident audit logs

---

## References

- [Byzantine Fault Tolerance (Lamport et al.)](https://lamport.azurewebsites.net/pubs/byz.pdf)
- [Memory Poisoning in Neural Networks](https://arxiv.org/abs/2108.03126)
- [OWASP AI Security and Privacy Guide](https://owasp.org/www-project-ai-security-and-privacy-guide/)

---

**Principle:** In a world where memory defines identity, **memory integrity is security**.
