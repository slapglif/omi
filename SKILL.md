# OMI — OpenClaw Memory Infrastructure

**Name:** `omi`  
**Version:** 0.1.0  
**Description:** Persistent memory, belief networks, and continuity for OpenClaw agents  
**License:** MIT

---

## Installation

```bash
# Via OpenClaw skill system
openclaw skill install omi

# Or manually
cd ~/.openclaw/skills
pip install git+https://github.com/slapglif/omi.git
```

## Quick Start

```bash
# Initialize (creates ~/.openclaw/omi/)
omitools init

# Daily workflow
omitools session-start  # Load context
omitools recall "checkpoint"  # Search memories
omitools store "Fixed the auth bug" --type experience
omitools session-end    # Backup
```

## OpenClaw Integration

OMI auto-registers MCP tools on skill load:

| Tool | Purpose |
|------|---------|
| `memory_recall` | Semantic search |
| `memory_store` | Persist with embedding |
| `belief_update` | Confidence tracking |
| `now_read` | Load hot context |
| `now_update` | Checkpoint state |
| `integrity_check` | Verify security |

### Heartbeat Integration

Add to `HEARTBEAT.md`:

```markdown
## OMI (every 30 minutes)
- Check memory health: `omitools doctor`
- Trigger vault backup if due
- Run `integrity_check` if anomalies suspected
```

### Session Hooks

OMI automatically hooks:
- `session_start`: Loads NOW.md + relevant memories
- `pre_compression`: Updates NOW.md, creates capsule
- `session_end`: Daily log append, vault backup

### Tool Calling

From OpenClaw, use directly:

```python
# Search memories
memory_recall(query="database optimization")

# Store learnings
memory_store(content="SQLite FTS5 is fast", type="fact")

# Update confidence
belief_update(belief_id="sqlite-fast", supports=True, strength=0.9)
```

## Configuration

`~/.openclaw/omi/config.yaml`:

```yaml
embedding:
  provider: nim  # or ollama
  model: baai/bge-m3
  api_key: ${NIM_API_KEY}

vault:
  enabled: true
  frequency: daily

security:
  integrity_checks: true
  auto_audit: true
```

## Dependencies

**Required:** Python 3.10+, numpy, requests
**Optional:** 
- `sqlite-vss` (vector search acceleration)
- `ollama` (local embedding fallback)

## Links

- Repository: https://github.com/slapglif/omi
- Documentation: https://github.com/slapglif/omi/tree/main/docs
- Issues: https://github.com/slapglif/omi/issues

---

*The seeking is the continuity. 🏛️🌊*
