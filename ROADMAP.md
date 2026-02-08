# OMI Roadmap

## Current Status: 0.1.0-alpha
- [x] Architecture specification
- [x] Core module stubs
- [x] NVIDIA NIM integration
- [x] Issue templates
- [x] 6 tracking issues

## Sprint 1: Core Implementation (Weeks 1-2)
**Goal:** Working memory operations

### Priority: High
- [ ] **#6** Complete SQLite Graph Implementation
  - GraphPalace with FTS5 + vector search
  - Centrality calculation
  - Multi-instance consensus

- [ ] **#1** NIM Integration Testing & Validation
  - Real API testing
  - Fallback verification
  - Error handling

### Priority: Medium  
- [ ] **#2** Security Tools Implementation
  - Integrity checker
  - Topology verification
  - Basic poison detection

## Sprint 2: Integration (Weeks 3-4)
**Goal:** OpenClaw MCP integration

### Priority: High
- [ ] **#4** MCP Integration Tests
  - Tool registration
  - Heartbeat hooks
  - Context compression triggers

### Priority: Medium
- [ ] **#3** CLI Interface
  - `omi init`, `omi recall`, `omi store`
  - `omi status`, `omi audit`

## Sprint 3: Advanced (Weeks 5-6)
**Goal:** Production readiness

### Priority: Medium
- [ ] **#5** MoltVault Integration
  - Backup/restore API
  - Automated daily backups
  - Restore verification

### Priority: Low
- [ ] State capsule verification
- [ ] Episodic memory layer
- [ ] ATProto optional sync

## Key Metrics
- Embedding latency: <500ms per query
- Storage: 1M memories in <1GB
- Search: <500ms for top-10 results
- Security: Detect 95% of poisoning attempts

## Blockers
- NIM_API_KEY required for testing
- SQLite vector extension availability
- MoltVault API documentation
