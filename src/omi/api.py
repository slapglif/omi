"""
MCP (Model Context Protocol) tool definitions
Integrates with OpenClaw
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import numpy as np

# Storage tier - import from new modular locations
from .storage.graph_palace import GraphPalace
from .storage.now import NowStorage
from .persistence import DailyLogStore, NOWEntry

# Belief system - using legacy module for now due to API compatibility
# TODO: Migrate to .graph.belief_network when API is unified
from .belief import BeliefNetwork, Evidence, ContradictionDetector, calculate_recency_score

# Embeddings
from .embeddings import OllamaEmbedder, EmbeddingCache

# Security
from .security import IntegrityChecker, TopologyVerifier, ConsensusManager
from .events import MemoryStoredEvent, MemoryRecalledEvent, BeliefUpdatedEvent, ContradictionDetectedEvent
from .event_bus import get_event_bus

# Vault
from .moltvault import MoltVault
# from .moltvault import MoltVault

# Snapshots
from .storage.snapshots import SnapshotManager


class MemoryTools:
    """
    Core memory operations (MCP tools)

    Recommended: memory_recall, memory_store
    """

    def __init__(self, palace_store: GraphPalace,
                 embedder: OllamaEmbedder,
                 cache: EmbeddingCache) -> None:
        self.palace: GraphPalace = palace_store
        self.embedder: OllamaEmbedder = embedder
        self.cache: EmbeddingCache = cache
    
    def recall(self,
              query: str,
              limit: int = 10,
              min_relevance: float = 0.7,
              memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        memory_recall: Semantic search with recency weighting

        Args:
            query: Natural language search query
            limit: Max results (default: 10)
            min_relevance: Similarity threshold (default: 0.7)
            memory_type: Filter by type (fact|experience|belief|decision)

        Returns:
            Memories sorted by relevance + recency
        """
        # Generate embedding for query
        query_embedding = self.cache.get_or_compute(query)

        # Get candidates from palace (returns List[Tuple[Memory, float]])
        candidate_tuples = self.palace.recall(query_embedding, limit=limit*2, min_relevance=min_relevance)

        # Convert tuples to dicts and filter by type
        candidates: List[Dict[str, Any]] = []
        for memory, relevance in candidate_tuples:
            mem_dict = memory.to_dict()
            mem_dict['relevance'] = relevance
            if memory_type is None or mem_dict.get('memory_type') == memory_type:
                candidates.append(mem_dict)


        # Apply recency weighting
        half_life = 30.0  # days

        weighted: List[Dict[str, Any]] = []
        for mem in candidates:
            created_at = mem.get('created_at')
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    created_at = datetime.now()
            elif created_at is None:
                created_at = datetime.now()
            days_ago = (datetime.now() - created_at).days
            recency = calculate_recency_score(days_ago, half_life)

            final_score = (mem.get('relevance', 0.7) * 0.7) + (recency * 0.3)
            mem['final_score'] = final_score
            weighted.append(mem)

        # Sort by final score
        weighted.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        results = weighted[:limit]

        # Emit event
        event = MemoryRecalledEvent(
            query=query,
            result_count=len(results),
            top_results=results
        )
        get_event_bus().publish(event)

        return results

    def memory_recall_at(self,
                        query: str,
                        timestamp: datetime,
                        limit: int = 10,
                        memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        memory_recall_at: Point-in-time recall with semantic search

        Queries memories as they existed at a specific timestamp, then ranks
        by relevance to the query with recency weighting (relative to timestamp).

        Args:
            query: Natural language search query
            timestamp: Point in time to query (datetime object)
            limit: Max results (default: 10)
            memory_type: Filter by type (fact|experience|belief|decision)

        Returns:
            Memories sorted by relevance + recency as they existed at timestamp
        """
        # Get all memories as they existed at the timestamp
        historical_memories = self.palace.recall_at(timestamp)

        # Filter by memory type if specified
        if memory_type:
            historical_memories = [
                mem for mem in historical_memories
                if mem.memory_type == memory_type
            ]

        # Generate query embedding for semantic search
        query_embedding = self.cache.get_or_compute(query)

        # Compute cosine similarity for each historical memory
        candidates: List[Dict[str, Any]] = []
        for memory in historical_memories:
            mem_dict = memory.to_dict()

            # Compute relevance via cosine similarity if embedding exists
            if memory.embedding:
                # Normalize embeddings
                query_norm = np.linalg.norm(query_embedding)
                mem_norm = np.linalg.norm(memory.embedding)

                if query_norm > 0 and mem_norm > 0:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, memory.embedding) / (query_norm * mem_norm)
                    relevance = float(similarity)
                else:
                    relevance = 0.0
            else:
                # No embedding available, use low baseline relevance
                relevance = 0.3

            mem_dict['relevance'] = relevance
            candidates.append(mem_dict)

        # Apply recency weighting relative to the query timestamp
        half_life = 30.0  # days

        weighted: List[Dict[str, Any]] = []
        for mem in candidates:
            created_at = mem.get('created_at')
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except (ValueError, TypeError):
                    created_at = timestamp
            elif created_at is None:
                created_at = timestamp

            # Calculate days from creation to query timestamp (not current time)
            days_ago = (timestamp - created_at).days
            recency = calculate_recency_score(days_ago, half_life)

            final_score = (mem.get('relevance', 0.7) * 0.7) + (recency * 0.3)
            mem['final_score'] = final_score
            weighted.append(mem)

        # Sort by final score
        weighted.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
        results = weighted[:limit]

        # Emit event
        event = MemoryRecalledEvent(
            query=f"{query} (at {timestamp.isoformat()})",
            result_count=len(results),
            top_results=results
        )
        get_event_bus().publish(event)

        return results

    def store(self,
             content: str,
             memory_type: str = 'experience',
             related_to: Optional[List[str]] = None,
             confidence: Optional[float] = None) -> str:
        """
        memory_store: Persist memory with embedding

        Args:
            content: Memory text to store
            memory_type: Type (fact|experience|belief|decision)
            related_to: IDs of related memories (optional)
            confidence: For beliefs, 0.0-1.0

        Returns:
            memory_id: UUID for created memory
        """
        # Generate embedding with caching
        embedding = self.cache.get_or_compute(content)

        # Store in palace
        memory_id = self.palace.store_memory(
            content=content,
            memory_type=memory_type,
            confidence=confidence
        )

        # Create relationships
        if related_to:
            for related_id in related_to:
                self.palace.create_edge(memory_id, related_id, 'RELATED_TO', 0.5)

        # Emit event
        event = MemoryStoredEvent(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            confidence=confidence
        )
        get_event_bus().publish(event)

        return memory_id


class BeliefTools:
    """
    Belief network operations
    """

    def __init__(self, belief_network: BeliefNetwork,
                 detector: ContradictionDetector) -> None:
        self.belief: BeliefNetwork = belief_network
        self.detector: ContradictionDetector = detector
    
    def create(self,
              content: str,
              initial_confidence: float = 0.5) -> str:
        """
        belief_create: Create new belief with confidence
        
        Args:
            content: Belief statement
            initial_confidence: Starting confidence 0.0-1.0
        
        Returns:
            belief_id: UUID for created belief
        """
        return self.belief.create_belief(content, initial_confidence)
    
    def update(self,
             belief_id: str,
             evidence_memory_id: str,
             supports: bool,
             strength: float) -> float:
        """
        belief_update: Add evidence, update confidence

        Uses EMA: Supporting (λ=0.15), Contradicting (λ=0.30)

        Args:
            belief_id: Belief to update
            evidence_memory_id: Source memory
            supports: True = supporting, False = contradicting
            strength: Evidence strength 0.0-1.0

        Returns:
            new_confidence: Updated confidence value
        """
        # Get old confidence before update
        current = self.belief.palace.get_belief(belief_id)
        old_confidence = current.get('confidence', 0.5)

        evidence = Evidence(
            memory_id=evidence_memory_id,
            supports=supports,
            strength=strength,
            timestamp=datetime.now()
        )

        new_confidence = self.belief.update_with_evidence(belief_id, evidence)

        # Emit event
        event = BeliefUpdatedEvent(
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            evidence_id=evidence_memory_id
        )
        get_event_bus().publish(event)

        return new_confidence
    
    def retrieve(self,
               query: str,
               min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        belief_retrieve: Get beliefs with confidence weighting
        
        High-confidence beliefs rank exponentially higher
        """
        return self.belief.retrieve_with_confidence_weighting(
            query, min_confidence
        )
    
    def check_contradiction(self, memory1_id: str, memory2_id: str) -> bool:
        """
        belief_check_contradiction: Detect conflicting evidence

        Patterns: "should always" vs "should never", etc.
        """
        mem1 = self.belief.palace.get_memory(memory1_id)
        mem2 = self.belief.palace.get_memory(memory2_id)

        is_contradiction, pattern = self.detector.detect_contradiction_with_pattern(
            mem1.get('content', ''),
            mem2.get('content', '')
        )

        # Emit event if contradiction detected
        if is_contradiction:
            event = ContradictionDetectedEvent(
                memory_id_1=memory1_id,
                memory_id_2=memory2_id,
                contradiction_pattern=pattern or "unknown"
            )
            get_event_bus().publish(event)

        return is_contradiction
    
    def get_evidence_chain(self, belief_id: str) -> List[Dict[str, Any]]:
        """
        belief_evidence_chain: Show supporting/contradicting evidence
        
        Returns evidence with timestamps for audit
        """
        evidence = self.belief.get_evidence_chain(belief_id)
        return [
            {
                'memory_id': e.memory_id,
                'supports': e.supports,
                'strength': e.strength,
                'timestamp': e.timestamp.isoformat()
            }
            for e in evidence
        ]


class CheckpointTools:
    """
    Session checkpoint and recovery
    """

    def __init__(self, now_store: NowStorage,
                 vault: MoltVault,
                 snapshot_manager: Optional[SnapshotManager] = None) -> None:
        # NOWStore is now an alias for NowStorage
        self.now: NowStorage = now_store
        self.vault: MoltVault = vault
        self.snapshot_manager: Optional[SnapshotManager] = snapshot_manager
    
    def now_read(self) -> Dict[str, Any]:
        """
        now_read: Load current operational context

        Read FIRST on session start
        """
        content = self.now.read()

        # Check if content exists and is not default
        if content and content != self.now._default_content():
            try:
                entry = NOWEntry.from_markdown(content)
                return {
                    'current_task': entry.current_task,
                    'recent_completions': entry.recent_completions,
                    'pending_decisions': entry.pending_decisions,
                    'key_files': entry.key_files,
                    'timestamp': entry.timestamp.isoformat()
                }
            except Exception:
                pass
        return {}
    
    def now_update(self,
                  current_task: Optional[str] = None,
                  recent_completions: Optional[List[str]] = None,
                  pending_decisions: Optional[List[str]] = None,
                  key_files: Optional[List[str]] = None) -> None:
        """
        now_update: Update operational state

        Trigger: 70% context threshold, task completion
        """
        # Use NowStorage.update() directly
        self.now.update(
            current_task=current_task,
            recent_completions=recent_completions,
            pending_decisions=pending_decisions,
            key_files=key_files
        )
    
    def create_capsule(self,
                      intent: str,
                      partial_plan: str) -> Dict[str, Any]:
        """
        capsule_create: Serialize state for recovery
        
        Args:
            intent: Current task/plan
            partial_plan: Checkpoint of progress
        
        Returns:
            Capsule with checksum and provenance
        """
        import hashlib
        
        capsule = {
            'version': '1.0',
            'intent_hash': hashlib.sha256(intent.encode()).hexdigest()[:16],
            'partial_plan': partial_plan,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate checksum
        content = json.dumps(capsule, sort_keys=True)
        capsule['checksum'] = hashlib.sha256(content.encode()).hexdigest()
        
        return capsule
    
    def vault_backup(self, full: bool = True) -> str:
        """
        vault_backup: Full backup to MoltVault

        Args:
            full: Create full backup (default: True)

        Returns:
            backup_id: ID of created backup
        """
        metadata = self.vault.backup(full=full)
        return metadata.backup_id

    def vault_restore(self, backup_id: str) -> str:
        """
        vault_restore: Restore from MoltVault

        Args:
            backup_id: ID of backup to restore

        Returns:
            Restored directory path as string
        """
        restore_path = self.vault.restore(backup_id)
        return str(restore_path)

    def snapshot_create(self,
                       description: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        snapshot_create: Create point-in-time memory snapshot

        Creates a delta-encoded snapshot of current memory state.
        First snapshot is full, subsequent are deltas.

        Args:
            description: Optional description for the snapshot
            metadata: Optional metadata dictionary

        Returns:
            Snapshot information dict with snapshot_id, created_at, etc.

        Raises:
            RuntimeError: If snapshot manager not initialized
        """
        if not self.snapshot_manager:
            raise RuntimeError("SnapshotManager not initialized")

        snapshot_info = self.snapshot_manager.create_snapshot(
            description=description,
            metadata=metadata
        )
        return snapshot_info.to_dict()

    def snapshot_list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        snapshot_list: List all snapshots

        Args:
            limit: Optional maximum number of snapshots to return

        Returns:
            List of snapshot information dicts, newest first

        Raises:
            RuntimeError: If snapshot manager not initialized
        """
        if not self.snapshot_manager:
            raise RuntimeError("SnapshotManager not initialized")

        snapshots = self.snapshot_manager.list_snapshots(limit=limit)
        return [s.to_dict() for s in snapshots]

    def snapshot_diff(self,
                     snapshot1_id: str,
                     snapshot2_id: str) -> Dict[str, Any]:
        """
        snapshot_diff: Compare two snapshots

        Shows what changed between two snapshots:
        added/modified/deleted memories.

        Args:
            snapshot1_id: First snapshot ID (older)
            snapshot2_id: Second snapshot ID (newer)

        Returns:
            Diff information dict with added, modified, deleted lists

        Raises:
            RuntimeError: If snapshot manager not initialized
            ValueError: If either snapshot doesn't exist
        """
        if not self.snapshot_manager:
            raise RuntimeError("SnapshotManager not initialized")

        diff = self.snapshot_manager.diff_snapshots(snapshot1_id, snapshot2_id)
        return diff.to_dict()

    def snapshot_rollback(self, snapshot_id: str) -> Dict[str, Any]:
        """
        snapshot_rollback: Rollback to previous snapshot

        WARNING: This is a destructive operation that modifies
        current memory state. Create a backup snapshot first.

        Args:
            snapshot_id: Snapshot ID to rollback to

        Returns:
            Dict with changes_applied count

        Raises:
            RuntimeError: If snapshot manager not initialized
            ValueError: If snapshot doesn't exist
        """
        if not self.snapshot_manager:
            raise RuntimeError("SnapshotManager not initialized")

        changes = self.snapshot_manager.rollback_to_snapshot(snapshot_id)
        return {
            'snapshot_id': snapshot_id,
            'changes_applied': changes,
            'timestamp': datetime.now().isoformat()
        }


class SecurityTools:
    """
    Security verification (MCP tools)
    """

    def __init__(self, integrity: IntegrityChecker,
                 topology: TopologyVerifier,
                 consensus: Optional[ConsensusManager] = None) -> None:
        self.integrity: IntegrityChecker = integrity
        self.topology: TopologyVerifier = topology
        self.consensus: Optional[ConsensusManager] = consensus
    
    def integrity_check(self, scope: str = 'all') -> Dict[str, Any]:
        """
        integrity_check: Verify memory files

        Scopes: 'now' | 'daily' | 'graph' | 'all'
        """
        results: Dict[str, Any] = {
            'now_md': self.integrity.check_now_md(),
            'memory_md': self.integrity.check_memory_md()
        }
        
        if scope in ['graph', 'all']:
            audit = self.topology.full_topology_audit()
            results['topology'] = {
                'orphan_nodes': len(audit.orphan_nodes),
                'sudden_cores': len(audit.sudden_cores),
                'warnings': audit.orphan_nodes[:5] + [c['id'] for c in audit.sudden_cores[:5]]
            }
        
        results['overall_safe'] = all([
            results['now_md'],
            results['memory_md'],
            results.get('topology', {}).get('orphan_nodes', 0) < 5
        ])
        
        return results
    
    def topology_audit(self) -> Dict[str, Any]:
        """
        topology_audit: Check graph anomalies
        
        Detects: orphan nodes, sudden cores, embedding drift
        """
        audit = self.topology.full_topology_audit()
        
        return {
            'orphan_nodes_count': len(audit.orphan_nodes),
            'orphan_nodes_sample': audit.orphan_nodes[:3],
            'sudden_cores_count': len(audit.sudden_cores),
            'sudden_cores_sample': audit.sudden_cores[:2],
            'semantic_anomalies': len(audit.semantic_anomalies),
            'safe': len(audit.orphan_nodes) < 5 and len(audit.sudden_cores) == 0
        }


class DailyLogTools:
    """
    Daily log operations
    """

    def __init__(self, daily_store: DailyLogStore) -> None:
        self.daily: DailyLogStore = daily_store
    
    def append(self, content: str) -> str:
        """
        daily_log_append: Add to today's log
        
        Pattern: Append throughout day, continuous capture
        """
        file_path = self.daily.append(content)
        return str(file_path)
    
    def read(self, days_ago: int = 0) -> str:
        """daily_log_read: Read specific day's log"""
        from datetime import datetime, timedelta
        
        target = datetime.now() - timedelta(days=days_ago)
        return self.daily.read_daily(target)
    
    def list_recent(self, days: int = 7) -> List[str]:
        """daily_log_list: Recent log files"""
        return [str(p) for p in self.daily.list_days(days)]


def get_all_mcp_tools(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all MCP tools with configuration

    Returns:
        Dictionary of tool instances for OpenClaw registration
    """
    from pathlib import Path

    base_path = Path(config.get('base_path', '~/.openclaw/omi'))
    db_path = base_path / 'palace.sqlite'

    # Initialize stores
    now_store = NowStorage(base_path)
    daily_store = DailyLogStore(base_path)
    palace = GraphPalace(db_path)
    vault = MoltVault(base_path=base_path)

    # Initialize snapshot manager
    try:
        snapshot_manager = SnapshotManager(db_path)
    except FileNotFoundError:
        snapshot_manager = None

    # Initialize embedders
    embedder = OllamaEmbedder(
        model=config.get('embedding_model', 'nomic-embed-text')
    )
    cache_path = base_path / 'embeddings'
    cache = EmbeddingCache(cache_path, embedder)

    # Initialize belief network
    # BeliefNetwork and ContradictionDetector already imported at module level
    belief_net = BeliefNetwork(palace)
    detector = ContradictionDetector()

    # Initialize security
    # IntegrityChecker and TopologyVerifier already imported at module level
    integrity = IntegrityChecker(base_path)
    topology = TopologyVerifier(palace)

    # Create checkpoint tools instance
    checkpoint_tools = CheckpointTools(now_store, vault, snapshot_manager)

    # Create tool instances
    tools = {
        'memory_recall': MemoryTools(palace, embedder, cache).recall,
        'memory_store': MemoryTools(palace, embedder, cache).store,
        'belief_create': BeliefTools(belief_net, detector).create,
        'belief_update': BeliefTools(belief_net, detector).update,
        'belief_retrieve': BeliefTools(belief_net, detector).retrieve,
        'belief_evidence_chain': BeliefTools(belief_net, detector).get_evidence_chain,
        'now_read': checkpoint_tools.now_read,
        'now_update': checkpoint_tools.now_update,
        'vault_backup': checkpoint_tools.vault_backup,
        'vault_restore': checkpoint_tools.vault_restore,
        'capsule_create': checkpoint_tools.create_capsule,
        'integrity_check': SecurityTools(integrity, topology).integrity_check,
        'topology_audit': SecurityTools(integrity, topology).topology_audit,
        'daily_log_append': DailyLogTools(daily_store).append
    }

    # Add snapshot tools if snapshot manager is available
    if snapshot_manager:
        tools.update({
            'snapshot_create': checkpoint_tools.snapshot_create,
            'snapshot_list': checkpoint_tools.snapshot_list,
            'snapshot_diff': checkpoint_tools.snapshot_diff,
            'snapshot_rollback': checkpoint_tools.snapshot_rollback
        })

    return tools
