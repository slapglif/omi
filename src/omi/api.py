"""
MCP (Model Context Protocol) tool definitions
Integrates with OpenClaw
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import json

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
from .events import MemoryStoredEvent, MemoryRecalledEvent
from .event_bus import get_event_bus

# Vault
from .moltvault import MoltVault
# from .moltvault import MoltVault


class MemoryTools:
    """
    Core memory operations (MCP tools)
    
    Recommended: memory_recall, memory_store
    """
    
    def __init__(self, palace_store: GraphPalace, 
                 embedder: OllamaEmbedder,
                 cache: EmbeddingCache):
        self.palace = palace_store
        self.embedder = embedder
        self.cache = cache
    
    def recall(self,
              query: str,
              limit: int = 10,
              min_relevance: float = 0.7,
              memory_type: Optional[str] = None) -> List[dict]:
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
        # Get candidates
        candidates = self.palace.recall(query, limit=limit*2)
        
        # Filter by type
        if memory_type:
            candidates = [c for c in candidates 
                         if c.get('memory_type') == memory_type]
        
        # Apply recency weighting
        # calculate_recency_score already imported at module level
        half_life = 30.0  # days
        
        weighted = []
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
        weighted.sort(key=lambda x: x['final_score'], reverse=True)
        results = weighted[:limit]

        # Emit event
        event = MemoryRecalledEvent(
            query=query,
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
                 detector: ContradictionDetector):
        self.belief = belief_network
        self.detector = detector
    
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
        evidence = Evidence(
            memory_id=evidence_memory_id,
            supports=supports,
            strength=strength,
            timestamp=datetime.now()
        )
        
        return self.belief.update_with_evidence(belief_id, evidence)
    
    def retrieve(self,
               query: str,
               min_confidence: Optional[float] = None) -> List[dict]:
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
        
        return self.detector.detect_contradiction(
            mem1.get('content', ''),
            mem2.get('content', '')
        )
    
    def get_evidence_chain(self, belief_id: str) -> List[dict]:
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

    def __init__(self, now_store,
                 vault: MoltVault):
        # NOWStore is now an alias for NowStorage
        self.now = now_store
        self.vault = vault
    
    def now_read(self) -> dict:
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
                      partial_plan: str) -> dict:
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
    
    def vault_backup(self, memory_content: str) -> str:
        """
        vault_backup: Full backup to MoltVault
        
        Calls: POST molt-vault.com/api/v1/vault/backup
        """
        return self.vault.backup(memory_content)
    
    def vault_restore(self, backup_id: str) -> str:
        """
        vault_restore: Restore from MoltVault
        
        Pattern: POST /vault/restore
        """
        return self.vault.restore(backup_id)


class SecurityTools:
    """
    Security verification (MCP tools)
    """
    
    def __init__(self, integrity: IntegrityChecker,
                 topology: TopologyVerifier,
                 consensus: ConsensusManager = None):
        self.integrity = integrity
        self.topology = topology
        self.consensus = consensus
    
    def integrity_check(self, scope: str = 'all') -> dict:
        """
        integrity_check: Verify memory files
        
        Scopes: 'now' | 'daily' | 'graph' | 'all'
        """
        results = {
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
    
    def topology_audit(self) -> dict:
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
    
    def __init__(self, daily_store: DailyLogStore):
        self.daily = daily_store
    
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


def get_all_mcp_tools(config: dict) -> dict:
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
    vault = MoltVault(api_key=config.get('vault_api_key'), base_path=base_path)
    
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
    
    # Create tool instances
    return {
        'memory_recall': MemoryTools(palace, embedder, cache).recall,
        'memory_store': MemoryTools(palace, embedder, cache).store,
        'belief_create': BeliefTools(belief_net, detector).create,
        'belief_update': BeliefTools(belief_net, detector).update,
        'belief_retrieve': BeliefTools(belief_net, detector).retrieve,
        'belief_evidence_chain': BeliefTools(belief_net, detector).get_evidence_chain,
        'now_read': CheckpointTools(now_store, vault).now_read,
        'now_update': CheckpointTools(now_store, vault).now_update,
        'vault_backup': CheckpointTools(now_store, vault).vault_backup,
        'vault_restore': CheckpointTools(now_store, vault).vault_restore,
        'integrity_check': SecurityTools(integrity, topology).integrity_check,
        'topology_audit': SecurityTools(integrity, topology).topology_audit,
        'daily_log_append': DailyLogTools(daily_store).append,
        'capsule_create': CheckpointTools(now_store, vault).create_capsule
    }
