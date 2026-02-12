"""
MCP (Model Context Protocol) tool definitions
Integrates with OpenClaw
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import base64
import hashlib

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

    @staticmethod
    def _encode_cursor(cursor_data: Dict[str, Any]) -> str:
        """
        Encode cursor data to a base64 string for pagination.

        Args:
            cursor_data: Dictionary containing cursor state (offset, query_hash, etc.)

        Returns:
            Base64-encoded cursor string
        """
        if not cursor_data:
            return ""

        # Serialize to JSON and encode to base64
        json_bytes = json.dumps(cursor_data, sort_keys=True).encode('utf-8')
        return base64.urlsafe_b64encode(json_bytes).decode('utf-8')

    @staticmethod
    def _decode_cursor(cursor: Optional[str]) -> Dict[str, Any]:
        """
        Decode a base64 cursor string to cursor data.

        Args:
            cursor: Base64-encoded cursor string (or None/empty for first page)

        Returns:
            Dictionary containing cursor state, or empty dict if cursor is invalid/empty
        """
        if not cursor:
            return {}

        try:
            # Decode from base64 and parse JSON
            json_bytes = base64.urlsafe_b64decode(cursor.encode('utf-8'))
            return json.loads(json_bytes.decode('utf-8'))
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            # Invalid cursor - return empty dict to start from beginning
            return {}
    
    def recall(self,
              query: str,
              limit: int = 10,
              min_relevance: float = 0.7,
              memory_type: Optional[str] = None,
              cursor: Optional[str] = None) -> Dict[str, Any]:
        """
        memory_recall: Semantic search with recency weighting

        Args:
            query: Natural language search query
            limit: Max results per page (default: 10, max: 500)
            min_relevance: Similarity threshold (default: 0.7)
            memory_type: Filter by type (fact|experience|belief|decision)
            cursor: Pagination cursor for next page (optional)

        Returns:
            Dictionary with:
                - memories: List of memories for this page
                - next_cursor: Cursor for next page (empty if no more results)
                - has_more: Boolean indicating if more results exist
        """
        # Validate and clamp limit
        limit = max(1, min(limit, 500))

        # Calculate query hash for cursor validation
        query_params = f"{query}|{min_relevance}|{memory_type or ''}"
        query_hash = hashlib.sha256(query_params.encode()).hexdigest()[:16]

        # Decode cursor to get offset
        cursor_data = self._decode_cursor(cursor)
        offset = cursor_data.get('offset', 0)

        # Validate cursor matches current query
        if cursor_data and cursor_data.get('query_hash') != query_hash:
            # Query changed - reset to beginning
            offset = 0

        # Generate embedding for query
        query_embedding = self.cache.get_or_compute(query)

        # Get more candidates than needed to ensure we have enough after filtering
        # Fetch enough to cover offset + limit for this page
        fetch_limit = (offset + limit) * 2
        candidate_tuples = self.palace.recall(query_embedding, limit=fetch_limit, min_relevance=min_relevance)

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

        # Apply pagination - get slice for this page
        page_results = weighted[offset:offset + limit]

        # Check if more results exist
        has_more = len(weighted) > (offset + limit)

        # Generate next cursor if more results exist
        next_cursor = ""
        if has_more:
            next_cursor_data = {
                'offset': offset + limit,
                'query_hash': query_hash,
                'limit': limit
            }
            next_cursor = self._encode_cursor(next_cursor_data)

        # Emit event
        event = MemoryRecalledEvent(
            query=query,
            result_count=len(page_results),
            top_results=page_results
        )
        get_event_bus().publish(event)

        return {
            'memories': page_results,
            'next_cursor': next_cursor,
            'has_more': has_more
        }
    
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
                 vault: MoltVault) -> None:
        # NOWStore is now an alias for NowStorage
        self.now: NowStorage = now_store
        self.vault: MoltVault = vault
    
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
