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
from .events import MemoryStoredEvent, MemoryRecalledEvent, BeliefUpdatedEvent, ContradictionDetectedEvent
from .event_bus import get_event_bus

# Vault
from .moltvault import MoltVault

# Multi-agent coordination
from .shared_namespace import SharedNamespace
from .permissions import PermissionManager, PermissionLevel
from .subscriptions import SubscriptionManager
from .audit_log import AuditLogger


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


class SharedNamespaceTools:
    """
    Shared namespace operations for multi-agent coordination

    Recommended: namespace_create, namespace_list, subscribe
    """

    def __init__(
        self,
        shared_namespace: SharedNamespace,
        permissions: PermissionManager,
        subscriptions: SubscriptionManager,
        audit_logger: AuditLogger
    ) -> None:
        self.shared_ns: SharedNamespace = shared_namespace
        self.permissions: PermissionManager = permissions
        self.subscriptions: SubscriptionManager = subscriptions
        self.audit: AuditLogger = audit_logger

    def create_namespace(
        self,
        namespace: str,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        namespace_create: Create shared namespace for multi-agent coordination

        Args:
            namespace: Namespace string (e.g., "team-alpha/research")
            created_by: Agent ID creating the namespace
            metadata: Optional metadata dictionary

        Returns:
            Created namespace information
        """
        try:
            ns_info = self.shared_ns.create(namespace, created_by, metadata)

            # Log creation
            self.audit.log(
                agent_id=created_by,
                action_type=AuditLogger.ACTION_CREATE_NAMESPACE,
                resource_type=AuditLogger.RESOURCE_NAMESPACE,
                resource_id=namespace,
                namespace=namespace,
                metadata=metadata
            )

            return ns_info.to_dict()
        except ValueError as e:
            return {'error': str(e)}

    def list_namespaces(
        self,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        namespace_list: List shared namespaces

        Args:
            agent_id: Optional agent ID to filter by creator

        Returns:
            List of namespace information dictionaries
        """
        if agent_id:
            namespaces = self.shared_ns.list_by_creator(agent_id)
        else:
            namespaces = self.shared_ns.list_all()

        return [ns.to_dict() for ns in namespaces]

    def get_namespace(self, namespace: str) -> Optional[Dict[str, Any]]:
        """
        namespace_get: Get information about a specific namespace

        Args:
            namespace: Namespace string

        Returns:
            Namespace information or None if not found
        """
        ns_info = self.shared_ns.get(namespace)
        return ns_info.to_dict() if ns_info else None

    def delete_namespace(
        self,
        namespace: str,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        namespace_delete: Delete a shared namespace (admin only)

        Requires ADMIN permission. Cascades to permissions and subscriptions.

        Args:
            namespace: Namespace string
            agent_id: Agent ID attempting deletion

        Returns:
            Success status
        """
        # Check admin permission
        if not self.permissions.can_admin(namespace, agent_id):
            return {'success': False, 'error': 'Admin permission required'}

        success = self.shared_ns.delete(namespace)

        if success:
            # Log deletion
            self.audit.log(
                agent_id=agent_id,
                action_type=AuditLogger.ACTION_DELETE_NAMESPACE,
                resource_type=AuditLogger.RESOURCE_NAMESPACE,
                resource_id=namespace,
                namespace=namespace
            )

        return {'success': success}

    def grant_permission(
        self,
        namespace: str,
        agent_id: str,
        target_agent_id: str,
        permission_level: str
    ) -> Dict[str, Any]:
        """
        permission_grant: Grant permission to agent for namespace

        Requires ADMIN permission.

        Args:
            namespace: Namespace string
            agent_id: Agent ID granting permission (must have ADMIN)
            target_agent_id: Agent ID to grant permission to
            permission_level: Permission level (read|write|admin)

        Returns:
            Permission information or error
        """
        # Check admin permission
        if not self.permissions.can_admin(namespace, agent_id):
            return {'error': 'Admin permission required'}

        try:
            level = PermissionLevel.from_string(permission_level)
            perm_info = self.permissions.grant(namespace, target_agent_id, level)

            # Log grant
            self.audit.log(
                agent_id=agent_id,
                action_type=AuditLogger.ACTION_GRANT_PERMISSION,
                resource_type=AuditLogger.RESOURCE_PERMISSION,
                resource_id=target_agent_id,
                namespace=namespace,
                metadata={'permission_level': permission_level}
            )

            return perm_info.to_dict()
        except ValueError as e:
            return {'error': str(e)}

    def revoke_permission(
        self,
        namespace: str,
        agent_id: str,
        target_agent_id: str
    ) -> Dict[str, Any]:
        """
        permission_revoke: Revoke agent permission from namespace

        Requires ADMIN permission.

        Args:
            namespace: Namespace string
            agent_id: Agent ID revoking permission (must have ADMIN)
            target_agent_id: Agent ID to revoke permission from

        Returns:
            Success status
        """
        # Check admin permission
        if not self.permissions.can_admin(namespace, agent_id):
            return {'success': False, 'error': 'Admin permission required'}

        success = self.permissions.revoke(namespace, target_agent_id)

        if success:
            # Log revoke
            self.audit.log(
                agent_id=agent_id,
                action_type=AuditLogger.ACTION_REVOKE_PERMISSION,
                resource_type=AuditLogger.RESOURCE_PERMISSION,
                resource_id=target_agent_id,
                namespace=namespace
            )

        return {'success': success}

    def list_permissions(
        self,
        namespace: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        permission_list: List permissions for namespace or agent

        Args:
            namespace: Optional namespace to filter by
            agent_id: Optional agent ID to filter by

        Returns:
            List of permission information dictionaries
        """
        if namespace:
            permissions = self.permissions.list_for_namespace(namespace)
        elif agent_id:
            permissions = self.permissions.list_for_agent(agent_id)
        else:
            return []

        return [perm.to_dict() for perm in permissions]

    def check_permission(
        self,
        namespace: str,
        agent_id: str,
        required_level: str
    ) -> bool:
        """
        permission_check: Check if agent has required permission level

        Args:
            namespace: Namespace string
            agent_id: Agent ID to check
            required_level: Required permission level (read|write|admin)

        Returns:
            True if agent has sufficient permission, False otherwise
        """
        try:
            level = PermissionLevel.from_string(required_level)
            return self.permissions.has_permission(namespace, agent_id, level)
        except ValueError:
            return False

    def subscribe(
        self,
        agent_id: str,
        event_types: List[str],
        namespace: Optional[str] = None,
        memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        subscribe: Subscribe agent to events

        Args:
            agent_id: Agent ID creating subscription
            event_types: Event types to subscribe to (e.g., ["memory.stored"])
            namespace: Optional namespace to filter by
            memory_id: Optional memory ID to filter by

        Returns:
            Subscription information or error
        """
        try:
            sub_info = self.subscriptions.subscribe(
                agent_id=agent_id,
                event_types=event_types,
                namespace=namespace,
                memory_id=memory_id
            )

            # Log subscription
            self.audit.log(
                agent_id=agent_id,
                action_type=AuditLogger.ACTION_SUBSCRIBE,
                resource_type=AuditLogger.RESOURCE_SUBSCRIPTION,
                resource_id=sub_info.id,
                namespace=namespace,
                metadata={
                    'event_types': event_types,
                    'memory_id': memory_id
                }
            )

            return sub_info.to_dict()
        except ValueError as e:
            return {'error': str(e)}

    def unsubscribe(
        self,
        agent_id: str,
        subscription_id: str
    ) -> Dict[str, Any]:
        """
        unsubscribe: Remove subscription

        Args:
            agent_id: Agent ID removing subscription
            subscription_id: Subscription ID to remove

        Returns:
            Success status
        """
        # Verify subscription belongs to agent
        sub_info = self.subscriptions.get(subscription_id)
        if not sub_info or sub_info.agent_id != agent_id:
            return {'success': False, 'error': 'Subscription not found or unauthorized'}

        success = self.subscriptions.unsubscribe(subscription_id)

        if success:
            # Log unsubscribe
            self.audit.log(
                agent_id=agent_id,
                action_type=AuditLogger.ACTION_UNSUBSCRIBE,
                resource_type=AuditLogger.RESOURCE_SUBSCRIPTION,
                resource_id=subscription_id,
                namespace=sub_info.namespace
            )

        return {'success': success}

    def list_subscriptions(
        self,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        subscription_list: List subscriptions for agent or namespace

        Args:
            agent_id: Optional agent ID to filter by
            namespace: Optional namespace to filter by

        Returns:
            List of subscription information dictionaries
        """
        if agent_id:
            subscriptions = self.subscriptions.list_for_agent(agent_id)
        elif namespace:
            subscriptions = self.subscriptions.list_for_namespace(namespace)
        else:
            return []

        return [sub.to_dict() for sub in subscriptions]

    def audit_recent(
        self,
        limit: int = 50,
        agent_id: Optional[str] = None,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        audit_recent: Get recent audit log entries

        Args:
            limit: Maximum entries to return (default: 50)
            agent_id: Optional agent ID to filter by
            namespace: Optional namespace to filter by

        Returns:
            List of audit entry dictionaries
        """
        if agent_id:
            entries = self.audit.get_by_agent(agent_id, limit=limit)
        elif namespace:
            entries = self.audit.get_by_namespace(namespace, limit=limit)
        else:
            entries = self.audit.get_recent(limit=limit)

        return [entry.to_dict() for entry in entries]


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
