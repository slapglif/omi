"""
Async MCP (Model Context Protocol) tool definitions
Async/await versions for non-blocking memory operations
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import aiofiles

# Async storage tier
from .storage.async_graph_palace import AsyncGraphPalace

# Storage tier - sync for now, wrapped in async for future compatibility
from .storage.now import NowStorage
from .persistence import NOWEntry, DailyLogStore

# Belief system (still sync for now)
from .belief import BeliefNetwork, Evidence, ContradictionDetector, calculate_recency_score

# Async embeddings
from .async_embeddings import AsyncNIMEmbedder, AsyncOllamaEmbedder, AsyncEmbeddingCache

# Vault (still sync for now)
from .moltvault import MoltVault

# Events
from .events import (
    MemoryStoredEvent,
    MemoryRecalledEvent,
    BeliefUpdatedEvent,
    ContradictionDetectedEvent,
    SessionStartedEvent,
    SessionEndedEvent
)
from .event_bus import get_event_bus


class AsyncMemoryTools:
    """
    Async core memory operations (MCP tools)

    Recommended: memory_recall, memory_store
    Non-blocking async/await versions using AsyncGraphPalace and AsyncEmbeddingCache
    """

    def __init__(self, palace_store: AsyncGraphPalace,
                 embedder: AsyncNIMEmbedder,
                 cache: AsyncEmbeddingCache):
        self.palace = palace_store
        self.embedder = embedder
        self.cache = cache

    async def recall(self,
                    query: str,
                    limit: int = 10,
                    min_relevance: float = 0.7,
                    memory_type: Optional[str] = None) -> List[dict]:
        """
        memory_recall: Async semantic search with recency weighting

        Args:
            query: Natural language search query
            limit: Max results (default: 10)
            min_relevance: Similarity threshold (default: 0.7)
            memory_type: Filter by type (fact|experience|belief|decision)

        Returns:
            Memories sorted by relevance + recency
        """
        # Embed the query string first (async)
        query_embedding = await self.cache.get_or_compute(query)

        # Get candidates (async)
        candidates = await self.palace.recall(query_embedding, limit=limit*2)

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

    async def store(self,
                   content: str,
                   memory_type: str = 'experience',
                   related_to: Optional[List[str]] = None,
                   confidence: Optional[float] = None) -> str:
        """
        memory_store: Async persist memory with embedding

        Args:
            content: Memory text to store
            memory_type: Type (fact|experience|belief|decision)
            related_to: IDs of related memories (optional)
            confidence: For beliefs, 0.0-1.0

        Returns:
            memory_id: UUID for created memory
        """
        # Generate embedding with caching (async)
        embedding = await self.cache.get_or_compute(content)

        # Store in palace (async)
        memory_id = await self.palace.store_memory(
            content=content,
            memory_type=memory_type,
            confidence=confidence
        )

        # Create relationships (async)
        if related_to:
            for related_id in related_to:
                await self.palace.create_edge(memory_id, related_id, 'RELATED_TO', 0.5)

        # Emit event
        event = MemoryStoredEvent(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            confidence=confidence
        )
        get_event_bus().publish(event)

        return memory_id


class AsyncBeliefTools:
    """
    Async belief network operations
    """

    def __init__(self, belief_network: BeliefNetwork,
                 detector: ContradictionDetector):
        self.belief = belief_network
        self.detector = detector

    async def create(self,
                    content: str,
                    initial_confidence: float = 0.5) -> str:
        """
        belief_create: Async create new belief with confidence

        Args:
            content: Belief statement
            initial_confidence: Starting confidence 0.0-1.0

        Returns:
            belief_id: UUID for created belief
        """
        # Direct async call to palace since BeliefNetwork is sync
        belief_id = await self.belief.palace.store_memory(
            content=content,
            memory_type='belief',
            confidence=initial_confidence
        )
        return belief_id

    async def update(self,
                    belief_id: str,
                    evidence_memory_id: str,
                    supports: bool,
                    strength: float) -> float:
        """
        belief_update: Async add evidence, update confidence

        Uses EMA: Supporting (λ=0.15), Contradicting (λ=0.30)

        Args:
            belief_id: Belief to update
            evidence_memory_id: Source memory
            supports: True = supporting, False = contradicting
            strength: Evidence strength 0.0-1.0

        Returns:
            new_confidence: Updated confidence value
        """
        # Get old confidence before update (await async call)
        current = await self.belief.palace.get_memory(belief_id)
        old_confidence = current.confidence if current.confidence is not None else 0.5

        # Calculate new confidence using EMA
        lambda_val = (self.belief.SUPPORT_LAMBDA if supports
                     else self.belief.CONTRADICT_LAMBDA)

        # Calculate target
        if supports:
            target = min(1.0, old_confidence + strength)
        else:
            target = max(0.0, old_confidence - strength)

        # EMA update
        new_confidence = old_confidence + lambda_val * (target - old_confidence)

        # Clamp to [0, 1]
        new_confidence = max(0.0, min(1.0, new_confidence))

        # Update in palace (await async call)
        await self.belief.palace.update_belief_confidence(belief_id, new_confidence)

        # Create evidence edge (await async call)
        edge_type = 'SUPPORTS' if supports else 'CONTRADICTS'
        await self.belief.palace.create_edge(belief_id, evidence_memory_id, edge_type, strength)

        # Emit event
        event = BeliefUpdatedEvent(
            belief_id=belief_id,
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            evidence_id=evidence_memory_id
        )
        get_event_bus().publish(event)

        return new_confidence

    async def retrieve(self,
                      query: str,
                      min_confidence: Optional[float] = None) -> List[dict]:
        """
        belief_retrieve: Async get beliefs with confidence weighting

        High-confidence beliefs rank exponentially higher
        """
        # Get candidates via semantic search (await async call)
        candidates = await self.belief.palace.recall(query, memory_type='belief')

        # Apply confidence weighting
        CONFIDENCE_EXPONENT = 1.5

        weighted = []
        for belief in candidates:
            confidence = belief.get('confidence', 0.5)

            # Filter by min confidence if specified
            if min_confidence and confidence < min_confidence:
                continue

            # Apply exponential weighting
            weight = CONFIDENCE_EXPONENT ** confidence
            belief['weighted_score'] = belief.get('relevance', 0.7) * weight
            weighted.append(belief)

        # Sort by weighted score
        return sorted(weighted, key=lambda x: x['weighted_score'], reverse=True)

    async def check_contradiction(self, memory1_id: str, memory2_id: str) -> bool:
        """
        belief_check_contradiction: Async detect conflicting evidence

        Patterns: "should always" vs "should never", etc.
        """
        # Await async get_memory calls
        mem1 = await self.belief.palace.get_memory(memory1_id)
        mem2 = await self.belief.palace.get_memory(memory2_id)

        is_contradiction, pattern = self.detector.detect_contradiction_with_pattern(
            mem1.content if mem1.content else '',
            mem2.content if mem2.content else ''
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

    async def get_evidence_chain(self, belief_id: str) -> List[dict]:
        """
        belief_evidence_chain: Async show supporting/contradicting evidence

        Returns evidence with timestamps for audit
        """
        # Get all edges from belief (await async call)
        edges = await self.belief.palace.get_edges(belief_id)

        evidence_chain = []
        for edge in edges:
            # If belief is the source, target is the evidence memory
            # If belief is the target, source is the evidence memory
            if edge.source_id == belief_id:
                memory_id = edge.target_id
            else:
                memory_id = edge.source_id

            evidence = Evidence(
                memory_id=memory_id,
                supports=(edge.edge_type == 'SUPPORTS'),
                strength=edge.strength if edge.strength is not None else 0.5,
                timestamp=edge.created_at
            )
            evidence_chain.append(evidence)

        # Sort by timestamp
        evidence_chain.sort(key=lambda e: e.timestamp)

        # Convert to dict format
        return [
            {
                'memory_id': e.memory_id,
                'supports': e.supports,
                'strength': e.strength,
                'timestamp': e.timestamp.isoformat()
            }
            for e in evidence_chain
        ]


class AsyncCheckpointTools:
    """
    Async session checkpoint and recovery
    """

    def __init__(self, now_store,
                 vault: MoltVault):
        # NOWStore is now an alias for NowStorage
        self.now = now_store
        self.vault = vault

    async def now_read(self) -> dict:
        """
        now_read: Async load current operational context

        Read FIRST on session start
        """
        # Note: now.read() is currently sync, wrapped in async for future compatibility
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

    async def now_update(self,
                        current_task: Optional[str] = None,
                        recent_completions: Optional[List[str]] = None,
                        pending_decisions: Optional[List[str]] = None,
                        key_files: Optional[List[str]] = None) -> None:
        """
        now_update: Async update operational state

        Trigger: 70% context threshold, task completion
        """
        # Note: Use NowStorage.update() directly (currently sync, wrapped for future compatibility)
        self.now.update(
            current_task=current_task,
            recent_completions=recent_completions,
            pending_decisions=pending_decisions,
            key_files=key_files
        )

    async def create_capsule(self,
                            intent: str,
                            partial_plan: str) -> dict:
        """
        capsule_create: Async serialize state for recovery

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

    async def vault_backup(self, memory_content: str) -> str:
        """
        vault_backup: Async full backup to MoltVault

        Calls: POST molt-vault.com/api/v1/vault/backup
        """
        # Note: vault.backup() is currently sync, wrapped in async for future compatibility
        return self.vault.backup(memory_content)

    async def vault_restore(self, backup_id: str) -> str:
        """
        vault_restore: Async restore from MoltVault

        Pattern: POST /vault/restore
        """
        # Note: vault.restore() is currently sync, wrapped in async for future compatibility
        return self.vault.restore(backup_id)


class AsyncDailyLogTools:
    """
    Async daily log operations
    """

    def __init__(self, daily_store: DailyLogStore):
        self.daily = daily_store

    async def append(self, content: str) -> str:
        """
        daily_log_append: Async add to today's log

        Pattern: Append throughout day, continuous capture
        """
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = self.daily.log_path / f"{today}.md"

        timestamp = datetime.now().isoformat()
        entry = f"\n\n## [{timestamp}]\n\n{content}\n"

        async with aiofiles.open(file_path, "a") as f:
            await f.write(entry)

        return str(file_path)

    async def read(self, days_ago: int = 0) -> str:
        """daily_log_read: Async read specific day's log"""
        target = datetime.now() - timedelta(days=days_ago)
        file_path = self.daily.log_path / f"{target.strftime('%Y-%m-%d')}.md"

        if file_path.exists():
            async with aiofiles.open(file_path, "r") as f:
                return await f.read()
        return ""

    async def list_recent(self, days: int = 7) -> List[str]:
        """daily_log_list: Async recent log files"""
        return [str(p) for p in self.daily.list_days(days)]


class AsyncSession:
    """
    Async session context manager for memory operations.

    Manages the lifecycle of a memory operation session:
    - Emits SessionStartedEvent on entry
    - Provides access to all async tools (memory, belief, checkpoint, daily log)
    - Emits SessionEndedEvent on exit
    - Handles cleanup and graceful shutdown

    Usage:
        async with async_session(
            memory_tools=memory_tools,
            belief_tools=belief_tools,
            checkpoint_tools=checkpoint_tools,
            daily_log_tools=daily_log_tools,
            session_id="user-123"
        ) as session:
            # Use session.memory, session.belief, etc.
            results = await session.memory.recall("query")
            await session.belief.create("belief content")
    """

    def __init__(self,
                 memory_tools: AsyncMemoryTools,
                 belief_tools: Optional[AsyncBeliefTools] = None,
                 checkpoint_tools: Optional[AsyncCheckpointTools] = None,
                 daily_log_tools: Optional[AsyncDailyLogTools] = None,
                 session_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize async session context manager.

        Args:
            memory_tools: AsyncMemoryTools instance (required)
            belief_tools: AsyncBeliefTools instance (optional)
            checkpoint_tools: AsyncCheckpointTools instance (optional)
            daily_log_tools: AsyncDailyLogTools instance (optional)
            session_id: Optional session identifier
            metadata: Optional metadata for the session
        """
        self.memory = memory_tools
        self.belief = belief_tools
        self.checkpoint = checkpoint_tools
        self.daily_log = daily_log_tools
        self.session_id = session_id or f"session-{datetime.now().timestamp()}"
        self.metadata = metadata or {}
        self._start_time = None

    async def __aenter__(self):
        """
        Enter async context, emit SessionStartedEvent.

        Returns:
            self: The session instance with access to all tools
        """
        self._start_time = datetime.now()

        # Emit session started event
        event = SessionStartedEvent(
            session_id=self.session_id,
            metadata=self.metadata
        )
        get_event_bus().publish(event)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit async context, emit SessionEndedEvent and cleanup.

        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred

        Returns:
            False to propagate exceptions (standard behavior)
        """
        # Calculate session duration
        duration = None
        if self._start_time:
            duration = (datetime.now() - self._start_time).total_seconds()

        # Prepare metadata with error if present
        metadata = dict(self.metadata) if self.metadata else {}
        if exc_val:
            metadata['error'] = str(exc_val)

        # Emit session ended event
        event = SessionEndedEvent(
            session_id=self.session_id,
            duration_seconds=duration,
            metadata=metadata
        )
        get_event_bus().publish(event)

        # Return False to propagate any exception
        return False


def async_session(memory_tools: AsyncMemoryTools,
                  belief_tools: Optional[AsyncBeliefTools] = None,
                  checkpoint_tools: Optional[AsyncCheckpointTools] = None,
                  daily_log_tools: Optional[AsyncDailyLogTools] = None,
                  session_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> AsyncSession:
    """
    Create an async session context manager for memory operations.

    Convenience factory function for creating AsyncSession instances.

    Args:
        memory_tools: AsyncMemoryTools instance (required)
        belief_tools: AsyncBeliefTools instance (optional)
        checkpoint_tools: AsyncCheckpointTools instance (optional)
        daily_log_tools: AsyncDailyLogTools instance (optional)
        session_id: Optional session identifier
        metadata: Optional metadata for the session

    Returns:
        AsyncSession: Context manager instance

    Example:
        async with async_session(memory_tools=tools) as session:
            results = await session.memory.recall("query")
            memory_id = await session.memory.store("content")
    """
    return AsyncSession(
        memory_tools=memory_tools,
        belief_tools=belief_tools,
        checkpoint_tools=checkpoint_tools,
        daily_log_tools=daily_log_tools,
        session_id=session_id,
        metadata=metadata
    )


__all__ = [
    'AsyncMemoryTools',
    'AsyncBeliefTools',
    'AsyncCheckpointTools',
    'AsyncDailyLogTools',
    'AsyncSession',
    'async_session'
]
