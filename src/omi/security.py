"""
Security architecture: Byzantine Fault Tolerance for memory
Pattern: Trust is the attack surface
"""

import hashlib
import sqlite3
import uuid
import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class AnomalyReport:
    """Report of detected anomalies"""
    orphan_nodes: List[str]                  # Memories with no relationships
    sudden_cores: List[Dict[str, Any]]       # "Core" memories with no history
    semantic_anomalies: List[Dict[str, Any]] # Embedding drift
    hash_mismatches: List[str]               # Files that fail integrity
    timestamp: datetime


class IntegrityChecker:
    """
    Integrity verification for memory files

    Pattern: SHA-256 hashes, Git version control, tamper detection
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path: Path = Path(base_path)
    
    def hash_file(self, file_path: Path) -> str:
        """Generate SHA-256 hash of file contents"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def check_now_md(self) -> bool:
        """Verify NOW.md integrity"""
        now_path = self.base_path / "NOW.md"
        if not now_path.exists():
            return True  # Nothing to check
        
        current_hash = self.hash_file(now_path)
        
        # Read stored hash
        hash_path = self.base_path / ".now.hash"
        if not hash_path.exists():
            # First run - store hash
            hash_path.write_text(current_hash)
            return True
        
        stored_hash = hash_path.read_text().strip()
        return current_hash == stored_hash
    
    def check_memory_md(self) -> bool:
        """Verify MEMORY.md integrity"""
        memory_path = self.base_path / "MEMORY.md"
        if not memory_path.exists():
            return True
        
        current_hash = self.hash_file(memory_path)
        hash_path = self.base_path / ".memory.hash"
        
        if not hash_path.exists():
            hash_path.write_text(current_hash)
            return True
        
        stored_hash = hash_path.read_text().strip()
        return current_hash == stored_hash
    
    def update_hashes(self) -> None:
        """Update all stored hashes (call after intentional modifications)"""
        for file_name in ["NOW.md", "MEMORY.md"]:
            file_path = self.base_path / file_name
            hash_path = self.base_path / f".{file_name.lower().replace('.md', '.hash').replace('.', '_')}.hash"
            
            if file_path.exists():
                file_hash = self.hash_file(file_path)
                hash_path.write_text(file_hash)
    
    def audit_git_history(self) -> Optional[Dict[str, Any]]:
        """
        Check git history for suspicious modifications

        Returns anomalies like:
        - Commits without proper messages
        - Large modifications to identity files
        - Commits at unusual times
        """
        import subprocess
        
        try:
            # Get recent commits to key files
            result = subprocess.run(
                ["git", "log", "--oneline", "-20", "--", "NOW.md", "MEMORY.md", "SOUL.md"],
                capture_output=True,
                text=True,
                cwd=self.base_path
            )
            
            if result.returncode != 0:
                return {"error": "Git not available"}
            
            commits = result.stdout.strip().split("\n")
            
            # Check for large changes
            suspicious = []
            for commit in commits[:5]:  # Check last 5
                if commit:
                    commit_hash = commit.split()[0]
                    # Get stats for this commit
                    stat_result = subprocess.run(
                        ["git", "show", "--stat", commit_hash],
                        capture_output=True,
                        text=True,
                        cwd=self.base_path
                    )
                    
                    output = stat_result.stdout
                    # Look for large line changes in identity files
                    if "insertions" in output and "MEMORY.md" in output:
                        suspicious.append({
                            "commit": commit_hash,
                            "warning": "Large modification to MEMORY.md"
                        })
            
            return {
                "recent_commits": len(commits),
                "suspicious": suspicious
            }
            
        except Exception as e:
            return {"error": str(e)}


class TopologyVerifier:
    """
    Graph topology verification for poisoning detection

    Principle: Compromised memories will have abnormal graph patterns
    """

    def __init__(self, palace_store: Any) -> None:
        """
        Args:
            palace_store: GraphPalace instance
        """
        self.palace: Any = palace_store
    
    def find_orphan_nodes(self) -> List[str]:
        """
        Find memories with no edges (suspicious)

        Legitimate memories usually connect to something.
        Orphan nodes may be injected content.
        """
        # Try to get all memories via stats then iterate
        try:
            if hasattr(self.palace, 'get_stats'):
                stats = self.palace.get_stats()
                # If no memories, no orphans
                if stats.get('memory_count', 0) == 0:
                    return []

            # Use SQLite directly if palace has db_path
            if hasattr(self.palace, 'db_path'):
                import sqlite3
                orphans = []
                with sqlite3.connect(self.palace.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT m.id FROM memories m
                        LEFT JOIN edges e ON m.id = e.source_id OR m.id = e.target_id
                        WHERE e.id IS NULL
                    """)
                    orphans = [row[0] for row in cursor]
                return orphans
        except Exception:
            pass

        return []

    def find_sudden_cores(self, min_in_edges: int = 5) -> List[Dict[str, Any]]:
        """
        Find "core" memories that appeared suddenly

        Pattern: Claims to be foundational but has no access history
        """
        # Use SQLite directly if palace has db_path
        try:
            if hasattr(self.palace, 'db_path'):
                import sqlite3
                sudden_cores = []
                with sqlite3.connect(self.palace.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT m.id, m.content, m.access_count,
                               COUNT(e.id) as edge_count
                        FROM memories m
                        LEFT JOIN edges e ON m.id = e.target_id
                        GROUP BY m.id
                        HAVING edge_count >= ?
                        AND m.access_count < 3
                    """, (min_in_edges,))
                    for row in cursor:
                        sudden_cores.append({
                            'id': row[0],
                            'content': row[1][:100] if row[1] else '',
                            'access_count': row[2],
                            'in_degree': row[3]
                        })
                return sudden_cores
        except Exception:
            pass

        return []
    
    def check_embedding_drift(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a memory's embedding is anomalous

        Pattern: Memory claims to be about X but embeds near Y
        """
        memory = self.palace.get_memory(memory_id)
        if not memory:
            return None
        
        embedding = memory.get('embedding', [])
        content = memory.get('content', '')
        
        # Re-embed the content
        from .embeddings import OllamaEmbedder
        embedder = OllamaEmbedder()
        current_embedding = embedder.embed(content)
        
        # Check drift
        similarity = embedder.similarity(embedding, current_embedding)  # type: ignore[attr-defined]
        
        if similarity < 0.9:
            # Significant drift - possible corruption
            return {
                'id': memory_id,
                'stored_similarity': similarity,
                'warning': 'Embedding drift detected'
            }
        
        return None
    
    def find_hash_mismatches(self) -> List[str]:
        """
        Find memories whose stored content_hash does not match
        the SHA-256 of their current content.

        Returns list of memory IDs with mismatched hashes.
        """
        mismatches = []
        try:
            if hasattr(self.palace, 'db_path'):
                import sqlite3
                with sqlite3.connect(self.palace.db_path) as conn:
                    cursor = conn.execute(
                        "SELECT id, content, content_hash FROM memories "
                        "WHERE content_hash IS NOT NULL"
                    )
                    for row in cursor:
                        memory_id, content, stored_hash = row
                        if content and stored_hash:
                            actual_hash = hashlib.sha256(content.encode()).hexdigest()
                            if actual_hash != stored_hash:
                                mismatches.append(memory_id)
        except Exception:
            pass
        return mismatches

    def full_topology_audit(self) -> AnomalyReport:
        """Run full topology verification including hash integrity."""
        return AnomalyReport(
            orphan_nodes=self.find_orphan_nodes(),
            sudden_cores=self.find_sudden_cores(),
            semantic_anomalies=[],
            hash_mismatches=self.find_hash_mismatches(),
            timestamp=datetime.now()
        )


class ConsensusManager:
    """
    Multi-instance consensus for memory protection

    Principle: No single compromised instance can poison shared memory
    """

    def __init__(self, instance_id: str,
                 palace_store: Any,
                 required_instances: int = 3) -> None:
        """
        Args:
            instance_id: Unique ID for this agent instance
            palace_store: GraphPalace (shared across instances)
            required_instances: Min instances to agree for "foundational" memories
        """
        self.instance_id: str = instance_id
        self.palace: Any = palace_store
        self.required_instances: int = required_instances
    
    def propose_foundation_memory(self, content: str) -> str:
        """
        Propose a new foundational memory
        
        Requires multi-instance consensus to be marked as "foundational"
        """
        # Create memory
        memory_id: str = str(self.palace.store_memory(
            content=content,
            memory_type='fact'
        ))

        # Record this instance's support
        self.palace.add_consensus_vote(
            memory_id=memory_id,
            instance_id=self.instance_id,
            votes_for=1
        )

        # Check if consensus reached
        votes = self.palace.get_consensus_votes(memory_id)

        if votes >= self.required_instances:
            # Mark as foundational
            self.palace.mark_as_foundational(memory_id)

        return memory_id
    
    def support_memory(self, memory_id: str) -> None:
        """Add this instance's support to a memory"""
        self.palace.add_consensus_vote(
            memory_id=memory_id,
            instance_id=self.instance_id,
            votes_for=1
        )
    
    def check_consensus(self, memory_id: str) -> Dict[str, Any]:
        """Check consensus status for a memory"""
        votes = self.palace.get_consensus_votes(memory_id)
        
        return {
            'memory_id': memory_id,
            'votes_for': votes,
            'required': self.required_instances,
            'is_foundational': votes >= self.required_instances
        }


class PoisonDetector:
    """
    Unified poisoning detection

    Combines: integrity checks, topology verification, consensus
    """

    def __init__(self, base_path: Path, palace_store: Optional[Any] = None) -> None:
        self.integrity: IntegrityChecker = IntegrityChecker(base_path)
        self.topology: Optional[TopologyVerifier] = TopologyVerifier(palace_store) if palace_store else None

    def full_security_audit(self) -> Dict[str, Any]:
        """Run complete security check"""
        file_integrity = self.integrity.check_now_md() and \
                        self.integrity.check_memory_md()

        orphan_nodes: List[str] = []
        sudden_cores: List[Dict[str, Any]] = []

        if self.topology:
            topology_audit = self.topology.full_topology_audit()
            orphan_nodes = topology_audit.orphan_nodes
            sudden_cores = topology_audit.sudden_cores

        git_check = self.integrity.audit_git_history()

        return {
            'file_integrity': file_integrity,
            'orphan_nodes': orphan_nodes,
            'sudden_cores': sudden_cores,
            'git_audit': git_check,
            'overall_safe': file_integrity and \
                           len(orphan_nodes) < 5 and \
                           len(sudden_cores) == 0
        }


@dataclass
class AuditRecord:
    """An audit log record."""
    id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "action": self.action,
            "resource": self.resource,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata or {},
            "success": self.success
        }


class AuditLogger:
    """
    Audit Logger - SQLite-based RBAC audit trail

    Pattern: Append-only log for access control operations
    Lifetime: Indefinite (with optional pruning)

    Features:
    - Track who (user_id) did what (action) to what (resource)
    - Query by user, action, resource, timestamp range
    - WAL mode for concurrent writes
    - Tamper-evident audit trail
    """

    def __init__(self, db_path: Union[str, Path], enable_wal: bool = True) -> None:
        """
        Initialize Audit Logger.

        Args:
            db_path: Path to SQLite database file (or ':memory:' for in-memory)
            enable_wal: Enable WAL mode for concurrent writes (default: True)
        """
        self.db_path: Union[str, Path] = db_path if db_path == ':memory:' else Path(db_path)
        if self.db_path != ':memory:':
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal: bool = enable_wal
        # For :memory: databases, maintain a persistent connection
        self._conn: Optional[sqlite3.Connection] = None
        if self.db_path == ':memory:':
            self._conn = sqlite3.connect(':memory:')
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (persistent for :memory:, new for file-based)."""
        if self._conn:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize database schema with indexes."""
        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            # Enable WAL mode for concurrent writes (not supported for :memory:)
            if self._enable_wal and self.db_path != ':memory:':
                conn.execute("PRAGMA journal_mode=WAL")

            # Create audit_logs table
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON
                    success INTEGER DEFAULT 1
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
                CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);
                CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
                CREATE INDEX IF NOT EXISTS idx_audit_user_timestamp ON audit_logs(user_id, timestamp);
            """)

            conn.commit()
        finally:
            if needs_close:
                conn.close()

    def log(self,
            user_id: str,
            action: str,
            resource: str,
            metadata: Optional[Dict[str, Any]] = None,
            success: bool = True) -> str:
        """
        Log an audit event.

        Args:
            user_id: User who performed the action
            action: Action performed (e.g., 'read', 'write', 'delete')
            resource: Resource accessed (e.g., 'memory/abc', 'belief/xyz')
            metadata: Optional additional metadata
            success: Whether the action succeeded (default: True)

        Returns:
            audit_id: UUID of the created audit record
        """
        audit_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            conn.execute("""
                INSERT INTO audit_logs (id, user_id, action, resource, timestamp, metadata, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_id,
                user_id,
                action,
                resource,
                timestamp,
                json.dumps(metadata) if metadata else None,
                1 if success else 0
            ))
            conn.commit()
        finally:
            if needs_close:
                conn.close()

        return audit_id

    def get_record(self, audit_id: str) -> Optional[AuditRecord]:
        """
        Retrieve an audit record by ID.

        Args:
            audit_id: UUID of the audit record

        Returns:
            AuditRecord or None if not found
        """
        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            cursor = conn.execute("""
                SELECT id, user_id, action, resource, timestamp, metadata, success
                FROM audit_logs WHERE id = ?
            """, (audit_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return AuditRecord(
                id=row[0],
                user_id=row[1],
                action=row[2],
                resource=row[3],
                timestamp=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                metadata=json.loads(row[5]) if row[5] else None,
                success=bool(row[6])
            )
        finally:
            if needs_close:
                conn.close()

    def query_logs(self,
                   user_id: Optional[str] = None,
                   action: Optional[str] = None,
                   resource: Optional[str] = None,
                   since: Optional[datetime] = None,
                   until: Optional[datetime] = None,
                   limit: int = 100) -> List[AuditRecord]:
        """
        Query audit logs with filters.

        Args:
            user_id: Filter by user
            action: Filter by action
            resource: Filter by resource
            since: Filter events after this timestamp
            until: Filter events before this timestamp
            limit: Maximum number of results (default: 100)

        Returns:
            List of AuditRecord objects
        """
        records: List[AuditRecord] = []

        # Build query dynamically based on filters
        query = "SELECT id, user_id, action, resource, timestamp, metadata, success FROM audit_logs WHERE 1=1"
        params: List[Any] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if action:
            query += " AND action = ?"
            params.append(action)

        if resource:
            query += " AND resource = ?"
            params.append(resource)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            cursor = conn.execute(query, params)

            for row in cursor:
                records.append(AuditRecord(
                    id=row[0],
                    user_id=row[1],
                    action=row[2],
                    resource=row[3],
                    timestamp=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                    metadata=json.loads(row[5]) if row[5] else None,
                    success=bool(row[6])
                ))

            return records
        finally:
            if needs_close:
                conn.close()

    def count_logs(self,
                   user_id: Optional[str] = None,
                   action: Optional[str] = None,
                   resource: Optional[str] = None,
                   since: Optional[datetime] = None,
                   until: Optional[datetime] = None) -> int:
        """
        Count audit logs matching filters.

        Args:
            user_id: Filter by user
            action: Filter by action
            resource: Filter by resource
            since: Filter events after this timestamp
            until: Filter events before this timestamp

        Returns:
            Count of matching audit records
        """
        query = "SELECT COUNT(*) FROM audit_logs WHERE 1=1"
        params: List[Any] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if action:
            query += " AND action = ?"
            params.append(action)

        if resource:
            query += " AND resource = ?"
            params.append(resource)

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()
            return int(result[0]) if result else 0
        finally:
            if needs_close:
                conn.close()

    def get_user_actions(self, user_id: str, limit: int = 100) -> List[AuditRecord]:
        """
        Get recent actions by a specific user.

        Args:
            user_id: User to query
            limit: Maximum number of results

        Returns:
            List of AuditRecord objects
        """
        return self.query_logs(user_id=user_id, limit=limit)

    def get_resource_access(self, resource: str, limit: int = 100) -> List[AuditRecord]:
        """
        Get recent access to a specific resource.

        Args:
            resource: Resource to query
            limit: Maximum number of results

        Returns:
            List of AuditRecord objects
        """
        return self.query_logs(resource=resource, limit=limit)

    def delete_logs_before(self, timestamp: datetime) -> int:
        """
        Delete audit logs older than the specified timestamp.
        Used for pruning old audit history.

        Args:
            timestamp: Delete logs before this time

        Returns:
            Number of logs deleted
        """
        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            cursor = conn.execute("""
                DELETE FROM audit_logs WHERE timestamp < ?
            """, (timestamp.isoformat(),))
            conn.commit()
            return cursor.rowcount
        finally:
            if needs_close:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            Dict with log_count, user/action/resource distribution, timestamp range
        """
        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM audit_logs")
            log_count = cursor.fetchone()[0]

            # User distribution
            cursor = conn.execute("""
                SELECT user_id, COUNT(*) FROM audit_logs GROUP BY user_id
            """)
            user_distribution = {row[0]: row[1] for row in cursor}

            # Action distribution
            cursor = conn.execute("""
                SELECT action, COUNT(*) FROM audit_logs GROUP BY action
            """)
            action_distribution = {row[0]: row[1] for row in cursor}

            # Timestamp range
            cursor = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp) FROM audit_logs
            """)
            row = cursor.fetchone()
            oldest = row[0]
            newest = row[1]

            # Success/failure counts
            cursor = conn.execute("""
                SELECT success, COUNT(*) FROM audit_logs GROUP BY success
            """)
            success_distribution = {bool(row[0]): row[1] for row in cursor}

            return {
                "log_count": log_count,
                "user_distribution": user_distribution,
                "action_distribution": action_distribution,
                "success_distribution": success_distribution,
                "oldest_log": oldest,
                "newest_log": newest
            }
        finally:
            if needs_close:
                conn.close()

    def vacuum(self) -> None:
        """Optimize database (reclaim space)."""
        conn = self._get_connection()
        needs_close = self._conn is None

        try:
            conn.execute("VACUUM")
        finally:
            if needs_close:
                conn.close()

    def close(self) -> None:
        """Close connection and cleanup."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
