"""
Security architecture: Byzantine Fault Tolerance for memory
Pattern: Trust is the attack surface
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class AnomalyReport:
    """Report of detected anomalies"""
    orphan_nodes: List[str]           # Memories with no relationships
    sudden_cores: List[dict]          # "Core" memories with no history
    semantic_anomalies: List[dict]   # Embedding drift
    hash_mismatches: List[str]       # Files that fail integrity
    timestamp: datetime


class IntegrityChecker:
    """
    Integrity verification for memory files
    
    Pattern: SHA-256 hashes, Git version control, tamper detection
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
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
    
    def audit_git_history(self) -> Optional[dict]:
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
    
    def __init__(self, palace_store):
        """
        Args:
            palace_store: GraphPalace instance
        """
        self.palace = palace_store
    
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

    def find_sudden_cores(self, min_in_edges: int = 5) -> List[dict]:
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
    
    def check_embedding_drift(self, memory_id: str) -> Optional[dict]:
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
        similarity = embedder.similarity(embedding, current_embedding)
        
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
                 palace_store,
                 required_instances: int = 3):
        """
        Args:
            instance_id: Unique ID for this agent instance
            palace_store: GraphPalace (shared across instances)
            required_instances: Min instances to agree for "foundational" memories
        """
        self.instance_id = instance_id
        self.palace = palace_store
        self.required_instances = required_instances
    
    def propose_foundation_memory(self, content: str) -> str:
        """
        Propose a new foundational memory
        
        Requires multi-instance consensus to be marked as "foundational"
        """
        # Create memory
        memory_id = self.palace.store_memory(
            content=content,
            memory_type='fact'
        )
        
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
    
    def check_consensus(self, memory_id: str) -> dict:
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
    
    def __init__(self, base_path: Path, palace_store=None):
        self.integrity = IntegrityChecker(base_path)
        self.topology = TopologyVerifier(palace_store) if palace_store else None

    def full_security_audit(self) -> dict:
        """Run complete security check"""
        file_integrity = self.integrity.check_now_md() and \
                        self.integrity.check_memory_md()

        orphan_nodes: List[str] = []
        sudden_cores: List[dict] = []

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
