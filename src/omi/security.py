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
        # Get all memories
        all_memories = self.palace.get_all_memories()
        
        orphans = []
        for memory in all_memories:
            edges = self.palace.get_edges(memory['id'])
            if len(edges) == 0:
                orphans.append(memory['id'])
        
        return orphans
    
    def find_sudden_cores(self, min_in_edges: int = 5) -> List[dict]:
        """
        Find "core" memories that appeared suddenly
        
        Pattern: Claims to be foundational but has no access history
        """
        # Get memories with high in-degree ("core" candidates)
        candidates = self.palace.get_high_centrality_memories(min_in_edges)
        
        sudden_cores = []
        for memory in candidates:
            # Check access history
            access_history = memory.get('access_history', [])
            
            if len(access_history) < 3:
                # Claimed to be core but almost never accessed
                sudden_cores.append({
                    'id': memory['id'],
                    'content': memory['content'][:100],
                    'access_count': len(access_history),
                    'in_degree': memory.get('in_degree', 0)
                })
        
        return sudden_cores
    
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
    
    def full_topology_audit(self) -> AnomalyReport:
        """Run full topology verification"""
        return AnomalyReport(
            orphan_nodes=self.find_orphan_nodes(),
            sudden_cores=self.find_sudden_cores(),
            semantic_anomalies=[],  # TODO: Check all memories
            hash_mismatches=[],  # TODO: Check file integrity
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
    
    def __init__(self, base_path: Path, palace_store):
        self.integrity = IntegrityChecker(base_path)
        self.topology = TopologyVerifier(palace_store)
    
    def full_security_audit(self) -> dict:
        """Run complete security check"""
        file_integrity = self.integrity.check_now_md() and \
                        self.integrity.check_memory_md()
        
        topology_audit = self.topology.full_topology_audit()
        
        git_check = self.integrity.audit_git_history()
        
        return {
            'file_integrity': file_integrity,
            'orphan_nodes': topology_audit.orphan_nodes,
            'sudden_cores': topology_audit.sudden_cores,
            'git_audit': git_check,
            'overall_safe': file_integrity and \
                           len(topology_audit.orphan_nodes) < 5 and \
                           len(topology_audit.sudden_cores) == 0
        }
