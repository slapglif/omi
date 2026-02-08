"""Belief networks with confidence tracking.

Based on VesperMolt's implementation + Hindsight paper (arxiv:2512.12818).
91.4% vs 39% baseline on LongMemEval.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Evidence:
    """Evidence that supports or contradicts a belief."""
    memory_id: str
    supports: bool
    strength: float  # 0.0 to 1.0
    timestamp: str = ""


class BeliefNetwork:
    """Manages beliefs with confidence tracking and evidence chains."""
    
    def __init__(self, db_path: Path, default_half_life_days: float = 60.0):
        self.db_path = Path(db_path)
        self.default_half_life_days = default_half_life_days
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize belief network database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    evidence_count INTEGER DEFAULT 0,
                    FOREIGN KEY (id) REFERENCES memories(id)
                );
                
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY,
                    belief_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    supports BOOLEAN NOT NULL,
                    strength REAL NOT NULL CHECK (strength >= 0 AND strength <= 1),
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (belief_id) REFERENCES beliefs(id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                );
                
                CREATE TABLE IF NOT EXISTS belief_evidence_edges (
                    belief_id TEXT NOT NULL,
                    evidence_id TEXT NOT NULL,
                    PRIMARY KEY (belief_id, evidence_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence);
                CREATE INDEX IF NOT EXISTS idx_evidence_belief ON evidence(belief_id);
            """)
    
    def create_belief(self, belief_id: str, content: str, 
                     initial_confidence: float = 0.5) -> None:
        """Create a new belief with initial confidence."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO beliefs
                (id, content, confidence, created_at, last_updated, evidence_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (belief_id, content, initial_confidence, now, now))
    
    def update_confidence(self, belief_id: str, evidence: Evidence) -> float:
        """Update belief confidence with new evidence.
        
        Returns new confidence level.
        """
        # Save evidence
        evidence_id = f"{belief_id}_{evidence.memory_id}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO evidence
                (id, belief_id, memory_id, supports, strength, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (evidence_id, belief_id, evidence.memory_id, 
                  evidence.supports, evidence.strength,
                  evidence.timestamp or datetime.utcnow().isoformat()))
            
            conn.execute("""
                INSERT INTO belief_evidence_edges (belief_id, evidence_id)
                VALUES (?, ?)
                ON CONFLICT DO NOTHING
            """, (belief_id, evidence_id))
        
        # Get current confidence
        current = self.get_confidence(belief_id)
        
        # EMA update
        if evidence.supports:
            # Supporting evidence: gentle nudge up (λ=0.15)
            new_confidence = self._ema_update(current, evidence.strength, 0.15)
        else:
            # Contradicting evidence: hits twice as hard (λ=0.30)
            new_confidence = self._ema_update(current, -evidence.strength, 0.30)
        
        # Clamp to [0, 1]
        new_confidence = max(0.0, min(1.0, new_confidence))
        
        # Persist updated confidence
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE beliefs
                SET confidence = ?, last_updated = ?, evidence_count = evidence_count + 1
                WHERE id = ?
            """, (new_confidence, datetime.utcnow().isoformat(), belief_id))
        
        return new_confidence
    
    def get_confidence(self, belief_id: str) -> float:
        """Get current confidence level for belief."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT confidence FROM beliefs WHERE id = ?",
                (belief_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else 0.0
    
    def get_evidence_chain(self, belief_id: str) -> List[Evidence]:
        """Get all evidence for a belief."""
        evidence = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT memory_id, supports, strength, timestamp
                FROM evidence
                WHERE belief_id = ?
                ORDER BY timestamp
            """, (belief_id,))
            
            for row in cursor:
                evidence.append(Evidence(
                    memory_id=row[0],
                    supports=row[1],
                    strength=row[2],
                    timestamp=row[3]
                ))
        return evidence
    
    def get_contradictions(self, belief_id: str) -> List[str]:
        """Get memory IDs that contradict this belief."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT memory_id FROM evidence
                WHERE belief_id = ? AND supports = 0
            """, (belief_id,))
            return [row[0] for row in cursor]
    
    def _ema_update(self, current: float, evidence_strength: float, 
                    lambda_param: float) -> float:
        """Exponential Moving Average update.
        
        new = (1 - λ) * current + λ * evidence
        
        For supporting evidence: λ = 0.15
        For contradicting evidence: λ = 0.30
        """
        return (1 - lambda_param) * current + lambda_param * evidence_strength
    
    def detect_contradictions(self) -> List[Tuple[str, str, str]]:
        """Detect contradictory beliefs.
        
        Returns: [(belief_a, belief_b, reason)]
        """
        contradictions = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Find beliefs with overlapping evidence but opposite conclusions
            cursor = conn.execute("""
                SELECT 
                    e1.belief_id as belief_a,
                    e2.belief_id as belief_b,
                    e1.memory_id,
                    e1.supports as a_supports,
                    e2.supports as b_supports
                FROM evidence e1
                JOIN evidence e2 ON e1.memory_id = e2.memory_id
                WHERE e1.belief_id != e2.belief_id
                  AND e1.supports != e2.supports
            """)
            
            for row in cursor:
                belief_a, belief_b, memory_id, a_supports, b_supports = row
                reason = f"Both reference memory {memory_id} but draw opposite conclusions"
                contradictions.append((belief_a, belief_b, reason))
        
        return contradictions
