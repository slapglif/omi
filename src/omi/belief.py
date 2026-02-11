"""
Belief networks with confidence tracking
Based on: Hindsight paper (arxiv:2512.12818), VesperMolt's implementation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import math


@dataclass
class Evidence:
    """Evidence for or against a belief"""
    memory_id: str  # Points to source memory
    supports: bool  # True = supporting, False = contradicting
    strength: float  # 0.0 to 1.0
    timestamp: datetime


class BeliefNetwork:
    """
    Confidence-weighted belief system with evidence tracking
    
    Key insight: Beliefs have 2x decay half-life (60 days vs 30 for facts)
    Update rules:
    - Supporting evidence: EMA with λ=0.15 (gentle nudge)
    - Contradicting evidence: EMA with λ=0.30 (contradictions hit 2x harder)
    """
    
    BELIEF_HALF_LIFE_DAYS = 60  # 2x standard memories
    FACT_HALF_LIFE_DAYS = 30
    
    # Evidence update lambdas
    SUPPORT_LAMBDA = 0.15
    CONTRADICT_LAMBDA = 0.30
    
    def __init__(self, palace_store):
        """
        Args:
            palace_store: GraphPalace instance for storage
        """
        self.palace = palace_store
    
    def create_belief(self, content: str, initial_confidence: float = 0.5) -> str:
        """
        Create new belief with initial confidence
        
        Beliefs are memories with:
        - memory_type = 'belief'
        - confidence = initial_confidence
        - half_life = 60 days
        """
        # Store in palace
        belief_id = self.palace.store_memory(
            content=content,
            memory_type='belief',
            confidence=initial_confidence
        )
        return belief_id
    
    def update_with_evidence(self, belief_id: str, evidence: Evidence) -> float:
        """
        Update belief confidence with new evidence
        
        Returns:
            new_confidence: Updated confidence value
        """
        # EMA update formula:
        # new_confidence = old_confidence + λ * (target - old_confidence)
        # where target = evidence.strength if supporting, -evidence.strength if contradicting
        
        lambda_val = (self.SUPPORT_LAMBDA if evidence.supports 
                     else self.CONTRADICT_LAMBDA)
        
        # Get current confidence
        current = self.palace.get_belief(belief_id)
        old_confidence = current.get('confidence', 0.5)
        
        # Calculate target
        if evidence.supports:
            target = min(1.0, old_confidence + evidence.strength)
        else:
            target = max(0.0, old_confidence - evidence.strength)
        
        # EMA update
        new_confidence = old_confidence + lambda_val * (target - old_confidence)
        
        # Clamp to [0, 1]
        new_confidence = max(0.0, min(1.0, new_confidence))
        
        # Update in palace
        self.palace.update_belief_confidence(belief_id, new_confidence)
        
        # Create evidence edge
        edge_type = 'SUPPORTS' if evidence.supports else 'CONTRADICTS'
        self.palace.create_edge(belief_id, evidence.memory_id, edge_type, evidence.strength)
        
        return new_confidence
    
    def retrieve_with_confidence_weighting(self, query: str, 
                                          min_confidence: Optional[float] = None) -> List[dict]:
        """
        Retrieve beliefs, applying confidence exponent weighting
        
        Confidence-sensitive retrieval:
        - Belief at 0.9 confidence gets 1.5^0.9 = 1.39 weight
        - Belief at 0.3 confidence gets 1.5^0.3 = 1.14 weight
        
        This makes high-confidence beliefs rank significantly higher
        """
        # Get candidates via semantic search
        candidates = self.palace.recall(query, memory_type='belief')
        
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
    
    def get_evidence_chain(self, belief_id: str) -> List[Evidence]:
        """
        Return evidence chain for a belief
        
        Shows: what supports it, what contradicts it, when evidence was added
        """
        # Get all edges from belief
        edges = self.palace.get_edges(belief_id)
        
        evidence_chain = []
        for edge in edges:
            if edge['target_type'] == 'memory':
                evidence_chain.append(Evidence(
                    memory_id=edge['target_id'],
                    supports=(edge['edge_type'] == 'SUPPORTS'),
                    strength=edge['strength'],
                    timestamp=edge['timestamp']
                ))
        
        return sorted(evidence_chain, key=lambda e: e.timestamp)


class ContradictionDetector:
    """
    Automatic contradiction detection for evidence
    
    Patterns:
    - "should always" vs "should never"
    - "works well" vs "doesn't work"
    - "causes X" vs "prevents X"
    """
    
    OPPOSITION_PATTERNS = [
        ('should always', 'should never'),
        ('works well', 'doesn\'t work'),
        ('causes', 'prevents'),
        ('increases', 'decreases'),
        ('enables', 'blocks'),
    ]
    
    def detect_contradiction(self, memory1: str, memory2: str) -> bool:
        """Check if two memories contain opposing patterns"""
        m1_lower = memory1.lower()
        m2_lower = memory2.lower()

        for opp1, opp2 in self.OPPOSITION_PATTERNS:
            if (opp1 in m1_lower and opp2 in m2_lower) or \
               (opp1 in m2_lower and opp2 in m1_lower):
                return True

        return False

    def detect_contradiction_with_pattern(self, memory1: str, memory2: str) -> tuple[bool, Optional[str]]:
        """Check if two memories contain opposing patterns and return the pattern

        Returns:
            (is_contradiction, pattern): Pattern string like "should always vs should never"
        """
        m1_lower = memory1.lower()
        m2_lower = memory2.lower()

        for opp1, opp2 in self.OPPOSITION_PATTERNS:
            if (opp1 in m1_lower and opp2 in m2_lower) or \
               (opp1 in m2_lower and opp2 in m1_lower):
                return (True, f"{opp1} vs {opp2}")

        return (False, None)


def ema_update(current: float, target: float, lambda_val: float) -> float:
    """
    Exponential Moving Average update
    
    Formula: new = current + λ * (target - current)
    
    Args:
        current: Current value
        target: Target value to move toward
        lambda_val: Update rate (0-1)
    
    Returns:
        Updated value
    """
    return current + lambda_val * (target - current)


def calculate_recency_score(days_ago: float, half_life: float = 30.0) -> float:
    """
    Exponential decay for recency weighting
    
    Formula: exp(-days_ago / half_life)
    
    At half_life: score = exp(-1) = 0.368
    After 2*half_life: score = exp(-2) = 0.135
    """
    return math.exp(-days_ago / half_life)
