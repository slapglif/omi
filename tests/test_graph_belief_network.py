"""
Comprehensive tests for graph.belief_network module

Tests cover:
- Belief creation with initial confidence
- Evidence tracking (supports/contradicts)
- Confidence updates with EMA
- Evidence chains
- Contradiction detection
"""

import pytest
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

from omi.graph.belief_network import BeliefNetwork, Evidence


class TestBeliefNetwork:
    """Test suite for BeliefNetwork class"""

    def test_init_creates_database(self, tmp_path):
        """Test that initialization creates database and tables"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        assert db_path.exists()
        assert bn.db_path == db_path
        assert bn.default_half_life_days == 60.0

    def test_create_belief_basic(self, tmp_path):
        """Test creating a basic belief"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        content = "Python is a good language"

        bn.create_belief(belief_id, content)

        # Verify it was created with default confidence
        confidence = bn.get_confidence(belief_id)
        assert confidence == 0.5

    def test_create_belief_with_initial_confidence(self, tmp_path):
        """Test creating belief with custom initial confidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        content = "TypeScript is the best"
        initial_confidence = 0.7

        bn.create_belief(belief_id, content, initial_confidence=initial_confidence)

        confidence = bn.get_confidence(belief_id)
        assert confidence == 0.7

    def test_get_confidence_nonexistent(self, tmp_path):
        """Test getting confidence for non-existent belief"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        fake_id = str(uuid.uuid4())
        confidence = bn.get_confidence(fake_id)

        assert confidence == 0.0

    def test_update_confidence_supporting_evidence(self, tmp_path):
        """Test updating confidence with supporting evidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test belief", initial_confidence=0.5)

        # Add supporting evidence
        evidence = Evidence(
            memory_id=str(uuid.uuid4()),
            supports=True,
            strength=0.8,
            timestamp=datetime.utcnow().isoformat()
        )

        new_confidence = bn.update_confidence(belief_id, evidence)

        # Supporting evidence should increase confidence
        assert new_confidence > 0.5
        assert new_confidence <= 1.0

    def test_update_confidence_contradicting_evidence(self, tmp_path):
        """Test updating confidence with contradicting evidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test belief", initial_confidence=0.7)

        # Add contradicting evidence
        evidence = Evidence(
            memory_id=str(uuid.uuid4()),
            supports=False,
            strength=0.6,
            timestamp=datetime.utcnow().isoformat()
        )

        new_confidence = bn.update_confidence(belief_id, evidence)

        # Contradicting evidence should decrease confidence
        assert new_confidence < 0.7
        assert new_confidence >= 0.0

    def test_update_confidence_clamping(self, tmp_path):
        """Test that confidence is clamped to [0, 1]"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test", initial_confidence=0.95)

        # Add strong supporting evidence - should clamp at 1.0
        evidence = Evidence(
            memory_id=str(uuid.uuid4()),
            supports=True,
            strength=1.0,
            timestamp=datetime.utcnow().isoformat()
        )

        new_confidence = bn.update_confidence(belief_id, evidence)
        assert new_confidence <= 1.0

    def test_get_evidence_chain_empty(self, tmp_path):
        """Test getting evidence chain for belief with no evidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test belief")

        evidence_chain = bn.get_evidence_chain(belief_id)

        assert len(evidence_chain) == 0

    def test_get_evidence_chain_with_evidence(self, tmp_path):
        """Test getting evidence chain with multiple pieces of evidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test belief")

        # Add multiple pieces of evidence
        evidences = [
            Evidence(memory_id=str(uuid.uuid4()), supports=True, strength=0.7),
            Evidence(memory_id=str(uuid.uuid4()), supports=False, strength=0.5),
            Evidence(memory_id=str(uuid.uuid4()), supports=True, strength=0.9),
        ]

        for ev in evidences:
            bn.update_confidence(belief_id, ev)

        # Get evidence chain
        chain = bn.get_evidence_chain(belief_id)

        assert len(chain) == 3
        assert all(isinstance(e, Evidence) for e in chain)

    def test_get_contradictions(self, tmp_path):
        """Test getting contradicting memories"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test belief")

        # Add supporting and contradicting evidence
        support_id = str(uuid.uuid4())
        contradict_id1 = str(uuid.uuid4())
        contradict_id2 = str(uuid.uuid4())

        bn.update_confidence(belief_id, Evidence(support_id, supports=True, strength=0.8))
        bn.update_confidence(belief_id, Evidence(contradict_id1, supports=False, strength=0.7))
        bn.update_confidence(belief_id, Evidence(contradict_id2, supports=False, strength=0.6))

        contradictions = bn.get_contradictions(belief_id)

        assert len(contradictions) == 2
        assert contradict_id1 in contradictions
        assert contradict_id2 in contradictions
        assert support_id not in contradictions

    def test_detect_contradictions_empty(self, tmp_path):
        """Test contradiction detection with no beliefs"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        contradictions = bn.detect_contradictions()

        assert len(contradictions) == 0

    def test_detect_contradictions_between_beliefs(self, tmp_path):
        """Test detecting contradictions between beliefs"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_a = str(uuid.uuid4())
        belief_b = str(uuid.uuid4())
        shared_memory = str(uuid.uuid4())

        bn.create_belief(belief_a, "Belief A")
        bn.create_belief(belief_b, "Belief B")

        # Both beliefs reference same memory but with opposite interpretations
        bn.update_confidence(belief_a, Evidence(shared_memory, supports=True, strength=0.8))
        bn.update_confidence(belief_b, Evidence(shared_memory, supports=False, strength=0.8))

        contradictions = bn.detect_contradictions()

        assert len(contradictions) > 0
        # Should find that belief_a and belief_b contradict each other
        contradiction_pairs = [(c[0], c[1]) for c in contradictions]
        assert (belief_a, belief_b) in contradiction_pairs or (belief_b, belief_a) in contradiction_pairs

    def test_ema_update_supporting(self, tmp_path):
        """Test EMA update formula for supporting evidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        current = 0.5
        evidence_strength = 0.8
        lambda_support = 0.15

        result = bn._ema_update(current, evidence_strength, lambda_support)

        # new = (1 - λ) * current + λ * evidence
        expected = (1 - lambda_support) * current + lambda_support * evidence_strength
        assert abs(result - expected) < 0.001

    def test_ema_update_contradicting(self, tmp_path):
        """Test EMA update formula for contradicting evidence"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        current = 0.7
        evidence_strength = -0.6  # Negative for contradiction
        lambda_contradict = 0.30

        result = bn._ema_update(current, evidence_strength, lambda_contradict)

        expected = (1 - lambda_contradict) * current + lambda_contradict * evidence_strength
        assert abs(result - expected) < 0.001

    def test_custom_half_life(self, tmp_path):
        """Test initializing with custom half-life"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path, default_half_life_days=90.0)

        assert bn.default_half_life_days == 90.0

    def test_evidence_without_timestamp(self, tmp_path):
        """Test adding evidence without explicit timestamp"""
        db_path = tmp_path / "beliefs.db"
        bn = BeliefNetwork(db_path)

        belief_id = str(uuid.uuid4())
        bn.create_belief(belief_id, "Test")

        # Evidence without timestamp (should use current time)
        evidence = Evidence(
            memory_id=str(uuid.uuid4()),
            supports=True,
            strength=0.7
        )

        new_confidence = bn.update_confidence(belief_id, evidence)

        assert new_confidence > 0
        chain = bn.get_evidence_chain(belief_id)
        assert len(chain) == 1
        assert chain[0].timestamp != ""
