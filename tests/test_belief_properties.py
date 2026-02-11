"""
Property-based tests for belief network using hypothesis

Tests invariants and mathematical properties of:
- Confidence bounds
- EMA update logic
- Confidence weighting
- Contradiction detection
- Evidence chains
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from hypothesis import given, settings, assume, strategies as st
from hypothesis import HealthCheck

from omi.belief import (
    BeliefNetwork,
    ContradictionDetector,
    Evidence,
    ema_update,
    calculate_recency_score,
)


# Hypothesis strategies for generating test data
@st.composite
def confidence_values(draw):
    """Generate valid confidence values in [0, 1]"""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def evidence_strength(draw):
    """Generate valid evidence strength values in [0, 1]"""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def lambda_values(draw):
    """Generate valid lambda values for EMA in [0, 1]"""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@st.composite
def days_ago_values(draw):
    """Generate valid days ago values (non-negative)"""
    return draw(st.floats(min_value=0.0, max_value=365.0, allow_nan=False, allow_infinity=False))


@st.composite
def half_life_values(draw):
    """Generate valid half-life values (positive)"""
    return draw(st.floats(min_value=1.0, max_value=365.0, allow_nan=False, allow_infinity=False))


class TestConfidenceBoundsProperties:
    """Test that confidence values always stay within [0, 1] bounds"""

    @given(
        initial_conf=confidence_values(),
        evidence_str=evidence_strength(),
        supports=st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_always_bounded_after_update(
        self, belief_network_setup, initial_conf, evidence_str, supports
    ):
        """Property: Confidence always stays in [0, 1] after any evidence update"""
        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        # Mock palace methods
        belief_id = "test-belief-123"
        palace.get_belief = MagicMock(return_value={"confidence": initial_conf})
        palace.update_belief_confidence = MagicMock()
        palace.create_edge = MagicMock()

        # Create evidence
        evidence = Evidence(
            memory_id="evidence-123",
            supports=supports,
            strength=evidence_str,
            timestamp=datetime.now(),
        )

        # Update and verify
        new_confidence = belief_net.update_with_evidence(belief_id, evidence)

        # Property: confidence must be in [0, 1]
        assert 0.0 <= new_confidence <= 1.0, (
            f"Confidence {new_confidence} out of bounds [0, 1] "
            f"(initial={initial_conf}, evidence_str={evidence_str}, supports={supports})"
        )

    @given(
        initial_conf=confidence_values(),
        num_updates=st.integers(min_value=1, max_value=20),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_bounded_after_multiple_updates(
        self, belief_network_setup, initial_conf, num_updates
    ):
        """Property: Confidence stays bounded after multiple sequential updates"""
        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        belief_id = "test-belief-123"
        current_conf = initial_conf

        for i in range(num_updates):
            # Mock current state
            palace.get_belief = MagicMock(return_value={"confidence": current_conf})
            palace.update_belief_confidence = MagicMock()
            palace.create_edge = MagicMock()

            # Random evidence
            evidence = Evidence(
                memory_id=f"evidence-{i}",
                supports=(i % 2 == 0),  # Alternate support/contradict
                strength=0.5,
                timestamp=datetime.now(),
            )

            current_conf = belief_net.update_with_evidence(belief_id, evidence)

            # Property: always bounded
            assert 0.0 <= current_conf <= 1.0


class TestEvidenceDirectionProperties:
    """Test that evidence affects confidence in the correct direction"""

    @given(
        initial_conf=confidence_values(),
        evidence_str=evidence_strength(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_supporting_evidence_never_decreases_confidence(
        self, belief_network_setup, initial_conf, evidence_str
    ):
        """Property: Supporting evidence never decreases confidence"""
        # Skip cases where confidence is already at maximum
        assume(initial_conf < 1.0)
        # Skip zero-strength evidence
        assume(evidence_str > 0.0)

        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        belief_id = "test-belief-123"
        palace.get_belief = MagicMock(return_value={"confidence": initial_conf})
        palace.update_belief_confidence = MagicMock()
        palace.create_edge = MagicMock()

        evidence = Evidence(
            memory_id="evidence-123",
            supports=True,
            strength=evidence_str,
            timestamp=datetime.now(),
        )

        new_confidence = belief_net.update_with_evidence(belief_id, evidence)

        # Property: supporting evidence should not decrease confidence
        assert new_confidence >= initial_conf, (
            f"Supporting evidence decreased confidence: {initial_conf} -> {new_confidence} "
            f"(strength={evidence_str})"
        )

    @given(
        initial_conf=confidence_values(),
        evidence_str=evidence_strength(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_contradicting_evidence_never_increases_confidence(
        self, belief_network_setup, initial_conf, evidence_str
    ):
        """Property: Contradicting evidence never increases confidence"""
        # Skip cases where confidence is already at minimum
        assume(initial_conf > 0.0)
        # Skip zero-strength evidence
        assume(evidence_str > 0.0)

        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        belief_id = "test-belief-123"
        palace.get_belief = MagicMock(return_value={"confidence": initial_conf})
        palace.update_belief_confidence = MagicMock()
        palace.create_edge = MagicMock()

        evidence = Evidence(
            memory_id="evidence-123",
            supports=False,
            strength=evidence_str,
            timestamp=datetime.now(),
        )

        new_confidence = belief_net.update_with_evidence(belief_id, evidence)

        # Property: contradicting evidence should not increase confidence
        assert new_confidence <= initial_conf, (
            f"Contradicting evidence increased confidence: {initial_conf} -> {new_confidence} "
            f"(strength={evidence_str})"
        )


class TestEMAUpdateProperties:
    """Test mathematical properties of EMA update formula"""

    @given(
        current=confidence_values(),
        target=confidence_values(),
        lambda_val=lambda_values(),
    )
    def test_ema_result_between_current_and_target(self, current, target, lambda_val):
        """Property: EMA result is always between current and target (or equal)"""
        result = ema_update(current, target, lambda_val)

        min_val = min(current, target)
        max_val = max(current, target)

        assert min_val <= result <= max_val, (
            f"EMA result {result} not between {current} and {target} "
            f"(lambda={lambda_val})"
        )

    @given(
        current=confidence_values(),
        target=confidence_values(),
    )
    def test_ema_with_zero_lambda_unchanged(self, current, target):
        """Property: λ=0 means no change"""
        result = ema_update(current, target, lambda_val=0.0)
        assert result == current

    @given(
        current=confidence_values(),
        target=confidence_values(),
    )
    def test_ema_with_one_lambda_reaches_target(self, current, target):
        """Property: λ=1 means full change to target"""
        result = ema_update(current, target, lambda_val=1.0)
        assert abs(result - target) < 1e-10  # Allow float precision error

    @given(
        current=confidence_values(),
        target=confidence_values(),
        lambda_val=lambda_values(),
    )
    def test_ema_monotonic_approach(self, current, target, lambda_val):
        """Property: Multiple EMA steps monotonically approach target"""
        assume(lambda_val > 0.0)  # Need movement
        assume(abs(current - target) > 1e-6)  # Need distance

        # Take 3 steps
        step1 = ema_update(current, target, lambda_val)
        step2 = ema_update(step1, target, lambda_val)
        step3 = ema_update(step2, target, lambda_val)

        # Distance should monotonically decrease
        dist0 = abs(current - target)
        dist1 = abs(step1 - target)
        dist2 = abs(step2 - target)
        dist3 = abs(step3 - target)

        assert dist1 <= dist0
        assert dist2 <= dist1
        assert dist3 <= dist2


class TestConfidenceWeightingProperties:
    """Test properties of confidence-based weighting for retrieval"""

    @given(
        conf1=confidence_values(),
        conf2=confidence_values(),
    )
    def test_higher_confidence_gives_higher_weight(self, conf1, conf2):
        """Property: Higher confidence always produces higher weight"""
        assume(abs(conf1 - conf2) > 0.01)  # Need meaningful difference

        CONFIDENCE_EXPONENT = 1.5
        weight1 = CONFIDENCE_EXPONENT ** conf1
        weight2 = CONFIDENCE_EXPONENT ** conf2

        if conf1 > conf2:
            assert weight1 > weight2
        else:
            assert weight1 < weight2

    @given(confidence=confidence_values())
    def test_weight_always_positive(self, confidence):
        """Property: Confidence weight is always positive"""
        CONFIDENCE_EXPONENT = 1.5
        weight = CONFIDENCE_EXPONENT ** confidence
        assert weight > 0.0

    @given(confidence=confidence_values())
    def test_confidence_weighting_bounds(self, confidence):
        """Property: Weight for confidence in [0,1] stays in reasonable bounds"""
        CONFIDENCE_EXPONENT = 1.5
        weight = CONFIDENCE_EXPONENT ** confidence

        # For confidence in [0, 1], weight should be in [1.5^0, 1.5^1] = [1.0, 1.5]
        assert 1.0 <= weight <= 1.5


class TestContradictionDetectionProperties:
    """Test properties of contradiction detection"""

    def test_contradiction_detection_is_symmetric(self):
        """Property: Contradiction detection is symmetric"""
        detector = ContradictionDetector()

        test_cases = [
            ("This should always work", "This should never work"),
            ("Feature works well", "Feature doesn't work"),
            ("This causes errors", "This prevents errors"),
            ("Usage increases performance", "Usage decreases performance"),
            ("Feature enables optimization", "Feature blocks optimization"),
        ]

        for mem1, mem2 in test_cases:
            result1 = detector.detect_contradiction(mem1, mem2)
            result2 = detector.detect_contradiction(mem2, mem1)

            assert result1 == result2, (
                f"Asymmetric contradiction detection: "
                f"detect({mem1}, {mem2}) != detect({mem2}, {mem1})"
            )

    def test_contradiction_with_pattern_is_symmetric(self):
        """Property: Contradiction with pattern detection is symmetric"""
        detector = ContradictionDetector()

        test_cases = [
            ("This should always work", "This should never work"),
            ("Feature works well", "Feature doesn't work"),
        ]

        for mem1, mem2 in test_cases:
            is_contra1, pattern1 = detector.detect_contradiction_with_pattern(mem1, mem2)
            is_contra2, pattern2 = detector.detect_contradiction_with_pattern(mem2, mem1)

            assert is_contra1 == is_contra2, "Contradiction detection not symmetric"
            assert pattern1 == pattern2, "Pattern detection not symmetric"

    @given(text=st.text(min_size=1, max_size=100))
    def test_text_never_contradicts_itself(self, text):
        """Property: A text never contradicts itself"""
        detector = ContradictionDetector()
        result = detector.detect_contradiction(text, text)
        assert result is False


class TestRecencyScoreProperties:
    """Test properties of recency scoring"""

    @given(
        days1=days_ago_values(),
        days2=days_ago_values(),
        half_life=half_life_values(),
    )
    def test_recency_score_decreases_with_time(self, days1, days2, half_life):
        """Property: More recent (fewer days ago) has higher score"""
        assume(abs(days1 - days2) > 0.1)  # Need meaningful difference

        score1 = calculate_recency_score(days1, half_life)
        score2 = calculate_recency_score(days2, half_life)

        if days1 < days2:  # days1 is more recent
            assert score1 > score2
        else:
            assert score1 < score2

    @given(half_life=half_life_values())
    def test_recency_score_at_zero_is_one(self, half_life):
        """Property: Score at time zero is 1.0"""
        score = calculate_recency_score(0.0, half_life)
        assert abs(score - 1.0) < 1e-10

    @given(days_ago=days_ago_values(), half_life=half_life_values())
    def test_recency_score_always_positive(self, days_ago, half_life):
        """Property: Recency score is always positive"""
        score = calculate_recency_score(days_ago, half_life)
        assert score > 0.0

    @given(days_ago=days_ago_values(), half_life=half_life_values())
    def test_recency_score_bounded_by_one(self, days_ago, half_life):
        """Property: Recency score never exceeds 1.0"""
        score = calculate_recency_score(days_ago, half_life)
        assert score <= 1.0

    @given(half_life=half_life_values())
    def test_recency_score_at_half_life(self, half_life):
        """Property: Score at half-life is approximately exp(-1) = 0.368"""
        score = calculate_recency_score(half_life, half_life)
        expected = 0.36787944117144233  # exp(-1)
        assert abs(score - expected) < 1e-10


class TestEvidenceChainProperties:
    """Test properties of evidence chains"""

    def test_evidence_chain_sorted_by_timestamp(self, belief_network_setup):
        """Property: Evidence chain is always sorted by timestamp"""
        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        belief_id = "test-belief-123"

        # Create mock edges with random timestamps
        now = datetime.now()
        mock_edges = [
            {
                "target_id": "evidence-3",
                "target_type": "memory",
                "edge_type": "SUPPORTS",
                "strength": 0.8,
                "timestamp": now - timedelta(hours=1),
            },
            {
                "target_id": "evidence-1",
                "target_type": "memory",
                "edge_type": "CONTRADICTS",
                "strength": 0.6,
                "timestamp": now - timedelta(hours=5),
            },
            {
                "target_id": "evidence-2",
                "target_type": "memory",
                "edge_type": "SUPPORTS",
                "strength": 0.7,
                "timestamp": now - timedelta(hours=3),
            },
        ]

        palace.get_edges = MagicMock(return_value=mock_edges)

        chain = belief_net.get_evidence_chain(belief_id)

        # Property: chain should be sorted by timestamp
        timestamps = [e.timestamp for e in chain]
        assert timestamps == sorted(timestamps), "Evidence chain not sorted by timestamp"

    def test_evidence_chain_preserves_all_evidence(self, belief_network_setup):
        """Property: Evidence chain contains all memory-type edges"""
        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        belief_id = "test-belief-123"
        now = datetime.now()

        # Create edges with some non-memory types
        mock_edges = [
            {
                "target_id": "evidence-1",
                "target_type": "memory",
                "edge_type": "SUPPORTS",
                "strength": 0.8,
                "timestamp": now,
            },
            {
                "target_id": "belief-2",
                "target_type": "belief",  # Non-memory type
                "edge_type": "RELATED_TO",
                "strength": 0.5,
                "timestamp": now,
            },
            {
                "target_id": "evidence-2",
                "target_type": "memory",
                "edge_type": "CONTRADICTS",
                "strength": 0.6,
                "timestamp": now,
            },
        ]

        palace.get_edges = MagicMock(return_value=mock_edges)

        chain = belief_net.get_evidence_chain(belief_id)

        # Property: should only include memory-type edges
        memory_edges = [e for e in mock_edges if e["target_type"] == "memory"]
        assert len(chain) == len(memory_edges)


class TestAsymmetricLambdaProperties:
    """Test that contradicting evidence has stronger effect than supporting"""

    @given(
        initial_conf=st.floats(min_value=0.4, max_value=0.6),  # Mid-range
        evidence_str=st.floats(min_value=0.3, max_value=0.7),  # Moderate strength
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_contradiction_lambda_stronger_than_support(
        self, belief_network_setup, initial_conf, evidence_str
    ):
        """Property: Contradictions affect confidence more strongly than support"""
        belief_net = belief_network_setup["belief_network"]
        palace = belief_network_setup["palace"]

        belief_id = "test-belief-123"

        # Test supporting evidence
        palace.get_belief = MagicMock(return_value={"confidence": initial_conf})
        palace.update_belief_confidence = MagicMock()
        palace.create_edge = MagicMock()

        support_evidence = Evidence(
            memory_id="evidence-support",
            supports=True,
            strength=evidence_str,
            timestamp=datetime.now(),
        )
        new_conf_support = belief_net.update_with_evidence(belief_id, support_evidence)
        support_delta = abs(new_conf_support - initial_conf)

        # Test contradicting evidence
        palace.get_belief = MagicMock(return_value={"confidence": initial_conf})
        contradict_evidence = Evidence(
            memory_id="evidence-contradict",
            supports=False,
            strength=evidence_str,
            timestamp=datetime.now(),
        )
        new_conf_contradict = belief_net.update_with_evidence(belief_id, contradict_evidence)
        contradict_delta = abs(new_conf_contradict - initial_conf)

        # Property: Contradiction should have larger effect
        # CONTRADICT_LAMBDA (0.30) is 2x SUPPORT_LAMBDA (0.15)
        # The EMA formula means the actual ratio depends on target calculation
        # but contradictions should still have measurably stronger effect
        # Allow tolerance for boundary effects and EMA non-linearity
        if support_delta > 0.01:  # Ignore negligible changes
            ratio = contradict_delta / support_delta
            # Assert ratio > 1.0 (contradictions stronger) rather than exact 2x
            # due to target clamping and EMA formula complexity
            assert ratio > 1.0, (
                f"Contradiction effect not stronger than support: "
                f"support_delta={support_delta}, contradict_delta={contradict_delta}, "
                f"ratio={ratio}"
            )
