"""
Unit tests for Policy Engine - Memory lifecycle management

Tests cover:
- Policy types and actions (enums)
- Policy data classes (PolicyRule, Policy, PolicyExecutionLog, PolicyExecutionResult)
- Retention policy (age-based)
- Usage policy (recall-based)
- Confidence policy (threshold-based)
- PolicyEngine execution
- Policy loading from config
- Default policies
- Policy action functions (archive, delete)
- Policy event handler
- Locked memory protection
"""

import unittest
import tempfile
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.policies import (
    PolicyType,
    PolicyAction,
    PolicyRule,
    Policy,
    PolicyExecutionLog,
    PolicyExecutionResult,
    RetentionPolicy,
    UsagePolicy,
    ConfidencePolicy,
    PolicyEngine,
    PolicyEventHandler,
    is_memory_locked,
    archive_memories,
    delete_memories,
    load_policies_from_config,
    get_default_policies,
)


# Mock Memory class for testing
@dataclass
class MockMemory:
    """Mock Memory object for testing policy evaluation."""
    id: str
    content: str
    memory_type: str = "fact"
    created_at: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    confidence: float = None
    locked: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = self.created_at


# Mock GraphPalace for testing
class MockGraphPalace:
    """Mock GraphPalace for testing PolicyEngine."""

    def __init__(self):
        self.memories = {}
        self.archived_memories = set()
        self.deleted_memories = set()

    def get_memory(self, memory_id: str):
        """Get memory by ID."""
        if memory_id in self.deleted_memories:
            return None
        return self.memories.get(memory_id)

    def archive_memories(self, memory_ids: List[str]) -> int:
        """Archive memories."""
        count = 0
        for mem_id in memory_ids:
            if mem_id in self.memories and mem_id not in self.deleted_memories:
                self.archived_memories.add(mem_id)
                count += 1
        return count

    def delete_memories(self, memory_ids: List[str]) -> int:
        """Delete memories."""
        count = 0
        for mem_id in memory_ids:
            if mem_id in self.memories and mem_id not in self.deleted_memories:
                self.deleted_memories.add(mem_id)
                del self.memories[mem_id]
                count += 1
        return count


class TestPolicyEnums(unittest.TestCase):
    """Test suite for policy enums."""

    def test_policy_type_values(self):
        """Test PolicyType enum values."""
        self.assertEqual(PolicyType.RETENTION.value, "retention")
        self.assertEqual(PolicyType.USAGE.value, "usage")
        self.assertEqual(PolicyType.CONFIDENCE.value, "confidence")
        self.assertEqual(PolicyType.SIZE.value, "size")

    def test_policy_action_values(self):
        """Test PolicyAction enum values."""
        self.assertEqual(PolicyAction.ARCHIVE.value, "archive")
        self.assertEqual(PolicyAction.DELETE.value, "delete")
        self.assertEqual(PolicyAction.COMPRESS.value, "compress")
        self.assertEqual(PolicyAction.PROMOTE.value, "promote")
        self.assertEqual(PolicyAction.DEMOTE.value, "demote")


class TestPolicyDataClasses(unittest.TestCase):
    """Test suite for policy data classes."""

    def test_policy_rule_creation(self):
        """Test creating a PolicyRule."""
        rule = PolicyRule(
            name="test-rule",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90},
            enabled=True,
            description="Test rule"
        )

        self.assertEqual(rule.name, "test-rule")
        self.assertEqual(rule.policy_type, PolicyType.RETENTION)
        self.assertEqual(rule.action, PolicyAction.ARCHIVE)
        self.assertEqual(rule.conditions["max_age_days"], 90)
        self.assertTrue(rule.enabled)
        self.assertEqual(rule.description, "Test rule")

    def test_policy_rule_to_dict(self):
        """Test PolicyRule serialization."""
        rule = PolicyRule(
            name="test-rule",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        rule_dict = rule.to_dict()
        self.assertEqual(rule_dict["name"], "test-rule")
        self.assertEqual(rule_dict["policy_type"], "retention")
        self.assertEqual(rule_dict["action"], "archive")
        self.assertEqual(rule_dict["conditions"]["max_age_days"], 90)

    def test_policy_creation(self):
        """Test creating a Policy."""
        rule = PolicyRule(
            name="test-rule",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        policy = Policy(
            name="test-policy",
            rules=[rule],
            enabled=True,
            description="Test policy",
            priority=1
        )

        self.assertEqual(policy.name, "test-policy")
        self.assertEqual(len(policy.rules), 1)
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.priority, 1)
        self.assertEqual(policy.execution_count, 0)

    def test_policy_to_dict(self):
        """Test Policy serialization."""
        rule = PolicyRule(
            name="test-rule",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        policy = Policy(
            name="test-policy",
            rules=[rule],
            enabled=True,
            priority=1
        )

        policy_dict = policy.to_dict()
        self.assertEqual(policy_dict["name"], "test-policy")
        self.assertEqual(len(policy_dict["rules"]), 1)
        self.assertTrue(policy_dict["enabled"])
        self.assertEqual(policy_dict["priority"], 1)

    def test_policy_execution_log_to_dict(self):
        """Test PolicyExecutionLog serialization."""
        log = PolicyExecutionLog(
            policy_name="test-policy",
            action="archive",
            memory_ids=["mem-1", "mem-2"],
            dry_run=True,
            result="success"
        )

        log_dict = log.to_dict()
        self.assertEqual(log_dict["policy_name"], "test-policy")
        self.assertEqual(log_dict["action"], "archive")
        self.assertEqual(len(log_dict["memory_ids"]), 2)
        self.assertTrue(log_dict["dry_run"])
        self.assertEqual(log_dict["result"], "success")

    def test_policy_execution_result_to_dict(self):
        """Test PolicyExecutionResult serialization."""
        result = PolicyExecutionResult(
            policy_name="test-policy",
            action=PolicyAction.ARCHIVE,
            affected_memory_ids=["mem-1", "mem-2"],
            dry_run=False
        )

        result_dict = result.to_dict()
        self.assertEqual(result_dict["policy_name"], "test-policy")
        self.assertEqual(result_dict["action"], "archive")
        self.assertEqual(len(result_dict["affected_memory_ids"]), 2)
        self.assertFalse(result_dict["dry_run"])


class TestRetentionPolicy(unittest.TestCase):
    """Test suite for RetentionPolicy."""

    def test_retention_policy_creation(self):
        """Test creating a RetentionPolicy."""
        policy = RetentionPolicy(max_age_days=90)
        self.assertEqual(policy.max_age_days, 90)
        self.assertIsNone(policy.memory_type_filter)

    def test_retention_policy_invalid_age(self):
        """Test RetentionPolicy rejects invalid age."""
        with self.assertRaises(ValueError):
            RetentionPolicy(max_age_days=0)

        with self.assertRaises(ValueError):
            RetentionPolicy(max_age_days=-10)

    def test_retention_policy_evaluate(self):
        """Test RetentionPolicy evaluation."""
        policy = RetentionPolicy(max_age_days=30)

        # Create test memories
        old_memory = MockMemory(
            id="mem-old",
            content="Old memory",
            created_at=datetime.now() - timedelta(days=60)
        )

        recent_memory = MockMemory(
            id="mem-recent",
            content="Recent memory",
            created_at=datetime.now() - timedelta(days=10)
        )

        memories = [old_memory, recent_memory]
        expired_ids = policy.evaluate(memories)

        self.assertEqual(len(expired_ids), 1)
        self.assertIn("mem-old", expired_ids)
        self.assertNotIn("mem-recent", expired_ids)

    def test_retention_policy_memory_type_filter(self):
        """Test RetentionPolicy with memory type filter."""
        policy = RetentionPolicy(max_age_days=30, memory_type_filter="fact")

        old_fact = MockMemory(
            id="mem-fact",
            content="Old fact",
            memory_type="fact",
            created_at=datetime.now() - timedelta(days=60)
        )

        old_experience = MockMemory(
            id="mem-exp",
            content="Old experience",
            memory_type="experience",
            created_at=datetime.now() - timedelta(days=60)
        )

        memories = [old_fact, old_experience]
        expired_ids = policy.evaluate(memories)

        # Only the fact should be selected (experience filtered out)
        self.assertEqual(len(expired_ids), 1)
        self.assertIn("mem-fact", expired_ids)

    def test_retention_policy_is_expired(self):
        """Test RetentionPolicy.is_expired()."""
        policy = RetentionPolicy(max_age_days=30)

        old_memory = MockMemory(
            id="mem-old",
            content="Old memory",
            created_at=datetime.now() - timedelta(days=60)
        )

        recent_memory = MockMemory(
            id="mem-recent",
            content="Recent memory",
            created_at=datetime.now() - timedelta(days=10)
        )

        self.assertTrue(policy.is_expired(old_memory))
        self.assertFalse(policy.is_expired(recent_memory))

    def test_retention_policy_days_until_expiry(self):
        """Test RetentionPolicy.days_until_expiry()."""
        policy = RetentionPolicy(max_age_days=30)

        memory = MockMemory(
            id="mem-1",
            content="Test memory",
            created_at=datetime.now() - timedelta(days=20)
        )

        days = policy.days_until_expiry(memory)
        self.assertIsNotNone(days)
        self.assertAlmostEqual(days, 10, delta=1)

    def test_retention_policy_locked_memory(self):
        """Test RetentionPolicy skips locked memories."""
        policy = RetentionPolicy(max_age_days=30)

        old_locked = MockMemory(
            id="mem-locked",
            content="Locked memory",
            created_at=datetime.now() - timedelta(days=60),
            locked=True
        )

        old_unlocked = MockMemory(
            id="mem-unlocked",
            content="Unlocked memory",
            created_at=datetime.now() - timedelta(days=60),
            locked=False
        )

        memories = [old_locked, old_unlocked]
        expired_ids = policy.evaluate(memories)

        # Only unlocked memory should be selected
        self.assertEqual(len(expired_ids), 1)
        self.assertIn("mem-unlocked", expired_ids)
        self.assertNotIn("mem-locked", expired_ids)


class TestUsagePolicy(unittest.TestCase):
    """Test suite for UsagePolicy."""

    def test_usage_policy_creation(self):
        """Test creating a UsagePolicy."""
        policy = UsagePolicy(min_access_count=3, max_age_days=90)
        self.assertEqual(policy.min_access_count, 3)
        self.assertEqual(policy.max_age_days, 90)
        self.assertIsNone(policy.memory_type_filter)

    def test_usage_policy_invalid_params(self):
        """Test UsagePolicy rejects invalid parameters."""
        with self.assertRaises(ValueError):
            UsagePolicy(min_access_count=-1, max_age_days=90)

        with self.assertRaises(ValueError):
            UsagePolicy(min_access_count=3, max_age_days=0)

    def test_usage_policy_evaluate(self):
        """Test UsagePolicy evaluation."""
        policy = UsagePolicy(min_access_count=5, max_age_days=30)

        # Old and underused (should match)
        unused_old = MockMemory(
            id="mem-unused",
            content="Unused memory",
            created_at=datetime.now() - timedelta(days=60),
            access_count=2
        )

        # Old but well-used (should NOT match)
        used_old = MockMemory(
            id="mem-used",
            content="Used memory",
            created_at=datetime.now() - timedelta(days=60),
            access_count=10
        )

        # Recent and underused (should NOT match - too new)
        unused_recent = MockMemory(
            id="mem-recent",
            content="Recent memory",
            created_at=datetime.now() - timedelta(days=10),
            access_count=2
        )

        memories = [unused_old, used_old, unused_recent]
        unused_ids = policy.evaluate(memories)

        # Only old and underused should match
        self.assertEqual(len(unused_ids), 1)
        self.assertIn("mem-unused", unused_ids)

    def test_usage_policy_is_unused(self):
        """Test UsagePolicy.is_unused()."""
        policy = UsagePolicy(min_access_count=5, max_age_days=30)

        unused_old = MockMemory(
            id="mem-1",
            content="Unused memory",
            created_at=datetime.now() - timedelta(days=60),
            access_count=2
        )

        used_old = MockMemory(
            id="mem-2",
            content="Used memory",
            created_at=datetime.now() - timedelta(days=60),
            access_count=10
        )

        self.assertTrue(policy.is_unused(unused_old))
        self.assertFalse(policy.is_unused(used_old))

    def test_usage_policy_days_since_last_access(self):
        """Test UsagePolicy.days_since_last_access()."""
        policy = UsagePolicy(min_access_count=5, max_age_days=30)

        memory = MockMemory(
            id="mem-1",
            content="Test memory",
            created_at=datetime.now() - timedelta(days=60),
            last_accessed=datetime.now() - timedelta(days=20)
        )

        days = policy.days_since_last_access(memory)
        self.assertIsNotNone(days)
        self.assertAlmostEqual(days, 20, delta=1)

    def test_usage_policy_usage_score(self):
        """Test UsagePolicy.usage_score()."""
        policy = UsagePolicy(min_access_count=5, max_age_days=30)

        # Well-used memory
        used_memory = MockMemory(
            id="mem-used",
            content="Used memory",
            created_at=datetime.now() - timedelta(days=60),
            last_accessed=datetime.now() - timedelta(days=5),
            access_count=10
        )

        # Unused memory
        unused_memory = MockMemory(
            id="mem-unused",
            content="Unused memory",
            created_at=datetime.now() - timedelta(days=60),
            last_accessed=datetime.now() - timedelta(days=60),
            access_count=1
        )

        used_score = policy.usage_score(used_memory)
        unused_score = policy.usage_score(unused_memory)

        self.assertIsNotNone(used_score)
        self.assertIsNotNone(unused_score)
        self.assertGreater(used_score, unused_score)

    def test_usage_policy_locked_memory(self):
        """Test UsagePolicy skips locked memories."""
        policy = UsagePolicy(min_access_count=5, max_age_days=30)

        unused_locked = MockMemory(
            id="mem-locked",
            content="Locked memory",
            created_at=datetime.now() - timedelta(days=60),
            access_count=2,
            locked=True
        )

        unused_unlocked = MockMemory(
            id="mem-unlocked",
            content="Unlocked memory",
            created_at=datetime.now() - timedelta(days=60),
            access_count=2,
            locked=False
        )

        memories = [unused_locked, unused_unlocked]
        unused_ids = policy.evaluate(memories)

        # Only unlocked memory should be selected
        self.assertEqual(len(unused_ids), 1)
        self.assertIn("mem-unlocked", unused_ids)


class TestConfidencePolicy(unittest.TestCase):
    """Test suite for ConfidencePolicy."""

    def test_confidence_policy_creation(self):
        """Test creating a ConfidencePolicy."""
        policy = ConfidencePolicy(min_confidence=0.3)
        self.assertEqual(policy.min_confidence, 0.3)
        self.assertIsNone(policy.memory_type_filter)

    def test_confidence_policy_invalid_threshold(self):
        """Test ConfidencePolicy rejects invalid confidence values."""
        with self.assertRaises(ValueError):
            ConfidencePolicy(min_confidence=-0.1)

        with self.assertRaises(ValueError):
            ConfidencePolicy(min_confidence=1.5)

    def test_confidence_policy_evaluate(self):
        """Test ConfidencePolicy evaluation."""
        policy = ConfidencePolicy(min_confidence=0.5)

        low_confidence = MockMemory(
            id="mem-low",
            content="Low confidence",
            memory_type="belief",
            confidence=0.2
        )

        high_confidence = MockMemory(
            id="mem-high",
            content="High confidence",
            memory_type="belief",
            confidence=0.9
        )

        no_confidence = MockMemory(
            id="mem-none",
            content="No confidence",
            memory_type="fact"
            # No confidence attribute
        )

        memories = [low_confidence, high_confidence, no_confidence]
        low_ids = policy.evaluate(memories)

        # Only low confidence memory should match
        self.assertEqual(len(low_ids), 1)
        self.assertIn("mem-low", low_ids)

    def test_confidence_policy_memory_type_filter(self):
        """Test ConfidencePolicy with memory type filter."""
        policy = ConfidencePolicy(min_confidence=0.5, memory_type_filter="belief")

        low_belief = MockMemory(
            id="mem-belief",
            content="Low confidence belief",
            memory_type="belief",
            confidence=0.2
        )

        low_fact = MockMemory(
            id="mem-fact",
            content="Low confidence fact",
            memory_type="fact",
            confidence=0.2
        )

        memories = [low_belief, low_fact]
        low_ids = policy.evaluate(memories)

        # Only belief should be selected
        self.assertEqual(len(low_ids), 1)
        self.assertIn("mem-belief", low_ids)

    def test_confidence_policy_is_below_threshold(self):
        """Test ConfidencePolicy.is_below_threshold()."""
        policy = ConfidencePolicy(min_confidence=0.5)

        low = MockMemory(id="mem-1", content="Low", confidence=0.2)
        high = MockMemory(id="mem-2", content="High", confidence=0.9)

        self.assertTrue(policy.is_below_threshold(low))
        self.assertFalse(policy.is_below_threshold(high))

    def test_confidence_policy_confidence_margin(self):
        """Test ConfidencePolicy.confidence_margin()."""
        policy = ConfidencePolicy(min_confidence=0.5)

        low = MockMemory(id="mem-1", content="Low", confidence=0.3)
        high = MockMemory(id="mem-2", content="High", confidence=0.8)

        low_margin = policy.confidence_margin(low)
        high_margin = policy.confidence_margin(high)

        self.assertIsNotNone(low_margin)
        self.assertIsNotNone(high_margin)
        self.assertAlmostEqual(low_margin, -0.2, places=5)
        self.assertAlmostEqual(high_margin, 0.3, places=5)

    def test_confidence_policy_confidence_score(self):
        """Test ConfidencePolicy.confidence_score()."""
        policy = ConfidencePolicy(min_confidence=0.5)

        memory = MockMemory(id="mem-1", content="Test", confidence=0.7)
        score = policy.confidence_score(memory)

        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, 0.7, places=5)

    def test_confidence_policy_locked_memory(self):
        """Test ConfidencePolicy skips locked memories."""
        policy = ConfidencePolicy(min_confidence=0.5)

        low_locked = MockMemory(
            id="mem-locked",
            content="Locked",
            confidence=0.2,
            locked=True
        )

        low_unlocked = MockMemory(
            id="mem-unlocked",
            content="Unlocked",
            confidence=0.2,
            locked=False
        )

        memories = [low_locked, low_unlocked]
        low_ids = policy.evaluate(memories)

        # Only unlocked memory should be selected
        self.assertEqual(len(low_ids), 1)
        self.assertIn("mem-unlocked", low_ids)


class TestPolicyEngine(unittest.TestCase):
    """Test suite for PolicyEngine."""

    def setUp(self):
        """Set up test engine."""
        self.palace = MockGraphPalace()
        self.engine = PolicyEngine(self.palace)

        # Add test memories to palace
        self.old_memory = MockMemory(
            id="mem-old",
            content="Old memory",
            created_at=datetime.now() - timedelta(days=200)
        )
        self.recent_memory = MockMemory(
            id="mem-recent",
            content="Recent memory",
            created_at=datetime.now() - timedelta(days=10)
        )

        self.palace.memories = {
            "mem-old": self.old_memory,
            "mem-recent": self.recent_memory
        }

        # Mock _fetch_memories to return our test memories
        self.engine._fetch_memories = lambda f=None: list(self.palace.memories.values())

    def test_engine_initialization(self):
        """Test PolicyEngine initialization."""
        engine = PolicyEngine(self.palace)
        self.assertEqual(engine.graph_palace, self.palace)
        self.assertEqual(len(engine.get_execution_history()), 0)

    def test_engine_execute_disabled_policy(self):
        """Test executing a disabled policy raises error."""
        policy = Policy(
            name="test-policy",
            enabled=False,
            rules=[]
        )

        with self.assertRaises(ValueError):
            self.engine.execute(policy)

    def test_engine_execute_dry_run(self):
        """Test PolicyEngine dry run execution."""
        rule = PolicyRule(
            name="archive-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        policy = Policy(
            name="test-policy",
            rules=[rule],
            enabled=True
        )

        results = self.engine.execute(policy, dry_run=True)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertTrue(result.dry_run)
        self.assertEqual(result.action, PolicyAction.ARCHIVE)
        self.assertEqual(len(result.affected_memory_ids), 1)
        self.assertIn("mem-old", result.affected_memory_ids)

        # Verify no actual changes were made
        self.assertEqual(len(self.palace.archived_memories), 0)

    def test_engine_execute_real(self):
        """Test PolicyEngine real execution."""
        rule = PolicyRule(
            name="archive-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        policy = Policy(
            name="test-policy",
            rules=[rule],
            enabled=True
        )

        results = self.engine.execute(policy, dry_run=False)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertFalse(result.dry_run)
        self.assertEqual(len(result.affected_memory_ids), 1)

        # Verify policy was executed (but _execute_action is a placeholder)
        # In real implementation, this would archive memories
        self.assertIsNotNone(policy.last_executed)
        self.assertEqual(policy.execution_count, 1)

    def test_engine_execute_multiple_rules(self):
        """Test executing policy with multiple rules."""
        rule1 = PolicyRule(
            name="archive-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        rule2 = PolicyRule(
            name="delete-very-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.DELETE,
            conditions={"max_age_days": 365}
        )

        policy = Policy(
            name="test-policy",
            rules=[rule1, rule2],
            enabled=True
        )

        results = self.engine.execute(policy, dry_run=True)

        # Should have results for both rules
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].action, PolicyAction.ARCHIVE)
        self.assertEqual(results[1].action, PolicyAction.DELETE)

    def test_engine_execute_disabled_rule(self):
        """Test disabled rules are skipped."""
        rule1 = PolicyRule(
            name="archive-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90},
            enabled=True
        )

        rule2 = PolicyRule(
            name="disabled-rule",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.DELETE,
            conditions={"max_age_days": 365},
            enabled=False
        )

        policy = Policy(
            name="test-policy",
            rules=[rule1, rule2],
            enabled=True
        )

        results = self.engine.execute(policy, dry_run=True)

        # Only enabled rule should execute
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].policy_name, "archive-old")

    def test_engine_execution_history(self):
        """Test execution history tracking."""
        rule = PolicyRule(
            name="archive-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90}
        )

        policy = Policy(
            name="test-policy",
            rules=[rule],
            enabled=True
        )

        # Execute twice
        self.engine.execute(policy, dry_run=True)
        self.engine.execute(policy, dry_run=True)

        history = self.engine.get_execution_history()
        self.assertEqual(len(history), 2)

        # Clear history
        self.engine.clear_execution_history()
        self.assertEqual(len(self.engine.get_execution_history()), 0)

    def test_engine_evaluate_retention_policy(self):
        """Test PolicyEngine evaluates retention policies."""
        rule = PolicyRule(
            name="archive-old",
            policy_type=PolicyType.RETENTION,
            action=PolicyAction.ARCHIVE,
            conditions={"max_age_days": 90},
            memory_type_filter="fact"
        )

        affected_ids = self.engine._evaluate_policy(
            rule,
            list(self.palace.memories.values())
        )

        self.assertIn("mem-old", affected_ids)
        self.assertNotIn("mem-recent", affected_ids)

    def test_engine_evaluate_usage_policy(self):
        """Test PolicyEngine evaluates usage policies."""
        # Add memory with low access count
        unused = MockMemory(
            id="mem-unused",
            content="Unused",
            created_at=datetime.now() - timedelta(days=100),
            access_count=1
        )
        self.palace.memories["mem-unused"] = unused

        rule = PolicyRule(
            name="archive-unused",
            policy_type=PolicyType.USAGE,
            action=PolicyAction.ARCHIVE,
            conditions={
                "min_access_count": 5,
                "max_age_days": 30
            }
        )

        affected_ids = self.engine._evaluate_policy(
            rule,
            list(self.palace.memories.values())
        )

        self.assertIn("mem-unused", affected_ids)

    def test_engine_evaluate_confidence_policy(self):
        """Test PolicyEngine evaluates confidence policies."""
        # Add memory with low confidence
        low_conf = MockMemory(
            id="mem-low",
            content="Low confidence",
            memory_type="belief",
            confidence=0.2
        )
        self.palace.memories["mem-low"] = low_conf

        rule = PolicyRule(
            name="delete-low-confidence",
            policy_type=PolicyType.CONFIDENCE,
            action=PolicyAction.DELETE,
            conditions={"min_confidence": 0.5}
        )

        affected_ids = self.engine._evaluate_policy(
            rule,
            list(self.palace.memories.values())
        )

        self.assertIn("mem-low", affected_ids)


class TestPolicyHelpers(unittest.TestCase):
    """Test suite for policy helper functions."""

    def test_is_memory_locked(self):
        """Test is_memory_locked() function."""
        locked = MockMemory(id="mem-1", content="Locked", locked=True)
        unlocked = MockMemory(id="mem-2", content="Unlocked", locked=False)
        no_attr = MockMemory(id="mem-3", content="No attr")

        self.assertTrue(is_memory_locked(locked))
        self.assertFalse(is_memory_locked(unlocked))
        self.assertFalse(is_memory_locked(no_attr))

    def test_archive_memories(self):
        """Test archive_memories() function."""
        palace = MockGraphPalace()
        palace.memories = {
            "mem-1": MockMemory(id="mem-1", content="Memory 1"),
            "mem-2": MockMemory(id="mem-2", content="Memory 2")
        }

        result = archive_memories(palace, ["mem-1", "mem-2"])

        self.assertTrue(result["success"])
        self.assertEqual(result["archived_count"], 2)
        self.assertEqual(len(result["memory_ids"]), 2)
        self.assertEqual(len(result["errors"]), 0)

    def test_archive_memories_empty_list(self):
        """Test archive_memories() with empty list."""
        palace = MockGraphPalace()
        result = archive_memories(palace, [])

        self.assertTrue(result["success"])
        self.assertEqual(result["archived_count"], 0)

    def test_delete_memories(self):
        """Test delete_memories() function."""
        palace = MockGraphPalace()
        palace.memories = {
            "mem-1": MockMemory(id="mem-1", content="Memory 1"),
            "mem-2": MockMemory(id="mem-2", content="Memory 2")
        }

        result = delete_memories(palace, ["mem-1", "mem-2"], safety_check=False)

        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_count"], 2)
        self.assertEqual(len(result["memory_ids"]), 2)
        self.assertEqual(len(result["errors"]), 0)

    def test_delete_memories_safety_check(self):
        """Test delete_memories() with safety check."""
        palace = MockGraphPalace()
        palace.memories = {
            "mem-1": MockMemory(id="mem-1", content="Memory 1")
        }

        # Try to delete both existing and non-existing
        result = delete_memories(
            palace,
            ["mem-1", "mem-nonexistent"],
            safety_check=True
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_count"], 1)
        self.assertEqual(len(result["skipped"]), 1)
        self.assertIn("mem-nonexistent", result["skipped"])

    def test_delete_memories_empty_list(self):
        """Test delete_memories() with empty list."""
        palace = MockGraphPalace()
        result = delete_memories(palace, [], safety_check=True)

        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_count"], 0)


class TestPolicyLoading(unittest.TestCase):
    """Test suite for policy loading and defaults."""

    def setUp(self):
        """Set up test config directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.yaml"

    def tearDown(self):
        """Clean up test directory."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_policies_from_config(self):
        """Test loading policies from YAML config."""
        config_data = {
            "policies": [
                {
                    "name": "test-policy",
                    "enabled": True,
                    "description": "Test policy",
                    "priority": 1,
                    "schedule": "daily",
                    "rules": [
                        {
                            "name": "test-rule",
                            "policy_type": "retention",
                            "action": "archive",
                            "enabled": True,
                            "conditions": {
                                "max_age_days": 90
                            }
                        }
                    ]
                }
            ]
        }

        self.config_path.write_text(yaml.dump(config_data))

        policies = load_policies_from_config(self.config_path)

        self.assertEqual(len(policies), 1)
        policy = policies[0]
        self.assertEqual(policy.name, "test-policy")
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.priority, 1)
        self.assertEqual(len(policy.rules), 1)

        rule = policy.rules[0]
        self.assertEqual(rule.name, "test-rule")
        self.assertEqual(rule.policy_type, PolicyType.RETENTION)
        self.assertEqual(rule.action, PolicyAction.ARCHIVE)
        self.assertEqual(rule.conditions["max_age_days"], 90)

    def test_load_policies_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with self.assertRaises(FileNotFoundError):
            load_policies_from_config(Path("/nonexistent/config.yaml"))

    def test_load_policies_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        self.config_path.write_text("invalid: yaml: content: [")

        with self.assertRaises(ValueError):
            load_policies_from_config(self.config_path)

    def test_load_policies_invalid_policy_type(self):
        """Test loading policy with invalid type raises error."""
        config_data = {
            "policies": [
                {
                    "name": "test-policy",
                    "rules": [
                        {
                            "name": "test-rule",
                            "policy_type": "invalid_type",
                            "action": "archive",
                            "conditions": {}
                        }
                    ]
                }
            ]
        }

        self.config_path.write_text(yaml.dump(config_data))

        with self.assertRaises(ValueError):
            load_policies_from_config(self.config_path)

    def test_get_default_policies(self):
        """Test getting default policies."""
        policies = get_default_policies()

        self.assertGreater(len(policies), 0)

        # Check for expected default policies
        policy_names = [p.name for p in policies]
        self.assertIn("archive-old-memories", policy_names)
        self.assertIn("archive-unused-memories", policy_names)
        self.assertIn("delete-low-confidence-beliefs", policy_names)
        self.assertIn("compliance-deletion", policy_names)

        # All default policies should be enabled
        for policy in policies:
            self.assertTrue(policy.enabled)
            self.assertGreater(len(policy.rules), 0)


class TestPolicyEventHandler(unittest.TestCase):
    """Test suite for PolicyEventHandler."""

    def test_event_handler_initialization(self):
        """Test PolicyEventHandler initialization."""
        engine = PolicyEngine(None)
        handler = PolicyEventHandler(engine)

        self.assertEqual(handler.policy_engine, engine)

    def test_event_handler_handle_event(self):
        """Test event handling (basic smoke test)."""
        engine = PolicyEngine(None)
        handler = PolicyEventHandler(engine)

        # Create mock event
        class MockEvent:
            event_type = "session.ended"

        # Should not raise error
        handler.handle(MockEvent())

    def test_event_handler_no_engine(self):
        """Test event handling with no engine."""
        handler = PolicyEventHandler(None)

        class MockEvent:
            event_type = "session.ended"

        # Should not raise error
        handler.handle(MockEvent())

    def test_event_handler_invalid_event(self):
        """Test handling invalid event."""
        handler = PolicyEventHandler(None)

        # Event without event_type attribute
        class BadEvent:
            pass

        # Should not raise error (silently skips)
        handler.handle(BadEvent())


if __name__ == "__main__":
    unittest.main()
