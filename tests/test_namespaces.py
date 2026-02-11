"""
Unit tests for OMI namespace functionality

Tests cover:
- Namespace parsing and validation
- Invalid namespace formats
- Namespace properties and methods
- Pattern matching
- Commons namespace behavior
- Hierarchy generation
- Helper functions
- Cross-namespace memory access with permissions
"""

import unittest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.storage.graph_palace import GraphPalace, Memory
from omi.namespaces import (
    Namespace,
    NamespaceComponents,
    validate_namespace,
    parse_namespace_pattern
)


class TestNamespaceValidation(unittest.TestCase):
    """Test namespace parsing and validation."""

    # ==================== Valid Namespaces ====================

    def test_org_only_namespace(self):
        """Test namespace with only organization level."""
        ns = Namespace("acme")
        self.assertEqual(ns.org, "acme")
        self.assertIsNone(ns.team)
        self.assertIsNone(ns.agent)
        self.assertTrue(ns.is_valid())

    def test_org_team_namespace(self):
        """Test namespace with organization and team levels."""
        ns = Namespace("acme/research")
        self.assertEqual(ns.org, "acme")
        self.assertEqual(ns.team, "research")
        self.assertIsNone(ns.agent)
        self.assertTrue(ns.is_valid())

    def test_full_namespace_hierarchy(self):
        """Test namespace with all three levels."""
        ns = Namespace("acme/research/reader")
        self.assertEqual(ns.org, "acme")
        self.assertEqual(ns.team, "research")
        self.assertEqual(ns.agent, "reader")
        self.assertTrue(ns.is_valid())

    def test_namespace_with_hyphens(self):
        """Test namespace components with hyphens."""
        ns = Namespace("my-org/my-team/my-agent")
        self.assertEqual(ns.org, "my-org")
        self.assertEqual(ns.team, "my-team")
        self.assertEqual(ns.agent, "my-agent")

    def test_namespace_with_underscores(self):
        """Test namespace components with underscores."""
        ns = Namespace("my_org/my_team/my_agent")
        self.assertEqual(ns.org, "my_org")
        self.assertEqual(ns.team, "my_team")
        self.assertEqual(ns.agent, "my_agent")

    def test_namespace_with_numbers(self):
        """Test namespace components with numbers."""
        ns = Namespace("org123/team456/agent789")
        self.assertEqual(ns.org, "org123")
        self.assertEqual(ns.team, "team456")
        self.assertEqual(ns.agent, "agent789")

    def test_namespace_mixed_alphanumeric(self):
        """Test namespace with mixed alphanumeric characters."""
        ns = Namespace("Acme-Corp_2024/Team-A1/Agent_99")
        self.assertEqual(ns.org, "Acme-Corp_2024")
        self.assertEqual(ns.team, "Team-A1")
        self.assertEqual(ns.agent, "Agent_99")

    # ==================== Invalid Namespaces ====================

    def test_empty_namespace(self):
        """Test empty namespace raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_whitespace_only_namespace(self):
        """Test whitespace-only namespace raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("   ")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_leading_slash(self):
        """Test namespace with leading slash raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("/acme/team")
        self.assertIn("cannot start or end with '/'", str(ctx.exception))

    def test_trailing_slash(self):
        """Test namespace with trailing slash raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("acme/team/")
        self.assertIn("cannot start or end with '/'", str(ctx.exception))

    def test_too_many_levels(self):
        """Test namespace with more than 3 levels raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("acme/team/agent/extra")
        self.assertIn("cannot have more than 3 levels", str(ctx.exception))

    def test_empty_component(self):
        """Test namespace with empty component raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("acme//agent")
        self.assertIn("cannot have empty components", str(ctx.exception))

    def test_special_characters(self):
        """Test namespace with special characters raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("acme@corp/team")
        self.assertIn("must contain only alphanumeric", str(ctx.exception))

    def test_dots_in_namespace(self):
        """Test namespace with dots raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("acme.corp/team")
        self.assertIn("must contain only alphanumeric", str(ctx.exception))

    def test_spaces_in_namespace(self):
        """Test namespace with spaces raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            Namespace("acme corp/team")
        self.assertIn("must contain only alphanumeric", str(ctx.exception))

    # ==================== Namespace Properties ====================

    def test_raw_property(self):
        """Test raw namespace string is preserved."""
        ns = Namespace("acme/research/reader")
        self.assertEqual(ns.raw, "acme/research/reader")

    def test_namespace_strips_whitespace(self):
        """Test namespace strips leading/trailing whitespace."""
        ns = Namespace("  acme/team  ")
        self.assertEqual(ns.raw, "acme/team")

    # ==================== Commons Namespace ====================

    def test_commons_namespace_detection(self):
        """Test commons namespace is correctly identified."""
        ns = Namespace("acme/commons")
        self.assertTrue(ns.is_commons())
        self.assertEqual(ns.org, "acme")
        self.assertEqual(ns.team, "commons")
        self.assertIsNone(ns.agent)

    def test_non_commons_namespace(self):
        """Test regular namespace is not identified as commons."""
        ns = Namespace("acme/research")
        self.assertFalse(ns.is_commons())

    def test_commons_with_agent_not_commons(self):
        """Test commons with agent level is not a commons namespace."""
        ns = Namespace("acme/commons/agent")
        self.assertFalse(ns.is_commons())

    def test_commons_org_only_not_commons(self):
        """Test org-only namespace is not commons."""
        ns = Namespace("commons")
        self.assertFalse(ns.is_commons())

    def test_get_commons_namespace(self):
        """Test getting commons namespace for an org."""
        ns = Namespace("acme/research/reader")
        self.assertEqual(ns.get_commons_namespace(), "acme/commons")

    def test_get_commons_namespace_from_org_only(self):
        """Test getting commons namespace from org-only namespace."""
        ns = Namespace("acme")
        self.assertEqual(ns.get_commons_namespace(), "acme/commons")

    # ==================== Organization Membership ====================

    def test_is_in_org_true(self):
        """Test namespace belongs to specified organization."""
        ns = Namespace("acme/team/agent")
        self.assertTrue(ns.is_in_org("acme"))

    def test_is_in_org_false(self):
        """Test namespace does not belong to different organization."""
        ns = Namespace("acme/team/agent")
        self.assertFalse(ns.is_in_org("other"))

    def test_is_in_org_case_sensitive(self):
        """Test organization matching is case-sensitive."""
        ns = Namespace("Acme/team/agent")
        self.assertFalse(ns.is_in_org("acme"))
        self.assertTrue(ns.is_in_org("Acme"))

    # ==================== Team Membership ====================

    def test_is_in_team_true(self):
        """Test namespace belongs to specified team."""
        ns = Namespace("acme/research/reader")
        self.assertTrue(ns.is_in_team("acme", "research"))

    def test_is_in_team_false_wrong_org(self):
        """Test namespace does not belong to team in different org."""
        ns = Namespace("acme/research/reader")
        self.assertFalse(ns.is_in_team("other", "research"))

    def test_is_in_team_false_wrong_team(self):
        """Test namespace does not belong to different team."""
        ns = Namespace("acme/research/reader")
        self.assertFalse(ns.is_in_team("acme", "other"))

    def test_is_in_team_false_no_team(self):
        """Test org-only namespace does not belong to any team."""
        ns = Namespace("acme")
        self.assertFalse(ns.is_in_team("acme", "research"))

    # ==================== Hierarchy Generation ====================

    def test_get_hierarchy_full(self):
        """Test hierarchy generation for full namespace."""
        ns = Namespace("acme/research/reader")
        hierarchy = ns.get_hierarchy()
        self.assertEqual(hierarchy, [
            "acme/research/reader",
            "acme/research",
            "acme"
        ])

    def test_get_hierarchy_org_team(self):
        """Test hierarchy generation for org/team namespace."""
        ns = Namespace("acme/research")
        hierarchy = ns.get_hierarchy()
        self.assertEqual(hierarchy, [
            "acme/research",
            "acme"
        ])

    def test_get_hierarchy_org_only(self):
        """Test hierarchy generation for org-only namespace."""
        ns = Namespace("acme")
        hierarchy = ns.get_hierarchy()
        self.assertEqual(hierarchy, ["acme"])

    # ==================== Pattern Matching ====================

    def test_matches_pattern_exact_match(self):
        """Test exact pattern matching."""
        ns = Namespace("acme/research/reader")
        self.assertTrue(ns.matches_pattern("acme/research/reader"))
        self.assertFalse(ns.matches_pattern("acme/research/writer"))

    def test_matches_pattern_org_wildcard(self):
        """Test pattern matching with org wildcard."""
        ns = Namespace("acme/research/reader")
        self.assertTrue(ns.matches_pattern("acme/*"))

    def test_matches_pattern_team_wildcard(self):
        """Test pattern matching with team wildcard."""
        ns = Namespace("acme/research/reader")
        self.assertTrue(ns.matches_pattern("acme/research/*"))

    def test_matches_pattern_no_match(self):
        """Test pattern does not match different namespace."""
        ns = Namespace("acme/research/reader")
        self.assertFalse(ns.matches_pattern("other/*"))
        self.assertFalse(ns.matches_pattern("acme/dev/*"))

    def test_matches_pattern_partial_no_wildcard(self):
        """Test partial match without wildcard does not match."""
        ns = Namespace("acme/research/reader")
        self.assertFalse(ns.matches_pattern("acme/research"))

    # ==================== String Representation ====================

    def test_str_representation(self):
        """Test string representation of namespace."""
        ns = Namespace("acme/research/reader")
        self.assertEqual(str(ns), "acme/research/reader")

    def test_repr_representation(self):
        """Test debug representation of namespace."""
        ns = Namespace("acme/research/reader")
        self.assertEqual(repr(ns), "Namespace('acme/research/reader')")

    # ==================== Equality and Hashing ====================

    def test_equality_with_namespace(self):
        """Test equality comparison between Namespace objects."""
        ns1 = Namespace("acme/research/reader")
        ns2 = Namespace("acme/research/reader")
        ns3 = Namespace("acme/research/writer")

        self.assertEqual(ns1, ns2)
        self.assertNotEqual(ns1, ns3)

    def test_equality_with_string(self):
        """Test equality comparison with string."""
        ns = Namespace("acme/research/reader")
        self.assertEqual(ns, "acme/research/reader")
        self.assertNotEqual(ns, "acme/research/writer")

    def test_equality_with_other_types(self):
        """Test equality comparison with other types returns False."""
        ns = Namespace("acme/research/reader")
        self.assertNotEqual(ns, 123)
        self.assertNotEqual(ns, None)
        self.assertNotEqual(ns, ["acme", "research", "reader"])

    def test_hash_consistency(self):
        """Test hash is consistent for same namespace."""
        ns1 = Namespace("acme/research/reader")
        ns2 = Namespace("acme/research/reader")
        self.assertEqual(hash(ns1), hash(ns2))

    def test_namespace_in_set(self):
        """Test namespace can be used in sets."""
        ns1 = Namespace("acme/research/reader")
        ns2 = Namespace("acme/research/reader")
        ns3 = Namespace("acme/research/writer")

        namespace_set = {ns1, ns2, ns3}
        self.assertEqual(len(namespace_set), 2)

    def test_namespace_as_dict_key(self):
        """Test namespace can be used as dictionary key."""
        ns1 = Namespace("acme/research/reader")
        ns2 = Namespace("acme/research/reader")

        namespace_dict = {ns1: "value1"}
        namespace_dict[ns2] = "value2"

        self.assertEqual(len(namespace_dict), 1)
        self.assertEqual(namespace_dict[ns1], "value2")


class TestNamespaceHelperFunctions(unittest.TestCase):
    """Test namespace helper functions."""

    # ==================== validate_namespace() ====================

    def test_validate_namespace_valid(self):
        """Test validate_namespace returns True for valid namespace."""
        self.assertTrue(validate_namespace("acme"))
        self.assertTrue(validate_namespace("acme/research"))
        self.assertTrue(validate_namespace("acme/research/reader"))
        self.assertTrue(validate_namespace("my-org_123/team-a/agent_99"))

    def test_validate_namespace_invalid(self):
        """Test validate_namespace returns False for invalid namespace."""
        self.assertFalse(validate_namespace(""))
        self.assertFalse(validate_namespace("/acme"))
        self.assertFalse(validate_namespace("acme/"))
        self.assertFalse(validate_namespace("acme//team"))
        self.assertFalse(validate_namespace("acme/team/agent/extra"))
        self.assertFalse(validate_namespace("acme@corp"))
        self.assertFalse(validate_namespace("acme.corp"))

    # ==================== parse_namespace_pattern() ====================

    def test_parse_namespace_pattern_org_only(self):
        """Test parsing org-only pattern."""
        result = parse_namespace_pattern("acme")
        self.assertEqual(result['org'], "acme")
        self.assertIsNone(result['team'])
        self.assertIsNone(result['agent'])
        self.assertFalse(result['wildcard'])

    def test_parse_namespace_pattern_org_team(self):
        """Test parsing org/team pattern."""
        result = parse_namespace_pattern("acme/research")
        self.assertEqual(result['org'], "acme")
        self.assertEqual(result['team'], "research")
        self.assertIsNone(result['agent'])
        self.assertFalse(result['wildcard'])

    def test_parse_namespace_pattern_full(self):
        """Test parsing full namespace pattern."""
        result = parse_namespace_pattern("acme/research/reader")
        self.assertEqual(result['org'], "acme")
        self.assertEqual(result['team'], "research")
        self.assertEqual(result['agent'], "reader")
        self.assertFalse(result['wildcard'])

    def test_parse_namespace_pattern_org_wildcard(self):
        """Test parsing org wildcard pattern."""
        result = parse_namespace_pattern("acme/*")
        self.assertEqual(result['org'], "acme")
        self.assertIsNone(result['team'])
        self.assertIsNone(result['agent'])
        self.assertTrue(result['wildcard'])

    def test_parse_namespace_pattern_team_wildcard(self):
        """Test parsing team wildcard pattern."""
        result = parse_namespace_pattern("acme/research/*")
        self.assertEqual(result['org'], "acme")
        self.assertEqual(result['team'], "research")
        self.assertIsNone(result['agent'])
        self.assertTrue(result['wildcard'])

    def test_parse_namespace_pattern_invalid(self):
        """Test parsing invalid pattern raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_namespace_pattern("invalid@pattern")
        self.assertIn("Invalid namespace pattern", str(ctx.exception))

    def test_parse_namespace_pattern_agent_wildcard(self):
        """Test parsing agent wildcard pattern."""
        result = parse_namespace_pattern("acme/team/agent/*")
        self.assertEqual(result['org'], "acme")
        self.assertEqual(result['team'], "team")
        self.assertEqual(result['agent'], "agent")
        self.assertTrue(result['wildcard'])


class TestNamespacePermissions(unittest.TestCase):
    """Test namespace permission system with GraphPalace."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_palace.sqlite"
        self.palace = GraphPalace(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        self.palace.close()
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_embedding(self, dim: int = 1024) -> List[float]:
        """Generate a random normalized embedding vector."""
        vec = np.random.randn(dim)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def _generate_similar_embedding(self, base: List[float], similarity: float = 0.9) -> List[float]:
        """Generate an embedding similar to base vector."""
        noise = np.random.randn(len(base))
        noise = noise / np.linalg.norm(noise)
        similar = np.array(base) * similarity + noise * (1 - similarity)
        similar = similar / np.linalg.norm(similar)
        return similar.tolist()

    # ==================== Permission Grants ====================

    def test_grant_permission(self):
        """Test granting read permission between namespaces."""
        memory_id = self.palace.store_memory(
            "Secret knowledge",
            namespace="acme/team/agent1"
        )

        perm_id = self.palace.grant_permission(
            "acme/team/agent1",
            "acme/team/agent2",
            "read"
        )

        self.assertIsNotNone(perm_id)
        self.assertTrue(self.palace.can_access("acme/team/agent2", memory_id))

    def test_grant_permission_cross_team(self):
        """Test granting permission across teams."""
        memory_id = self.palace.store_memory(
            "Team A knowledge",
            namespace="acme/teamA/agent1"
        )

        perm_id = self.palace.grant_permission(
            "acme/teamA/agent1",
            "acme/teamB/agent2",
            "read"
        )

        self.assertIsNotNone(perm_id)
        self.assertTrue(self.palace.can_access("acme/teamB/agent2", memory_id))

    def test_grant_permission_to_team_level(self):
        """Test granting permission to team-level namespace."""
        memory_id = self.palace.store_memory(
            "Agent knowledge",
            namespace="acme/team/agent1"
        )

        perm_id = self.palace.grant_permission(
            "acme/team/agent1",
            "acme/team",
            "read"
        )

        self.assertIsNotNone(perm_id)

    # ==================== Permission Revokes ====================

    def test_revoke_permission(self):
        """Test revoking permission."""
        memory_id = self.palace.store_memory(
            "Secret knowledge",
            namespace="acme/team/agent1"
        )

        self.palace.grant_permission(
            "acme/team/agent1",
            "acme/team/agent2",
            "read"
        )
        self.assertTrue(self.palace.can_access("acme/team/agent2", memory_id))

        self.palace.revoke_permission("acme/team/agent1", "acme/team/agent2")
        self.assertFalse(self.palace.can_access("acme/team/agent2", memory_id))

    def test_revoke_nonexistent_permission(self):
        """Test revoking non-existent permission does not error."""
        # Should not raise error
        self.palace.revoke_permission("acme/team/agent1", "acme/team/agent2")

    # ==================== Commons Namespace Access ====================

    def test_commons_namespace_readable_by_org(self):
        """Test that commons namespace is readable by all agents in org."""
        memory_id = self.palace.store_memory(
            "Shared organizational knowledge",
            namespace="acme/commons"
        )

        self.assertTrue(self.palace.can_access("acme/team/agent1", memory_id))
        self.assertTrue(self.palace.can_access("acme/team/agent2", memory_id))
        self.assertTrue(self.palace.can_access("acme/other-team/agent3", memory_id))
        self.assertTrue(self.palace.can_access("acme/commons", memory_id))

    def test_commons_namespace_not_readable_by_other_org(self):
        """Test that commons namespace is not readable by other orgs."""
        memory_id = self.palace.store_memory(
            "Acme shared knowledge",
            namespace="acme/commons"
        )

        self.assertFalse(self.palace.can_access("other-org/team/agent", memory_id))
        self.assertFalse(self.palace.can_access("other-org/commons", memory_id))

    def test_multiple_commons_namespaces(self):
        """Test multiple organizations have separate commons namespaces."""
        acme_memory_id = self.palace.store_memory(
            "Acme shared knowledge",
            namespace="acme/commons"
        )

        other_memory_id = self.palace.store_memory(
            "Other org shared knowledge",
            namespace="other-org/commons"
        )

        # Acme agents can access acme/commons but not other-org/commons
        self.assertTrue(self.palace.can_access("acme/team/agent", acme_memory_id))
        self.assertFalse(self.palace.can_access("acme/team/agent", other_memory_id))

        # Other org agents can access other-org/commons but not acme/commons
        self.assertTrue(self.palace.can_access("other-org/team/agent", other_memory_id))
        self.assertFalse(self.palace.can_access("other-org/team/agent", acme_memory_id))

    # ==================== Cross-Namespace Recall ====================

    def test_cross_namespace_recall_with_permissions(self):
        """Test recall with cross-namespace permissions."""
        base_embedding = self._generate_embedding()

        # Agent1's memory
        mem1_id = self.palace.store_memory(
            "Agent 1 knowledge",
            embedding=base_embedding,
            namespace="acme/team/agent1"
        )

        # Agent2's memory (similar embedding)
        similar_emb = self._generate_similar_embedding(base_embedding, similarity=0.95)
        mem2_id = self.palace.store_memory(
            "Agent 2 knowledge",
            embedding=similar_emb,
            namespace="acme/team/agent2"
        )

        # Commons memory (similar embedding)
        commons_emb = self._generate_similar_embedding(base_embedding, similarity=0.93)
        commons_id = self.palace.store_memory(
            "Shared knowledge",
            embedding=commons_emb,
            namespace="acme/commons"
        )

        # Test 1: Agent1 should only see their own memory and commons
        results = self.palace.recall(
            query_embedding=base_embedding,
            namespace="acme/team/agent1",
            min_relevance=0.7,
            limit=10
        )

        result_ids = [m.id for m, score in results]
        self.assertIn(mem1_id, result_ids, "Agent1 should see their own memory")
        self.assertIn(commons_id, result_ids, "Agent1 should see commons memory")
        self.assertNotIn(mem2_id, result_ids, "Agent1 should NOT see agent2's memory without permission")

        # Test 2: Grant agent1 permission to read agent2's memories
        self.palace.grant_permission("acme/team/agent2", "acme/team/agent1", "read")

        results = self.palace.recall(
            query_embedding=base_embedding,
            namespace="acme/team/agent1",
            min_relevance=0.7,
            limit=10
        )

        result_ids = [m.id for m, score in results]
        self.assertIn(mem1_id, result_ids, "Agent1 should see their own memory")
        self.assertIn(mem2_id, result_ids, "Agent1 should see agent2's memory with permission")
        self.assertIn(commons_id, result_ids, "Agent1 should see commons memory")

        # Test 3: Verify agent2 cannot see agent1's memory without permission
        results = self.palace.recall(
            query_embedding=base_embedding,
            namespace="acme/team/agent2",
            min_relevance=0.7,
            limit=10
        )

        result_ids = [m.id for m, score in results]
        self.assertIn(mem2_id, result_ids, "Agent2 should see their own memory")
        self.assertIn(commons_id, result_ids, "Agent2 should see commons memory")
        self.assertNotIn(mem1_id, result_ids, "Agent2 should NOT see agent1's memory without permission")

    def test_namespace_isolation(self):
        """Test that namespaces are properly isolated without permissions."""
        base_embedding = self._generate_embedding()

        # Create memories in different namespaces
        mem_acme = self.palace.store_memory(
            "Acme knowledge",
            embedding=base_embedding,
            namespace="acme/team/agent"
        )

        mem_other = self.palace.store_memory(
            "Other org knowledge",
            embedding=self._generate_similar_embedding(base_embedding, 0.95),
            namespace="other-org/team/agent"
        )

        # Acme agent should not see other org's memory
        results = self.palace.recall(
            query_embedding=base_embedding,
            namespace="acme/team/agent",
            min_relevance=0.7,
            limit=10
        )

        result_ids = [m.id for m, score in results]
        self.assertIn(mem_acme, result_ids)
        self.assertNotIn(mem_other, result_ids)

        # Other org agent should not see acme's memory
        results = self.palace.recall(
            query_embedding=base_embedding,
            namespace="other-org/team/agent",
            min_relevance=0.7,
            limit=10
        )

        result_ids = [m.id for m, score in results]
        self.assertIn(mem_other, result_ids)
        self.assertNotIn(mem_acme, result_ids)

    def test_bidirectional_permissions(self):
        """Test bidirectional permission grants."""
        base_embedding = self._generate_embedding()

        mem1 = self.palace.store_memory(
            "Agent 1 knowledge",
            embedding=base_embedding,
            namespace="acme/team/agent1"
        )

        mem2 = self.palace.store_memory(
            "Agent 2 knowledge",
            embedding=self._generate_similar_embedding(base_embedding, 0.95),
            namespace="acme/team/agent2"
        )

        # Grant bidirectional permissions
        self.palace.grant_permission("acme/team/agent1", "acme/team/agent2", "read")
        self.palace.grant_permission("acme/team/agent2", "acme/team/agent1", "read")

        # Both agents should see both memories
        results1 = self.palace.recall(
            query_embedding=base_embedding,
            namespace="acme/team/agent1",
            min_relevance=0.7,
            limit=10
        )
        result_ids1 = [m.id for m, score in results1]
        self.assertIn(mem1, result_ids1)
        self.assertIn(mem2, result_ids1)

        results2 = self.palace.recall(
            query_embedding=base_embedding,
            namespace="acme/team/agent2",
            min_relevance=0.7,
            limit=10
        )
        result_ids2 = [m.id for m, score in results2]
        self.assertIn(mem1, result_ids2)
        self.assertIn(mem2, result_ids2)


class TestNamespaceComponents(unittest.TestCase):
    """Test NamespaceComponents dataclass."""

    def test_components_creation(self):
        """Test creating NamespaceComponents."""
        components = NamespaceComponents(org="acme", team="research", agent="reader")
        self.assertEqual(components.org, "acme")
        self.assertEqual(components.team, "research")
        self.assertEqual(components.agent, "reader")

    def test_components_optional_fields(self):
        """Test NamespaceComponents with optional fields."""
        components = NamespaceComponents(org="acme")
        self.assertEqual(components.org, "acme")
        self.assertIsNone(components.team)
        self.assertIsNone(components.agent)


if __name__ == "__main__":
    unittest.main()
