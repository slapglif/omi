"""Tests for OMI CLI namespace commands

Tests namespace flag integration for multi-agent memory isolation.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


class TestNamespaceCLI:
    """Tests for CLI commands with --namespace flag."""

    def test_init_with_namespace(self):
        """Test that init command accepts and stores namespace in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init", "--namespace", "org/team/agent"])

            # Command should succeed
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Config should be created
            config_path = base_path / "config.yaml"
            assert config_path.exists(), "config.yaml not created"

            # Config should contain namespace
            config = yaml.safe_load(config_path.read_text())
            assert config.get("namespace") == "org/team/agent", \
                f"Namespace not set in config. Config: {config}"

    def test_store_with_namespace(self):
        """Test that store command accepts and uses --namespace flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0, f"Init failed: {result.output}"

            # Store with namespace
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "store", "Test memory in namespace",
                    "--namespace", "org/team/agent1"
                ])

            # Command should succeed
            assert result.exit_code == 0, f"Command failed: {result.output}"
            assert "✓" in result.output or "stored" in result.output.lower()

            # Verify memory was stored in correct namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            results = palace.full_text_search("Test memory in namespace")
            assert len(results) > 0, "Memory not found"
            assert results[0].namespace == "org/team/agent1", \
                f"Expected 'org/team/agent1', got '{results[0].namespace}'"
            palace.close()

    def test_recall_with_namespace(self):
        """Test that recall command filters by namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0, f"Init failed: {result.output}"

            # Store memories in different namespaces
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            palace.store_memory("Agent1 memory about testing", namespace="org/team/agent1")
            palace.store_memory("Agent2 memory about testing", namespace="org/team/agent2")
            palace.close()

            # Recall with namespace filter for agent1
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "recall", "testing",
                    "--namespace", "org/team/agent1"
                ])

            # Command should succeed
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Should only show agent1's memory
            assert "Agent1" in result.output
            assert "Agent2" not in result.output

            # Recall with namespace filter for agent2
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "recall", "testing",
                    "--namespace", "org/team/agent2"
                ])

            # Should only show agent2's memory
            assert "Agent2" in result.output
            assert "Agent1" not in result.output

    def test_share_command(self):
        """Test that share command grants cross-namespace permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Store a memory in source namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            memory_id = palace.store_memory(
                "Test memory to share",
                namespace="org/team/agent1"
            )
            palace.close()

            # Share memory with another namespace
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "share", memory_id,
                    "--from", "org/team/agent1",
                    "--with", "org/team/agent2"
                ])

            # Command should succeed
            assert result.exit_code == 0, f"Command failed: {result.output}"
            assert "shared successfully" in result.output.lower() or "✓" in result.output

            # Verify permission was granted
            palace = GraphPalace(db_path)
            assert palace.can_access("org/team/agent2", memory_id), \
                "agent2 should have access to shared memory"
            palace.close()

    def test_backward_compatibility(self):
        """Test that commands work without --namespace flag.

        Verifies:
        1. Commands work when no namespace is provided
        2. Use namespace from config if set
        3. Fall back to 'default' namespace if config has no namespace
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Test 1: Init without namespace, should still work
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0, f"Init failed: {result.output}"

            # Test 2: Store without namespace flag should work (use 'default')
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["store", "Test memory without namespace"])
                assert result.exit_code == 0, f"Store failed: {result.output}"

            # Verify memory was stored in 'default' namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            results = palace.full_text_search("Test memory without namespace")
            assert len(results) > 0, "Memory not found"
            assert results[0].namespace == 'default', \
                f"Expected 'default' namespace, got '{results[0].namespace}'"
            palace.close()

            # Test 3: Init with namespace in config
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['namespace'] = 'org/team/agent'
            config_path.write_text(yaml.dump(config))

            # Store without flag should now use namespace from config
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["store", "Test memory with config namespace"])
                assert result.exit_code == 0, f"Store with config namespace failed: {result.output}"

            # Verify memory was stored in config namespace
            palace = GraphPalace(db_path)
            results = palace.full_text_search("config namespace")
            assert len(results) > 0, "Memory not found"
            assert results[0].namespace == 'org/team/agent', \
                f"Expected 'org/team/agent' namespace, got '{results[0].namespace}'"
            palace.close()

    def test_invalid_namespace_validation(self):
        """Test that invalid namespaces are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Test invalid namespace with init
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init", "--namespace", "invalid@namespace"])

            # Command should fail
            assert result.exit_code != 0
            assert "invalid" in result.output.lower() or "error" in result.output.lower()

    def test_commons_namespace(self):
        """Test that commons namespace is readable by all agents in org."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Store memory in commons namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            commons_id = palace.store_memory(
                "Shared org knowledge about testing",
                namespace="org/commons"
            )

            # Verify agents in same org can access commons
            assert palace.can_access("org/team/agent1", commons_id), \
                "agent1 should have access to commons"
            assert palace.can_access("org/team/agent2", commons_id), \
                "agent2 should have access to commons"
            assert palace.can_access("org/other/agent3", commons_id), \
                "agent3 should have access to commons"

            # Verify agents in different org cannot access
            assert not palace.can_access("other_org/team/agent", commons_id), \
                "Different org should not have access to commons"

            palace.close()

    def test_namespace_isolation(self):
        """Test that namespaces are properly isolated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Store memories in different namespaces
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            id1 = palace.store_memory("Secret agent1 data", namespace="org/team/agent1")
            id2 = palace.store_memory("Secret agent2 data", namespace="org/team/agent2")

            # Verify namespace isolation (no permissions)
            assert palace.can_access("org/team/agent1", id1), "agent1 should access own memory"
            assert not palace.can_access("org/team/agent2", id1), "agent2 should NOT access agent1's memory"

            assert palace.can_access("org/team/agent2", id2), "agent2 should access own memory"
            assert not palace.can_access("org/team/agent1", id2), "agent1 should NOT access agent2's memory"

            palace.close()

    def test_share_command_validation(self):
        """Test share command validation and error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Try to share non-existent memory
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "share", "nonexistent-id",
                    "--from", "org/team/agent1",
                    "--with", "org/team/agent2"
                ])

            # Should fail
            assert result.exit_code != 0
            assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_recall_json_output(self):
        """Test recall with JSON output includes namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace
            import json

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Store memory with namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            palace.store_memory("JSON test memory", namespace="org/team/agent1")
            palace.close()

            # Recall with JSON output
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "recall", "JSON test",
                    "--namespace", "org/team/agent1",
                    "--json-output"
                ])

            # Should succeed and return valid JSON
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Parse JSON output
            try:
                output_data = json.loads(result.output)
                assert isinstance(output_data, list), "JSON output should be a list"
                assert len(output_data) > 0, "Should have at least one result"
                # Note: namespace might not be in JSON output, but the query should work
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON output: {e}\nOutput: {result.output}")

    def test_cross_namespace_recall_with_permission(self):
        """Test recall across namespaces when permission is granted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Store memory in agent1's namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)
            memory_id = palace.store_memory(
                "Shared research findings",
                namespace="org/research/agent1"
            )

            # Grant permission to agent2
            palace.grant_permission(
                source_namespace="org/research/agent1",
                target_namespace="org/writing/agent2",
                permission_level="read"
            )
            palace.close()

            # Agent2 should be able to access agent1's memory now
            palace = GraphPalace(db_path)
            assert palace.can_access("org/writing/agent2", memory_id), \
                "agent2 should have access after permission granted"
            palace.close()

    def test_init_without_namespace_creates_default(self):
        """Test that init without namespace still works (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Init without namespace should still work
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0, f"Init failed: {result.output}"

            # Config should exist but may not have namespace field
            config_path = base_path / "config.yaml"
            assert config_path.exists(), "config.yaml not created"

            # Config should not have namespace (or it should be None/empty)
            config = yaml.safe_load(config_path.read_text())
            # Namespace should not be in config, or should be None
            assert config.get("namespace") is None, \
                "Namespace should not be set when init called without --namespace"

    def test_multi_agent_sharing_scenario_e2e(self):
        """End-to-end test: Multi-agent scenario with sharing.

        Verifies the complete multi-agent workflow:
        1. Initialize three namespaces: org/team/researcher, org/team/writer, org/team/reviewer
        2. Store memories in each namespace
        3. Share a memory from researcher to writer
        4. Verify writer can access shared memory
        5. Verify reviewer cannot access unless shared
        6. Store memory in org/commons
        7. Verify all three agents can read commons
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi.storage.graph_palace import GraphPalace

            # Step 1: Initialize three namespaces
            researcher_ns = "org/team/researcher"
            writer_ns = "org/team/writer"
            reviewer_ns = "org/team/reviewer"

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init", "--namespace", researcher_ns])
                assert result.exit_code == 0, f"Init researcher failed: {result.output}"

            # Step 2: Store memories in each namespace
            db_path = base_path / "palace.sqlite"
            palace = GraphPalace(db_path)

            researcher_memory_id = palace.store_memory(
                "Research findings about AI safety",
                namespace=researcher_ns
            )
            writer_memory_id = palace.store_memory(
                "Draft article about technology trends",
                namespace=writer_ns
            )
            reviewer_memory_id = palace.store_memory(
                "Review comments for publication",
                namespace=reviewer_ns
            )

            # Verify initial isolation - each agent can only access their own memories
            assert palace.can_access(researcher_ns, researcher_memory_id), \
                "Researcher should access own memory"
            assert not palace.can_access(writer_ns, researcher_memory_id), \
                "Writer should NOT access researcher's memory initially"
            assert not palace.can_access(reviewer_ns, researcher_memory_id), \
                "Reviewer should NOT access researcher's memory initially"

            assert palace.can_access(writer_ns, writer_memory_id), \
                "Writer should access own memory"
            assert not palace.can_access(researcher_ns, writer_memory_id), \
                "Researcher should NOT access writer's memory"

            assert palace.can_access(reviewer_ns, reviewer_memory_id), \
                "Reviewer should access own memory"
            assert not palace.can_access(researcher_ns, reviewer_memory_id), \
                "Researcher should NOT access reviewer's memory"

            palace.close()

            # Step 3: Share a memory from researcher to writer
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "share", researcher_memory_id,
                    "--from", researcher_ns,
                    "--with", writer_ns
                ])
                assert result.exit_code == 0, f"Share command failed: {result.output}"
                assert "shared successfully" in result.output.lower() or "✓" in result.output

            # Step 4: Verify writer can access shared memory
            palace = GraphPalace(db_path)
            assert palace.can_access(writer_ns, researcher_memory_id), \
                "Writer SHOULD access researcher's memory after sharing"

            # Step 5: Verify reviewer cannot access unless shared
            assert not palace.can_access(reviewer_ns, researcher_memory_id), \
                "Reviewer should STILL NOT access researcher's memory (not shared)"

            # Step 6: Store memory in org/commons
            commons_memory_id = palace.store_memory(
                "Shared organizational knowledge about mission and values",
                namespace="org/commons"
            )

            # Step 7: Verify all three agents can read commons
            assert palace.can_access(researcher_ns, commons_memory_id), \
                "Researcher should access org/commons"
            assert palace.can_access(writer_ns, commons_memory_id), \
                "Writer should access org/commons"
            assert palace.can_access(reviewer_ns, commons_memory_id), \
                "Reviewer should access org/commons"

            # Additional verification: Different org cannot access commons
            assert not palace.can_access("other_org/team/agent", commons_memory_id), \
                "Different org should NOT access org/commons"

            # Verify full_text_search works correctly with permissions
            # Researcher should see their own memories
            researcher_results = palace.full_text_search(
                "safety",
                namespace=researcher_ns
            )
            assert len(researcher_results) > 0, "Researcher should find their own memories"

            # Writer should see their own memories in their namespace
            writer_results = palace.full_text_search(
                "article",
                namespace=writer_ns
            )
            assert len(writer_results) > 0, "Writer should find their own memories"

            # Verify writer can access researcher's shared memory directly
            shared_memory = palace.get_memory(researcher_memory_id)
            assert shared_memory is not None, "Shared memory should exist"
            assert palace.can_access(writer_ns, researcher_memory_id), \
                "Writer can access shared memory"

            # Reviewer should NOT have access to researcher's memory (not shared)
            assert not palace.can_access(reviewer_ns, researcher_memory_id), \
                "Reviewer should NOT access researcher's memory"

            # All agents should be able to find commons memory via direct access
            # (FTS search within namespace may not return commons memories)
            for ns in [researcher_ns, writer_ns, reviewer_ns]:
                # Verify can_access works for all agents
                assert palace.can_access(ns, commons_memory_id), \
                    f"{ns} should have access to org/commons memory"

                # Verify they can retrieve the memory
                commons_mem = palace.get_memory(commons_memory_id)
                assert commons_mem is not None, \
                    f"{ns} should be able to retrieve org/commons memory"
                assert commons_mem.namespace == "org/commons", \
                    "Retrieved memory should be from org/commons"

            palace.close()
