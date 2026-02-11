"""Tests for OMI Graph CLI Commands

Tests for 'omi graph' command group including neighbors and edges commands.
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner


class TestCLIGraphGroup:
    """Tests for 'omi graph' command group."""

    def test_graph_group_exists(self):
        """Test that graph command group exists."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["graph", "--help"])

        assert result.exit_code == 0
        assert "graph" in result.output.lower()
        assert "traversal" in result.output.lower() or "exploration" in result.output.lower()

    def test_graph_group_shows_commands(self):
        """Test that graph group shows available commands."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["graph", "--help"])

        assert result.exit_code == 0
        assert "neighbors" in result.output
        assert "edges" in result.output


class TestCLIGraphNeighbors:
    """Tests for 'omi graph neighbors' command."""

    def test_neighbors_requires_init(self):
        """Test that graph neighbors requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_neighbors_with_valid_memory_id(self):
        """Test neighbors command with valid memory ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_connected
            mock_mem1 = MagicMock()
            mock_mem1.id = "neighbor1-id"
            mock_mem1.content = "Neighbor memory 1"
            mock_mem1.memory_type = "fact"
            mock_mem1.confidence = None
            mock_mem1.created_at = datetime.now()

            mock_mem2 = MagicMock()
            mock_mem2.id = "neighbor2-id"
            mock_mem2.content = "Neighbor memory 2"
            mock_mem2.memory_type = "experience"
            mock_mem2.confidence = None
            mock_mem2.created_at = datetime.now()

            mock_neighbors = [mock_mem1, mock_mem2]

            with patch.object(GraphPalace, 'get_connected', return_value=mock_neighbors):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id"])

            assert result.exit_code == 0
            assert "Connected Memories" in result.output
            assert "2 found" in result.output

    def test_neighbors_no_results(self):
        """Test neighbors command when no neighbors found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock empty result
            with patch.object(GraphPalace, 'get_connected', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id"])

            assert result.exit_code == 0
            assert "No neighbors found" in result.output

    def test_neighbors_with_depth_flag(self):
        """Test neighbors command with --depth flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_connected
            mock_mem = MagicMock()
            mock_mem.id = "neighbor-id"
            mock_mem.content = "Neighbor memory"
            mock_mem.memory_type = "fact"
            mock_mem.confidence = None
            mock_mem.created_at = datetime.now()

            with patch.object(GraphPalace, 'get_connected', return_value=[mock_mem]) as mock_get_connected:
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id", "--depth", "3"])

            assert result.exit_code == 0
            assert "depth=3" in result.output

    def test_neighbors_with_type_flag(self):
        """Test neighbors command with --type flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_neighbors
            mock_mem = MagicMock()
            mock_mem.id = "neighbor-id"
            mock_mem.content = "Related memory"
            mock_mem.memory_type = "fact"
            mock_mem.confidence = None
            mock_mem.created_at = datetime.now()

            with patch.object(GraphPalace, 'get_neighbors', return_value=[mock_mem]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id", "--type", "RELATED_TO"])

            assert result.exit_code == 0
            assert "edge_type=RELATED_TO" in result.output

    def test_neighbors_with_json_output(self):
        """Test neighbors command with --json flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_connected
            mock_mem = MagicMock()
            mock_mem.id = "neighbor-id"
            mock_mem.content = "Neighbor memory"
            mock_mem.memory_type = "fact"
            mock_mem.confidence = 0.95
            mock_mem.created_at = datetime(2024, 1, 1, 12, 0, 0)

            with patch.object(GraphPalace, 'get_connected', return_value=[mock_mem]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id", "--json"])

            assert result.exit_code == 0
            assert '"id": "neighbor-id"' in result.output
            assert '"content": "Neighbor memory"' in result.output
            assert '"memory_type": "fact"' in result.output

    def test_neighbors_invalid_edge_type(self):
        """Test neighbors command with invalid edge type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id", "--type", "INVALID_TYPE"])

            assert result.exit_code != 0
            assert "Invalid value" in result.output or "Error" in result.output


class TestCLIGraphEdges:
    """Tests for 'omi graph edges' command."""

    def test_edges_requires_init(self):
        """Test that graph edges requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["graph", "edges", "test-memory-id"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_edges_with_valid_memory_id(self):
        """Test edges command with valid memory ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_edges
            mock_edge1 = MagicMock()
            mock_edge1.id = "edge1-id"
            mock_edge1.source_id = "source-memory-id"
            mock_edge1.target_id = "target-memory-id"
            mock_edge1.edge_type = "SUPPORTS"
            mock_edge1.strength = 0.85
            mock_edge1.created_at = datetime.now()

            mock_edge2 = MagicMock()
            mock_edge2.id = "edge2-id"
            mock_edge2.source_id = "source-memory-id"
            mock_edge2.target_id = "target2-memory-id"
            mock_edge2.edge_type = "RELATED_TO"
            mock_edge2.strength = 0.70
            mock_edge2.created_at = datetime.now()

            mock_edges = [mock_edge1, mock_edge2]

            with patch.object(GraphPalace, 'get_edges', return_value=mock_edges):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "edges", "source-memory-id"])

            assert result.exit_code == 0
            assert "Edges for Memory" in result.output
            assert "2 found" in result.output

    def test_edges_no_results(self):
        """Test edges command when no edges found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock empty result
            with patch.object(GraphPalace, 'get_edges', return_value=[]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "edges", "test-memory-id"])

            assert result.exit_code == 0
            assert "No edges found" in result.output

    def test_edges_with_type_flag(self):
        """Test edges command with --type flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_edges
            mock_edge = MagicMock()
            mock_edge.id = "edge-id"
            mock_edge.source_id = "source-memory-id"
            mock_edge.target_id = "target-memory-id"
            mock_edge.edge_type = "CONTRADICTS"
            mock_edge.strength = 0.60
            mock_edge.created_at = datetime.now()

            with patch.object(GraphPalace, 'get_edges', return_value=[mock_edge]) as mock_get_edges:
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "edges", "source-memory-id", "--type", "CONTRADICTS"])

            assert result.exit_code == 0
            assert "edge_type=CONTRADICTS" in result.output
            # Verify that get_edges was called with the correct edge_type
            mock_get_edges.assert_called_once()
            call_args = mock_get_edges.call_args
            assert call_args[1]['edge_type'] == 'CONTRADICTS'

    def test_edges_with_json_output(self):
        """Test edges command with --json flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock GraphPalace.get_edges
            mock_edge = MagicMock()
            mock_edge.id = "edge-id"
            mock_edge.source_id = "source-memory-id"
            mock_edge.target_id = "target-memory-id"
            mock_edge.edge_type = "SUPPORTS"
            mock_edge.strength = 0.85
            mock_edge.created_at = datetime(2024, 1, 1, 12, 0, 0)

            with patch.object(GraphPalace, 'get_edges', return_value=[mock_edge]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "edges", "source-memory-id", "--json"])

            assert result.exit_code == 0
            assert '"id": "edge-id"' in result.output
            assert '"source_id": "source-memory-id"' in result.output
            assert '"target_id": "target-memory-id"' in result.output
            assert '"edge_type": "SUPPORTS"' in result.output
            assert '"strength": 0.85' in result.output

    def test_edges_invalid_edge_type(self):
        """Test edges command with invalid edge type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph", "edges", "test-memory-id", "--type", "INVALID_TYPE"])

            assert result.exit_code != 0
            assert "Invalid value" in result.output or "Error" in result.output

    def test_edges_shows_direction(self):
        """Test that edges command shows edge direction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock outgoing edge
            mock_edge_out = MagicMock()
            mock_edge_out.id = "edge-out-id"
            mock_edge_out.source_id = "query-memory-id"
            mock_edge_out.target_id = "target-memory-id"
            mock_edge_out.edge_type = "SUPPORTS"
            mock_edge_out.strength = 0.85
            mock_edge_out.created_at = datetime.now()

            with patch.object(GraphPalace, 'get_edges', return_value=[mock_edge_out]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["graph", "edges", "query-memory-id"])

            assert result.exit_code == 0
            # Check for direction indicator (→ for outgoing)
            assert "→" in result.output or "Source:" in result.output


class TestCLIGraphIntegration:
    """Integration tests for graph commands."""

    def test_graph_commands_help_text(self):
        """Test that all graph commands have proper help text."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        # Test neighbors help
        result = runner.invoke(cli, ["graph", "neighbors", "--help"])
        assert result.exit_code == 0
        assert "memory_id" in result.output.lower() or "MEMORY_ID" in result.output
        assert "--depth" in result.output
        assert "--type" in result.output
        assert "--json" in result.output

        # Test edges help
        result = runner.invoke(cli, ["graph", "edges", "--help"])
        assert result.exit_code == 0
        assert "memory_id" in result.output.lower() or "MEMORY_ID" in result.output
        assert "--type" in result.output
        assert "--json" in result.output

    def test_neighbors_and_edges_consistency(self):
        """Test that neighbors and edges commands work together consistently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli
            from omi import GraphPalace

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Mock data
            mock_mem = MagicMock()
            mock_mem.id = "neighbor-id"
            mock_mem.content = "Connected memory"
            mock_mem.memory_type = "fact"
            mock_mem.confidence = None
            mock_mem.created_at = datetime.now()

            mock_edge = MagicMock()
            mock_edge.id = "edge-id"
            mock_edge.source_id = "test-memory-id"
            mock_edge.target_id = "neighbor-id"
            mock_edge.edge_type = "SUPPORTS"
            mock_edge.strength = 0.85
            mock_edge.created_at = datetime.now()

            # Test neighbors
            with patch.object(GraphPalace, 'get_connected', return_value=[mock_mem]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    neighbors_result = runner.invoke(cli, ["graph", "neighbors", "test-memory-id"])

            # Test edges
            with patch.object(GraphPalace, 'get_edges', return_value=[mock_edge]):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    edges_result = runner.invoke(cli, ["graph", "edges", "test-memory-id"])

            assert neighbors_result.exit_code == 0
            assert edges_result.exit_code == 0
