"""Tests for OMI CLI graph command

Tests the 'omi graph' command which visualizes memory relationships.
"""

import os
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


class TestGraph:
    """Tests for 'omi graph' command."""

    def test_graph_requires_init(self):
        """Test that graph requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_graph_requires_database(self):
        """Test that graph requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 1
            assert "database not found" in result.output.lower()

    def test_graph_empty_database(self):
        """Test graph with empty database (no memories)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            assert "no memories" in result.output.lower()

    def test_graph_displays_memories_no_edges(self):
        """Test graph displays memories when no edges exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add test memories without edges
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-001', 'Test memory 1', 'fact')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-002', 'Test memory 2', 'experience')
            """)
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            assert "Memory Graph Visualization" in result.output
            assert "Test memory 1" in result.output
            assert "Test memory 2" in result.output
            assert "Legend" in result.output

    def test_graph_displays_edges(self):
        """Test graph displays edges between memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add test memories with edges
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-001', 'Test memory 1', 'fact')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-002', 'Test memory 2', 'belief')
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-001', 'mem-002', 'SUPPORTS', 0.85)
            """)
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            assert "SUPPORTS" in result.output
            assert "0.85" in result.output
            assert "1 edges" in result.output

    def test_graph_respects_limit(self):
        """Test graph respects --limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add multiple memories
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for i in range(5):
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type)
                    VALUES (?, ?, 'experience')
                """, (f"mem-{i:03d}", f"Test memory {i}"))
            conn.commit()
            conn.close()

            # Test with limit=2
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph", "--limit", "2"])

            assert result.exit_code == 0
            assert "showing 2" in result.output

    def test_graph_filters_by_edge_type(self):
        """Test graph filters by edge type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add memories with different edge types
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-001', 'Memory 1', 'fact')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-002', 'Memory 2', 'fact')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-003', 'Memory 3', 'belief')
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-001', 'mem-002', 'SUPPORTS', 0.9)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-002', 'mem-003', 'CONTRADICTS', 0.7)
            """)
            conn.commit()
            conn.close()

            # Filter by SUPPORTS only
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph", "--type", "SUPPORTS"])

            assert result.exit_code == 0
            assert "Filter: SUPPORTS" in result.output
            assert "SUPPORTS" in result.output

    def test_graph_shows_legend(self):
        """Test graph shows legend with memory types and edge types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add a simple memory
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-001', 'Test', 'fact')
            """)
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            assert "Legend" in result.output
            assert "Memory Types:" in result.output
            assert "fact" in result.output.lower()
            assert "experience" in result.output.lower()
            assert "belief" in result.output.lower()
            assert "decision" in result.output.lower()
            assert "Edge Types:" in result.output
            assert "SUPPORTS" in result.output
            assert "CONTRADICTS" in result.output
            assert "RELATED_TO" in result.output

    def test_graph_truncates_long_content(self):
        """Test graph truncates long memory content for display."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add memory with very long content
            long_content = "A" * 100  # Content longer than 50 chars
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-001', ?, 'fact')
            """, (long_content,))
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            # Should be truncated with "..."
            assert "..." in result.output

    def test_graph_shows_centrality_ranking(self):
        """Test graph ranks nodes by centrality (connections)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add memories with different centrality
            # mem-001 is central (3 connections)
            # mem-002, mem-003, mem-004 are peripheral (1 connection each)
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for i in range(1, 5):
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type)
                    VALUES (?, ?, 'fact')
                """, (f"mem-{i:03d}", f"Memory {i}"))

            # mem-001 is connected to all others
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-001', 'mem-002', 'RELATED_TO', 0.8)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-001', 'mem-003', 'RELATED_TO', 0.8)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-001', 'mem-004', 'RELATED_TO', 0.8)
            """)
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            assert "centrality" in result.output.lower()

    def test_graph_help_shows_options(self):
        """Test graph --help shows all available options."""
        runner = CliRunner()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from omi.cli import cli

        result = runner.invoke(cli, ["graph", "--help"])

        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--type" in result.output
        assert "--depth" in result.output
        assert "Maximum number of nodes" in result.output

    def test_graph_with_multiple_memory_types(self):
        """Test graph displays different memory types with colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add memories of different types
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-001', 'Fact memory', 'fact')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-002', 'Experience memory', 'experience')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-003', 'Belief memory', 'belief')
            """)
            cursor.execute("""
                INSERT INTO memories (id, content, memory_type)
                VALUES ('mem-004', 'Decision memory', 'decision')
            """)
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            # All memory types should be shown
            assert "(fact)" in result.output.lower()
            assert "(experience)" in result.output.lower()
            assert "(belief)" in result.output.lower()
            assert "(decision)" in result.output.lower()

    def test_graph_with_multiple_edge_types(self):
        """Test graph displays different edge types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Add memories with different edge types
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Create a chain of memories with different edge types
            for i in range(1, 7):
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type)
                    VALUES (?, ?, 'fact')
                """, (f"mem-{i:03d}", f"Memory {i}"))

            # Different edge types
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-001', 'mem-002', 'SUPPORTS', 0.9)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-002', 'mem-003', 'CONTRADICTS', 0.8)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-003', 'mem-004', 'RELATED_TO', 0.7)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-004', 'mem-005', 'DEPENDS_ON', 0.6)
            """)
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, edge_type, strength)
                VALUES ('mem-005', 'mem-006', 'POSTED', 0.5)
            """)
            conn.commit()
            conn.close()

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["graph"])

            assert result.exit_code == 0
            # Check that various edge types appear
            assert "SUPPORTS" in result.output or "CONTRADICTS" in result.output or "RELATED_TO" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
