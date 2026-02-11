"""Tests for 'omi inspect' command.

Uses Click's test runner for command testing.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner


class TestInspect:
    """Tests for 'omi inspect' command."""

    def test_inspect_requires_init(self):
        """Test that inspect requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["inspect"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    def test_inspect_empty_database(self):
        """Test that inspect works on empty database without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                init_result = runner.invoke(cli, ["init"])
                assert init_result.exit_code == 0

                # Run inspect on empty database
                result = runner.invoke(cli, ["inspect"])

            assert result.exit_code == 0
            assert "Total Memories:" in result.output
            assert "0" in result.output
            assert "No memories stored yet" in result.output or "Database Size:" in result.output

    def test_inspect_shows_memory_stats(self):
        """Test that inspect shows memory statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Add some test data directly to database
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Insert test memories
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, ("test-1", "Test fact", "fact", "2024-01-01T00:00:00"))

                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, ("test-2", "Test experience", "experience", "2024-01-01T00:00:00"))

                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, ("test-3", "Test belief", "belief", "2024-01-01T00:00:00"))

                conn.commit()
                conn.close()

                # Run inspect
                result = runner.invoke(cli, ["inspect"])

            assert result.exit_code == 0
            assert "Memory Inspection Report" in result.output
            assert "Total Memories:" in result.output
            assert "3" in result.output
            assert "Database Size:" in result.output
            assert "KB" in result.output
            assert "Breakdown by Type:" in result.output
            assert "fact" in result.output.lower() or "Fact" in result.output
            assert "experience" in result.output.lower() or "Experience" in result.output
            assert "belief" in result.output.lower() or "Belief" in result.output

    def test_inspect_json_output(self):
        """Test that inspect --json-output returns valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Add test memory
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, ("test-1", "Test memory", "fact", "2024-01-01T00:00:00"))
                conn.commit()
                conn.close()

                # Run inspect with JSON output
                result = runner.invoke(cli, ["inspect", "--json-output"])

            assert result.exit_code == 0

            # Parse JSON output
            output_data = json.loads(result.output)

            # Verify JSON structure
            assert "total_memories" in output_data
            assert output_data["total_memories"] == 1
            assert "memory_types" in output_data
            assert "fact" in output_data["memory_types"]
            assert output_data["memory_types"]["fact"] == 1
            assert "database_size_kb" in output_data
            assert isinstance(output_data["database_size_kb"], (int, float))
            assert "last_session" in output_data

    def test_inspect_json_output_empty_database(self):
        """Test that inspect --json-output works on empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])
                result = runner.invoke(cli, ["inspect", "--json-output"])

            assert result.exit_code == 0

            # Parse JSON output
            output_data = json.loads(result.output)

            # Verify JSON structure for empty database
            assert output_data["total_memories"] == 0
            assert output_data["memory_types"] == {}
            assert "database_size_kb" in output_data
            assert "last_session" in output_data

    def test_inspect_beliefs_flag(self):
        """Test that inspect --beliefs shows belief network summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create beliefs table and add test data
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Create beliefs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS beliefs (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        evidence_count INTEGER DEFAULT 0
                    )
                """)

                # Insert test beliefs
                cursor.execute("""
                    INSERT INTO beliefs (id, content, confidence, evidence_count)
                    VALUES (?, ?, ?, ?)
                """, ("belief-1", "Python is great for data science", 0.95, 5))

                cursor.execute("""
                    INSERT INTO beliefs (id, content, confidence, evidence_count)
                    VALUES (?, ?, ?, ?)
                """, ("belief-2", "Testing is important", 0.75, 3))

                conn.commit()
                conn.close()

                # Run inspect with --beliefs flag
                result = runner.invoke(cli, ["inspect", "--beliefs"])

            assert result.exit_code == 0
            assert "Belief Network Summary:" in result.output
            assert "Total Beliefs:" in result.output
            assert "2" in result.output
            assert "Confidence:" in result.output
            assert "Evidence:" in result.output
            assert "Python is great for data science" in result.output
            assert "Testing is important" in result.output

    def test_inspect_beliefs_empty_network(self):
        """Test that inspect --beliefs handles empty belief network."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create empty beliefs table
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS beliefs (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        evidence_count INTEGER DEFAULT 0
                    )
                """)
                conn.commit()
                conn.close()

                # Run inspect with --beliefs flag
                result = runner.invoke(cli, ["inspect", "--beliefs"])

            assert result.exit_code == 0
            assert "No beliefs in network yet" in result.output

    def test_inspect_beliefs_not_initialized(self):
        """Test that inspect --beliefs handles missing beliefs table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Run inspect with --beliefs flag (beliefs table doesn't exist)
                result = runner.invoke(cli, ["inspect", "--beliefs"])

            assert result.exit_code == 0
            assert "Belief network not initialized" in result.output

    def test_inspect_beliefs_json_output(self):
        """Test that inspect --beliefs --json-output returns beliefs in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Create beliefs table and add test data
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS beliefs (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        evidence_count INTEGER DEFAULT 0
                    )
                """)

                cursor.execute("""
                    INSERT INTO beliefs (id, content, confidence, evidence_count)
                    VALUES (?, ?, ?, ?)
                """, ("belief-1", "Test belief", 0.85, 2))

                conn.commit()
                conn.close()

                # Run inspect with both flags
                result = runner.invoke(cli, ["inspect", "--beliefs", "--json-output"])

            assert result.exit_code == 0

            # Parse JSON output
            output_data = json.loads(result.output)

            # Verify beliefs in JSON
            assert "beliefs" in output_data
            assert isinstance(output_data["beliefs"], list)
            assert len(output_data["beliefs"]) == 1
            assert output_data["beliefs"][0]["id"] == "belief-1"
            assert output_data["beliefs"][0]["content"] == "Test belief"
            assert output_data["beliefs"][0]["confidence"] == 0.85
            assert output_data["beliefs"][0]["evidence_count"] == 2

    def test_inspect_shows_last_activity(self):
        """Test that inspect shows last activity timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Add memory with last_accessed timestamp
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?)
                """, ("test-1", "Test", "fact", "2024-01-01T00:00:00", "2024-01-15T12:00:00"))
                conn.commit()
                conn.close()

                # Run inspect
                result = runner.invoke(cli, ["inspect"])

            assert result.exit_code == 0
            assert "Last Activity:" in result.output
            assert "2024-01-15" in result.output

    def test_inspect_no_activity(self):
        """Test that inspect handles no activity gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Add memory without last_accessed
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, ("test-1", "Test", "fact", "2024-01-01T00:00:00"))
                conn.commit()
                conn.close()

                # Run inspect
                result = runner.invoke(cli, ["inspect"])

            assert result.exit_code == 0
            assert "Last Activity:" in result.output
            assert ("No activity recorded" in result.output or "None" in result.output)

    def test_inspect_shows_type_percentages(self):
        """Test that inspect shows percentage breakdown by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

                # Add memories with different types
                import sqlite3
                db_path = base_path / "palace.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # 3 facts, 1 experience
                for i in range(3):
                    cursor.execute("""
                        INSERT INTO memories (id, content, memory_type, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (f"fact-{i}", f"Test fact {i}", "fact", "2024-01-01T00:00:00"))

                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, created_at)
                    VALUES (?, ?, ?, ?)
                """, ("exp-1", "Test experience", "experience", "2024-01-01T00:00:00"))

                conn.commit()
                conn.close()

                # Run inspect
                result = runner.invoke(cli, ["inspect"])

            assert result.exit_code == 0
            assert "75.0%" in result.output or "75%" in result.output  # 3/4 = 75% for facts
            assert "25.0%" in result.output or "25%" in result.output  # 1/4 = 25% for experience


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
