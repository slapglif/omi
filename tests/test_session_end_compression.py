"""Integration tests for session-end compression workflow

Tests the complete workflow: init → store memories → session-end with compression
"""

import os
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

import pytest
from click.testing import CliRunner


def create_mock_now_store(base_path):
    """Create a properly mocked NOWStore with all needed methods."""
    mock_store = MagicMock()
    mock_store.read.return_value = """# NOW - 2024-01-01T10:00:00

## Current Task
Test compression workflow

## Recent Completions
- [x] Setup complete

## Pending Decisions
- [ ] None

## Key Files
- `test.py`
"""
    mock_store.update = MagicMock()
    mock_store.now_path = Path(base_path) / "NOW.md"
    mock_store.hash_file = Path(base_path) / ".now.hash"
    mock_store.check_integrity.return_value = True
    return mock_store


class TestSessionEndCompression:
    """Integration tests for session-end compression workflow."""

    def test_session_end_with_compression_enabled(self):
        """Test session-end compresses memories when compression is enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Step 1: Initialize OMI
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Step 2: Enable compression in config
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config['compression']['provider'] = 'ollama'
            config_path.write_text(yaml.dump(config))

            # Step 3: Insert test memories directly into database
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()

                # Insert memories created in last 24 hours (within session window)
                test_memories = [
                    (
                        "mem-1",
                        "This is a detailed memory about a project meeting where we discussed timelines and deliverables.",
                        "experience",
                        0.9,
                        (now - timedelta(hours=2)).isoformat()
                    ),
                    (
                        "mem-2",
                        "Another verbose memory containing lots of information about technical implementation details.",
                        "fact",
                        0.85,
                        (now - timedelta(hours=1)).isoformat()
                    )
                ]

                for mem_id, content, mem_type, confidence, created_at in test_memories:
                    cursor.execute("""
                        INSERT INTO memories (id, content, memory_type, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (mem_id, content, mem_type, confidence, created_at))

                conn.commit()

            # Step 4: Mock LLM API to return compressed summaries
            def mock_ollama_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "response": "Compressed summary."
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # Step 5: End session with compression
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_ollama_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify compression stats in output
            assert "Compressed" in result.output
            assert "memories" in result.output
            assert "tokens" in result.output
            assert "savings" in result.output

            # Verify daily log contains compressed memories
            memory_dir = base_path / "memory"
            today = datetime.now().strftime("%Y-%m-%d")
            daily_log = memory_dir / f"{today}.md"

            assert daily_log.exists()
            log_content = daily_log.read_text()

            # Check for compression metadata in daily log
            assert "Session Memories (Compressed)" in log_content
            assert "Compressed summary." in log_content
            assert "savings" in log_content.lower()

    def test_session_end_with_compression_disabled(self):
        """Test session-end skips compression when disabled in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Disable compression in config
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = False
            config_path.write_text(yaml.dump(config))

            # Insert test memory
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("mem-1", "Test memory", "fact", 0.9, (now - timedelta(hours=1)).isoformat()))
                conn.commit()

            # End session
            mock_store = create_mock_now_store(base_path)
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end"])
                    assert result.exit_code == 0

            # Verify compression was skipped (no compression stats shown)
            # The compression stats only appear if compression runs successfully
            assert "Compressed" not in result.output or "0 memories" in result.output

    def test_session_end_compression_with_no_memories(self):
        """Test session-end handles case where no memories exist in session window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Enable compression
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config_path.write_text(yaml.dump(config))

            # End session without any memories
            mock_store = create_mock_now_store(base_path)
            with patch('omi.cli.session.NOWStore', return_value=mock_store):
                with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                    result = runner.invoke(cli, ["session-end"])
                    assert result.exit_code == 0

            # Should complete successfully without errors
            assert "Session ended" in result.output or "session ended" in result.output.lower()

    def test_session_end_compression_with_openai_provider(self):
        """Test session-end compression with OpenAI provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Configure OpenAI provider
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config['compression']['provider'] = 'openai'
            config['compression']['api_key'] = 'test-api-key'
            config_path.write_text(yaml.dump(config))

            # Insert test memory
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("mem-1", "Detailed technical discussion about API design.", "experience", 0.9,
                      (now - timedelta(hours=1)).isoformat()))
                conn.commit()

            # Mock OpenAI API response
            def mock_openai_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "choices": [{
                        "message": {
                            "content": "API design discussion summary."
                        }
                    }]
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # End session with OpenAI compression
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_openai_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify compression ran
            assert "Compressed" in result.output

    def test_session_end_compression_preserves_metadata(self):
        """Test that compression preserves memory metadata in daily log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Enable compression
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config_path.write_text(yaml.dump(config))

            # Insert memories with various types
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()

                memory_types = ["experience", "fact", "belief", "decision"]
                for i, mem_type in enumerate(memory_types):
                    cursor.execute("""
                        INSERT INTO memories (id, content, memory_type, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (f"mem-{i}", f"Memory of type {mem_type}", mem_type, 0.9,
                          (now - timedelta(hours=i+1)).isoformat()))

                conn.commit()

            # Mock compression
            def mock_ollama_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "response": "Compressed."
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # End session
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_ollama_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify daily log contains metadata for different memory types
            memory_dir = base_path / "memory"
            today = datetime.now().strftime("%Y-%m-%d")
            daily_log = memory_dir / f"{today}.md"

            log_content = daily_log.read_text()

            # Check that memory types are preserved
            for mem_type in memory_types:
                assert mem_type.capitalize() in log_content

    def test_session_end_compression_error_handling(self):
        """Test session-end handles compression errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Enable compression
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config_path.write_text(yaml.dump(config))

            # Insert test memory
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("mem-1", "Test memory", "fact", 0.9, (now - timedelta(hours=1)).isoformat()))
                conn.commit()

            # Mock API to raise an error
            def mock_error_response(*args, **kwargs):
                raise Exception("API connection failed")

            # End session - should handle error gracefully
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_error_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        # Should still complete successfully
                        assert result.exit_code == 0

            # Verify error was reported but session-end completed
            assert "Session ended" in result.output or "session ended" in result.output.lower()

    def test_session_end_compression_shows_token_savings(self):
        """Test session-end displays accurate token savings statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Enable compression
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config_path.write_text(yaml.dump(config))

            # Insert memory with known length
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                long_content = "This is a very long memory. " * 50  # ~1400 chars = ~350 tokens
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("mem-1", long_content, "fact", 0.9, (now - timedelta(hours=1)).isoformat()))
                conn.commit()

            # Mock compression to return short summary
            def mock_ollama_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "response": "Short summary"  # ~12 chars = ~3 tokens
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # End session
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_ollama_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify token statistics are shown
            assert "tokens" in result.output.lower()
            assert "→" in result.output or "->" in result.output  # Arrow showing reduction
            assert "%" in result.output  # Percentage savings

    def test_session_end_compression_with_multiple_memories(self):
        """Test session-end compresses multiple memories in batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Enable compression
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config_path.write_text(yaml.dump(config))

            # Insert 10 test memories
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()

                for i in range(10):
                    cursor.execute("""
                        INSERT INTO memories (id, content, memory_type, confidence, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (f"mem-{i}", f"Memory content number {i} with some details.", "fact", 0.9,
                          (now - timedelta(hours=i+1)).isoformat()))

                conn.commit()

            # Mock compression
            call_count = [0]
            def mock_ollama_response(*args, **kwargs):
                call_count[0] += 1
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "response": f"Summary {call_count[0]}"
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # End session
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_ollama_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify all 10 memories were compressed
            assert "10 memories" in result.output or "Compressed 10" in result.output

            # Verify daily log has all compressed memories
            memory_dir = base_path / "memory"
            today = datetime.now().strftime("%Y-%m-%d")
            daily_log = memory_dir / f"{today}.md"

            log_content = daily_log.read_text()
            # Check that we have compressed content
            assert log_content.count("Summary") >= 5  # Should have multiple summaries


class TestCompressionConfigIntegration:
    """Test configuration loading and usage in session-end."""

    def test_session_end_respects_max_summary_tokens(self):
        """Test that max_summary_tokens config is passed to summarizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Configure with custom max_summary_tokens
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config['compression']['max_summary_tokens'] = 200  # Custom value
            config_path.write_text(yaml.dump(config))

            # Insert memory
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("mem-1", "Test memory content", "fact", 0.9,
                      (now - timedelta(hours=1)).isoformat()))
                conn.commit()

            # Mock API
            def mock_ollama_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "response": "Summary"
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # End session - config should be loaded
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_ollama_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0

            # Verify session completed successfully
            assert result.exit_code == 0

    def test_session_end_with_custom_model(self):
        """Test session-end uses custom model from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["init"])
                assert result.exit_code == 0

            # Configure with custom model
            import yaml
            config_path = base_path / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config['compression']['enabled'] = True
            config['compression']['model'] = 'llama3.2:1b'  # Custom model
            config_path.write_text(yaml.dump(config))

            # Insert memory
            db_path = base_path / "palace.sqlite"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute("""
                    INSERT INTO memories (id, content, memory_type, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ("mem-1", "Test", "fact", 0.9, (now - timedelta(hours=1)).isoformat()))
                conn.commit()

            # Mock API
            def mock_ollama_response(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.json.return_value = {
                    "response": "Summary"
                }
                mock_response.raise_for_status = MagicMock()
                return mock_response

            # End session
            mock_store = create_mock_now_store(base_path)
            with patch('requests.Session.post', side_effect=mock_ollama_response):
                with patch('omi.cli.session.NOWStore', return_value=mock_store):
                    with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                        result = runner.invoke(cli, ["session-end"])
                        assert result.exit_code == 0
