"""End-to-end tests for the compress command

Tests the complete compression workflow:
- Dry run mode
- Live compression with backup
- Memory summarization
- Embedding regeneration
- Date filtering
"""

import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.cli import cli


class TestCompressE2E:
    """End-to-end tests for compress command."""

    def _init_omi(self, runner, base_path):
        """Initialize OMI in a test directory."""
        with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        return base_path

    def _store_memory_with_date(self, base_path, content, memory_type, created_at):
        """Store a memory with a specific creation date."""
        import sqlite3
        import hashlib

        db_path = base_path / "palace.sqlite"
        memory_id = hashlib.sha256(content.encode()).hexdigest()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, content, memory_type, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (memory_id, content, memory_type, created_at.isoformat(), created_at.isoformat()))
        conn.commit()
        conn.close()

        return memory_id

    def test_compress_dry_run(self):
        """Test compress command in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store some old memories (40 days ago)
            old_date = datetime.now() - timedelta(days=40)
            for i in range(5):
                self._store_memory_with_date(
                    base_path,
                    f"Old memory {i}",
                    "experience",
                    old_date
                )

            # Store some recent memories (10 days ago)
            recent_date = datetime.now() - timedelta(days=10)
            for i in range(3):
                self._store_memory_with_date(
                    base_path,
                    f"Recent memory {i}",
                    "experience",
                    recent_date
                )

            # Run compress in dry-run mode (default 30 days)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run"])

            assert result.exit_code == 0
            assert "DRY RUN" in result.output
            assert "Preview mode - no changes will be made" in result.output
            assert "5" in result.output  # Should show 5 old memories to compress
            assert "Would perform:" in result.output
            assert "Create MoltVault backup" in result.output
            assert "Summarize 5 memories" in result.output
            assert "Dry run complete - no changes made" in result.output

    def test_compress_dry_run_with_age_days_filter(self):
        """Test compress command with --age-days filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store memories at different ages
            for days_ago in [5, 15, 25, 35]:
                date = datetime.now() - timedelta(days=days_ago)
                self._store_memory_with_date(
                    base_path,
                    f"Memory from {days_ago} days ago",
                    "fact",
                    date
                )

            # Compress with --age-days 20 (should get 2 memories: 25 and 35 days old)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run", "--age-days", "20"])

            assert result.exit_code == 0
            assert "older than 20 days" in result.output
            assert "Old memories" in result.output

    def test_compress_dry_run_with_before_filter(self):
        """Test compress command with --before date filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store memories with known dates
            cutoff_date = datetime(2024, 1, 15)
            old_date = datetime(2024, 1, 10)
            new_date = datetime(2024, 1, 20)

            for i in range(3):
                self._store_memory_with_date(
                    base_path,
                    f"Old memory {i}",
                    "fact",
                    old_date
                )

            for i in range(2):
                self._store_memory_with_date(
                    base_path,
                    f"New memory {i}",
                    "fact",
                    new_date
                )

            # Compress with --before 2024-01-15 (should get 3 old memories)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run", "--before", "2024-01-15"])

            assert result.exit_code == 0
            assert "before 2024-01-15" in result.output

    def test_compress_invalid_date_format(self):
        """Test compress command with invalid date format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Try to compress with invalid date format
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run", "--before", "2024/01/15"])

            assert result.exit_code == 1
            assert "Invalid date format" in result.output
            assert "YYYY-MM-DD" in result.output

    def test_compress_conflicting_filters(self):
        """Test compress command with both --before and --age-days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Try to use both filters
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "compress",
                    "--dry-run",
                    "--before", "2024-01-15",
                    "--age-days", "30"
                ])

            assert result.exit_code == 1
            assert "Cannot use both --before and --age-days" in result.output

    def test_compress_no_memories_to_compress(self):
        """Test compress when no old memories exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store only recent memories (within default 30 days)
            recent_date = datetime.now() - timedelta(days=5)
            for i in range(3):
                self._store_memory_with_date(
                    base_path,
                    f"Recent memory {i}",
                    "experience",
                    recent_date
                )

            # Run compress
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run"])

            assert result.exit_code == 0
            assert "No memories to compress" in result.output

    def test_compress_requires_init(self):
        """Test that compress requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "not_initialized"

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run"])

            assert result.exit_code == 1
            assert "not initialized" in result.output.lower()

    @patch('omi.moltvault.MoltVault')
    @patch('omi.summarizer.MemorySummarizer')
    @patch('omi.embeddings.NIMEmbedder')
    def test_compress_live_mode(self, mock_embedder_cls, mock_summarizer_cls, mock_vault_cls):
        """Test compress command in live mode with mocked services."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store old memories
            old_date = datetime.now() - timedelta(days=40)
            original_contents = []
            for i in range(3):
                content = f"This is a detailed memory about event {i} with lots of information."
                original_contents.append(content)
                self._store_memory_with_date(
                    base_path,
                    content,
                    "experience",
                    old_date
                )

            # Mock MoltVault backup
            mock_vault = MagicMock()
            mock_metadata = MagicMock()
            mock_metadata.backup_id = "test-backup-123"
            mock_metadata.file_size = 1024
            mock_vault.backup.return_value = mock_metadata
            mock_vault_cls.return_value = mock_vault

            # Mock MemorySummarizer
            mock_summarizer = MagicMock()
            compressed_summaries = [f"Summary {i}" for i in range(3)]
            mock_summarizer.batch_summarize.return_value = compressed_summaries
            mock_summarizer_cls.return_value = mock_summarizer

            # Mock NIMEmbedder
            mock_embedder = MagicMock()
            mock_embeddings = [[0.1] * 1024 for _ in range(3)]
            mock_embedder.embed_batch.return_value = mock_embeddings
            mock_embedder_cls.return_value = mock_embedder

            # Run compress in live mode
            with patch.dict(os.environ, {
                "OMI_BASE_PATH": str(base_path),
                "ANTHROPIC_API_KEY": "test-key"
            }):
                result = runner.invoke(cli, ["compress"])

            assert result.exit_code == 0
            assert "LIVE" in result.output
            assert "Creating backup..." in result.output
            assert "Backup created:" in result.output
            assert "test-backup-123" in result.output
            assert "Summarizing memories..." in result.output
            assert "Regenerating embeddings..." in result.output
            assert "Updating Graph Palace..." in result.output
            assert "Compression Complete!" in result.output
            assert "Memories compressed:" in result.output
            assert "Savings:" in result.output

            # Verify mocks were called correctly
            mock_vault.backup.assert_called_once_with(full=True)
            mock_summarizer.batch_summarize.assert_called()
            mock_embedder.embed_batch.assert_called_once_with(compressed_summaries)

            # Verify database was updated
            import sqlite3
            db_path = base_path / "palace.sqlite"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM memories")
            updated_contents = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Contents should be updated to summaries
            for summary in compressed_summaries:
                assert summary in updated_contents

    @patch('omi.moltvault.MoltVault')
    @patch('omi.summarizer.MemorySummarizer')
    def test_compress_missing_api_key(self, mock_summarizer_cls, mock_vault_cls):
        """Test compress command fails gracefully when API key is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store old memories
            old_date = datetime.now() - timedelta(days=40)
            for i in range(2):
                self._store_memory_with_date(
                    base_path,
                    f"Old memory {i}",
                    "experience",
                    old_date
                )

            # Mock backup to succeed
            mock_vault = MagicMock()
            mock_metadata = MagicMock()
            mock_metadata.backup_id = "test-backup-123"
            mock_metadata.file_size = 1024
            mock_vault.backup.return_value = mock_metadata
            mock_vault_cls.return_value = mock_vault

            # Run compress without API key
            env = {"OMI_BASE_PATH": str(base_path)}
            # Ensure no API key in environment
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("OPENAI_API_KEY", None)

            with patch.dict(os.environ, env, clear=True):
                result = runner.invoke(cli, ["compress"])

            assert result.exit_code == 1
            assert "Missing ANTHROPIC_API_KEY" in result.output

    def test_compress_llm_provider_option(self):
        """Test compress command with different LLM providers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store old memories
            old_date = datetime.now() - timedelta(days=40)
            for i in range(2):
                self._store_memory_with_date(
                    base_path,
                    f"Old memory {i}",
                    "experience",
                    old_date
                )

            # Test with OpenAI provider (dry run)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "compress",
                    "--dry-run",
                    "--llm-provider", "openai"
                ])

            assert result.exit_code == 0
            assert "LLM Provider: openai" in result.output

            # Test with Anthropic provider (default, dry run)
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, [
                    "compress",
                    "--dry-run",
                    "--llm-provider", "anthropic"
                ])

            assert result.exit_code == 0
            assert "LLM Provider: anthropic" in result.output

    @patch('omi.summarizer.MemorySummarizer')
    @patch('omi.embeddings.NIMEmbedder')
    @patch('omi.moltvault.MoltVault')
    def test_compress_with_local_backup_fallback(self, mock_vault_cls, mock_embedder_cls, mock_summarizer_cls):
        """Test compress creates local backup when MoltVault raises ImportError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Store old memories
            old_date = datetime.now() - timedelta(days=40)
            for i in range(2):
                self._store_memory_with_date(
                    base_path,
                    f"Old memory {i}",
                    "experience",
                    old_date
                )

            # Mock summarizer and embedder
            mock_summarizer = MagicMock()
            mock_summarizer.batch_summarize.return_value = ["Summary 0", "Summary 1"]
            mock_summarizer_cls.return_value = mock_summarizer

            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
            mock_embedder_cls.return_value = mock_embedder

            # Mock MoltVault constructor to raise ImportError
            mock_vault_cls.side_effect = ImportError("MoltVault not found")

            with patch.dict(os.environ, {
                "OMI_BASE_PATH": str(base_path),
                "ANTHROPIC_API_KEY": "test-key"
            }):
                result = runner.invoke(cli, ["compress"])

            # Should succeed with local backup
            assert result.exit_code == 0
            assert "MoltVault not available" in result.output or "Local backup created" in result.output

            # Verify local backup was created
            backup_dir = base_path / "backups"
            assert backup_dir.exists()
            backup_files = list(backup_dir.glob("pre_compress_*.db"))
            assert len(backup_files) > 0

    def test_compress_empty_database(self):
        """Test compress with empty database (no memories)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize (empty database)
            self._init_omi(runner, base_path)

            # Run compress
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run"])

            assert result.exit_code == 0
            assert "No memories to compress" in result.output

    def test_compress_invalid_age_days(self):
        """Test compress with invalid --age-days value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = CliRunner()
            base_path = Path(tmpdir) / "omi"

            # Initialize
            self._init_omi(runner, base_path)

            # Try negative age-days
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["compress", "--dry-run", "--age-days", "-5"])

            assert result.exit_code == 1
            assert "--age-days must be a positive integer" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
