"""
Tests for MoltVault complete backup/restore roundtrip cycles.

Covers:
- Full backup -> delete -> restore -> verify cycle
- Incremental backup -> restore cycle
- Directory structure preservation
- Multiple files and nested directories
- Encrypted backup roundtrip (when available)
- Checksum verification
"""

import pytest
import json
import hashlib
import tempfile
import tarfile
import io
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Ensure omi modules are importable
def ensure_omi_importable():
    """Add src to path if needed."""
    test_dir = Path(__file__).resolve().parent
    src_path = test_dir.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

ensure_omi_importable()


class TestBasicRoundtrip:
    """Test basic backup and restore roundtrip cycles."""

    @pytest.fixture
    def source_omi_dir(self, tmp_path):
        """Create source OMI directory with test data."""
        omi_dir = tmp_path / "source_omi"
        omi_dir.mkdir(parents=True)

        # Critical files
        (omi_dir / "palace.sqlite").write_text("SQLite database content")
        (omi_dir / "NOW.md").write_text("# NOW\n\nCurrent task")
        (omi_dir / "config.yaml").write_text("embedding:\n  provider: ollama\n")
        (omi_dir / "MEMORY.md").write_text("# Long-term Memory")

        # Memory logs
        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "2025-02-01.md").write_text("## Morning\n\nWork started.")
        (memory_dir / "2025-02-02.md").write_text("## Evening\n\nWork continued.")

        # Hash files
        (omi_dir / ".now.hash").write_text("abc123")
        (omi_dir / ".memory.hash").write_text("xyz789")

        return omi_dir

    def test_full_backup_delete_restore_verify(self, source_omi_dir, tmp_path):
        """Test complete cycle: backup -> delete -> restore -> verify."""
        from omi.moltvault import MoltVault

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(source_omi_dir)

            # Save original content
            original_palace = (source_omi_dir / "palace.sqlite").read_text()
            original_now = (source_omi_dir / "NOW.md").read_text()
            original_config = (source_omi_dir / "config.yaml").read_text()
            original_log1 = (source_omi_dir / "memory" / "2025-02-01.md").read_text()
            original_log2 = (source_omi_dir / "memory" / "2025-02-02.md").read_text()

            # Create backup archive
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Delete all original files
                (source_omi_dir / "palace.sqlite").unlink()
                (source_omi_dir / "NOW.md").unlink()
                (source_omi_dir / "config.yaml").unlink()
                shutil.rmtree(source_omi_dir / "memory")

                # Verify deletion
                assert not (source_omi_dir / "palace.sqlite").exists()
                assert not (source_omi_dir / "NOW.md").exists()
                assert not (source_omi_dir / "memory").exists()

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir(parents=True)

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify restored content matches original
                assert (target_dir / "palace.sqlite").read_text() == original_palace
                assert (target_dir / "NOW.md").read_text() == original_now
                assert (target_dir / "config.yaml").read_text() == original_config
                assert (target_dir / "memory" / "2025-02-01.md").read_text() == original_log1
                assert (target_dir / "memory" / "2025-02-02.md").read_text() == original_log2

    def test_backup_preserves_directory_structure(self, source_omi_dir, tmp_path):
        """Test that directory structure is preserved in backup."""
        from omi.moltvault import MoltVault

        # Create memory directory structure
        # Note: MoltVault only backs up .md files directly in memory/, not nested subdirs
        memory_dir = source_omi_dir / "memory"

        # Add more log files to test structure
        (memory_dir / "2025-02-03.md").write_text("Day 3")
        (memory_dir / "2025-02-04.md").write_text("Day 4")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(source_omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify directory structure preserved
                assert (target_dir / "memory").exists()
                assert (target_dir / "memory" / "2025-02-01.md").exists()
                assert (target_dir / "memory" / "2025-02-02.md").exists()
                assert (target_dir / "memory" / "2025-02-03.md").exists()
                assert (target_dir / "memory" / "2025-02-04.md").read_text() == "Day 4"

    def test_backup_with_empty_directories(self, tmp_path):
        """Test backup handles empty directories correctly."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi_with_empty"
        omi_dir.mkdir(parents=True)

        # Create files
        (omi_dir / "palace.sqlite").write_text("db")
        (omi_dir / "NOW.md").write_text("# NOW")

        # Create empty directory
        empty_dir = omi_dir / "embeddings"
        empty_dir.mkdir()

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Files should be restored
                assert (target_dir / "palace.sqlite").exists()
                assert (target_dir / "NOW.md").exists()


class TestIncrementalRoundtrip:
    """Test incremental backup and restore cycles."""

    @pytest.fixture
    def source_omi_dir(self, tmp_path):
        """Create source OMI directory."""
        omi_dir = tmp_path / "source_omi"
        omi_dir.mkdir(parents=True)

        (omi_dir / "palace.sqlite").write_text("db v1")
        (omi_dir / "NOW.md").write_text("# NOW v1")
        (omi_dir / "config.yaml").write_text("config v1")

        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "2025-02-01.md").write_text("Log 1")

        return omi_dir

    def test_incremental_backup_after_full(self, source_omi_dir, tmp_path):
        """Test incremental backup captures only changed files."""
        from omi.moltvault import MoltVault

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(source_omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Full backup
                files_full = vault._get_files_to_backup(incremental=False)
                archive_full = vault._create_backup_archive(files_full, "full", temp_dir)

                # Mark backup time in the past (1 hour ago)
                past_time = datetime.now() - timedelta(hours=1)
                vault._last_backup_path.write_text(past_time.isoformat())

                # Modify one file
                (source_omi_dir / "NOW.md").write_text("# NOW v2 - modified")

                # Incremental backup
                files_incr = vault._get_files_to_backup(incremental=True)
                file_names = {f.name for f in files_incr}

                # Should include modified file
                assert "NOW.md" in file_names
                # Should include palace.sqlite (always included)
                assert "palace.sqlite" in file_names

    def test_incremental_restore(self, source_omi_dir, tmp_path):
        """Test restoring from incremental backup."""
        from omi.moltvault import MoltVault

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(source_omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Full backup
                files_full = vault._get_files_to_backup(incremental=False)
                archive_full = vault._create_backup_archive(files_full, "full", temp_dir)

                # Restore full backup
                full_restore_dir = tmp_path / "full_restore"
                full_restore_dir.mkdir()

                with tarfile.open(archive_full, "r:gz") as tar:
                    tar.extractall(path=full_restore_dir)

                # Modify and create incremental
                (source_omi_dir / "NOW.md").write_text("# NOW v2")
                vault._last_backup_path.write_text(
                    (datetime.now() - timedelta(hours=1)).isoformat()
                )

                files_incr = vault._get_files_to_backup(incremental=True)
                archive_incr = vault._create_backup_archive(files_incr, "incremental", temp_dir)

                # Apply incremental on top of full restore
                with tarfile.open(archive_incr, "r:gz") as tar:
                    tar.extractall(path=full_restore_dir)

                # Verify incremental changes applied
                assert (full_restore_dir / "NOW.md").read_text() == "# NOW v2"


class TestContentIntegrity:
    """Test content integrity during backup/restore cycles."""

    def test_binary_file_roundtrip(self, tmp_path):
        """Test backup and restore of binary files."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        # Create binary file with specific byte pattern
        binary_data = bytes([i % 256 for i in range(1000)])
        (omi_dir / "palace.sqlite").write_bytes(binary_data)
        (omi_dir / "NOW.md").write_text("# NOW")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify binary data preserved exactly
                restored_data = (target_dir / "palace.sqlite").read_bytes()
                assert restored_data == binary_data

    def test_large_file_roundtrip(self, tmp_path):
        """Test backup and restore of larger files."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        # Create larger file (100KB)
        large_content = "x" * 100000
        (omi_dir / "palace.sqlite").write_text(large_content)
        (omi_dir / "NOW.md").write_text("# NOW")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify large file preserved
                restored_content = (target_dir / "palace.sqlite").read_text()
                assert restored_content == large_content
                assert len(restored_content) == 100000

    def test_unicode_content_roundtrip(self, tmp_path):
        """Test backup and restore of files with unicode content."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        # Unicode content
        unicode_content = "# NOW\n\n日本語 Español 中文 Русский 🚀 ✨"
        (omi_dir / "NOW.md").write_text(unicode_content, encoding="utf-8")
        (omi_dir / "palace.sqlite").write_text("db")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify unicode preserved
                restored_content = (target_dir / "NOW.md").read_text(encoding="utf-8")
                assert restored_content == unicode_content

    def test_checksum_verification(self, tmp_path):
        """Test that checksums are calculated correctly during backup."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        content = "test content for checksum"
        (omi_dir / "palace.sqlite").write_text(content)
        (omi_dir / "NOW.md").write_text("# NOW")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Calculate checksum
                checksum = vault._calculate_checksum(archive_path)

                # Verify checksum format
                assert len(checksum) == 64  # SHA256 hex

                # Verify checksum is consistent
                checksum2 = vault._calculate_checksum(archive_path)
                assert checksum == checksum2


class TestEncryptedRoundtrip:
    """Test encrypted backup/restore roundtrip cycles."""

    @pytest.fixture
    def source_omi_dir(self, tmp_path):
        """Create source OMI directory."""
        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        (omi_dir / "palace.sqlite").write_text("sensitive data")
        (omi_dir / "NOW.md").write_text("# Secret task")

        return omi_dir

    def test_encrypted_backup_roundtrip(self, source_omi_dir, tmp_path, monkeypatch):
        """Test complete encrypted backup and restore cycle."""
        from omi.moltvault import MoltVault

        # Skip if crypto not available
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography not available")

        monkeypatch.setenv("MOLTVAULT_KEY", "test-encryption-key-12345")
        monkeypatch.setattr("omi.moltvault.BOTO3_AVAILABLE", True)
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")

        vault = MoltVault(source_omi_dir)

        original_palace = (source_omi_dir / "palace.sqlite").read_text()
        original_now = (source_omi_dir / "NOW.md").read_text()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create backup
            files = vault._get_files_to_backup(incremental=False)
            archive_path = vault._create_backup_archive(files, "full", temp_dir)

            # Encrypt
            encrypted_path = vault._encrypt_file(archive_path, temp_dir)

            # Verify encrypted content is different
            encrypted_content = encrypted_path.read_bytes()
            original_content = archive_path.read_bytes()
            assert encrypted_content != original_content

            # Decrypt
            decrypted_path = vault._decrypt_file(encrypted_path, temp_dir)

            # Restore from decrypted
            target_dir = tmp_path / "restored"
            target_dir.mkdir()

            with tarfile.open(decrypted_path, "r:gz") as tar:
                tar.extractall(path=target_dir)

            # Verify content matches original
            assert (target_dir / "palace.sqlite").read_text() == original_palace
            assert (target_dir / "NOW.md").read_text() == original_now


class TestMultipleBackups:
    """Test multiple backup cycles and restore points."""

    @pytest.fixture
    def source_omi_dir(self, tmp_path):
        """Create source OMI directory."""
        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        (omi_dir / "palace.sqlite").write_text("db v1")
        (omi_dir / "NOW.md").write_text("# NOW v1")

        return omi_dir

    def test_multiple_backup_versions(self, source_omi_dir, tmp_path):
        """Test creating multiple backup versions and restoring each."""
        from omi.moltvault import MoltVault

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(source_omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Create version 1 backup
                files_v1 = vault._get_files_to_backup(incremental=False)
                archive_v1 = vault._create_backup_archive(files_v1, "full", temp_dir)
                archive_v1_copy = temp_dir / "backup_v1.tar.gz"
                shutil.copy(archive_v1, archive_v1_copy)

                # Modify files
                (source_omi_dir / "NOW.md").write_text("# NOW v2")

                # Create version 2 backup
                files_v2 = vault._get_files_to_backup(incremental=False)
                archive_v2 = vault._create_backup_archive(files_v2, "full", temp_dir)
                archive_v2_copy = temp_dir / "backup_v2.tar.gz"
                shutil.copy(archive_v2, archive_v2_copy)

                # Modify again
                (source_omi_dir / "NOW.md").write_text("# NOW v3")

                # Create version 3 backup
                files_v3 = vault._get_files_to_backup(incremental=False)
                archive_v3 = vault._create_backup_archive(files_v3, "full", temp_dir)
                archive_v3_copy = temp_dir / "backup_v3.tar.gz"
                shutil.copy(archive_v3, archive_v3_copy)

                # Restore v1
                restore_v1 = tmp_path / "restore_v1"
                restore_v1.mkdir()
                with tarfile.open(archive_v1_copy, "r:gz") as tar:
                    tar.extractall(path=restore_v1)
                assert (restore_v1 / "NOW.md").read_text() == "# NOW v1"

                # Restore v2
                restore_v2 = tmp_path / "restore_v2"
                restore_v2.mkdir()
                with tarfile.open(archive_v2_copy, "r:gz") as tar:
                    tar.extractall(path=restore_v2)
                assert (restore_v2 / "NOW.md").read_text() == "# NOW v2"

                # Restore v3
                restore_v3 = tmp_path / "restore_v3"
                restore_v3.mkdir()
                with tarfile.open(archive_v3_copy, "r:gz") as tar:
                    tar.extractall(path=restore_v3)
                assert (restore_v3 / "NOW.md").read_text() == "# NOW v3"


class TestEdgeCases:
    """Test edge cases in backup/restore cycles."""

    def test_empty_file_roundtrip(self, tmp_path):
        """Test backup and restore of empty files."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        # Create empty files
        (omi_dir / "palace.sqlite").write_text("")
        (omi_dir / "NOW.md").write_text("")
        (omi_dir / "config.yaml").write_text("# config")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify empty files preserved
                assert (target_dir / "palace.sqlite").exists()
                assert (target_dir / "palace.sqlite").read_text() == ""
                assert (target_dir / "NOW.md").exists()
                assert (target_dir / "NOW.md").read_text() == ""

    def test_special_characters_in_filenames(self, tmp_path):
        """Test backup and restore of files with special characters."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        # Create files with special characters (avoid OS-restricted chars)
        (omi_dir / "palace.sqlite").write_text("db")
        (omi_dir / "NOW.md").write_text("# NOW")

        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "file-with-dashes.md").write_text("content 1")
        (memory_dir / "file_with_underscores.md").write_text("content 2")
        (memory_dir / "file.with.dots.md").write_text("content 3")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify all files with special characters restored
                assert (target_dir / "memory" / "file-with-dashes.md").exists()
                assert (target_dir / "memory" / "file_with_underscores.md").exists()
                assert (target_dir / "memory" / "file.with.dots.md").exists()

    def test_many_files_roundtrip(self, tmp_path):
        """Test backup and restore of many files."""
        from omi.moltvault import MoltVault

        omi_dir = tmp_path / "omi"
        omi_dir.mkdir(parents=True)

        (omi_dir / "palace.sqlite").write_text("db")
        (omi_dir / "NOW.md").write_text("# NOW")

        # Create many memory log files
        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()

        for i in range(100):
            (memory_dir / f"2025-02-{i:02d}.md").write_text(f"Log entry {i}")

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(omi_dir)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)

                # Backup
                files = vault._get_files_to_backup(incremental=False)
                assert len(files) >= 100  # At least all the log files

                archive_path = vault._create_backup_archive(files, "full", temp_dir)

                # Restore
                target_dir = tmp_path / "restored"
                target_dir.mkdir()

                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)

                # Verify all files restored
                restored_memory = target_dir / "memory"
                assert restored_memory.exists()

                restored_files = list(restored_memory.glob("*.md"))
                assert len(restored_files) == 100

                # Verify content of sample files
                assert (restored_memory / "2025-02-00.md").read_text() == "Log entry 0"
                assert (restored_memory / "2025-02-50.md").read_text() == "Log entry 50"
                assert (restored_memory / "2025-02-99.md").read_text() == "Log entry 99"
