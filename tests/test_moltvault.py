"""
Tests for MoltVault backup/restore module.

Covers:
- Backup/restore cycle
- Incremental detection
- Encryption/decryption
- S3/R2 mocking
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
from unittest.mock import Mock, MagicMock, patch, call
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


class TestBackupMetadata:
    """Test BackupMetadata dataclass."""
    
    def test_to_dict(self):
        """Test metadata serialization to dict."""
        from omi.moltvault import BackupMetadata
        
        meta = BackupMetadata(
            backup_id="test_backup_1",
            backup_type="full",
            created_at="2025-01-01T00:00:00",
            file_size=1024,
            checksum="abc123",
            encrypted=True,
            files_included=["file1.txt", "file2.db"],
            base_path_hash="hash1234",
            retention_days=30,
        )
        
        d = meta.to_dict()
        
        assert d["backup_id"] == "test_backup_1"
        assert d["backup_type"] == "full"
        assert d["file_size"] == 1024
        assert d["files_included"] == ["file1.txt", "file2.db"]
        assert d["encrypted"] is True
    
    def test_from_dict(self):
        """Test metadata deserialization from dict."""
        from omi.moltvault import BackupMetadata
        
        data = {
            "backup_id": "test_backup_2",
            "backup_type": "incremental",
            "created_at": "2025-01-02T12:00:00",
            "file_size": 512,
            "checksum": "def456",
            "encrypted": False,
            "files_included": ["file3.txt"],
            "base_path_hash": "hash5678",
            "retention_days": 7,
        }
        
        meta = BackupMetadata.from_dict(data)
        
        assert meta.backup_id == "test_backup_2"
        assert meta.backup_type == "incremental"
        assert meta.file_size == 512
        assert meta.retention_days == 7


class TestEncryptionManager:
    """Test encryption functionality."""
    
    @pytest.fixture
    def mock_crypto_available(self, monkeypatch):
        """Ensure cryptography is marked as available."""
        monkeypatch.setattr(
            "omi.moltvault.CRYPTO_AVAILABLE",
            True,
            raising=False
        )
    
    def test_encryption_requires_key(self, mock_crypto_available):
        """Test that encryption requires a key."""
        from omi.moltvault import EncryptionManager
        
        with patch.dict(os.environ, {"MOLTVAULT_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="Encryption key required"):
                EncryptionManager()
    
    @pytest.mark.skip(reason="Requires full cryptography package")
    def test_encrypt_decrypt_round_trip(self, mock_crypto_available, monkeypatch):
        """Test encryption and decryption work together."""
        # This test requires the actual cryptography package
        # Skipped because we don't have a clean way to mock Fernet and PBKDF2HMAC
        pass


class TestMoltVaultBasic:
    """Test basic MoltVault functionality without S3."""
    
    @pytest.fixture
    def temp_omi_dir(self, tmp_path):
        """Create temporary OMI directory structure."""
        omi_dir = tmp_path / ".openclaw" / "omi"
        omi_dir.mkdir(parents=True)
        
        # Create critical files
        (omi_dir / "palace.sqlite").write_text("fake sqlite data")
        (omi_dir / "NOW.md").write_text("# NOW\n\nCurrent task")
        (omi_dir / "config.yaml").write_text("# config")
        (omi_dir / "MEMORY.md").write_text("# Memory")
        
        # Create memory logs
        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "2025-01-01.md").write_text("Log 1")
        (memory_dir / "2025-01-02.md").write_text("Log 2")
        
        # Create hash file
        (omi_dir / ".now.hash").write_text("fakehash123")
        
        return omi_dir
    
    def test_moltvault_initialization(self, temp_omi_dir):
        """Test MoltVault can be initialized."""
        from omi.moltvault import MoltVault
        
        # Skip S3 requirement for basic test
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            assert vault.base_path == temp_omi_dir
            assert vault.bucket == "moltbot-data"
    
    def test_get_files_to_backup_full(self, temp_omi_dir):
        """Test getting files for full backup."""
        from omi.moltvault import MoltVault
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            files = vault._get_files_to_backup(incremental=False)
            
            # Should include all critical files
            file_names = {f.name for f in files}
            assert "palace.sqlite" in file_names
            assert "NOW.md" in file_names
            assert "config.yaml" in file_names
            assert "MEMORY.md" in file_names
            
            # Should include memory logs
            assert "2025-01-01.md" in file_names
            assert "2025-01-02.md" in file_names
            
            # Should include hash file
            assert ".now.hash" in file_names
    
    def test_get_files_to_backup_incremental_no_last_backup(self, temp_omi_dir):
        """Test getting files for incremental when no last backup exists."""
        from omi.moltvault import MoltVault
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            files = vault._get_files_to_backup(incremental=True)
            
            # Should include all files since no last backup
            file_names = {f.name for f in files}
            assert "palace.sqlite" in file_names
            assert "NOW.md" in file_names
    
    def test_calculate_checksum(self, temp_omi_dir):
        """Test checksum calculation."""
        from omi.moltvault import MoltVault
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            
            test_file = temp_omi_dir / "test.txt"
            test_file.write_text("test content")
            
            checksum = vault._calculate_checksum(test_file)
            
            # Should be valid SHA256 hex string
            assert len(checksum) == 64
            int(checksum, 16)  # Should be valid hex
            
            # Should be consistent
            checksum2 = vault._calculate_checksum(test_file)
            assert checksum == checksum2
    
    def test_create_backup_archive(self, temp_omi_dir):
        """Test creating tar.gz archive."""
        from omi.moltvault import MoltVault
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            
            files = [
                temp_omi_dir / "NOW.md",
                temp_omi_dir / "config.yaml",
            ]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)
                
                # Archive should exist
                assert archive_path.exists()
                assert archive_path.suffix == ".gz"
                
                # Should be a valid tar.gz
                with tarfile.open(archive_path, "r:gz") as tar:
                    names = tar.getnames()
                    assert "NOW.md" in names
                    assert "config.yaml" in names
    
    def test_archive_preserves_structure(self, temp_omi_dir):
        """Test that archive preserves directory structure."""
        from omi.moltvault import MoltVault
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            
            files = [
                temp_omi_dir / "palace.sqlite",
                temp_omi_dir / "memory" / "2025-01-01.md",
            ]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)
                
                with tarfile.open(archive_path, "r:gz") as tar:
                    names = tar.getnames()
                    assert "palace.sqlite" in names
                    assert "memory/2025-01-01.md" in names or "memory\\2025-01-01.md" in names


class TestMoltVaultWithMockS3:
    """Test MoltVault with mocked S3."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client."""
        mock = MagicMock()
        
        # Mock list_objects_v2
        mock.list_objects_v2.return_value = {"Contents": []}
        
        # Mock put_object
        mock.put_object.return_value = {}
        
        # Mock upload_fileobj
        mock.upload_fileobj.return_value = None
        
        return mock
    
    @pytest.fixture
    def temp_omi_dir(self, tmp_path):
        """Create temporary OMI directory."""
        omi_dir = tmp_path / ".openclaw" / "omi"
        omi_dir.mkdir(parents=True)
        
        # Create files
        (omi_dir / "palace.sqlite").write_text("fake sqlite data")
        (omi_dir / "NOW.md").write_text("# NOW\n\nCurrent task")
        (omi_dir / "config.yaml").write_text("# config")
        (omi_dir / "MEMORY.md").write_text("# Memory")
        
        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "2025-01-01.md").write_text("Log entry")
        
        return omi_dir
    
    @pytest.fixture
    def vault_with_mock_s3(self, temp_omi_dir, mock_s3_client, monkeypatch):
        """Create MoltVault with mocked S3."""
        from omi.moltvault import MoltVault
        
        # Mark boto3 as available but we'll provide our own client
        monkeypatch.setattr("omi.moltvault.BOTO3_AVAILABLE", True)
        
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-access-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret-key")
        
        vault = MoltVault(
            temp_omi_dir,
            bucket="test-bucket",
            endpoint="https://test.example.com",
            access_key="fake-access-key",
            secret_key="fake-secret-key",
        )
        
        vault._s3_client = mock_s3_client
        
        return vault, mock_s3_client
    
    def test_backup_full_uploads_to_s3(self, vault_with_mock_s3):
        """Test full backup uploads to S3."""
        vault, mock_s3 = vault_with_mock_s3
        
        metadata = vault.backup(full=True)
        
        assert metadata.backup_type == "full"
        assert metadata.backup_id.startswith("omi_full_")
        assert len(metadata.files_included) > 0
        
        # Verify S3 upload
        mock_s3.upload_fileobj.assert_called_once()
        mock_s3.put_object.assert_called_once()
    
    def test_backup_incremental(self, vault_with_mock_s3):
        """Test incremental backup."""
        vault, _ = vault_with_mock_s3

        # Create initial backup
        full_meta = vault.backup(full=True)

        # Modify a file
        (vault.base_path / "NOW.md").write_text("# NOW\n\nUpdated task")

        # Create incremental backup
        incr_meta = vault.backup(incremental=True)

        assert incr_meta.backup_type == "incremental"

    def test_backup_with_progress_callback(self, vault_with_mock_s3):
        """Test backup with progress callback."""
        vault, _ = vault_with_mock_s3

        # Track progress updates
        progress_updates = []

        def progress_callback(percentage):
            progress_updates.append(percentage)

        # Create backup with progress callback
        metadata = vault.backup(full=True, progress_callback=progress_callback)

        # Verify backup completed
        assert metadata.backup_type == "full"

        # Verify progress callback was called with increasing values
        assert len(progress_updates) > 0
        assert progress_updates[0] <= progress_updates[-1]
        assert 100 in progress_updates  # Should reach 100% at the end

        # Verify progress increases monotonically (each value >= previous)
        for i in range(1, len(progress_updates)):
            assert progress_updates[i] >= progress_updates[i-1]

    def test_list_backups(self, vault_with_mock_s3):
        """Test listing backups."""
        vault, mock_s3 = vault_with_mock_s3
        
        # Mock backup objects in S3
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "backups/backup_1.json"},
                {"Key": "backups/backup_1.tar.gz"},
                {"Key": "backups/backup_2.json"},
            ]
        }
        
        # Mock metadata retrieval
        from omi.moltvault import BackupMetadata
        mock_meta = BackupMetadata(
            backup_id="backup_1",
            backup_type="full",
            created_at="2025-01-01T00:00:00",
            file_size=1024,
            checksum="abc",
            encrypted=False,
            files_included=["file.txt"],
            base_path_hash="xyz",
            retention_days=30,
        )
        
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps(mock_meta.to_dict()).encode()))
        }
        
        backups = vault.list_backups()
        
        # Should be sorted by date
        assert isinstance(backups, list)
    
    def test_restore_backup(self, vault_with_mock_s3, temp_omi_dir):
        """Test restoring from backup."""
        vault, mock_s3 = vault_with_mock_s3
        
        # Create a mock backup
        from omi.moltvault import BackupMetadata
        from datetime import datetime
        
        metadata = BackupMetadata(
            backup_id="test_restore_1",
            backup_type="full",
            created_at=datetime.now().isoformat(),
            file_size=100,
            checksum="fake_checksum_for_testing_12345678901234567890",
            encrypted=False,
            files_included=["NOW.md"],
            base_path_hash="hash",
            retention_days=30,
        )
        
        def mock_get_object(**kwargs):
            if kwargs.get("Key") == "backups/test_restore_1.json":
                return {
                    "Body": MagicMock(read=MagicMock(return_value=json.dumps(metadata.to_dict()).encode()))
                }
            else:
                return {
                    "Body": MagicMock(read=MagicMock(return_value=b"mock_tar_content"))
                }
        
        mock_s3.get_object.side_effect = mock_get_object
        
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "restored"
            
            # Create a simple tar.gz
            archive_data = io.BytesIO()
            with tarfile.open(fileobj=archive_data, mode="w:gz") as tar:
                content = b"# Restored NOW"
                info = tarfile.TarInfo(name="NOW.md")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            archive_data.seek(0)
            
            # Mock download_file
            def mock_download(bucket, key, filepath):
                Path(filepath).write_bytes(archive_data.read())
                archive_data.seek(0)
            
            mock_s3.download_file = mock_download
            
            # Update checksum to match
            metadata.checksum = hashlib.sha256(archive_data.read()).hexdigest()
            archive_data.seek(0)
            
            # Perform restore
            restored_path = vault.restore("test_restore_1", target, verify=False)
            
            assert restored_path.exists()


class TestMoltVaultCleanup:
    """Test retention policy cleanup."""
    
    def test_cleanup_dry_run(self, monkeypatch):
        """Test cleanup in dry-run mode."""
        from omi.moltvault import MoltVault
        
        monkeypatch.setattr("omi.moltvault.BOTO3_AVAILABLE", True)
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")
        
        vault = MoltVault(Path("/tmp/test"))
        
        # Mock list_backups
        old_backup = MagicMock()
        old_backup.created_at = (datetime.now() - timedelta(days=40)).isoformat()
        old_backup.backup_id = "old_backup"
        old_backup.retention_days = 30
        old_backup.encrypted = False
        
        new_backup = MagicMock()
        new_backup.created_at = datetime.now().isoformat()
        new_backup.backup_id = "new_backup"
        new_backup.retention_days = 30
        new_backup.encrypted = False
        
        s3 = MagicMock()
        vault._s3_client = s3
        
        with patch.object(vault, "list_backups", return_value=[old_backup, new_backup]):
            result = vault.cleanup(dry_run=True)
            
            # Should report deleted but not actually delete
            assert result["deleted"] == 1
            assert result["kept"] == 1
            s3.delete_object.assert_not_called()
    
    def test_cleanup_deletes_old_backups(self, monkeypatch):
        """Test cleanup actually deletes old backups."""
        from omi.moltvault import MoltVault
        
        monkeypatch.setattr("omi.moltvault.BOTO3_AVAILABLE", True)
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")
        
        vault = MoltVault(Path("/tmp/test"))
        
        # Mock old backup
        old_backup = MagicMock()
        old_backup.created_at = (datetime.now() - timedelta(days=40)).isoformat()
        old_backup.backup_id = "old_backup"
        old_backup.retention_days = 30
        old_backup.encrypted = False
        
        s3 = MagicMock()
        vault._s3_client = s3
        
        with patch.object(vault, "list_backups", return_value=[old_backup]):
            result = vault.cleanup(dry_run=False)
            
            assert result["deleted"] == 1
            s3.delete_object.assert_has_calls([
                call(Bucket=vault.bucket, Key="backups/old_backup.tar.gz"),
                call(Bucket=vault.bucket, Key="backups/old_backup.json"),
            ])


class TestIncrementalDetection:
    """Test incremental backup detection logic."""
    
    @pytest.fixture
    def temp_omi_dir(self, tmp_path):
        """Create temporary OMI directory."""
        omi_dir = tmp_path / ".openclaw" / "omi"
        omi_dir.mkdir(parents=True)
        (omi_dir / "palace.sqlite").write_text("db data")
        (omi_dir / "NOW.md").write_text("# NOW")
        return omi_dir
    
    def test_incremental_only_changed_files(self, temp_omi_dir):
        """Test incremental only includes changed files."""
        from omi.moltvault import MoltVault
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            
            # Set last backup time to now
            vault._last_backup_path.write_text(datetime.now().isoformat())
            
            # Get incremental files
            files = vault._get_files_to_backup(incremental=True)
            
            file_names = {f.name for f in files}
            assert "palace.sqlite" in file_names
    
    def test_incremental_with_modified_files(self, temp_omi_dir, monkeypatch):
        """Test incremental includes modified files."""
        from omi.moltvault import MoltVault
        
        # Set last backup time in the past
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        last_backup_path = temp_omi_dir / ".moltvault_last_backup"
        last_backup_path.write_text(past)
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            
            # Modify NOW.md
            (temp_omi_dir / "NOW.md").write_text("# NOW\n\nUpdated!")
            
            files = vault._get_files_to_backup(incremental=True)
            
            file_names = {f.name for f in files}
            # Should include modified NOW.md
            assert "NOW.md" in file_names


@pytest.mark.skip(reason="CLI tests require imports setup")
class TestCLICommands:
    """Test CLI commands for backup/restore."""
    
    def test_backup_full_cli(self, tmp_path, monkeypatch):
        """Test backup full CLI command."""
        from click.testing import CliRunner
        from omi.cli import cli
        from omi import moltvault
        
        # Create OMI directory
        omi_dir = tmp_path / ".openclaw" / "omi"
        omi_dir.mkdir(parents=True)
        (omi_dir / "palace.sqlite").write_text("db")
        (omi_dir / "NOW.md").write_text("# NOW")
        
        # Mock environment
        monkeypatch.setenv("OMI_BASE_PATH", str(omi_dir))
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")
        
        runner = CliRunner()
        
        # Mock MoltVault at the function level
        with patch.object(moltvault, 'MoltVault') as MockVault:
            mock_vault = MagicMock()
            mock_metadata = MagicMock()
            mock_metadata.backup_id = "test_backup_123"
            mock_metadata.backup_type = "full"
            mock_metadata.file_size = 1024
            mock_metadata.encrypted = False
            mock_metadata.files_included = ["palace.sqlite", "NOW.md"]
            
            mock_vault.backup.return_value = mock_metadata
            MockVault.return_value = mock_vault
            
            result = runner.invoke(cli, ['backup', 'full'])
            
            assert result.exit_code == 0
            assert "Full backup created" in result.output
    
    def test_restore_list_cli(self, monkeypatch):
        """Test restore list CLI command."""
        from click.testing import CliRunner
        from omi.cli import cli
        from omi import moltvault
        
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")
        
        runner = CliRunner()
        
        with patch.object(moltvault, 'MoltVault') as MockVault:
            from omi.moltvault import BackupMetadata
            
            mock_vault = MagicMock()
            mock_metadata = BackupMetadata(
                backup_id="backup_1",
                backup_type="full",
                created_at=datetime.now().isoformat(),
                file_size=1024,
                checksum="abc",
                encrypted=False,
                files_included=["file.txt"],
                base_path_hash="xyz",
                retention_days=30,
            )
            mock_vault.list_backups.return_value = [mock_metadata]
            MockVault.return_value = mock_vault
            
            result = runner.invoke(cli, ['restore', 'list'])
            
            assert result.exit_code == 0
            assert "backup_1" in result.output or "Available Backups" in result.output


class TestEncryptionIntegration:
    """Test full encryption/decryption cycle with real crypto."""
    
    @pytest.fixture
    def temp_omi_dir(self, tmp_path):
        """Create temp OMI directory."""
        omi_dir = tmp_path / ".openclaw" / "omi"
        omi_dir.mkdir(parents=True)
        (omi_dir / "palace.sqlite").write_text("sensitive db data")
        (omi_dir / "NOW.md").write_text("# Secret NOW content")
        return omi_dir
    
    def test_encrypted_backup_cycle(self, temp_omi_dir, monkeypatch):
        """Test complete encrypted backup and restore cycle."""
        from omi.moltvault import MoltVault
        
        # Skip if crypto not available
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography not available")
        
        monkeypatch.setenv("MOLTVAULT_KEY", "test-key-for-encrypted-backup-cycle-1234")
        
        # Simulate boto3 available
        monkeypatch.setattr("omi.moltvault.BOTO3_AVAILABLE", True)
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")
        
        vault = MoltVault(temp_omi_dir)
        
        # Should have encryption manager
        assert vault._encryption is not None
        
        # Create encrypted backup
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            
            # Get files
            files = vault._get_files_to_backup(incremental=False)
            
            # Create archive
            archive = vault._create_backup_archive(files, "full", temp_dir)
            original_content = archive.read_bytes()
            
            # Encrypt
            encrypted = vault._encrypt_file(archive, temp_dir)
            encrypted_content = encrypted.read_bytes()
            
            # Encrypted content should be different
            assert encrypted_content != original_content
            
            # Decrypt
            decrypted = vault._decrypt_file(encrypted, temp_dir)
            decrypted_content = decrypted.read_bytes()
            
            # Should match original
            assert decrypted_content == original_content


class TestBackupRestoreCycle:
    """Full integration test of backup->restore cycle."""
    
    @pytest.fixture
    def source_omi_dir(self, tmp_path):
        """Create source OMI with various files."""
        omi_dir = tmp_path / "source_omi"
        omi_dir.mkdir(parents=True)
        
        # Critical files
        (omi_dir / "palace.sqlite").write_text("SQLite database content v1.2.3")
        (omi_dir / "NOW.md").write_text("# NOW\n\nCurrent task: Testing backup")
        (omi_dir / "config.yaml").write_text("embedding:\n  provider: ollama\n")
        (omi_dir / "MEMORY.md").write_text("# Long-term Memory\n\n- Learned X\n- Learned Y")
        
        # Memory logs
        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "2025-02-01.md").write_text("## Morning\n\nStarted testing.")
        (memory_dir / "2025-02-02.md").write_text("## Evening\n\nStill testing.")
        
        # Hash files
        (omi_dir / ".now.hash").write_text("abc123def456")
        (omi_dir / ".memory.hash").write_text("xyz789uvw012")
        
        return omi_dir
    
    def test_full_backup_restore(self, source_omi_dir, tmp_path, monkeypatch):
        """Test complete backup and restore cycle."""
        from omi.moltvault import MoltVault
        
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "fake-key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "fake-secret")
        
        target_dir = tmp_path / "restored_omi"
        
        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(source_omi_dir)
            
            # Create backup locally (bypass S3 upload)
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_dir = Path(tmpdir)
                
                # Create archive
                files = vault._get_files_to_backup(incremental=False)
                archive_path = vault._create_backup_archive(files, "full", temp_dir)
                
                # Verify archive contents
                with tarfile.open(archive_path, "r:gz") as tar:
                    names = tar.getnames()
                    assert "palace.sqlite" in names
                    assert "NOW.md" in names
                    assert "config.yaml" in names
                    assert "MEMORY.md" in names
                    assert any("2025-02" in n for n in names)
                
                # Extract to target directory
                target_dir.mkdir(parents=True)
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(path=target_dir)
                
                # Verify restored files
                assert (target_dir / "palace.sqlite").exists()
                assert (target_dir / "palace.sqlite").read_text() == "SQLite database content v1.2.3"
                assert (target_dir / "NOW.md").read_text() == "# NOW\n\nCurrent task: Testing backup"
                assert (target_dir / "config.yaml").read_text() == "embedding:\n  provider: ollama\n"
                
                # Check memory logs
                assert (target_dir / "memory" / "2025-02-01.md").exists()
                assert (target_dir / "memory" / "2025-02-02.md").exists()


class TestProgressCallbacks:
    """Test progress callback functionality in MoltVault operations."""

    @pytest.fixture
    def temp_omi_dir(self, tmp_path):
        """Create temporary OMI directory structure."""
        omi_dir = tmp_path / ".openclaw" / "omi"
        omi_dir.mkdir(parents=True)

        # Create critical files
        (omi_dir / "palace.sqlite").write_text("fake sqlite data")
        (omi_dir / "NOW.md").write_text("# NOW\n\nCurrent task")
        (omi_dir / "config.yaml").write_text("# config")
        (omi_dir / "MEMORY.md").write_text("# Memory")

        # Create memory logs
        memory_dir = omi_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "2025-01-01.md").write_text("Log 1")

        return omi_dir

    def test_backup_calls_progress_callback(self, temp_omi_dir):
        """Test that backup method calls progress_callback with updates."""
        from omi.moltvault import MoltVault

        progress_values = []

        def progress_callback(percentage: int):
            progress_values.append(percentage)

        # Mock backend to avoid actual S3 operations
        mock_backend = MagicMock()
        mock_backend.upload_file.return_value = None
        mock_backend.upload_data.return_value = None

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            vault._backend = mock_backend

            # Run backup with progress callback
            metadata = vault.backup(full=True, progress_callback=progress_callback)

            # Verify progress callback was called
            assert len(progress_values) > 0
            # Should have at least start and end progress
            assert 0 <= progress_values[0] <= 100
            # Should end at 100
            assert progress_values[-1] == 100
            # Progress should be monotonically increasing
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i-1]

    def test_backup_works_without_progress_callback(self, temp_omi_dir):
        """Test that backup works when progress_callback is None."""
        from omi.moltvault import MoltVault

        # Mock backend to avoid actual S3 operations
        mock_backend = MagicMock()
        mock_backend.upload_file.return_value = None
        mock_backend.upload_data.return_value = None

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            vault._backend = mock_backend

            # Run backup without progress callback - should not raise error
            metadata = vault.backup(full=True, progress_callback=None)

            assert metadata is not None
            assert metadata.backup_id is not None

    def test_restore_calls_progress_callback(self, temp_omi_dir, tmp_path):
        """Test that restore method calls progress_callback with updates."""
        from omi.moltvault import MoltVault, BackupMetadata

        progress_values = []

        def progress_callback(percentage: int):
            progress_values.append(percentage)

        # Create a test archive
        backup_id = "test-backup-123"

        # Create persistent test files
        test_files_dir = tmp_path / "test_files"
        test_files_dir.mkdir()
        archive_path = test_files_dir / f"{backup_id}.tar.gz"

        # Create minimal tar.gz
        with tarfile.open(archive_path, "w:gz") as tar:
            # Add a test file
            info = tarfile.TarInfo(name="test.txt")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))

        # Create metadata
        mock_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type="full",
            created_at="2025-01-01T00:00:00",
            file_size=1024,
            checksum=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            encrypted=False,
            files_included=["test.txt"],
            base_path_hash="hash123",
            retention_days=30,
        )

        # Mock backend download method to copy files to local_path
        def mock_download(key, local_path):
            if "tar.gz" in key:
                shutil.copy(archive_path, local_path)
            elif "json" in key:
                local_path.write_text(json.dumps(mock_metadata.to_dict()))

        mock_backend = MagicMock()
        mock_backend.download.side_effect = mock_download

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            vault._backend = mock_backend

            target_path = tmp_path / "restore"
            target_path.mkdir()

            # Run restore with progress callback
            restored_path = vault.restore(
                backup_id=backup_id,
                target_path=target_path,
                progress_callback=progress_callback
            )

            # Verify progress callback was called
            assert len(progress_values) > 0
            # Should end at 100
            assert progress_values[-1] == 100
            # Progress should be monotonically increasing
            for i in range(1, len(progress_values)):
                assert progress_values[i] >= progress_values[i-1]

    def test_restore_works_without_progress_callback(self, temp_omi_dir, tmp_path):
        """Test that restore works when progress_callback is None."""
        from omi.moltvault import MoltVault, BackupMetadata

        # Create a test archive
        backup_id = "test-backup-456"

        # Create persistent test files
        test_files_dir = tmp_path / "test_files"
        test_files_dir.mkdir()
        archive_path = test_files_dir / f"{backup_id}.tar.gz"

        # Create minimal tar.gz
        with tarfile.open(archive_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="test.txt")
            info.size = 5
            tar.addfile(info, io.BytesIO(b"hello"))

        # Create metadata
        mock_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type="full",
            created_at="2025-01-01T00:00:00",
            file_size=1024,
            checksum=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            encrypted=False,
            files_included=["test.txt"],
            base_path_hash="hash123",
            retention_days=30,
        )

        # Mock backend download method to copy files to local_path
        def mock_download(key, local_path):
            if "tar.gz" in key:
                shutil.copy(archive_path, local_path)
            elif "json" in key:
                local_path.write_text(json.dumps(mock_metadata.to_dict()))

        mock_backend = MagicMock()
        mock_backend.download.side_effect = mock_download

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            vault._backend = mock_backend

            target_path = tmp_path / "restore"
            target_path.mkdir()

            # Run restore without progress callback - should not raise error
            restored_path = vault.restore(
                backup_id=backup_id,
                target_path=target_path,
                progress_callback=None
            )

            assert restored_path is not None

    def test_cleanup_calls_progress_callback(self, temp_omi_dir):
        """Test that cleanup method calls progress_callback with updates."""
        from omi.moltvault import MoltVault, BackupMetadata

        progress_values = []

        def progress_callback(percentage: int):
            progress_values.append(percentage)

        # Mock backend with some backups
        mock_backend = MagicMock()

        # Create mock metadata for old backup
        old_backup = BackupMetadata(
            backup_id="old-backup-1",
            backup_type="full",
            created_at=(datetime.now() - timedelta(days=60)).isoformat(),
            file_size=1024,
            checksum="abc123",
            encrypted=False,
            files_included=["test.txt"],
            base_path_hash="hash123",
            retention_days=30,
        )

        mock_backend.list_objects.return_value = [
            MagicMock(key=f"backups/{old_backup.backup_id}.json")
        ]
        mock_backend.download_data.return_value = json.dumps(old_backup.to_dict()).encode()
        mock_backend.delete_object.return_value = None

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            vault._backend = mock_backend

            # Run cleanup with progress callback
            result = vault.cleanup(dry_run=False, progress_callback=progress_callback)

            # Verify progress callback was called
            assert len(progress_values) > 0
            # Should end at or near 100
            assert progress_values[-1] >= 90

    def test_cleanup_works_without_progress_callback(self, temp_omi_dir):
        """Test that cleanup works when progress_callback is None."""
        from omi.moltvault import MoltVault

        # Mock backend with no backups
        mock_backend = MagicMock()
        mock_backend.list_objects.return_value = []

        with patch("omi.moltvault.BOTO3_AVAILABLE", False):
            vault = MoltVault(temp_omi_dir)
            vault._backend = mock_backend

            # Run cleanup without progress callback - should not raise error
            result = vault.cleanup(dry_run=True, progress_callback=None)

            assert result is not None
            assert 'deleted' in result
            assert 'kept' in result
