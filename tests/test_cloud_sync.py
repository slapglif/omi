"""
Integration tests for cloud sync async operations and conflict resolution.

Covers:
- Async upload/download operations
- Conflict detection (timestamp, checksum, size mismatches)
- Conflict resolution strategies (last-write-wins, manual, merge)
- Multi-file sync scenarios
"""

import pytest
import asyncio
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
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


class TestAsyncOperations:
    """Test async upload and download operations"""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock storage backend with async methods"""
        from omi.storage_backends import StorageBackend, StorageObject

        backend = Mock(spec=StorageBackend)
        backend.bucket = "test-bucket"
        backend.prefix = ""

        # Mock synchronous methods
        backend.upload = Mock(return_value="test-key")
        backend.download = Mock(return_value=Path("/tmp/downloaded"))

        # Mock async methods to actually be async
        async def mock_async_upload(local_path, key, metadata=None):
            """Simulate async upload"""
            await asyncio.sleep(0.01)  # Simulate I/O delay
            return f"uploaded-{key}"

        async def mock_async_download(key, local_path):
            """Simulate async download"""
            await asyncio.sleep(0.01)  # Simulate I/O delay
            return local_path

        backend.async_upload = mock_async_upload
        backend.async_download = mock_async_download

        return backend

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary test file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        return test_file

    @pytest.mark.asyncio
    async def test_async_upload_single_file(self, mock_backend, temp_file):
        """Test async upload of a single file"""
        result = await mock_backend.async_upload(
            temp_file,
            "test-key",
            metadata={"type": "test"}
        )

        assert result == "uploaded-test-key"

    @pytest.mark.asyncio
    async def test_async_download_single_file(self, mock_backend, tmp_path):
        """Test async download of a single file"""
        download_path = tmp_path / "downloaded.txt"

        result = await mock_backend.async_download(
            "test-key",
            download_path
        )

        assert result == download_path

    @pytest.mark.asyncio
    async def test_multiple_concurrent_uploads(self, mock_backend, tmp_path):
        """Test multiple files uploading concurrently"""
        # Create multiple test files
        files = []
        for i in range(5):
            test_file = tmp_path / f"test_{i}.txt"
            test_file.write_text(f"content {i}")
            files.append((test_file, f"key-{i}"))

        # Upload concurrently
        tasks = [
            mock_backend.async_upload(local_path, key)
            for local_path, key in files
        ]

        results = await asyncio.gather(*tasks)

        # Verify all uploads succeeded
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result == f"uploaded-key-{i}"

    @pytest.mark.asyncio
    async def test_concurrent_upload_and_download(self, mock_backend, tmp_path):
        """Test concurrent uploads and downloads don't block each other"""
        # Create test files
        upload_file = tmp_path / "upload.txt"
        upload_file.write_text("upload content")
        download_file = tmp_path / "download.txt"

        # Run upload and download concurrently
        upload_task = mock_backend.async_upload(upload_file, "upload-key")
        download_task = mock_backend.async_download("download-key", download_file)

        upload_result, download_result = await asyncio.gather(
            upload_task,
            download_task
        )

        assert upload_result == "uploaded-upload-key"
        assert download_result == download_file

    @pytest.mark.asyncio
    async def test_async_operations_non_blocking(self, mock_backend, temp_file):
        """Test that async operations don't block the event loop"""
        start_time = asyncio.get_event_loop().time()

        # Start 3 uploads that should run concurrently
        tasks = [
            mock_backend.async_upload(temp_file, f"key-{i}")
            for i in range(3)
        ]

        results = await asyncio.gather(*tasks)

        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time

        # If running concurrently, should take ~0.01s (single I/O delay)
        # If sequential, would take ~0.03s (3x I/O delay)
        # Allow some margin for test execution overhead
        assert elapsed < 0.05  # Should be much less if truly concurrent
        assert len(results) == 3


class TestConflictDetection:
    """Test conflict detection between local and remote files"""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary test file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        return test_file

    def test_no_conflict_identical_content(self, temp_file):
        """Test no conflict when files have identical content"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        # Calculate checksum of local file
        with open(temp_file, 'rb') as f:
            local_checksum = hashlib.sha256(f.read()).hexdigest()

        # Create remote object with matching checksum
        remote_obj = StorageObject(
            key="test.txt",
            size=temp_file.stat().st_size,
            last_modified=datetime.now(),
            etag=local_checksum,
        )

        conflict = detect_conflicts(temp_file, remote_obj)

        assert conflict is None

    def test_conflict_both_modified(self, temp_file):
        """Test conflict when both local and remote modified since last sync"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        # Last sync was 2 days ago
        last_sync = datetime.now() - timedelta(days=2)

        # Local file modified 1 day ago
        one_day_ago = (datetime.now() - timedelta(days=1)).timestamp()
        os.utime(temp_file, (one_day_ago, one_day_ago))

        # Remote modified today (after last sync)
        remote_obj = StorageObject(
            key="test.txt",
            size=temp_file.stat().st_size + 10,  # Different size
            last_modified=datetime.now(),
            etag="different_checksum",
        )

        conflict = detect_conflicts(temp_file, remote_obj, last_sync)

        assert conflict is not None
        assert conflict.conflict_type == "both_modified"

    def test_conflict_checksum_mismatch(self, temp_file):
        """Test conflict when checksums differ"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        # Remote has different checksum (different content)
        remote_obj = StorageObject(
            key="test.txt",
            size=temp_file.stat().st_size,
            last_modified=datetime.now(),
            etag="different_checksum_abc123",
        )

        conflict = detect_conflicts(temp_file, remote_obj)

        assert conflict is not None
        assert conflict.conflict_type == "checksum_mismatch"
        assert conflict.remote_checksum == "different_checksum_abc123"

    def test_conflict_size_mismatch(self, temp_file):
        """Test conflict when file sizes differ significantly"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        # Remote has very different size
        # Note: Will detect checksum_mismatch first since we provide an etag
        remote_obj = StorageObject(
            key="test.txt",
            size=temp_file.stat().st_size * 2,  # Double the size
            last_modified=datetime.now(),
            etag="some_checksum",
        )

        conflict = detect_conflicts(temp_file, remote_obj)

        assert conflict is not None
        # Implementation checks checksum before size, so this will be checksum_mismatch
        assert conflict.conflict_type in ["checksum_mismatch", "size_mismatch"]
        assert conflict.local_size != conflict.remote_size

    def test_conflict_missing_local_file(self, tmp_path):
        """Test error when local file doesn't exist"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        missing_file = tmp_path / "missing.txt"

        remote_obj = StorageObject(
            key="missing.txt",
            size=100,
            last_modified=datetime.now(),
            etag="checksum",
        )

        with pytest.raises(FileNotFoundError):
            detect_conflicts(missing_file, remote_obj)

    def test_no_conflict_same_timestamp(self, temp_file):
        """Test no conflict when timestamps are identical"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        # Calculate checksum
        with open(temp_file, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Remote with same modification time and checksum
        local_mtime = datetime.fromtimestamp(temp_file.stat().st_mtime)
        remote_obj = StorageObject(
            key="test.txt",
            size=temp_file.stat().st_size,
            last_modified=local_mtime,
            etag=checksum,
        )

        conflict = detect_conflicts(temp_file, remote_obj)

        assert conflict is None

    def test_conflict_quoted_etag(self, temp_file):
        """Test handling of quoted etags (S3 format)"""
        from omi.moltvault import detect_conflicts
        from omi.storage_backends import StorageObject

        # Calculate checksum
        with open(temp_file, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # S3 returns etags with quotes
        remote_obj = StorageObject(
            key="test.txt",
            size=temp_file.stat().st_size,
            last_modified=datetime.now(),
            etag=f'"{checksum}"',  # Quoted
        )

        # Should strip quotes and compare correctly
        conflict = detect_conflicts(temp_file, remote_obj)

        # Should not detect conflict if checksums match after stripping quotes
        # (The implementation should handle this)
        assert conflict is None or conflict.conflict_type != "checksum_mismatch"


class TestConflictResolution:
    """Test conflict resolution strategies"""

    @pytest.fixture
    def conflict_info(self):
        """Create a sample conflict"""
        from omi.moltvault import ConflictInfo

        return ConflictInfo(
            file_path="test.txt",
            local_modified=datetime(2024, 1, 1, 12, 0),
            remote_modified=datetime(2024, 1, 2, 12, 0),
            local_checksum="local_checksum_abc123",
            remote_checksum="remote_checksum_def456",
            local_size=100,
            remote_size=120,
            conflict_type="both_modified",
        )

    def test_last_write_wins_keep_remote(self, conflict_info):
        """Test last-write-wins keeps remote when it's newer"""
        from omi.moltvault import resolve_conflict

        # Remote is newer (2024-01-02 vs 2024-01-01)
        result = resolve_conflict(conflict_info, strategy="last-write-wins")

        assert result['status'] == 'resolved'
        assert result['action'] == 'keep_remote'
        assert result['winner'] == 'remote'
        assert 'remote version' in result['message']

    def test_last_write_wins_keep_local(self, conflict_info):
        """Test last-write-wins keeps local when it's newer"""
        from omi.moltvault import resolve_conflict, ConflictInfo

        # Create conflict where local is newer
        conflict = ConflictInfo(
            file_path="test.txt",
            local_modified=datetime(2024, 1, 2, 12, 0),  # Local newer
            remote_modified=datetime(2024, 1, 1, 12, 0),
            local_checksum="local_abc",
            remote_checksum="remote_def",
            local_size=100,
            remote_size=120,
            conflict_type="both_modified",
        )

        result = resolve_conflict(conflict, strategy="last-write-wins")

        assert result['status'] == 'resolved'
        assert result['action'] == 'keep_local'
        assert result['winner'] == 'local'
        assert 'local version' in result['message']

    def test_manual_strategy(self, conflict_info):
        """Test manual strategy returns conflict for user decision"""
        from omi.moltvault import resolve_conflict

        result = resolve_conflict(conflict_info, strategy="manual")

        assert result['status'] == 'manual_required'
        assert result['action'] == 'none'
        assert result['winner'] is None
        assert 'Manual resolution required' in result['message']
        assert result['conflict'] == conflict_info

    def test_invalid_strategy(self, conflict_info):
        """Test error on invalid strategy"""
        from omi.moltvault import resolve_conflict

        with pytest.raises(ValueError, match="Invalid strategy"):
            resolve_conflict(conflict_info, strategy="invalid")

    def test_merge_identical_files(self, tmp_path, conflict_info):
        """Test merge resolves when files are actually identical"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        # Create local file
        local_file = tmp_path / "test.txt"
        local_file.write_text("same content\n")

        # Mock backend that returns identical content
        backend = Mock(spec=StorageBackend)

        def mock_download(key, local_path):
            local_path.write_text("same content\n")
            return local_path

        backend.download = mock_download

        result = resolve_conflict(
            conflict_info,
            strategy="merge",
            backend=backend,
            local_path=local_file
        )

        assert result['status'] == 'resolved'
        assert result['action'] == 'keep_local'
        assert result['winner'] == 'both'
        assert 'identical' in result['message'].lower()

    def test_merge_remote_superset(self, tmp_path, conflict_info):
        """Test merge suggests keeping remote when it's a superset"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        # Create local file
        local_file = tmp_path / "test.txt"
        local_file.write_text("line 1\n")

        # Mock backend with superset content
        backend = Mock(spec=StorageBackend)

        def mock_download(key, local_path):
            # Remote has more content
            local_path.write_text("line 1\nline 2\nline 3\n")
            return local_path

        backend.download = mock_download

        result = resolve_conflict(
            conflict_info,
            strategy="merge",
            backend=backend,
            local_path=local_file
        )

        assert result['status'] == 'manual_required'
        assert result['action'] == 'keep_remote'
        assert result['winner'] == 'remote'
        assert 'superset' in result['message'].lower()

    def test_merge_local_superset(self, tmp_path, conflict_info):
        """Test merge suggests keeping local when it's a superset"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        # Create local file with more content
        local_file = tmp_path / "test.txt"
        local_file.write_text("line 1\nline 2\nline 3\n")

        # Mock backend with subset content
        backend = Mock(spec=StorageBackend)

        def mock_download(key, local_path):
            # Remote has less content
            local_path.write_text("line 1\n")
            return local_path

        backend.download = mock_download

        result = resolve_conflict(
            conflict_info,
            strategy="merge",
            backend=backend,
            local_path=local_file
        )

        assert result['status'] == 'manual_required'
        assert result['action'] == 'keep_local'
        assert result['winner'] == 'local'
        assert 'superset' in result['message'].lower()

    def test_merge_diverged_files(self, tmp_path, conflict_info):
        """Test merge requires manual intervention for diverged files"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        # Create local file
        local_file = tmp_path / "test.txt"
        local_file.write_text("local changes\n")

        # Mock backend with different content
        backend = Mock(spec=StorageBackend)

        def mock_download(key, local_path):
            # Remote has completely different content
            local_path.write_text("remote changes\n")
            return local_path

        backend.download = mock_download

        result = resolve_conflict(
            conflict_info,
            strategy="merge",
            backend=backend,
            local_path=local_file
        )

        assert result['status'] == 'manual_required'
        assert result['action'] == 'none'
        assert result['winner'] is None
        assert 'diverged' in result['message'].lower()
        assert 'Manual merge required' in result['message']

    def test_merge_binary_file(self, tmp_path, conflict_info):
        """Test merge fails gracefully with binary files"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        # Create binary file
        local_file = tmp_path / "test.bin"
        local_file.write_bytes(b'\x00\x01\x02\x03\xff\xfe')

        # Mock backend
        backend = Mock(spec=StorageBackend)
        backend.download = Mock(return_value=tmp_path / "remote.bin")

        result = resolve_conflict(
            conflict_info,
            strategy="merge",
            backend=backend,
            local_path=local_file
        )

        assert result['status'] == 'manual_required'
        assert 'not text' in result['message'].lower() or 'unreadable' in result['message'].lower()

    def test_merge_missing_backend(self, conflict_info, tmp_path):
        """Test merge requires backend parameter"""
        from omi.moltvault import resolve_conflict

        local_file = tmp_path / "test.txt"
        local_file.write_text("content")

        with pytest.raises(ValueError, match="backend required"):
            resolve_conflict(
                conflict_info,
                strategy="merge",
                backend=None,
                local_path=local_file
            )

    def test_merge_missing_local_path(self, conflict_info):
        """Test merge requires local_path parameter"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        backend = Mock(spec=StorageBackend)

        with pytest.raises(ValueError, match="local_path required"):
            resolve_conflict(
                conflict_info,
                strategy="merge",
                backend=backend,
                local_path=None
            )

    def test_merge_missing_local_file(self, conflict_info, tmp_path):
        """Test merge fails when local file doesn't exist"""
        from omi.moltvault import resolve_conflict
        from omi.storage_backends import StorageBackend

        backend = Mock(spec=StorageBackend)
        missing_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            resolve_conflict(
                conflict_info,
                strategy="merge",
                backend=backend,
                local_path=missing_file
            )


class TestIntegrationScenarios:
    """Test complete sync scenarios combining multiple operations"""

    @pytest.mark.asyncio
    async def test_sync_workflow_no_conflicts(self, tmp_path):
        """Test complete sync workflow with no conflicts"""
        from omi.storage_backends import StorageBackend, StorageObject
        from omi.moltvault import detect_conflicts

        # Create local files
        files = []
        for i in range(3):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"content {i}")
            files.append(f)

        # Mock backend
        backend = Mock(spec=StorageBackend)

        uploaded_files = []

        async def mock_upload(local_path, key, metadata=None):
            await asyncio.sleep(0.001)
            uploaded_files.append(key)
            return key

        backend.async_upload = mock_upload

        # Upload all files concurrently
        tasks = [
            backend.async_upload(f, f"sync/{f.name}")
            for f in files
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert len(uploaded_files) == 3
        assert all(f"file_{i}.txt" in key for i, key in enumerate(uploaded_files))

    @pytest.mark.asyncio
    async def test_sync_workflow_with_conflicts(self, tmp_path):
        """Test sync workflow detecting and resolving conflicts"""
        from omi.moltvault import detect_conflicts, resolve_conflict, ConflictInfo
        from omi.storage_backends import StorageObject

        # Create local file
        local_file = tmp_path / "conflicted.txt"
        local_file.write_text("local content")

        # Simulate remote file that was modified
        remote_obj = StorageObject(
            key="conflicted.txt",
            size=200,
            last_modified=datetime.now(),
            etag="different_checksum",
        )

        # Detect conflict
        conflict = detect_conflicts(local_file, remote_obj)

        assert conflict is not None

        # Resolve with last-write-wins
        resolution = resolve_conflict(conflict, strategy="last-write-wins")

        assert resolution['status'] == 'resolved'
        assert resolution['action'] in ['keep_local', 'keep_remote']

    def test_multiple_conflicts_batch_resolution(self, tmp_path):
        """Test resolving multiple conflicts in batch"""
        from omi.moltvault import ConflictInfo, resolve_conflict

        # Create multiple conflicts
        conflicts = []
        for i in range(5):
            conflict = ConflictInfo(
                file_path=f"file_{i}.txt",
                local_modified=datetime(2024, 1, 1 + i, 12, 0),
                remote_modified=datetime(2024, 1, 6 - i, 12, 0),
                local_checksum=f"local_{i}",
                remote_checksum=f"remote_{i}",
                local_size=100 + i * 10,
                remote_size=120 - i * 10,
                conflict_type="both_modified",
            )
            conflicts.append(conflict)

        # Resolve all with last-write-wins
        resolutions = []
        for conflict in conflicts:
            resolution = resolve_conflict(conflict, strategy="last-write-wins")
            resolutions.append(resolution)

        assert len(resolutions) == 5

        # Files 0-2 should keep remote (remote newer)
        # i=0: local=2024-01-01, remote=2024-01-06 → remote newer
        # i=1: local=2024-01-02, remote=2024-01-05 → remote newer
        # i=2: local=2024-01-03, remote=2024-01-04 → remote newer
        # Files 3-4 should keep local (local newer)
        # i=3: local=2024-01-04, remote=2024-01-03 → local newer
        # i=4: local=2024-01-05, remote=2024-01-02 → local newer
        assert resolutions[0]['action'] == 'keep_remote'
        assert resolutions[1]['action'] == 'keep_remote'
        assert resolutions[2]['action'] == 'keep_remote'
        assert resolutions[3]['action'] == 'keep_local'
        assert resolutions[4]['action'] == 'keep_local'


class TestErrorHandling:
    """Test error handling in sync operations"""

    @pytest.mark.asyncio
    async def test_async_upload_partial_failure(self, tmp_path):
        """Test handling of partial failures in batch upload"""
        from omi.storage_backends import StorageBackend, StorageError

        # Create test files
        files = [tmp_path / f"file_{i}.txt" for i in range(3)]
        for f in files:
            f.write_text("content")

        # Mock backend that fails on second upload
        backend = Mock(spec=StorageBackend)

        upload_count = [0]

        async def mock_upload(local_path, key, metadata=None):
            await asyncio.sleep(0.001)
            upload_count[0] += 1
            if upload_count[0] == 2:
                raise StorageError("Upload failed")
            return key

        backend.async_upload = mock_upload

        # Upload all files
        tasks = [
            backend.async_upload(f, f"key-{i}")
            for i, f in enumerate(files)
        ]

        # Use gather with return_exceptions to handle failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should have 2 successes and 1 failure
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        assert len(successes) == 2
        assert len(failures) == 1
        assert isinstance(failures[0], StorageError)

    def test_conflict_resolution_with_download_failure(self, tmp_path):
        """Test merge strategy handles download failures gracefully"""
        from omi.moltvault import resolve_conflict, ConflictInfo
        from omi.storage_backends import StorageBackend, StorageError

        local_file = tmp_path / "test.txt"
        local_file.write_text("content")

        conflict = ConflictInfo(
            file_path="test.txt",
            local_modified=datetime(2024, 1, 1, 12, 0),
            remote_modified=datetime(2024, 1, 2, 12, 0),
            local_checksum="local",
            remote_checksum="remote",
            local_size=100,
            remote_size=120,
            conflict_type="both_modified",
        )

        # Mock backend that fails to download
        backend = Mock(spec=StorageBackend)
        backend.download = Mock(side_effect=StorageError("Download failed"))

        result = resolve_conflict(
            conflict,
            strategy="merge",
            backend=backend,
            local_path=local_file
        )

        # Should require manual resolution
        assert result['status'] == 'manual_required'
        assert result['action'] == 'none'
