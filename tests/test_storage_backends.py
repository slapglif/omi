"""
Tests for Storage Backends (S3, GCS, Azure)

Covers:
- Backend initialization and authentication
- Upload/download operations
- List/delete/exists operations
- Metadata handling
- Error handling
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
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


class TestStorageObject:
    """Test StorageObject dataclass."""

    def test_to_dict(self):
        """Test storage object serialization to dict."""
        from omi.storage_backends import StorageObject

        obj = StorageObject(
            key="backups/test.tar.gz",
            size=1024,
            last_modified=datetime(2025, 1, 1, 12, 0, 0),
            etag="abc123",
            metadata={"backup_id": "test_1", "encrypted": "true"},
        )

        d = obj.to_dict()

        assert d["key"] == "backups/test.tar.gz"
        assert d["size"] == 1024
        assert d["last_modified"] == "2025-01-01T12:00:00"
        assert d["etag"] == "abc123"
        assert d["metadata"]["backup_id"] == "test_1"

    def test_to_dict_no_metadata(self):
        """Test serialization when metadata is None."""
        from omi.storage_backends import StorageObject

        obj = StorageObject(
            key="test.txt",
            size=512,
            last_modified=datetime(2025, 1, 2, 0, 0, 0),
        )

        d = obj.to_dict()

        assert d["key"] == "test.txt"
        assert d["size"] == 512
        assert d["metadata"] == {}
        assert d["etag"] is None


class TestStorageBackend:
    """Test abstract StorageBackend base class."""

    def test_prefix_handling(self):
        """Test prefix normalization."""
        from omi.storage_backends import StorageBackend

        # Create a concrete implementation for testing
        class TestBackend(StorageBackend):
            def upload(self, local_path, key, metadata=None):
                pass
            def download(self, key, local_path):
                pass
            def list(self, prefix="", max_keys=None):
                pass
            def delete(self, key):
                pass
            def exists(self, key):
                pass
            def get_metadata(self, key):
                pass

        # Test with trailing slash
        backend = TestBackend("test-bucket", "backups/")
        assert backend.prefix == "backups/"

        # Test without trailing slash
        backend = TestBackend("test-bucket", "backups")
        assert backend.prefix == "backups/"

        # Test empty prefix
        backend = TestBackend("test-bucket", "")
        assert backend.prefix == ""

    def test_make_key(self):
        """Test key construction with prefix."""
        from omi.storage_backends import StorageBackend

        class TestBackend(StorageBackend):
            def upload(self, local_path, key, metadata=None):
                pass
            def download(self, key, local_path):
                pass
            def list(self, prefix="", max_keys=None):
                pass
            def delete(self, key):
                pass
            def exists(self, key):
                pass
            def get_metadata(self, key):
                pass

        backend = TestBackend("test-bucket", "backups")

        # Test normal key
        assert backend._make_key("file.txt") == "backups/file.txt"

        # Test key with leading slash
        assert backend._make_key("/file.txt") == "backups/file.txt"

        # Test nested key
        assert backend._make_key("2025/01/file.txt") == "backups/2025/01/file.txt"


class TestS3Backend:
    """Test S3Backend functionality."""

    @pytest.fixture
    def mock_boto3_available(self, monkeypatch):
        """Mock boto3 availability."""
        monkeypatch.setattr("omi.storage_backends.BOTO3_AVAILABLE", True, raising=False)

    @pytest.fixture
    def mock_s3_client(self):
        """Create mock S3 client."""
        client = MagicMock()
        return client

    def test_s3_import_error(self, monkeypatch):
        """Test S3Backend raises error when boto3 not available."""
        from omi.storage_backends import S3Backend

        monkeypatch.setattr("omi.storage_backends.BOTO3_AVAILABLE", False, raising=False)

        with pytest.raises(ImportError, match="boto3 package required"):
            S3Backend("test-bucket")

    def test_s3_initialization(self, mock_boto3_available):
        """Test S3Backend initialization."""
        from omi.storage_backends import S3Backend

        mock_client = MagicMock()
        with patch.object(S3Backend, "_create_client", return_value=mock_client):
            backend = S3Backend(
                bucket="test-bucket",
                prefix="backups",
                endpoint="https://example.r2.cloudflarestorage.com",
                access_key="test_key",
                secret_key="test_secret",
                region="auto",
            )

            assert backend.bucket == "test-bucket"
            assert backend.prefix == "backups/"
            assert backend.endpoint == "https://example.r2.cloudflarestorage.com"
            assert backend.region == "auto"
            assert backend._client == mock_client

    def test_s3_upload_success(self, mock_boto3_available, tmp_path):
        """Test successful file upload."""
        from omi.storage_backends import S3Backend

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        mock_client = MagicMock()
        with patch.object(S3Backend, "_create_client", return_value=mock_client):
            backend = S3Backend("test-bucket")
            result = backend.upload(
                test_file,
                "test.txt",
                metadata={"backup_id": "test_1"},
            )

            # Verify upload was called
            mock_client.upload_file.assert_called_once()
            assert result == "test.txt"

    def test_s3_upload_file_not_found(self, mock_boto3_available):
        """Test upload with non-existent file."""
        from omi.storage_backends import S3Backend

        mock_client = MagicMock()
        with patch.object(S3Backend, "_create_client", return_value=mock_client):
            backend = S3Backend("test-bucket")

            with pytest.raises(FileNotFoundError):
                backend.upload(Path("/nonexistent/file.txt"), "test.txt")

    def test_s3_download_success(self, mock_boto3_available, tmp_path):
        """Test successful file download."""
        from omi.storage_backends import S3Backend

        download_path = tmp_path / "downloaded.txt"

        mock_client = MagicMock()
        with patch.object(S3Backend, "_create_client", return_value=mock_client):
            backend = S3Backend("test-bucket")
            result = backend.download("test.txt", download_path)

            # Verify download was called
            mock_client.download_file.assert_called_once()
            assert result == download_path

    def test_s3_list_objects(self, mock_boto3_available):
        """Test listing objects."""
        from omi.storage_backends import S3Backend

        mock_client = MagicMock()
        # Mock list response
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "test1.txt",
                    "Size": 100,
                    "LastModified": datetime(2025, 1, 1, 12, 0, 0),
                    "ETag": '"abc123"',
                },
                {
                    "Key": "test2.txt",
                    "Size": 200,
                    "LastModified": datetime(2025, 1, 2, 12, 0, 0),
                    "ETag": '"def456"',
                },
            ]
        }

        with patch.object(S3Backend, "_create_client", return_value=mock_client):
            backend = S3Backend("test-bucket")
            objects = backend.list()

            assert len(objects) == 2
            assert objects[0].key == "test1.txt"
            assert objects[0].size == 100
            assert objects[1].key == "test2.txt"
            assert objects[1].size == 200


class TestGCSBackend:
    """Test GCS Backend functionality."""

    @pytest.fixture
    def mock_gcs_available(self, monkeypatch):
        """Mock GCS availability."""
        monkeypatch.setattr("omi.storage_backends.GCS_AVAILABLE", True, raising=False)

    def test_gcs_import_error(self, monkeypatch):
        """Test GCSBackend raises error when google-cloud-storage not available."""
        from omi.storage_backends import GCSBackend

        monkeypatch.setattr("omi.storage_backends.GCS_AVAILABLE", False, raising=False)

        with pytest.raises(ImportError, match="google-cloud-storage package required"):
            GCSBackend("test-bucket")

    def test_gcs_initialization_default_credentials(self, mock_gcs_available):
        """Test GCSBackend initialization with default credentials."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend(bucket="test-bucket", prefix="backups")

            assert backend.bucket == "test-bucket"
            assert backend.prefix == "backups/"
            assert backend._client == mock_client

    def test_gcs_initialization_with_credentials_file(self, mock_gcs_available):
        """Test GCSBackend initialization with explicit credentials file."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend(
                bucket="test-bucket",
                credentials_file="/path/to/creds.json",
                project="test-project",
            )

            assert backend.bucket == "test-bucket"
            assert backend.project == "test-project"
            assert backend.credentials_file == "/path/to/creds.json"

    def test_gcs_initialization_with_env_credentials(self, mock_gcs_available):
        """Test GCSBackend initialization with GOOGLE_APPLICATION_CREDENTIALS env var."""
        from omi.storage_backends import GCSBackend

        with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/env_creds.json"}):
            mock_client = MagicMock()
            with patch.object(GCSBackend, "_create_client", return_value=mock_client):
                backend = GCSBackend(bucket="test-bucket")

                # Verify credentials path was picked up from env
                assert backend.credentials_file == "/path/to/env_creds.json"

    def test_gcs_upload_success(self, mock_gcs_available, tmp_path):
        """Test successful file upload to GCS."""
        from omi.storage_backends import GCSBackend

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.upload(
                test_file,
                "test.txt",
                metadata={"backup_id": "test_1"},
            )

            # Verify blob was created and uploaded
            mock_bucket.blob.assert_called_once_with("test.txt")
            assert mock_blob.metadata == {"backup_id": "test_1"}
            mock_blob.upload_from_filename.assert_called_once_with(str(test_file))
            assert result == "test.txt"

    def test_gcs_upload_with_prefix(self, mock_gcs_available, tmp_path):
        """Test upload with prefix."""
        from omi.storage_backends import GCSBackend

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket", prefix="backups")
            result = backend.upload(test_file, "test.txt")

            # Verify full key with prefix was used
            mock_bucket.blob.assert_called_once_with("backups/test.txt")
            assert result == "backups/test.txt"

    def test_gcs_upload_file_not_found(self, mock_gcs_available):
        """Test upload with non-existent file."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")

            with pytest.raises(FileNotFoundError, match="Local file not found"):
                backend.upload(Path("/nonexistent/file.txt"), "test.txt")

    def test_gcs_download_success(self, mock_gcs_available, tmp_path):
        """Test successful file download from GCS."""
        from omi.storage_backends import GCSBackend

        download_path = tmp_path / "downloads" / "test.txt"

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.download("test.txt", download_path)

            # Verify blob was fetched and downloaded
            mock_bucket.blob.assert_called_once_with("test.txt")
            mock_blob.exists.assert_called_once()
            mock_blob.download_to_filename.assert_called_once_with(str(download_path))
            assert result == download_path

    def test_gcs_download_not_found(self, mock_gcs_available, tmp_path):
        """Test download when object doesn't exist."""
        from omi.storage_backends import GCSBackend

        download_path = tmp_path / "test.txt"

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")

            with pytest.raises(KeyError, match="Object not found"):
                backend.download("test.txt", download_path)

    def test_gcs_list_objects(self, mock_gcs_available):
        """Test listing objects in GCS."""
        from omi.storage_backends import GCSBackend

        # Create mock blobs
        mock_blob1 = MagicMock()
        mock_blob1.name = "test1.txt"
        mock_blob1.size = 100
        mock_blob1.updated = datetime(2025, 1, 1, 12, 0, 0)
        mock_blob1.etag = "abc123"
        mock_blob1.metadata = {"key": "value1"}

        mock_blob2 = MagicMock()
        mock_blob2.name = "test2.txt"
        mock_blob2.size = 200
        mock_blob2.updated = datetime(2025, 1, 2, 12, 0, 0)
        mock_blob2.etag = "def456"
        mock_blob2.metadata = None

        mock_client = MagicMock()
        mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            objects = backend.list()

            assert len(objects) == 2
            assert objects[0].key == "test1.txt"
            assert objects[0].size == 100
            assert objects[0].etag == "abc123"
            assert objects[0].metadata == {"key": "value1"}
            assert objects[1].key == "test2.txt"
            assert objects[1].size == 200

    def test_gcs_list_with_prefix(self, mock_gcs_available):
        """Test listing objects with prefix."""
        from omi.storage_backends import GCSBackend

        mock_blob = MagicMock()
        mock_blob.name = "backups/2025/test.txt"
        mock_blob.size = 100
        mock_blob.updated = datetime(2025, 1, 1, 12, 0, 0)
        mock_blob.etag = "abc123"
        mock_blob.metadata = None

        mock_client = MagicMock()
        mock_client.list_blobs.return_value = [mock_blob]

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket", prefix="backups")
            objects = backend.list(prefix="2025")

            # Verify list was called with combined prefix
            mock_client.list_blobs.assert_called_once_with(
                "test-bucket",
                prefix="backups/2025",
                max_results=None,
            )

            # Verify prefix was stripped from returned key
            assert len(objects) == 1
            assert objects[0].key == "2025/test.txt"

    def test_gcs_list_with_max_keys(self, mock_gcs_available):
        """Test listing objects with max_keys limit."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_client.list_blobs.return_value = []

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            backend.list(max_keys=10)

            # Verify max_results was passed
            mock_client.list_blobs.assert_called_once_with(
                "test-bucket",
                prefix="",
                max_results=10,
            )

    def test_gcs_delete_success(self, mock_gcs_available):
        """Test successful object deletion."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.delete("test.txt")

            # Verify blob was deleted
            mock_bucket.blob.assert_called_once_with("test.txt")
            mock_blob.exists.assert_called_once()
            mock_blob.delete.assert_called_once()
            assert result is True

    def test_gcs_delete_not_found(self, mock_gcs_available):
        """Test deleting non-existent object."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.delete("test.txt")

            # Verify delete was not called
            mock_blob.delete.assert_not_called()
            assert result is False

    def test_gcs_exists_true(self, mock_gcs_available):
        """Test exists returns True for existing object."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.exists("test.txt")

            assert result is True
            mock_bucket.blob.assert_called_once_with("test.txt")
            mock_blob.exists.assert_called_once()

    def test_gcs_exists_false(self, mock_gcs_available):
        """Test exists returns False for non-existent object."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.exists("test.txt")

            assert result is False

    def test_gcs_get_metadata_success(self, mock_gcs_available):
        """Test getting metadata for an object."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True
        mock_blob.size = 1024
        mock_blob.updated = datetime(2025, 1, 1, 12, 0, 0)
        mock_blob.etag = "abc123"
        mock_blob.metadata = {"backup_id": "test_1"}

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.get_metadata("test.txt")

            # Verify metadata was fetched
            mock_bucket.blob.assert_called_once_with("test.txt")
            mock_blob.exists.assert_called_once()
            mock_blob.reload.assert_called_once()

            assert result is not None
            assert result.key == "test.txt"
            assert result.size == 1024
            assert result.etag == "abc123"
            assert result.metadata == {"backup_id": "test_1"}

    def test_gcs_get_metadata_not_found(self, mock_gcs_available):
        """Test getting metadata for non-existent object."""
        from omi.storage_backends import GCSBackend

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False

        with patch.object(GCSBackend, "_create_client", return_value=mock_client):
            backend = GCSBackend("test-bucket")
            result = backend.get_metadata("test.txt")

            # Should return None for non-existent object
            assert result is None
            mock_blob.reload.assert_not_called()
