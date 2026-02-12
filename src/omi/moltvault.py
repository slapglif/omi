"""
MoltVault - Backup/Restore module for OMI

Cloudflare R2/S3 compatible backup system with encryption.
The palace remembers what the river forgets.
"""

import os
import tarfile
import hashlib
import json
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, BinaryIO, Iterator
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Encryption imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# S3/R2 imports
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Storage backend imports
from .storage_backends import (
    S3Backend,
    GCSBackend,
    AzureBackend,
    StorageBackend,
    StorageObject,
    StorageError,
    StorageAuthError
)


def create_backend_from_config(config: Dict[str, Any]) -> StorageBackend:
    """
    Create a storage backend from configuration.

    Args:
        config: Configuration dictionary with structure:
            {
                'backup': {
                    'backend': 's3' | 'gcs' | 'azure',
                    'bucket': 'bucket-name',
                    'prefix': 'optional/prefix/',
                    # S3-specific options:
                    'endpoint': 'https://...', # for R2, MinIO, etc.
                    'access_key': 'key',
                    'secret_key': 'secret',
                    'region': 'us-east-1',
                    # GCS-specific options:
                    'credentials_file': '/path/to/creds.json',
                    'project': 'my-project',
                    # Azure-specific options:
                    'connection_string': 'DefaultEndpointsProtocol=...',
                    'account_name': 'myaccount',
                    'account_key': 'key',
                    'sas_token': 'token',
                }
            }

    Returns:
        StorageBackend instance (S3Backend, GCSBackend, or AzureBackend)

    Raises:
        ValueError: If backend type is missing or unsupported
        ValueError: If required configuration is missing
        ImportError: If required package is not installed

    Examples:
        >>> config = {'backup': {'backend': 's3', 'bucket': 'my-bucket'}}
        >>> backend = create_backend_from_config(config)
        >>> isinstance(backend, S3Backend)
        True
    """
    # Extract backup config
    if 'backup' not in config:
        raise ValueError(
            "Missing 'backup' section in configuration. "
            "Run 'omi config set backup.backend s3' to configure."
        )

    backup_config = config['backup']

    # Get backend type
    backend_type = backup_config.get('backend')
    if not backend_type:
        raise ValueError(
            "Missing 'backend' in backup configuration. "
            "Run 'omi config set backup.backend s3' to set backend type."
        )

    # Get bucket name
    bucket = backup_config.get('bucket')
    if not bucket:
        raise ValueError(
            "Missing 'bucket' in backup configuration. "
            "Run 'omi config set backup.bucket my-bucket' to set bucket name."
        )

    # Get optional prefix
    prefix = backup_config.get('prefix', '')

    # Create backend based on type
    if backend_type == 's3':
        return S3Backend(
            bucket=bucket,
            prefix=prefix,
            endpoint=backup_config.get('endpoint'),
            access_key=backup_config.get('access_key'),
            secret_key=backup_config.get('secret_key'),
            region=backup_config.get('region', 'auto'),
        )

    elif backend_type == 'gcs':
        return GCSBackend(
            bucket=bucket,
            prefix=prefix,
            credentials_file=backup_config.get('credentials_file'),
            project=backup_config.get('project'),
        )

    elif backend_type == 'azure':
        return AzureBackend(
            bucket=bucket,
            prefix=prefix,
            connection_string=backup_config.get('connection_string'),
            account_name=backup_config.get('account_name'),
            account_key=backup_config.get('account_key'),
            sas_token=backup_config.get('sas_token'),
        )

    else:
        raise ValueError(
            f"Unsupported backend type: '{backend_type}'. "
            f"Supported types: 's3', 'gcs', 'azure'"
        )


class _MockS3BackendWrapper(StorageBackend):
    """
    Wrapper for mocked S3 client to provide StorageBackend interface
    Used for backward compatibility with tests that mock _s3_client
    """

    def __init__(self, s3_client: Any, bucket: str) -> None:
        super().__init__(bucket, prefix="")
        self._client: Any = s3_client

    def upload(self, local_path: Path, key: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Upload file using raw S3 client"""
        full_key = self._make_key(key)
        # For JSON files, use put_object to match old behavior
        if str(local_path).endswith('.json'):
            with open(local_path, 'r') as f:
                self._client.put_object(
                    Bucket=self.bucket,
                    Key=full_key,
                    Body=f.read(),
                    ContentType="application/json",
                )
        else:
            # For other files, use upload_fileobj
            with open(local_path, "rb") as f:
                extra_args = {}
                if metadata:
                    extra_args["Metadata"] = metadata
                self._client.upload_fileobj(
                    f, self.bucket, full_key,
                    ExtraArgs=extra_args if extra_args else None
                )
        return full_key

    def download(self, key: str, local_path: Path) -> Path:
        """Download file using raw S3 client"""
        full_key = self._make_key(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # For JSON files, use get_object (for test compatibility)
        # For other files, use download_file
        if str(key).endswith('.json'):
            response = self._client.get_object(Bucket=self.bucket, Key=full_key)
            content = response["Body"].read()
            local_path.write_bytes(content)
        else:
            self._client.download_file(self.bucket, full_key, str(local_path))
        return local_path

    def list(self, prefix: str = "", max_keys: Optional[int] = None) -> List[StorageObject]:
        """List objects using raw S3 client"""
        from .storage_backends import StorageObject
        full_prefix = self._make_key(prefix)
        response = self._client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=full_prefix,
            **({"MaxKeys": max_keys} if max_keys else {})
        )
        objects: List[StorageObject] = []
        for item in response.get("Contents", []):
            key = item.get("Key", "")
            if self.prefix and key.startswith(self.prefix):
                key = key[len(self.prefix):]
            objects.append(StorageObject(
                key=key,
                size=item.get("Size", 0),
                last_modified=item.get("LastModified"),
                etag=item.get("ETag", "").strip('"'),
            ))
        return objects

    def delete(self, key: str) -> bool:
        """Delete object using raw S3 client"""
        full_key = self._make_key(key)
        self._client.delete_object(Bucket=self.bucket, Key=full_key)
        return True

    def exists(self, key: str) -> bool:
        """Check if object exists using raw S3 client"""
        full_key = self._make_key(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except:
            return False

    def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata using raw S3 client"""
        from .storage_backends import StorageObject
        full_key = self._make_key(key)
        try:
            response = self._client.head_object(Bucket=self.bucket, Key=full_key)
            return StorageObject(
                key=key,
                size=response["ContentLength"],
                last_modified=response["LastModified"],
                etag=response.get("ETag", "").strip('"'),
                metadata=response.get("Metadata"),
            )
        except:
            return None

    async def async_upload(self, local_path: Path, key: str, metadata: Optional[Dict[str, str]] = None) -> str:
        """Async upload - delegates to sync implementation for mock"""
        return self.upload(local_path, key, metadata)

    async def async_download(self, key: str, local_path: Path) -> Path:
        """Async download - delegates to sync implementation for mock"""
        return self.download(key, local_path)


@dataclass
class BackupMetadata:
    """Metadata for a stored backup"""
    backup_id: str
    backup_type: str  # 'full' or 'incremental'
    created_at: str  # ISO format timestamp
    file_size: int  # size in bytes
    checksum: str  # SHA-256 checksum
    encrypted: bool
    files_included: List[str]
    base_path_hash: str  # Hash of base path for verification
    retention_days: int  # Days to keep this backup

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        return cls(**data)


@dataclass
class SyncSnapshotMetadata:
    """Metadata for a sync snapshot used in distributed synchronization"""
    snapshot_id: str  # Unique identifier for this snapshot
    instance_id: str  # ID of the instance that created this snapshot
    created_at: str  # ISO format timestamp
    file_size: int  # Size in bytes
    checksum: str  # SHA-256 checksum
    encrypted: bool  # Whether snapshot is encrypted
    files_included: List[str]  # List of files in snapshot
    vector_clock: Dict[str, int]  # Vector clock at snapshot time
    sync_metadata: Dict[str, Any]  # Additional sync-specific metadata
    retention_days: int  # Days to keep this snapshot (default: 7)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncSnapshotMetadata":
        return cls(**data)


@dataclass
class ConflictInfo:
    """Information about a file conflict between local and remote versions"""
    file_path: str  # Path to the conflicting file
    local_modified: datetime  # Local file modification time
    remote_modified: datetime  # Remote file modification time
    local_checksum: Optional[str]  # Local file checksum (SHA-256)
    remote_checksum: Optional[str]  # Remote file etag/checksum
    local_size: int  # Local file size in bytes
    remote_size: int  # Remote file size in bytes
    conflict_type: str  # 'both_modified', 'checksum_mismatch', 'size_mismatch'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "file_path": self.file_path,
            "local_modified": self.local_modified.isoformat(),
            "remote_modified": self.remote_modified.isoformat(),
            "local_checksum": self.local_checksum,
            "remote_checksum": self.remote_checksum,
            "local_size": self.local_size,
            "remote_size": self.remote_size,
            "conflict_type": self.conflict_type,
        }


def detect_conflicts(
    local_path: Path,
    remote_obj: StorageObject,
    last_sync_time: Optional[datetime] = None,
) -> Optional[ConflictInfo]:
    """
    Detect conflicts between local and remote file versions using metadata.

    A conflict is detected when:
    1. Both local and remote files have been modified since last sync
    2. Checksums differ (indicating different content)
    3. File sizes differ significantly

    Args:
        local_path: Path to local file
        remote_obj: StorageObject with remote file metadata
        last_sync_time: Optional timestamp of last successful sync

    Returns:
        ConflictInfo if conflict detected, None otherwise

    Raises:
        FileNotFoundError: If local_path doesn't exist

    Examples:
        >>> from pathlib import Path
        >>> from datetime import datetime
        >>> from src.omi.storage_backends import StorageObject
        >>> local = Path("test.txt")
        >>> remote = StorageObject(
        ...     key="test.txt",
        ...     size=100,
        ...     last_modified=datetime.now(),
        ...     etag="abc123"
        ... )
        >>> conflict = detect_conflicts(local, remote)
        >>> conflict is None or isinstance(conflict, ConflictInfo)
        True
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Get local file metadata
    local_stat = local_path.stat()
    local_modified = datetime.fromtimestamp(local_stat.st_mtime)
    local_size = local_stat.st_size

    # Calculate local file checksum
    local_checksum = None
    try:
        with open(local_path, 'rb') as f:
            local_checksum = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        # If we can't read the file, we can't calculate checksum
        pass

    # Extract remote metadata
    remote_modified = remote_obj.last_modified
    remote_size = remote_obj.size
    remote_checksum = remote_obj.etag

    # Check for conflicts
    conflict_type = None

    # Case 1: Both modified since last sync (if last_sync_time provided)
    if last_sync_time:
        local_modified_since_sync = local_modified > last_sync_time
        remote_modified_since_sync = remote_modified > last_sync_time

        if local_modified_since_sync and remote_modified_since_sync:
            # Both files changed - check if they're actually different
            if local_checksum and remote_checksum:
                # Remove quotes from etag if present (S3 etags are quoted)
                remote_etag_clean = remote_checksum.strip('"')
                if local_checksum != remote_etag_clean:
                    conflict_type = "both_modified"
            elif local_size != remote_size:
                conflict_type = "both_modified"

    # Case 2: Checksums differ (even without last_sync_time)
    if not conflict_type and local_checksum and remote_checksum:
        remote_etag_clean = remote_checksum.strip('"')
        if local_checksum != remote_etag_clean:
            conflict_type = "checksum_mismatch"

    # Case 3: Size mismatch (strong indicator of different content)
    if not conflict_type and local_size != remote_size:
        conflict_type = "size_mismatch"

    # Return conflict info if conflict detected
    if conflict_type:
        return ConflictInfo(
            file_path=str(local_path),
            local_modified=local_modified,
            remote_modified=remote_modified,
            local_checksum=local_checksum,
            remote_checksum=remote_checksum,
            local_size=local_size,
            remote_size=remote_size,
            conflict_type=conflict_type,
        )

    return None


def resolve_conflict(
    conflict: ConflictInfo,
    strategy: str = "last-write-wins",
    backend: Optional[StorageBackend] = None,
    local_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Resolve a file conflict using the specified strategy.

    Strategies:
    - 'last-write-wins': Keep the most recently modified version
    - 'manual': Return conflict details for manual resolution (no automatic action)
    - 'merge': Attempt to merge both versions (for text files only)

    Args:
        conflict: ConflictInfo object describing the conflict
        strategy: Resolution strategy to use ('last-write-wins', 'manual', 'merge')
        backend: Optional StorageBackend for downloading remote version
        local_path: Optional Path to local file (required for merge strategy)

    Returns:
        Dict with resolution result:
        {
            'status': 'resolved' | 'manual_required',
            'action': 'keep_local' | 'keep_remote' | 'merged' | 'none',
            'winner': 'local' | 'remote' | 'both' | None,
            'message': str,
            'conflict': ConflictInfo (for manual strategy)
        }

    Raises:
        ValueError: If strategy is invalid or required parameters are missing

    Examples:
        >>> from datetime import datetime
        >>> conflict = ConflictInfo(
        ...     file_path="test.txt",
        ...     local_modified=datetime(2024, 1, 1, 12, 0),
        ...     remote_modified=datetime(2024, 1, 2, 12, 0),
        ...     local_checksum="abc123",
        ...     remote_checksum="def456",
        ...     local_size=100,
        ...     remote_size=120,
        ...     conflict_type="both_modified"
        ... )
        >>> result = resolve_conflict(conflict, strategy="last-write-wins")
        >>> result['action']
        'keep_remote'
    """
    # Validate strategy
    valid_strategies = ['last-write-wins', 'manual', 'merge']
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Valid options: {', '.join(valid_strategies)}"
        )

    # Manual strategy - return conflict for user decision
    if strategy == 'manual':
        return {
            'status': 'manual_required',
            'action': 'none',
            'winner': None,
            'message': (
                f"Conflict detected in {conflict.file_path}. "
                f"Local modified: {conflict.local_modified.isoformat()}, "
                f"Remote modified: {conflict.remote_modified.isoformat()}. "
                f"Manual resolution required."
            ),
            'conflict': conflict,
        }

    # Last-write-wins strategy - keep the newest version
    if strategy == 'last-write-wins':
        if conflict.remote_modified > conflict.local_modified:
            return {
                'status': 'resolved',
                'action': 'keep_remote',
                'winner': 'remote',
                'message': (
                    f"Resolved {conflict.file_path}: keeping remote version "
                    f"(modified {conflict.remote_modified.isoformat()}, "
                    f"newer than local {conflict.local_modified.isoformat()})"
                ),
            }
        else:
            return {
                'status': 'resolved',
                'action': 'keep_local',
                'winner': 'local',
                'message': (
                    f"Resolved {conflict.file_path}: keeping local version "
                    f"(modified {conflict.local_modified.isoformat()}, "
                    f"newer than or equal to remote {conflict.remote_modified.isoformat()})"
                ),
            }

    # Merge strategy - attempt to merge text files
    if strategy == 'merge':
        # Validate required parameters
        if not local_path:
            raise ValueError(
                "local_path required for merge strategy"
            )
        if not backend:
            raise ValueError(
                "backend required for merge strategy to download remote version"
            )

        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_file}")

        # Check if file appears to be text (simple heuristic)
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                local_content = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            # Binary file or unreadable - can't merge
            return {
                'status': 'manual_required',
                'action': 'none',
                'winner': None,
                'message': (
                    f"Cannot merge {conflict.file_path}: file is not text or unreadable. "
                    f"Error: {e}. Manual resolution required."
                ),
                'conflict': conflict,
            }

        # Download remote version for comparison
        try:
            with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
                remote_tmp_path = Path(tmp.name)

            try:
                # Extract the key from file_path (relative to storage root)
                remote_key = conflict.file_path
                backend.download(key=remote_key, local_path=remote_tmp_path)

                # Read remote content
                try:
                    with open(remote_tmp_path, 'r', encoding='utf-8') as f:
                        remote_content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    return {
                        'status': 'manual_required',
                        'action': 'none',
                        'winner': None,
                        'message': (
                            f"Cannot merge {conflict.file_path}: "
                            f"remote file is not text or unreadable. "
                            f"Manual resolution required."
                        ),
                        'conflict': conflict,
                    }

                # Simple merge: check if files are identical (despite different metadata)
                if local_content == remote_content:
                    return {
                        'status': 'resolved',
                        'action': 'keep_local',
                        'winner': 'both',
                        'message': (
                            f"Resolved {conflict.file_path}: files are identical "
                            f"(false conflict due to metadata differences)"
                        ),
                    }

                # Check if one is a superset of the other (simple append case)
                if local_content in remote_content:
                    return {
                        'status': 'manual_required',
                        'action': 'keep_remote',
                        'winner': 'remote',
                        'message': (
                            f"Merge suggested for {conflict.file_path}: "
                            f"remote appears to be superset of local. "
                            f"Consider keeping remote version."
                        ),
                        'conflict': conflict,
                    }
                elif remote_content in local_content:
                    return {
                        'status': 'manual_required',
                        'action': 'keep_local',
                        'winner': 'local',
                        'message': (
                            f"Merge suggested for {conflict.file_path}: "
                            f"local appears to be superset of remote. "
                            f"Consider keeping local version."
                        ),
                        'conflict': conflict,
                    }

                # Complex conflict - need manual merge
                return {
                    'status': 'manual_required',
                    'action': 'none',
                    'winner': None,
                    'message': (
                        f"Cannot auto-merge {conflict.file_path}: "
                        f"files have diverged. Manual merge required. "
                        f"Local: {len(local_content)} chars, "
                        f"Remote: {len(remote_content)} chars"
                    ),
                    'conflict': conflict,
                }

            finally:
                # Cleanup temp file
                remote_tmp_path.unlink(missing_ok=True)

        except Exception as e:
            return {
                'status': 'manual_required',
                'action': 'none',
                'winner': None,
                'message': (
                    f"Error during merge of {conflict.file_path}: {e}. "
                    f"Manual resolution required."
                ),
                'conflict': conflict,
            }

    # Should never reach here due to validation above
    raise ValueError(f"Unhandled strategy: {strategy}")


class EncryptionManager:
    """Handle encryption/decryption of backups"""

    def __init__(self, key: Optional[str] = None) -> None:
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography package required. Install with: pip install cryptography"
            )

        self.key: str = key or os.getenv("MOLTVAULT_KEY")  # type: ignore
        if not self.key:
            raise ValueError(
                "Encryption key required. Set MOLTVAULT_KEY environment variable."
            )
        self.fernet: "Fernet" = self._get_fernet()
    
    def _get_fernet(self) -> "Fernet":
        """Derive Fernet key from passphrase"""
        # Use a constant salt - safe because this is for local encryption
        # Not for passwords stored in databases
        salt = b"moltvault_backup_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.key.encode()))
        return Fernet(key)
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data"""
        return self.fernet.encrypt(data)
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data"""
        return self.fernet.decrypt(data)


class MoltVault:
    """
    OMI Backup/Restore System
    
    Features:
    - Full backup: tar.gz of palace.sqlite + NOW.md + config + memory logs
    - Incremental: only files changed since last full backup
    - Encryption support via MOLTVAULT_KEY
    - R2/S3 integration using boto3
    - Restore with integrity verification
    - Retention policy cleanup
    """
    
    def __init__(
        self,
        base_path: Path,
        config: Optional[Dict[str, Any]] = None,
        bucket: Optional[str] = None,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        """
        Initialize MoltVault

        Supports two initialization patterns:
        1. Direct parameters: MoltVault(path, bucket="my-bucket", endpoint="...")
        2. Config dict: MoltVault(path, {"backup": {"bucket": "my-bucket", ...}})

        Args:
            base_path: Path to OMI data directory
            config: Optional configuration dict (takes precedence over individual params)
            bucket: S3/R2 bucket name
            endpoint: S3-compatible endpoint URL
            access_key: S3 access key (default: R2_ACCESS_KEY_ID env var)
            secret_key: S3 secret key (default: R2_SECRET_ACCESS_KEY env var)
            region: S3 region

        Example:
            >>> # Direct parameters
            >>> vault = MoltVault(Path("/data"), bucket="my-bucket")
            >>> # Config dict
            >>> vault = MoltVault(Path("/data"), {"backup": {"bucket": "my-bucket"}})
        """
        self.base_path: Path = Path(base_path)

        # Extract configuration from config dict if provided
        if config and isinstance(config, dict) and 'backup' in config:
            backup_config = config['backup']
            self.bucket = backup_config.get('bucket', "moltbot-data")
            self.endpoint = backup_config.get('endpoint', "https://4c2932bc3381be38d5266241b16be092.r2.cloudflarestorage.com")
            self.region = backup_config.get('region', "auto")
            self.access_key = backup_config.get('access_key') or os.getenv("R2_ACCESS_KEY_ID")
            self.secret_key = backup_config.get('secret_key') or os.getenv("R2_SECRET_ACCESS_KEY")
        else:
            # Use direct parameters with defaults
            self.bucket: str = bucket or "moltbot-data"
            self.endpoint: str = endpoint or "https://4c2932bc3381be38d5266241b16be092.r2.cloudflarestorage.com"
            self.region: str = region or "auto"
            self.access_key: Optional[str] = access_key or os.getenv("R2_ACCESS_KEY_ID")
            self.secret_key: Optional[str] = secret_key or os.getenv("R2_SECRET_ACCESS_KEY")

        # State tracking
        self._backend: Optional[StorageBackend] = None
        self._s3_client: Optional[Any] = None  # Kept for backward compatibility with tests
        self._encryption: Optional[EncryptionManager] = None
        self._last_backup_path: Path = self.base_path / ".moltvault_last_backup"

        # Check for encryption key
        if os.getenv("MOLTVAULT_KEY") and CRYPTO_AVAILABLE:
            self._encryption = EncryptionManager()
    
    def _get_backend(self) -> StorageBackend:
        """Get or create storage backend"""
        # If _s3_client is set directly (e.g., by tests), use mock wrapper
        if self._s3_client is not None:
            if self._backend is None:
                # Use mock wrapper for backward compatibility with tests
                self._backend = _MockS3BackendWrapper(self._s3_client, self.bucket)
            return self._backend

        if self._backend is None:
            if not BOTO3_AVAILABLE:
                raise ImportError(
                    "boto3 package required. Install with: pip install boto3"
                )

            if not self.access_key or not self.secret_key:
                raise ValueError(
                    "R2 credentials required. Set R2_ACCESS_KEY_ID and "
                    "R2_SECRET_ACCESS_KEY environment variables."
                )

            self._backend = S3Backend(
                bucket=self.bucket,
                prefix="",
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                region=self.region,
            )

        return self._backend

    def _get_s3_client(self) -> Optional[Any]:
        """Get or create S3 client (deprecated - for backward compatibility)"""
        backend = self._get_backend()
        if hasattr(backend, '_client'):
            return backend._client
        return None
    
    def _get_files_to_backup(self, incremental: bool = False) -> List[Path]:
        """
        Get list of files to backup
        
        Args:
            incremental: If True, only include files changed since last backup
            
        Returns:
            List of file paths to include in backup
        """
        files = []
        
        # Critical files to always include
        critical_files = [
            self.base_path / "palace.sqlite",
            self.base_path / "NOW.md",
            self.base_path / "config.yaml",
            self.base_path / "MEMORY.md",
        ]
        
        # Add critical files that exist
        for file_path in critical_files:
            if file_path.exists():
                files.append(file_path)
        
        # Include memory logs directory
        memory_dir = self.base_path / "memory"
        if memory_dir.exists() and memory_dir.is_dir():
            for log_file in memory_dir.glob("*.md"):
                files.append(log_file)
        
        # Include any hash integrity files
        for hash_file in self.base_path.glob(".*.hash"):
            files.append(hash_file)
        
        # Filter for incremental backups
        if incremental and self._last_backup_path.exists():
            last_backup_time = datetime.fromisoformat(
                self._last_backup_path.read_text().strip()
            )
            
            filtered_files = []
            for file_path in files:
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime > last_backup_time:
                        filtered_files.append(file_path)
                except (OSError, FileNotFoundError):
                    continue
            
            # Always include critical database even in incremental
            if self.base_path / "palace.sqlite" not in filtered_files:
                db_path = self.base_path / "palace.sqlite"
                if db_path.exists():
                    filtered_files.insert(0, db_path)
            
            return filtered_files
        
        return files
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _create_backup_archive(
        self,
        files: List[Path],
        backup_type: str,
        temp_dir: Path,
    ) -> Path:
        """
        Create tar.gz archive from files
        
        Args:
            files: List of files to archive
            backup_type: 'full' or 'incremental'
            temp_dir: Directory for temporary files
            
        Returns:
            Path to created archive
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"omi_backup_{backup_type}_{timestamp}.tar.gz"
        archive_path = temp_dir / archive_name
        
        with tarfile.open(archive_path, "w:gz") as tar:
            for file_path in files:
                if file_path.exists():
                    # Store relative to base_path
                    arcname = file_path.relative_to(self.base_path)
                    tar.add(file_path, arcname=arcname)
        
        return archive_path
    
    def _encrypt_file(self, file_path: Path, temp_dir: Path) -> Path:
        """Encrypt file using encryption manager"""
        if not self._encryption:
            return file_path
        
        encrypted_path = temp_dir / f"{file_path.name}.enc"
        
        with open(file_path, "rb") as f:
            data = f.read()
        
        encrypted_data = self._encryption.encrypt(data)
        encrypted_path.write_bytes(encrypted_data)
        
        return encrypted_path
    
    def _decrypt_file(self, file_path: Path, temp_dir: Path) -> Path:
        """Decrypt file using encryption manager"""
        if not self._encryption:
            return file_path
        
        decrypted_path = temp_dir / file_path.name.replace(".enc", "")
        
        with open(file_path, "rb") as f:
            data = f.read()
        
        decrypted_data = self._encryption.decrypt(data)
        decrypted_path.write_bytes(decrypted_data)
        
        return decrypted_path
    
    def backup(
        self,
        full: bool = False,
        incremental: bool = False,
        encrypt: Optional[bool] = None,
    ) -> BackupMetadata:
        """
        Create backup
        
        Args:
            full: Create full backup (all critical files)
            incremental: Create incremental backup (only changed files)
            encrypt: Force encryption (default: auto-detect from MOLTVAULT_KEY)
            
        Returns:
            BackupMetadata with details of created backup
        """
        backup_type = "full" if full or not incremental else "incremental"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            
            # Get files to backup
            files = self._get_files_to_backup(incremental=backup_type == "incremental")
            
            if not files:
                raise ValueError("No files found to backup")
            
            # Create archive
            archive_path = self._create_backup_archive(files, backup_type, temp_dir)
            
            # Encrypt if needed
            use_encryption = encrypt if encrypt is not None else self._encryption is not None
            if use_encryption:
                if not CRYPTO_AVAILABLE:
                    raise ImportError("cryptography package required for encryption")
                if not self._encryption:
                    raise ValueError("MOLTVAULT_KEY required for encryption")
                archive_path = self._encrypt_file(archive_path, temp_dir)
            
            # Calculate metadata
            backup_id = f"omi_{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_size = archive_path.stat().st_size
            checksum = self._calculate_checksum(archive_path)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type=backup_type,
                created_at=datetime.now().isoformat(),
                file_size=file_size,
                checksum=checksum,
                encrypted=use_encryption,
                files_included=[str(f.relative_to(self.base_path)) for f in files],
                base_path_hash=hashlib.sha256(str(self.base_path).encode()).hexdigest()[:16],
                retention_days=30 if backup_type == "full" else 7,
            )
            
            # Upload to R2/S3 using backend abstraction
            backend = self._get_backend()

            s3_key = f"backups/{backup_id}.tar.gz"
            if use_encryption:
                s3_key += ".enc"
            meta_key = f"backups/{backup_id}.json"

            # Upload archive with metadata
            backend.upload(
                local_path=archive_path,
                key=s3_key,
                metadata={
                    "backup-type": backup_type,
                    "created-at": metadata.created_at,
                    "checksum": checksum,
                }
            )

            # Upload metadata JSON
            meta_file = temp_dir / f"{backup_id}.json"
            meta_file.write_text(json.dumps(metadata.to_dict(), indent=2))
            backend.upload(
                local_path=meta_file,
                key=meta_key,
            )

            # Update last backup time
            self._last_backup_path.write_text(datetime.now().isoformat())

            return metadata
    
    def list_backups(self) -> List[BackupMetadata]:
        """
        List all available backups with metadata

        Returns:
            List of BackupMetadata sorted by creation time (newest first)
        """
        backend = self._get_backend()

        backups = []
        prefix = "backups/"

        try:
            objects = backend.list(prefix=prefix)

            if not objects:
                return []

            # Find metadata files
            for obj in objects:
                key = obj.key
                if not key.endswith(".json"):
                    continue

                try:
                    # Download metadata to temporary file
                    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        backend.download(key=f"{prefix}{key}", local_path=tmp_path)
                        meta_data = json.loads(tmp_path.read_text())
                        backups.append(BackupMetadata.from_dict(meta_data))
                    finally:
                        tmp_path.unlink(missing_ok=True)

                except Exception:
                    continue

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.created_at, reverse=True)

        except StorageError as e:
            raise RuntimeError(f"Failed to list backups: {e}")

        return backups
    
    def restore(
        self,
        backup_id: str,
        target_path: Optional[Path] = None,
        verify: bool = True,
    ) -> Path:
        """
        Restore from backup

        Args:
            backup_id: ID of backup to restore
            target_path: Where to restore (default: original backup location)
            verify: Verify checksum integrity

        Returns:
            Path to restored base directory
        """
        backend = self._get_backend()

        # Determine paths
        s3_key = f"backups/{backup_id}.tar.gz"
        meta_key = f"backups/{backup_id}.json"

        # Download metadata first
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            meta_path = temp_dir / f"{backup_id}.json"

            try:
                backend.download(key=meta_key, local_path=meta_path)
                metadata = BackupMetadata.from_dict(
                    json.loads(meta_path.read_text())
                )
            except (KeyError, StorageError) as e:
                raise ValueError(f"Backup {backup_id} not found: {e}")

            # Adjust for encrypted backups
            if metadata.encrypted:
                if not CRYPTO_AVAILABLE:
                    raise ImportError("cryptography package required for decryption")
                if not self._encryption:
                    raise ValueError("MOLTVAULT_KEY required to restore encrypted backup")
                s3_key += ".enc"

            # Use base_path from metadata or provided target
            if target_path:
                restore_path = Path(target_path)
            else:
                restore_path = self.base_path

            archive_path = temp_dir / f"{backup_id}.tar.gz"

            # Download archive
            try:
                backend.download(key=s3_key, local_path=archive_path)
            except (KeyError, StorageError) as e:
                raise RuntimeError(f"Failed to download backup: {e}")

            # Verify checksum
            if verify:
                actual_checksum = self._calculate_checksum(archive_path)
                if actual_checksum != metadata.checksum:
                    raise ValueError(
                        f"Checksum mismatch! Expected {metadata.checksum}, "
                        f"got {actual_checksum}"
                    )

            # Decrypt if needed
            if metadata.encrypted:
                archive_path = self._decrypt_file(archive_path, temp_dir)

            # Ensure restore path exists
            restore_path.mkdir(parents=True, exist_ok=True)

            # Extract archive
            with tarfile.open(archive_path, "r:gz") as tar:
                # Security: validate members
                for member in tar.getmembers():
                    # Check for path traversal
                    member_path = restore_path / member.name
                    try:
                        member_path.relative_to(restore_path)
                    except ValueError:
                        raise ValueError(
                            f"Malicious archive: path traversal detected in {member.name}"
                        )
                tar.extractall(path=restore_path)

        return restore_path
    
    def cleanup(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Apply retention policy cleanup

        - Full backups: keep for 30 days
        - Incremental backups: keep for 7 days

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Dict with 'deleted' and 'kept' counts
        """
        backend = self._get_backend()
        backups = self.list_backups()

        now = datetime.now()
        deleted = 0
        kept = 0

        for backup in backups:
            try:
                created = datetime.fromisoformat(backup.created_at)
                age_days = (now - created).days

                # Determine retention
                max_age = backup.retention_days

                if age_days > max_age:
                    if not dry_run:
                        # Delete archive and metadata
                        s3_key = f"backups/{backup.backup_id}.tar.gz"
                        meta_key = f"backups/{backup.backup_id}.json"

                        if backup.encrypted:
                            s3_key += ".enc"

                        try:
                            backend.delete(s3_key)
                            backend.delete(meta_key)
                        except StorageError:
                            pass
                    deleted += 1
                else:
                    kept += 1
            except Exception:
                kept += 1

        return {"deleted": deleted, "kept": kept}

    def create_sync_snapshot(
        self,
        instance_id: str,
        vector_clock: Optional[Dict[str, int]] = None,
        sync_metadata: Optional[Dict[str, Any]] = None,
        encrypt: Optional[bool] = None,
    ) -> SyncSnapshotMetadata:
        """
        Create a sync-specific snapshot for distributed synchronization.

        Similar to backup() but optimized for sync operations:
        - Always includes palace.sqlite (critical for sync)
        - Optionally includes NOW.md and memory logs
        - Includes vector clock and sync metadata
        - Uses 'sync/' prefix instead of 'backups/' prefix

        Args:
            instance_id: ID of the instance creating this snapshot
            vector_clock: Optional vector clock state at snapshot time
            sync_metadata: Optional additional sync-specific metadata
            encrypt: Force encryption (default: auto-detect from MOLTVAULT_KEY)

        Returns:
            SyncSnapshotMetadata with details of created snapshot

        Raises:
            ValueError: If no files found to snapshot
            ImportError: If encryption requested but cryptography unavailable
            RuntimeError: If upload to storage fails

        Example:
            >>> vault = MoltVault(Path("/data/omi"))
            >>> metadata = vault.create_sync_snapshot(
            ...     instance_id="instance-1",
            ...     vector_clock={"instance-1": 42, "instance-2": 30}
            ... )
            >>> print(metadata.snapshot_id)
            'sync_instance-1_20240101_120000'
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Get files for sync snapshot (always include database)
            files = []

            # Critical: palace.sqlite is required for sync
            db_path = self.base_path / "palace.sqlite"
            if not db_path.exists():
                raise ValueError(
                    f"Database not found at {db_path}. Cannot create sync snapshot."
                )
            files.append(db_path)

            # Optional: Include NOW.md if it exists
            now_path = self.base_path / "NOW.md"
            if now_path.exists():
                files.append(now_path)

            # Optional: Include recent memory logs (last 7 days)
            memory_dir = self.base_path / "memory"
            if memory_dir.exists() and memory_dir.is_dir():
                cutoff_date = datetime.now() - timedelta(days=7)
                for log_file in memory_dir.glob("*.md"):
                    try:
                        mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                        if mtime > cutoff_date:
                            files.append(log_file)
                    except (OSError, FileNotFoundError):
                        continue

            # Include integrity hash files
            for hash_file in self.base_path.glob(".*.hash"):
                if hash_file.exists():
                    files.append(hash_file)

            if not files:
                raise ValueError("No files found for sync snapshot")

            # Create archive
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"omi_sync_{instance_id}_{timestamp}.tar.gz"
            archive_path = temp_dir / archive_name

            with tarfile.open(archive_path, "w:gz") as tar:
                for file_path in files:
                    if file_path.exists():
                        arcname = file_path.relative_to(self.base_path)
                        tar.add(file_path, arcname=arcname)

            # Encrypt if needed
            use_encryption = encrypt if encrypt is not None else self._encryption is not None
            if use_encryption:
                if not CRYPTO_AVAILABLE:
                    raise ImportError("cryptography package required for encryption")
                if not self._encryption:
                    raise ValueError("MOLTVAULT_KEY required for encryption")
                archive_path = self._encrypt_file(archive_path, temp_dir)

            # Calculate metadata
            snapshot_id = f"sync_{instance_id}_{timestamp}"
            file_size = archive_path.stat().st_size
            checksum = self._calculate_checksum(archive_path)

            # Create metadata
            metadata = SyncSnapshotMetadata(
                snapshot_id=snapshot_id,
                instance_id=instance_id,
                created_at=datetime.now().isoformat(),
                file_size=file_size,
                checksum=checksum,
                encrypted=use_encryption,
                files_included=[str(f.relative_to(self.base_path)) for f in files],
                vector_clock=vector_clock or {},
                sync_metadata=sync_metadata or {},
                retention_days=7,  # Sync snapshots have shorter retention
            )

            # Upload to storage using 'sync/' prefix
            backend = self._get_backend()

            s3_key = f"sync/{snapshot_id}.tar.gz"
            if use_encryption:
                s3_key += ".enc"
            meta_key = f"sync/{snapshot_id}.json"

            # Upload archive with metadata
            backend.upload(
                local_path=archive_path,
                key=s3_key,
                metadata={
                    "snapshot-type": "sync",
                    "instance-id": instance_id,
                    "created-at": metadata.created_at,
                    "checksum": checksum,
                }
            )

            # Upload metadata JSON
            meta_file = temp_dir / f"{snapshot_id}.json"
            meta_file.write_text(json.dumps(metadata.to_dict(), indent=2))
            backend.upload(
                local_path=meta_file,
                key=meta_key,
            )

            logger.info(
                f"Created sync snapshot {snapshot_id} for instance {instance_id} "
                f"({file_size} bytes, {len(files)} files)"
            )

            return metadata

    def list_sync_snapshots(
        self,
        instance_id: Optional[str] = None
    ) -> List[SyncSnapshotMetadata]:
        """
        List available sync snapshots.

        Args:
            instance_id: Optional filter by instance ID

        Returns:
            List of SyncSnapshotMetadata sorted by creation time (newest first)

        Example:
            >>> vault = MoltVault(Path("/data/omi"))
            >>> snapshots = vault.list_sync_snapshots(instance_id="instance-1")
            >>> for snapshot in snapshots:
            ...     print(f"{snapshot.snapshot_id}: {snapshot.file_size} bytes")
        """
        backend = self._get_backend()

        snapshots = []
        prefix = "sync/"

        try:
            objects = backend.list(prefix=prefix)

            if not objects:
                return []

            # Find metadata files
            for obj in objects:
                key = obj.key
                if not key.endswith(".json"):
                    continue

                # Filter by instance_id if specified
                if instance_id and f"sync_{instance_id}_" not in key:
                    continue

                try:
                    # Download metadata to temporary file
                    with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as tmp:
                        tmp_path = Path(tmp.name)

                    try:
                        backend.download(key=f"{prefix}{key}", local_path=tmp_path)
                        meta_data = json.loads(tmp_path.read_text())
                        snapshots.append(SyncSnapshotMetadata.from_dict(meta_data))
                    finally:
                        tmp_path.unlink(missing_ok=True)

                except Exception as e:
                    logger.warning(f"Failed to read sync snapshot metadata {key}: {e}")
                    continue

            # Sort by creation time (newest first)
            snapshots.sort(key=lambda x: x.created_at, reverse=True)

        except StorageError as e:
            logger.error(f"Failed to list sync snapshots: {e}")
            raise RuntimeError(f"Failed to list sync snapshots: {e}")

        return snapshots

    def download_sync_snapshot(
        self,
        snapshot_id: str,
        target_path: Optional[Path] = None,
        verify: bool = True,
    ) -> Path:
        """
        Download and extract a sync snapshot.

        Args:
            snapshot_id: ID of snapshot to download
            target_path: Where to extract (default: base_path)
            verify: Verify checksum integrity

        Returns:
            Path to extracted directory

        Raises:
            ValueError: If snapshot not found or checksum mismatch
            RuntimeError: If download or extraction fails

        Example:
            >>> vault = MoltVault(Path("/data/omi"))
            >>> restored_path = vault.download_sync_snapshot(
            ...     snapshot_id="sync_instance-1_20240101_120000"
            ... )
            >>> print(f"Restored to {restored_path}")
        """
        backend = self._get_backend()

        # Determine paths
        s3_key = f"sync/{snapshot_id}.tar.gz"
        meta_key = f"sync/{snapshot_id}.json"

        # Download metadata first
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            meta_path = temp_dir / f"{snapshot_id}.json"

            try:
                backend.download(key=meta_key, local_path=meta_path)
                metadata = SyncSnapshotMetadata.from_dict(
                    json.loads(meta_path.read_text())
                )
            except (KeyError, StorageError) as e:
                raise ValueError(f"Sync snapshot {snapshot_id} not found: {e}")

            # Adjust for encrypted snapshots
            if metadata.encrypted:
                if not CRYPTO_AVAILABLE:
                    raise ImportError("cryptography package required for decryption")
                if not self._encryption:
                    raise ValueError("MOLTVAULT_KEY required to decrypt snapshot")
                s3_key += ".enc"

            # Use base_path or provided target
            if target_path:
                extract_path = Path(target_path)
            else:
                extract_path = self.base_path

            archive_path = temp_dir / f"{snapshot_id}.tar.gz"

            # Download archive
            try:
                backend.download(key=s3_key, local_path=archive_path)
            except (KeyError, StorageError) as e:
                raise RuntimeError(f"Failed to download sync snapshot: {e}")

            # Verify checksum
            if verify:
                actual_checksum = self._calculate_checksum(archive_path)
                if actual_checksum != metadata.checksum:
                    raise ValueError(
                        f"Checksum mismatch! Expected {metadata.checksum}, "
                        f"got {actual_checksum}"
                    )

            # Decrypt if needed
            if metadata.encrypted:
                archive_path = self._decrypt_file(archive_path, temp_dir)

            # Ensure extract path exists
            extract_path.mkdir(parents=True, exist_ok=True)

            # Extract archive
            with tarfile.open(archive_path, "r:gz") as tar:
                # Security: validate members
                for member in tar.getmembers():
                    # Check for path traversal
                    member_path = extract_path / member.name
                    try:
                        member_path.relative_to(extract_path)
                    except ValueError:
                        raise ValueError(
                            f"Malicious archive: path traversal detected in {member.name}"
                        )
                tar.extractall(path=extract_path)

            logger.info(
                f"Downloaded and extracted sync snapshot {snapshot_id} to {extract_path}"
            )

        return extract_path

    def delete_sync_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a sync snapshot from storage.

        Args:
            snapshot_id: ID of snapshot to delete

        Returns:
            True if successfully deleted, False otherwise

        Example:
            >>> vault = MoltVault(Path("/data/omi"))
            >>> success = vault.delete_sync_snapshot("sync_instance-1_20240101_120000")
        """
        backend = self._get_backend()

        try:
            # Delete archive and metadata
            s3_key = f"sync/{snapshot_id}.tar.gz"
            meta_key = f"sync/{snapshot_id}.json"

            # Try both encrypted and non-encrypted versions
            try:
                backend.delete(s3_key)
            except StorageError:
                try:
                    backend.delete(f"{s3_key}.enc")
                except StorageError:
                    pass

            backend.delete(meta_key)

            logger.info(f"Deleted sync snapshot {snapshot_id}")
            return True

        except StorageError as e:
            logger.error(f"Failed to delete sync snapshot {snapshot_id}: {e}")
            return False

    def cleanup_sync_snapshots(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Apply retention policy cleanup to sync snapshots.

        Sync snapshots are kept for 7 days by default.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Dict with 'deleted' and 'kept' counts

        Example:
            >>> vault = MoltVault(Path("/data/omi"))
            >>> result = vault.cleanup_sync_snapshots(dry_run=True)
            >>> print(f"Would delete {result['deleted']} snapshots")
        """
        snapshots = self.list_sync_snapshots()

        now = datetime.now()
        deleted = 0
        kept = 0

        for snapshot in snapshots:
            try:
                created = datetime.fromisoformat(snapshot.created_at)
                age_days = (now - created).days

                if age_days > snapshot.retention_days:
                    if not dry_run:
                        self.delete_sync_snapshot(snapshot.snapshot_id)
                    deleted += 1
                else:
                    kept += 1
            except Exception as e:
                logger.warning(
                    f"Failed to process snapshot {snapshot.snapshot_id} for cleanup: {e}"
                )
                kept += 1

        logger.info(
            f"Sync snapshot cleanup: deleted={deleted}, kept={kept} (dry_run={dry_run})"
        )

        return {"deleted": deleted, "kept": kept}


# Import base64 for Fernet key generation
import base64


def create_backup(
    base_path: Optional[Path] = None,
    full: bool = False,
    incremental: bool = False,
) -> str:
    """
    Convenience function for CLI: Create backup
    
    Args:
        base_path: Optional custom base path
        full: Create full backup
        incremental: Create incremental backup
        
    Returns:
        Backup ID as string
    """
    if base_path is None:
        base_path = Path.home() / ".openclaw" / "omi"
    
    vault = MoltVault(base_path)
    metadata = vault.backup(full=full, incremental=incremental)
    return metadata.backup_id


def restore_backup(
    backup_id: str,
    target_path: Optional[Path] = None,
) -> Path:
    """
    Convenience function for CLI: Restore backup
    
    Args:
        backup_id: ID of backup to restore
        target_path: Optional restore location
        
    Returns:
        Path to restored directory
    """
    if target_path is None:
        target_path = Path.home() / ".openclaw" / "omi"
    
    vault = MoltVault(target_path)
    return vault.restore(backup_id, target_path)


def list_backups_cli() -> List[Dict[str, Any]]:
    """
    Convenience function for CLI: List backups
    
    Returns:
        List of backup metadata as dicts
    """
    base_path = Path.home() / ".openclaw" / "omi"
    vault = MoltVault(base_path)
    backups = vault.list_backups()
    return [b.to_dict() for b in backups]
