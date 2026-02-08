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
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, BinaryIO
from dataclasses import dataclass, asdict
from contextlib import contextmanager

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


class EncryptionManager:
    """Handle encryption/decryption of backups"""
    
    def __init__(self, key: Optional[str] = None):
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography package required. Install with: pip install cryptography"
            )
        
        self.key = key or os.getenv("MOLTVAULT_KEY")
        if not self.key:
            raise ValueError(
                "Encryption key required. Set MOLTVAULT_KEY environment variable."
            )
        self.fernet = self._get_fernet()
    
    def _get_fernet(self):
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
        bucket: str = "moltbot-data",
        endpoint: str = "https://4c2932bc3381be38d5266241b16be092.r2.cloudflarestorage.com",
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "auto",
    ):
        """
        Initialize MoltVault
        
        Args:
            base_path: Path to OMI data directory
            bucket: S3/R2 bucket name
            endpoint: S3-compatible endpoint URL
            access_key: S3 access key (default: R2_ACCESS_KEY_ID env var)
            secret_key: S3 secret key (default: R2_SECRET_ACCESS_KEY env var)
            region: S3 region
        """
        self.base_path = Path(base_path)
        self.bucket = bucket
        self.endpoint = endpoint
        self.region = region
        
        # Get credentials from env or parameters
        self.access_key = access_key or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("R2_SECRET_ACCESS_KEY")
        
        # State tracking
        self._s3_client: Optional[Any] = None
        self._encryption: Optional[EncryptionManager] = None
        self._last_backup_path = self.base_path / ".moltvault_last_backup"
        
        # Check for encryption key
        if os.getenv("MOLTVAULT_KEY") and CRYPTO_AVAILABLE:
            self._encryption = EncryptionManager()
    
    def _get_s3_client(self) -> Any:
        """Get or create S3 client"""
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 package required. Install with: pip install boto3"
            )
        
        if self._s3_client is None:
            if not self.access_key or not self.secret_key:
                raise ValueError(
                    "R2 credentials required. Set R2_ACCESS_KEY_ID and "
                    "R2_SECRET_ACCESS_KEY environment variables."
                )
            
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
            )
        return self._s3_client
    
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
            
            # Upload to R2/S3
            s3_key = f"backups/{backup_id}.tar.gz"
            if use_encryption:
                s3_key += ".enc"
            meta_key = f"backups/{backup_id}.json"
            
            s3 = self._get_s3_client()
            
            # Upload archive
            with open(archive_path, "rb") as f:
                s3.upload_fileobj(
                    f,
                    self.bucket,
                    s3_key,
                    ExtraArgs={
                        "Metadata": {
                            "backup-type": backup_type,
                            "created-at": metadata.created_at,
                            "checksum": checksum,
                        }
                    }
                )
            
            # Upload metadata
            s3.put_object(
                Bucket=self.bucket,
                Key=meta_key,
                Body=json.dumps(metadata.to_dict(), indent=2),
                ContentType="application/json",
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
        s3 = self._get_s3_client()
        
        backups = []
        prefix = "backups/"
        
        try:
            response = s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
            
            if "Contents" not in response:
                return []
            
            # Find metadata files
            for obj in response["Contents"]:
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                
                try:
                    meta_obj = s3.get_object(Bucket=self.bucket, Key=key)
                    meta_data = json.loads(meta_obj["Body"].read())
                    backups.append(BackupMetadata.from_dict(meta_data))
                except Exception:
                    continue
            
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.created_at, reverse=True)
            
        except ClientError as e:
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
        s3 = self._get_s3_client()
        
        # Determine paths
        s3_key = f"backups/{backup_id}.tar.gz"
        meta_key = f"backups/{backup_id}.json"
        
        # Download metadata first
        try:
            meta_obj = s3.get_object(Bucket=self.bucket, Key=meta_key)
            metadata = BackupMetadata.from_dict(
                json.loads(meta_obj["Body"].read())
            )
        except ClientError as e:
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            archive_path = temp_dir / f"{backup_id}.tar.gz"
            
            # Download archive
            try:
                s3.download_file(self.bucket, s3_key, str(archive_path))
            except ClientError as e:
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
        s3 = self._get_s3_client()
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
                            s3.delete_object(Bucket=self.bucket, Key=s3_key)
                            s3.delete_object(Bucket=self.bucket, Key=meta_key)
                        except ClientError:
                            pass
                    deleted += 1
                else:
                    kept += 1
            except Exception:
                kept += 1
        
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
