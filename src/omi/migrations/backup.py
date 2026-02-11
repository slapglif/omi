"""
Backup Manager for OMI Database

Handles database backup and restore operations for safe migrations.

Features:
- Timestamped backups with metadata
- WAL-safe backup using SQLite backup API
- Automatic old backup cleanup
- Restore from backup on migration failure
- Backup verification
"""

import sqlite3
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class BackupInfo:
    """Information about a database backup."""
    path: Path
    original_db: Path
    created_at: datetime
    size_bytes: int
    schema_version: int
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": str(self.path),
            "original_db": str(self.original_db),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "size_bytes": self.size_bytes,
            "schema_version": self.schema_version,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupInfo':
        """Create BackupInfo from dictionary."""
        return cls(
            path=Path(data["path"]),
            original_db=Path(data["original_db"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            size_bytes=data["size_bytes"],
            schema_version=data["schema_version"],
            metadata=data.get("metadata")
        )


class BackupManager:
    """
    Backup Manager - Safe database backup and restore for migrations

    Pattern: Timestamped backups with metadata, WAL-safe operations
    Lifetime: Backups persist until manually cleaned

    Features:
    - WAL-safe backup using SQLite backup API
    - Timestamped backup files with metadata
    - Restore from backup
    - List and query backups
    - Automatic cleanup of old backups
    - Backup verification

    Example:
        manager = BackupManager(db_path, backup_dir)
        backup_info = manager.create_backup()
        # ... perform migration ...
        if migration_failed:
            manager.restore_backup(backup_info.path)
    """

    # Backup filename pattern: {db_name}_backup_{timestamp}.db
    BACKUP_SUFFIX = "_backup_"
    METADATA_SUFFIX = ".meta.json"

    def __init__(self, db_path: Path, backup_dir: Optional[Path] = None):
        """
        Initialize Backup Manager.

        Args:
            db_path: Path to SQLite database file to backup
            backup_dir: Directory to store backups (default: {db_path.parent}/backups)
        """
        self.db_path = Path(db_path)

        # Default backup directory: {db_path.parent}/backups
        if backup_dir is None:
            self.backup_dir = self.db_path.parent / "backups"
        else:
            self.backup_dir = Path(backup_dir)

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, metadata: Optional[Dict[str, Any]] = None) -> BackupInfo:
        """
        Create a timestamped backup of the database using SQLite backup API.

        This method uses SQLite's online backup API which is WAL-safe and
        ensures a consistent snapshot even with concurrent writes.

        Args:
            metadata: Optional metadata to store with backup

        Returns:
            BackupInfo object with backup details

        Raises:
            sqlite3.Error: If backup fails
            OSError: If backup file cannot be created
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        # Generate timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_name = self.db_path.stem
        backup_name = f"{db_name}{self.BACKUP_SUFFIX}{timestamp}.db"
        backup_path = self.backup_dir / backup_name

        # Get schema version before backup
        schema_version = self._get_schema_version()

        # Perform WAL-safe backup using SQLite backup API
        self._sqlite_backup(self.db_path, backup_path)

        # Get backup file size
        size_bytes = backup_path.stat().st_size

        # Create backup info
        backup_info = BackupInfo(
            path=backup_path,
            original_db=self.db_path,
            created_at=datetime.now(),
            size_bytes=size_bytes,
            schema_version=schema_version,
            metadata=metadata
        )

        # Save metadata to sidecar file
        self._save_metadata(backup_info)

        return backup_info

    def _sqlite_backup(self, source: Path, dest: Path) -> None:
        """
        Perform WAL-safe backup using SQLite backup API.

        The backup API copies the database page-by-page, ensuring consistency
        even with concurrent writes in WAL mode.

        Args:
            source: Source database path
            dest: Destination backup path

        Raises:
            sqlite3.Error: If backup fails
        """
        # Connect to source database
        source_conn = sqlite3.connect(source)

        # Connect to destination database
        dest_conn = sqlite3.connect(dest)

        try:
            # Perform backup using SQLite backup API
            # This is WAL-safe and handles concurrent writes
            with source_conn:
                source_conn.backup(dest_conn)
        finally:
            source_conn.close()
            dest_conn.close()

    def _get_schema_version(self) -> int:
        """
        Get current schema version from database.

        Returns:
            Schema version from PRAGMA user_version
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA user_version")
            return cursor.fetchone()[0]

    def _save_metadata(self, backup_info: BackupInfo) -> None:
        """
        Save backup metadata to sidecar JSON file.

        Args:
            backup_info: Backup information to save
        """
        metadata_path = backup_info.path.with_suffix(
            backup_info.path.suffix + self.METADATA_SUFFIX
        )

        with open(metadata_path, 'w') as f:
            json.dump(backup_info.to_dict(), f, indent=2)

    def _load_metadata(self, backup_path: Path) -> Optional[BackupInfo]:
        """
        Load backup metadata from sidecar JSON file.

        Args:
            backup_path: Path to backup database file

        Returns:
            BackupInfo if metadata exists, None otherwise
        """
        metadata_path = backup_path.with_suffix(
            backup_path.suffix + self.METADATA_SUFFIX
        )

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return BackupInfo.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def restore_backup(self, backup_path: Path) -> None:
        """
        Restore database from backup.

        WARNING: This overwrites the current database. Use with caution.

        Args:
            backup_path: Path to backup file to restore

        Raises:
            FileNotFoundError: If backup file doesn't exist
            OSError: If restore fails
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Close any connections to the database
        # In production, caller should ensure no active connections

        # Restore backup using copy
        # For WAL mode databases, also need to handle -wal and -shm files
        shutil.copy2(backup_path, self.db_path)

        # Remove WAL files if they exist (force clean state)
        wal_file = self.db_path.with_suffix(self.db_path.suffix + "-wal")
        shm_file = self.db_path.with_suffix(self.db_path.suffix + "-shm")

        if wal_file.exists():
            wal_file.unlink()
        if shm_file.exists():
            shm_file.unlink()

    def list_backups(self) -> List[BackupInfo]:
        """
        List all available backups for this database.

        Returns:
            List of BackupInfo objects, ordered by creation time (newest first)
        """
        backups = []

        # Find all backup files matching pattern
        db_name = self.db_path.stem
        pattern = f"{db_name}{self.BACKUP_SUFFIX}*.db"

        for backup_file in self.backup_dir.glob(pattern):
            # Try to load metadata
            backup_info = self._load_metadata(backup_file)

            # If no metadata, create basic info from file
            if backup_info is None:
                backup_info = BackupInfo(
                    path=backup_file,
                    original_db=self.db_path,
                    created_at=datetime.fromtimestamp(backup_file.stat().st_mtime),
                    size_bytes=backup_file.stat().st_size,
                    schema_version=-1,  # Unknown
                    metadata=None
                )

            backups.append(backup_info)

        # Sort by creation time, newest first
        backups.sort(key=lambda b: b.created_at, reverse=True)

        return backups

    def get_latest_backup(self) -> Optional[BackupInfo]:
        """
        Get the most recent backup.

        Returns:
            BackupInfo for latest backup, or None if no backups exist
        """
        backups = self.list_backups()
        return backups[0] if backups else None

    def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """
        Remove old backups, keeping only the most recent N backups.

        Args:
            keep_count: Number of recent backups to keep (default: 5)

        Returns:
            Number of backups deleted
        """
        if keep_count < 1:
            raise ValueError(f"keep_count must be >= 1, got {keep_count}")

        backups = self.list_backups()

        # Keep the most recent N backups
        to_delete = backups[keep_count:]

        deleted_count = 0
        for backup_info in to_delete:
            try:
                # Delete backup file
                backup_info.path.unlink()

                # Delete metadata file if exists
                metadata_path = backup_info.path.with_suffix(
                    backup_info.path.suffix + self.METADATA_SUFFIX
                )
                if metadata_path.exists():
                    metadata_path.unlink()

                deleted_count += 1
            except OSError:
                # Continue on error, best effort cleanup
                pass

        return deleted_count

    def verify_backup(self, backup_path: Path) -> bool:
        """
        Verify that a backup file is a valid SQLite database.

        Args:
            backup_path: Path to backup file to verify

        Returns:
            True if backup is valid, False otherwise
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            return False

        try:
            # Try to open and query the backup database
            with sqlite3.connect(backup_path) as conn:
                conn.execute("PRAGMA integrity_check")
                conn.execute("SELECT name FROM sqlite_master LIMIT 1")
            return True
        except sqlite3.Error:
            return False

    def get_backup_info(self, backup_path: Path) -> Optional[BackupInfo]:
        """
        Get information about a specific backup.

        Args:
            backup_path: Path to backup file

        Returns:
            BackupInfo if backup exists, None otherwise
        """
        backup_path = Path(backup_path)

        if not backup_path.exists():
            return None

        # Try to load metadata
        backup_info = self._load_metadata(backup_path)

        # If no metadata, create basic info from file
        if backup_info is None:
            try:
                with sqlite3.connect(backup_path) as conn:
                    cursor = conn.execute("PRAGMA user_version")
                    schema_version = cursor.fetchone()[0]
            except sqlite3.Error:
                schema_version = -1

            backup_info = BackupInfo(
                path=backup_path,
                original_db=self.db_path,
                created_at=datetime.fromtimestamp(backup_path.stat().st_mtime),
                size_bytes=backup_path.stat().st_size,
                schema_version=schema_version,
                metadata=None
            )

        return backup_info
