"""
Storage Backends - Abstract interface for cloud storage

Provides plugin architecture for different cloud storage providers (S3, GCS, Azure).
Each backend implements upload, download, list, and delete operations.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StorageObject:
    """Metadata for a stored object"""
    key: str  # Object key/path in storage
    size: int  # Size in bytes
    last_modified: datetime  # Last modification timestamp
    etag: Optional[str] = None  # Entity tag (checksum/version identifier)
    metadata: Optional[Dict[str, str]] = None  # Custom metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "size": self.size,
            "last_modified": self.last_modified.isoformat(),
            "etag": self.etag,
            "metadata": self.metadata or {},
        }


class StorageBackend(ABC):
    """
    Abstract base class for cloud storage backends

    All storage backends (S3, GCS, Azure) must implement these methods.
    This enables plugin-based architecture for MoltVault backups.
    """

    def __init__(self, bucket: str, prefix: str = ""):
        """
        Initialize storage backend

        Args:
            bucket: Bucket/container name
            prefix: Optional prefix for all keys (e.g., "backups/")
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

    def _make_key(self, key: str) -> str:
        """Apply prefix to key"""
        return self.prefix + key.lstrip("/")

    @abstractmethod
    def upload(
        self,
        local_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Upload a file to storage

        Args:
            local_path: Path to local file to upload
            key: Storage key (path) for the object
            metadata: Optional metadata to attach to object

        Returns:
            The storage key of the uploaded object

        Raises:
            FileNotFoundError: If local_path doesn't exist
            StorageError: If upload fails
        """
        pass

    @abstractmethod
    def download(
        self,
        key: str,
        local_path: Path,
    ) -> Path:
        """
        Download a file from storage

        Args:
            key: Storage key of object to download
            local_path: Path where file should be saved

        Returns:
            Path to downloaded file

        Raises:
            KeyError: If key doesn't exist in storage
            StorageError: If download fails
        """
        pass

    @abstractmethod
    def list(
        self,
        prefix: str = "",
        max_keys: Optional[int] = None,
    ) -> List[StorageObject]:
        """
        List objects in storage

        Args:
            prefix: Only list objects with this prefix (relative to bucket prefix)
            max_keys: Maximum number of objects to return

        Returns:
            List of StorageObject metadata

        Raises:
            StorageError: If listing fails
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete an object from storage

        Args:
            key: Storage key of object to delete

        Returns:
            True if object was deleted, False if it didn't exist

        Raises:
            StorageError: If deletion fails
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if an object exists in storage

        Args:
            key: Storage key to check

        Returns:
            True if object exists, False otherwise

        Raises:
            StorageError: If check fails
        """
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[StorageObject]:
        """
        Get metadata for an object without downloading it

        Args:
            key: Storage key of object

        Returns:
            StorageObject with metadata, or None if object doesn't exist

        Raises:
            StorageError: If metadata retrieval fails
        """
        pass


class StorageError(Exception):
    """Base exception for storage backend errors"""
    pass


class StorageAuthError(StorageError):
    """Exception for authentication/authorization errors"""
    pass


class StorageConnectionError(StorageError):
    """Exception for connection/network errors"""
    pass
