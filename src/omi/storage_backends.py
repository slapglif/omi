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


# S3/Boto3 imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3Backend(StorageBackend):
    """
    AWS S3 / Cloudflare R2 storage backend

    Supports any S3-compatible storage service including:
    - AWS S3
    - Cloudflare R2
    - MinIO
    - DigitalOcean Spaces

    Authentication via:
    - Explicit credentials (access_key, secret_key)
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - IAM roles (when running on AWS)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "auto",
    ):
        """
        Initialize S3 backend

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix for all keys (e.g., "backups/")
            endpoint: S3-compatible endpoint URL (for R2, MinIO, etc.)
            access_key: AWS access key ID (default: AWS_ACCESS_KEY_ID env var)
            secret_key: AWS secret access key (default: AWS_SECRET_ACCESS_KEY env var)
            region: AWS region (default: "auto" for R2, "us-east-1" for AWS)
        """
        super().__init__(bucket, prefix)

        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 package required for S3 backend. "
                "Install with: pip install boto3"
            )

        self.endpoint = endpoint
        self.region = region

        # Get credentials from parameters or environment
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        # Initialize S3 client
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create and configure S3 client"""
        try:
            client_kwargs = {
                "service_name": "s3",
                "region_name": self.region,
            }

            # Add endpoint if specified (for R2, MinIO, etc.)
            if self.endpoint:
                client_kwargs["endpoint_url"] = self.endpoint

            # Add explicit credentials if provided
            if self.access_key and self.secret_key:
                client_kwargs["aws_access_key_id"] = self.access_key
                client_kwargs["aws_secret_access_key"] = self.secret_key

            return boto3.client(**client_kwargs)

        except NoCredentialsError as e:
            raise StorageAuthError(
                "No AWS credentials found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables or pass "
                "access_key and secret_key parameters."
            ) from e
        except Exception as e:
            raise StorageConnectionError(
                f"Failed to create S3 client: {e}"
            ) from e

    def upload(
        self,
        local_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload a file to S3"""
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = self._make_key(key)

        try:
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            self._client.upload_file(
                str(local_path),
                self.bucket,
                full_key,
                ExtraArgs=extra_args if extra_args else None,
            )

            return full_key

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
            elif error_code in ["AccessDenied", "InvalidAccessKeyId"]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to upload {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error uploading {key}: {e}") from e

    def download(
        self,
        key: str,
        local_path: Path,
    ) -> Path:
        """Download a file from S3"""
        full_key = self._make_key(key)

        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self._client.download_file(
                self.bucket,
                full_key,
                str(local_path),
            )

            return local_path

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise KeyError(f"Object not found: {key}") from e
            elif error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
            elif error_code in ["AccessDenied", "InvalidAccessKeyId"]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to download {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error downloading {key}: {e}") from e

    def list(
        self,
        prefix: str = "",
        max_keys: Optional[int] = None,
    ) -> List[StorageObject]:
        """List objects in S3"""
        full_prefix = self._make_key(prefix)

        try:
            list_kwargs = {
                "Bucket": self.bucket,
                "Prefix": full_prefix,
            }
            if max_keys is not None:
                list_kwargs["MaxKeys"] = max_keys

            response = self._client.list_objects_v2(**list_kwargs)

            objects = []
            for item in response.get("Contents", []):
                # Remove bucket prefix to get relative key
                key = item["Key"]
                if self.prefix and key.startswith(self.prefix):
                    key = key[len(self.prefix):]

                objects.append(StorageObject(
                    key=key,
                    size=item["Size"],
                    last_modified=item["LastModified"],
                    etag=item.get("ETag", "").strip('"'),
                    metadata=None,  # Metadata not included in list
                ))

            return objects

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
            elif error_code in ["AccessDenied", "InvalidAccessKeyId"]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to list objects: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error listing objects: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete an object from S3"""
        full_key = self._make_key(key)

        try:
            # Check if object exists first
            exists = self.exists(key)

            if exists:
                self._client.delete_object(
                    Bucket=self.bucket,
                    Key=full_key,
                )
                return True
            else:
                return False

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
            elif error_code in ["AccessDenied", "InvalidAccessKeyId"]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to delete {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error deleting {key}: {e}") from e

    def exists(self, key: str) -> bool:
        """Check if an object exists in S3"""
        full_key = self._make_key(key)

        try:
            self._client.head_object(
                Bucket=self.bucket,
                Key=full_key,
            )
            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                return False
            elif error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
            elif error_code in ["AccessDenied", "InvalidAccessKeyId"]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to check existence of {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error checking {key}: {e}") from e

    def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata for an object without downloading it"""
        full_key = self._make_key(key)

        try:
            response = self._client.head_object(
                Bucket=self.bucket,
                Key=full_key,
            )

            # Remove bucket prefix to get relative key
            relative_key = key

            return StorageObject(
                key=relative_key,
                size=response["ContentLength"],
                last_modified=response["LastModified"],
                etag=response.get("ETag", "").strip('"'),
                metadata=response.get("Metadata"),
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "404":
                return None
            elif error_code == "NoSuchBucket":
                raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
            elif error_code in ["AccessDenied", "InvalidAccessKeyId"]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to get metadata for {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error getting metadata for {key}: {e}") from e
