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


# Google Cloud Storage imports
try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound, Forbidden, GoogleCloudError
    from google.api_core.exceptions import Unauthenticated
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class GCSBackend(StorageBackend):
    """
    Google Cloud Storage backend

    Supports Google Cloud Storage with automatic authentication via:
    - Explicit credentials file path (credentials_file)
    - GOOGLE_APPLICATION_CREDENTIALS environment variable
    - Default credentials (gcloud CLI, GCE/GKE service account, etc.)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        credentials_file: Optional[str] = None,
        project: Optional[str] = None,
    ):
        """
        Initialize GCS backend

        Args:
            bucket: GCS bucket name
            prefix: Optional prefix for all keys (e.g., "backups/")
            credentials_file: Path to service account credentials JSON file
            project: GCP project ID (optional, can be inferred from credentials)
        """
        super().__init__(bucket, prefix)

        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage package required for GCS backend. "
                "Install with: pip install google-cloud-storage"
            )

        self.project = project
        self.credentials_file = credentials_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        # Initialize GCS client
        self._client = self._create_client()
        self._bucket = self._client.bucket(self.bucket)

    def _create_client(self) -> Any:
        """Create and configure GCS client"""
        try:
            client_kwargs = {}

            if self.project:
                client_kwargs["project"] = self.project

            # Load credentials from file if specified
            if self.credentials_file:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_file
                )
                client_kwargs["credentials"] = credentials

            return storage.Client(**client_kwargs)

        except Exception as e:
            if "could not be automatically determined" in str(e).lower():
                raise StorageAuthError(
                    "No GCS credentials found. Set GOOGLE_APPLICATION_CREDENTIALS "
                    "environment variable or pass credentials_file parameter."
                ) from e
            else:
                raise StorageConnectionError(
                    f"Failed to create GCS client: {e}"
                ) from e

    def upload(
        self,
        local_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload a file to GCS"""
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = self._make_key(key)

        try:
            blob = self._bucket.blob(full_key)

            # Add metadata if provided
            if metadata:
                blob.metadata = metadata

            blob.upload_from_filename(str(local_path))

            return full_key

        except NotFound as e:
            raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
        except (Forbidden, Unauthenticated) as e:
            raise StorageAuthError(f"Access denied: {e}") from e
        except GoogleCloudError as e:
            raise StorageError(f"Failed to upload {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error uploading {key}: {e}") from e

    def download(
        self,
        key: str,
        local_path: Path,
    ) -> Path:
        """Download a file from GCS"""
        full_key = self._make_key(key)

        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob = self._bucket.blob(full_key)

            # Check if blob exists
            if not blob.exists():
                raise KeyError(f"Object not found: {key}")

            blob.download_to_filename(str(local_path))

            return local_path

        except KeyError:
            raise  # Re-raise KeyError as-is
        except NotFound as e:
            raise KeyError(f"Object not found: {key}") from e
        except (Forbidden, Unauthenticated) as e:
            raise StorageAuthError(f"Access denied: {e}") from e
        except GoogleCloudError as e:
            raise StorageError(f"Failed to download {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error downloading {key}: {e}") from e

    def list(
        self,
        prefix: str = "",
        max_keys: Optional[int] = None,
    ) -> List[StorageObject]:
        """List objects in GCS"""
        full_prefix = self._make_key(prefix)

        try:
            # Get blobs with prefix
            blobs = self._client.list_blobs(
                self.bucket,
                prefix=full_prefix,
                max_results=max_keys,
            )

            objects = []
            for blob in blobs:
                # Remove bucket prefix to get relative key
                key = blob.name
                if self.prefix and key.startswith(self.prefix):
                    key = key[len(self.prefix):]

                objects.append(StorageObject(
                    key=key,
                    size=blob.size,
                    last_modified=blob.updated,
                    etag=blob.etag,
                    metadata=blob.metadata,
                ))

            return objects

        except NotFound as e:
            raise StorageError(f"Bucket '{self.bucket}' does not exist") from e
        except (Forbidden, Unauthenticated) as e:
            raise StorageAuthError(f"Access denied: {e}") from e
        except GoogleCloudError as e:
            raise StorageError(f"Failed to list objects: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error listing objects: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete an object from GCS"""
        full_key = self._make_key(key)

        try:
            blob = self._bucket.blob(full_key)

            # Check if object exists first
            if not blob.exists():
                return False

            blob.delete()
            return True

        except NotFound:
            return False
        except (Forbidden, Unauthenticated) as e:
            raise StorageAuthError(f"Access denied: {e}") from e
        except GoogleCloudError as e:
            raise StorageError(f"Failed to delete {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error deleting {key}: {e}") from e

    def exists(self, key: str) -> bool:
        """Check if an object exists in GCS"""
        full_key = self._make_key(key)

        try:
            blob = self._bucket.blob(full_key)
            return blob.exists()

        except NotFound:
            return False
        except (Forbidden, Unauthenticated) as e:
            raise StorageAuthError(f"Access denied: {e}") from e
        except GoogleCloudError as e:
            raise StorageError(f"Failed to check existence of {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error checking {key}: {e}") from e

    def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata for an object without downloading it"""
        full_key = self._make_key(key)

        try:
            blob = self._bucket.blob(full_key)

            # Reload to get metadata
            if not blob.exists():
                return None

            blob.reload()

            # Remove bucket prefix to get relative key
            relative_key = key

            return StorageObject(
                key=relative_key,
                size=blob.size,
                last_modified=blob.updated,
                etag=blob.etag,
                metadata=blob.metadata,
            )

        except NotFound:
            return None
        except (Forbidden, Unauthenticated) as e:
            raise StorageAuthError(f"Access denied: {e}") from e
        except GoogleCloudError as e:
            raise StorageError(f"Failed to get metadata for {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error getting metadata for {key}: {e}") from e


# Azure Blob Storage imports
try:
    from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
    from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class AzureBackend(StorageBackend):
    """
    Azure Blob Storage backend

    Supports Azure Blob Storage with authentication via:
    - Connection string (connection_string)
    - Account name + account key (account_name, account_key)
    - AZURE_STORAGE_CONNECTION_STRING environment variable
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
    ):
        """
        Initialize Azure Blob Storage backend

        Args:
            bucket: Azure container name (called bucket for consistency with other backends)
            prefix: Optional prefix for all keys (e.g., "backups/")
            connection_string: Azure storage connection string
            account_name: Azure storage account name
            account_key: Azure storage account key
        """
        super().__init__(bucket, prefix)

        if not AZURE_AVAILABLE:
            raise ImportError(
                "azure-storage-blob package required for Azure backend. "
                "Install with: pip install azure-storage-blob"
            )

        self.connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

        # Initialize Azure Blob Storage client
        self._client = self._create_client()

    def _create_client(self) -> Any:
        """Create and configure Azure Blob Storage client"""
        try:
            # Prefer connection string if available
            if self.connection_string:
                return BlobServiceClient.from_connection_string(self.connection_string)
            elif self.account_name and self.account_key:
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                return BlobServiceClient(
                    account_url=account_url,
                    credential=self.account_key,
                )
            else:
                raise StorageAuthError(
                    "No Azure credentials found. Set AZURE_STORAGE_CONNECTION_STRING "
                    "environment variable or pass connection_string parameter, or provide "
                    "account_name and account_key parameters."
                )

        except Exception as e:
            if isinstance(e, StorageAuthError):
                raise
            raise StorageConnectionError(
                f"Failed to create Azure client: {e}"
            ) from e

    def upload(
        self,
        local_path: Path,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload a file to Azure Blob Storage"""
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_key = self._make_key(key)

        try:
            blob_client = self._client.get_blob_client(
                container=self.bucket,
                blob=full_key,
            )

            with open(local_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    metadata=metadata,
                )

            return full_key

        except ResourceNotFoundError as e:
            raise StorageError(f"Container '{self.bucket}' does not exist") from e
        except HttpResponseError as e:
            if e.status_code in [401, 403]:
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
        """Download a file from Azure Blob Storage"""
        full_key = self._make_key(key)

        try:
            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            blob_client = self._client.get_blob_client(
                container=self.bucket,
                blob=full_key,
            )

            with open(local_path, "wb") as data:
                download_stream = blob_client.download_blob()
                data.write(download_stream.readall())

            return local_path

        except ResourceNotFoundError as e:
            raise KeyError(f"Object not found: {key}") from e
        except HttpResponseError as e:
            if e.status_code in [401, 403]:
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
        """List objects in Azure Blob Storage"""
        full_prefix = self._make_key(prefix)

        try:
            container_client = self._client.get_container_client(self.bucket)

            # List blobs with prefix
            blobs = container_client.list_blobs(
                name_starts_with=full_prefix,
            )

            objects = []
            for blob in blobs:
                # Remove bucket prefix to get relative key
                key = blob.name
                if self.prefix and key.startswith(self.prefix):
                    key = key[len(self.prefix):]

                objects.append(StorageObject(
                    key=key,
                    size=blob.size,
                    last_modified=blob.last_modified,
                    etag=blob.etag.strip('"') if blob.etag else None,
                    metadata=blob.metadata,
                ))

                # Apply max_keys limit if specified
                if max_keys is not None and len(objects) >= max_keys:
                    break

            return objects

        except ResourceNotFoundError as e:
            raise StorageError(f"Container '{self.bucket}' does not exist") from e
        except HttpResponseError as e:
            if e.status_code in [401, 403]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to list objects: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error listing objects: {e}") from e

    def delete(self, key: str) -> bool:
        """Delete an object from Azure Blob Storage"""
        full_key = self._make_key(key)

        try:
            blob_client = self._client.get_blob_client(
                container=self.bucket,
                blob=full_key,
            )

            # Check if object exists first
            if not blob_client.exists():
                return False

            blob_client.delete_blob()
            return True

        except ResourceNotFoundError:
            return False
        except HttpResponseError as e:
            if e.status_code in [401, 403]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to delete {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error deleting {key}: {e}") from e

    def exists(self, key: str) -> bool:
        """Check if an object exists in Azure Blob Storage"""
        full_key = self._make_key(key)

        try:
            blob_client = self._client.get_blob_client(
                container=self.bucket,
                blob=full_key,
            )
            return blob_client.exists()

        except ResourceNotFoundError:
            return False
        except HttpResponseError as e:
            if e.status_code in [401, 403]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to check existence of {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error checking {key}: {e}") from e

    def get_metadata(self, key: str) -> Optional[StorageObject]:
        """Get metadata for an object without downloading it"""
        full_key = self._make_key(key)

        try:
            blob_client = self._client.get_blob_client(
                container=self.bucket,
                blob=full_key,
            )

            # Check if blob exists
            if not blob_client.exists():
                return None

            # Get blob properties
            properties = blob_client.get_blob_properties()

            # Remove bucket prefix to get relative key
            relative_key = key

            return StorageObject(
                key=relative_key,
                size=properties.size,
                last_modified=properties.last_modified,
                etag=properties.etag.strip('"') if properties.etag else None,
                metadata=properties.metadata,
            )

        except ResourceNotFoundError:
            return None
        except HttpResponseError as e:
            if e.status_code in [401, 403]:
                raise StorageAuthError(f"Access denied: {e}") from e
            else:
                raise StorageError(f"Failed to get metadata for {key}: {e}") from e
        except Exception as e:
            raise StorageError(f"Unexpected error getting metadata for {key}: {e}") from e
