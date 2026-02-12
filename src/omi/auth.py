"""
API Key Authentication and Management for OMI REST API.

Provides secure API key generation, storage, validation, and revocation.
Keys are hashed with SHA-256 before storage (plaintext never persisted).

Usage:
    manager = APIKeyManager(db_path)
    api_key = manager.generate_key("my-app", rate_limit=100)
    is_valid = manager.validate_key(api_key)
    manager.revoke_key("my-app")
"""

import sqlite3
import hashlib
import secrets
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class APIKey:
    """Represents an API key with metadata."""
    id: str
    name: str
    key_hash: str
    rate_limit: int
    created_at: datetime
    last_used: Optional[datetime] = None
    revoked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "key_hash": self.key_hash,
            "rate_limit": self.rate_limit,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "revoked": self.revoked
        }


class APIKeyManager:
    """
    Manages API keys for authentication.

    Features:
    - Secure key generation using secrets.token_urlsafe
    - SHA-256 hashing (plaintext keys never stored)
    - SQLite storage with WAL mode for concurrency
    - Per-key rate limiting configuration
    - Key revocation support
    - Thread-safe operations
    """

    def __init__(self, db_path: Path, enable_wal: bool = True):
        """
        Initialize API Key Manager.

        Args:
            db_path: Path to SQLite database file (can be shared with GraphPalace)
            enable_wal: Enable WAL mode for concurrent writes (default: True)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._enable_wal = enable_wal

        # Create persistent connection
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row

        # Thread lock for serializing database operations
        self._db_lock = threading.Lock()

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema with indexes."""
        with self._db_lock:
            # Enable WAL mode for concurrent writes
            if self._enable_wal:
                self._conn.execute("PRAGMA journal_mode=WAL")

            # Foreign key constraints
            self._conn.execute("PRAGMA foreign_keys=ON")

            # Create api_keys table
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    key_hash TEXT NOT NULL UNIQUE,
                    rate_limit INTEGER NOT NULL DEFAULT 100,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    revoked INTEGER DEFAULT 0 CHECK(revoked IN (0, 1))
                );

                -- Indexes for performance
                CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
                CREATE INDEX IF NOT EXISTS idx_api_keys_name ON api_keys(name);
                CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked);
            """)

    def _hash_key(self, api_key: str) -> str:
        """
        Hash an API key using SHA-256.

        Args:
            api_key: The plaintext API key

        Returns:
            str: Hexadecimal SHA-256 hash
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def generate_key(self, name: str, rate_limit: int = 100) -> str:
        """
        Generate a new API key.

        Args:
            name: Human-readable name for the key (must be unique)
            rate_limit: Requests per minute allowed (default: 100)

        Returns:
            str: The generated API key (only shown once, not stored)

        Raises:
            ValueError: If a key with this name already exists
        """
        # Generate secure random key (32 bytes = 43 chars in base64)
        api_key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(api_key)
        key_id = secrets.token_urlsafe(16)

        with self._db_lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO api_keys (id, name, key_hash, rate_limit, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key_id, name, key_hash, rate_limit, datetime.now().isoformat())
                )
            except sqlite3.IntegrityError as e:
                if "name" in str(e).lower():
                    raise ValueError(f"API key with name '{name}' already exists")
                elif "key_hash" in str(e).lower():
                    # Extremely unlikely, but handle hash collision
                    raise ValueError("Key generation collision (retry)")
                raise

        return api_key

    def validate_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return its metadata.

        Args:
            api_key: The API key to validate

        Returns:
            APIKey object if valid and not revoked, None otherwise
        """
        key_hash = self._hash_key(api_key)

        with self._db_lock:
            row = self._conn.execute(
                """
                SELECT id, name, key_hash, rate_limit, created_at, last_used, revoked
                FROM api_keys
                WHERE key_hash = ? AND revoked = 0
                """,
                (key_hash,)
            ).fetchone()

            if row is None:
                return None

            # Update last_used timestamp
            self._conn.execute(
                "UPDATE api_keys SET last_used = ? WHERE key_hash = ?",
                (datetime.now().isoformat(), key_hash)
            )

            return APIKey(
                id=row["id"],
                name=row["name"],
                key_hash=row["key_hash"],
                rate_limit=row["rate_limit"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
                revoked=bool(row["revoked"])
            )

    def revoke_key(self, name: Optional[str] = None, api_key: Optional[str] = None) -> bool:
        """
        Revoke an API key by name or key value.

        Args:
            name: Name of the key to revoke
            api_key: The API key value to revoke

        Returns:
            bool: True if a key was revoked, False if not found

        Raises:
            ValueError: If neither name nor api_key is provided
        """
        if name is None and api_key is None:
            raise ValueError("Must provide either name or api_key")

        with self._db_lock:
            if name is not None:
                cursor = self._conn.execute(
                    "UPDATE api_keys SET revoked = 1 WHERE name = ? AND revoked = 0",
                    (name,)
                )
            else:
                key_hash = self._hash_key(api_key)
                cursor = self._conn.execute(
                    "UPDATE api_keys SET revoked = 1 WHERE key_hash = ? AND revoked = 0",
                    (key_hash,)
                )

            return cursor.rowcount > 0

    def list_keys(self, include_revoked: bool = False) -> List[APIKey]:
        """
        List all API keys.

        Args:
            include_revoked: Include revoked keys in the list (default: False)

        Returns:
            List of APIKey objects
        """
        with self._db_lock:
            if include_revoked:
                rows = self._conn.execute(
                    """
                    SELECT id, name, key_hash, rate_limit, created_at, last_used, revoked
                    FROM api_keys
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT id, name, key_hash, rate_limit, created_at, last_used, revoked
                    FROM api_keys
                    WHERE revoked = 0
                    ORDER BY created_at DESC
                    """
                ).fetchall()

            return [
                APIKey(
                    id=row["id"],
                    name=row["name"],
                    key_hash=row["key_hash"],
                    rate_limit=row["rate_limit"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
                    revoked=bool(row["revoked"])
                )
                for row in rows
            ]

    def get_key_by_name(self, name: str) -> Optional[APIKey]:
        """
        Get an API key by its name.

        Args:
            name: Name of the key

        Returns:
            APIKey object if found, None otherwise
        """
        with self._db_lock:
            row = self._conn.execute(
                """
                SELECT id, name, key_hash, rate_limit, created_at, last_used, revoked
                FROM api_keys
                WHERE name = ?
                """,
                (name,)
            ).fetchone()

            if row is None:
                return None

            return APIKey(
                id=row["id"],
                name=row["name"],
                key_hash=row["key_hash"],
                rate_limit=row["rate_limit"],
                created_at=datetime.fromisoformat(row["created_at"]),
                last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
                revoked=bool(row["revoked"])
            )

    def close(self) -> None:
        """Close the database connection."""
        with self._db_lock:
            self._conn.close()


class RateLimiter:
    """
    Sliding window rate limiter for API requests.

    Features:
    - In-memory storage using deque for efficient FIFO operations
    - Sliding window algorithm (60 second window)
    - Thread-safe operations
    - Automatic cleanup of expired timestamps
    - Per-API-key rate limiting

    Usage:
        limiter = RateLimiter()
        allowed, retry_after = limiter.check_rate_limit("api_key_123", limit=100)
        if not allowed:
            # Return 429 with Retry-After header
            pass
    """

    def __init__(self, window_seconds: int = 60):
        """
        Initialize the rate limiter.

        Args:
            window_seconds: Size of the sliding window in seconds (default: 60)
        """
        self.window_seconds = window_seconds
        # Map of api_key -> deque of timestamps (float from time.time())
        self._requests: Dict[str, deque] = {}
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

    def check_rate_limit(self, api_key: str, limit: int) -> Tuple[bool, int]:
        """
        Check if a request is allowed under the rate limit.

        Implements a sliding window algorithm:
        1. Remove timestamps older than window_seconds
        2. Check if number of remaining timestamps < limit
        3. If allowed, add current timestamp
        4. Return (allowed, retry_after)

        Args:
            api_key: The API key making the request
            limit: Maximum requests allowed per window

        Returns:
            Tuple of (allowed: bool, retry_after: int):
            - allowed: True if request is within rate limit
            - retry_after: Seconds until rate limit resets (0 if allowed, >0 if denied)
        """
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - self.window_seconds

            # Initialize deque for new API keys
            if api_key not in self._requests:
                self._requests[api_key] = deque()

            request_times = self._requests[api_key]

            # Remove timestamps outside the sliding window
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()

            # Check if under rate limit
            if len(request_times) < limit:
                # Allow the request and record timestamp
                request_times.append(current_time)
                return (True, 0)
            else:
                # Rate limit exceeded - calculate retry_after
                # The oldest request in the window will expire at: oldest_time + window_seconds
                # So retry_after = (oldest_time + window_seconds) - current_time
                oldest_time = request_times[0]
                retry_after = int(oldest_time + self.window_seconds - current_time) + 1
                return (False, retry_after)

    def reset(self, api_key: Optional[str] = None) -> None:
        """
        Reset rate limit tracking for an API key or all keys.

        Args:
            api_key: Specific API key to reset, or None to reset all keys
        """
        with self._lock:
            if api_key is None:
                self._requests.clear()
            elif api_key in self._requests:
                del self._requests[api_key]

    def get_remaining(self, api_key: str, limit: int) -> int:
        """
        Get the number of remaining requests for an API key.

        Args:
            api_key: The API key to check
            limit: Maximum requests allowed per window

        Returns:
            int: Number of requests remaining before hitting rate limit
        """
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - self.window_seconds

            if api_key not in self._requests:
                return limit

            request_times = self._requests[api_key]

            # Remove expired timestamps
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()

            return max(0, limit - len(request_times))
