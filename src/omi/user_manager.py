"""
User Management for RBAC System

This module handles Create, Read, Update, Delete operations for users:
- create_user: Create new users with username and email
- assign_role: Assign roles to users (global or namespace-specific)
- revoke_role: Remove roles from users
- list_users: List all users in the system
- get_user_permissions: Get all permissions for a user
- create_api_key: Generate API keys for users
- revoke_api_key: Revoke API keys
"""

import sqlite3
import json
import hashlib
import uuid
import secrets
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .storage.schema import init_database
from .rbac import Role, RBACManager


@dataclass
class User:
    """
    User model for RBAC system.

    Attributes:
        id: Unique user identifier (UUID)
        username: Unique username for login
        email: User email address (optional)
        created_at: User creation timestamp
    """
    id: str
    username: str
    email: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class APIKey:
    """
    API Key model for authentication.

    Attributes:
        id: Unique key identifier (UUID)
        key_hash: SHA-256 hash of the API key
        user_id: User who owns this key
        created_at: Key creation timestamp
        last_used: Last time key was used (optional)
    """
    id: str
    key_hash: str
    user_id: str
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "key_hash": self.key_hash,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


class UserManager:
    """
    User management operations for RBAC system.

    Handles basic create, read, update, delete operations for users
    with support for:
    - User creation with unique username validation
    - Role assignment (global and namespace-specific)
    - API key generation and revocation
    - Thread-safe database operations
    - Integration with RBACManager for permission checking
    """

    def __init__(self, db_path: str, enable_wal: bool = True, conn: Optional[sqlite3.Connection] = None):
        """
        Initialize User Manager.

        Args:
            db_path: Path to SQLite database file (or ':memory:' for in-memory)
            enable_wal: Enable WAL mode for concurrent writes (default: True)
            conn: Optional shared connection (for :memory: databases in facade pattern)
        """
        self.db_path = Path(db_path) if db_path != ':memory:' else db_path
        self._owns_connection = conn is None

        if conn is not None:
            # Use shared connection (facade pattern)
            self._conn = conn
            self._db_lock = threading.Lock()
        else:
            # Create parent directory if needed (skip for :memory:)
            if self.db_path != ':memory:':
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create persistent connection
            # check_same_thread=False allows multi-threaded access (safe with WAL mode)
            # isolation_level=None enables autocommit mode for better concurrency
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
                timeout=30.0
            )

            # Thread lock for serializing database operations
            self._db_lock = threading.Lock()

            # Initialize database schema
            init_database(self._conn, enable_wal)

        # Initialize RBAC manager (note: for :memory: databases, RBACManager
        # creates its own connection, so roles need to be initialized here too)
        self.rbac = RBACManager(db_path)

        # For :memory: databases, we need to ensure roles are initialized in our connection
        if db_path == ':memory:':
            self._init_default_roles()

    def _init_default_roles(self) -> None:
        """
        Initialize default roles in database if they don't exist.

        Creates the four built-in roles: admin, developer, reader, auditor
        This is needed for :memory: databases where each connection is separate.
        """
        roles_data = [
            ("admin", "admin", "Full access to all operations"),
            ("developer", "developer", "Read-write access to own namespaces"),
            ("reader", "reader", "Read-only access"),
            ("auditor", "auditor", "Read-only access + security reports"),
        ]

        with self._db_lock:
            for role_id, role_name, description in roles_data:
                self._conn.execute(
                    "INSERT OR IGNORE INTO roles (id, name, description) VALUES (?, ?, ?)",
                    (role_id, role_name, description)
                )
            self._conn.commit()

    def create_user(self, username: str, email: Optional[str] = None) -> str:
        """
        Create a new user.

        Args:
            username: Unique username for the user
            email: User email address (optional)

        Returns:
            user_id: UUID of the created user

        Raises:
            ValueError: If username already exists or is invalid
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")

        user_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Use lock for thread-safe database access
        with self._db_lock:
            try:
                self._conn.execute("""
                    INSERT INTO users (id, username, email, created_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, username.strip(), email, now))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: users.username" in str(e):
                    raise ValueError(f"Username '{username}' already exists")
                raise

        return user_id

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Retrieve a user by ID.

        Args:
            user_id: UUID of the user

        Returns:
            User object or None if not found
        """
        cursor = self._conn.execute("""
            SELECT id, username, email, created_at
            FROM users WHERE id = ?
        """, (user_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return User(
            id=row[0],
            username=row[1],
            email=row[2],
            created_at=datetime.fromisoformat(row[3]) if row[3] else None
        )

    def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Retrieve a user by username.

        Args:
            username: Username to search for

        Returns:
            User object or None if not found
        """
        cursor = self._conn.execute("""
            SELECT id, username, email, created_at
            FROM users WHERE username = ?
        """, (username,))

        row = cursor.fetchone()
        if not row:
            return None

        return User(
            id=row[0],
            username=row[1],
            email=row[2],
            created_at=datetime.fromisoformat(row[3]) if row[3] else None
        )

    def list_users(self) -> List[User]:
        """
        List all users in the system.

        Returns:
            List of User objects
        """
        cursor = self._conn.execute("""
            SELECT id, username, email, created_at
            FROM users
            ORDER BY username
        """)

        users = []
        for row in cursor:
            users.append(User(
                id=row[0],
                username=row[1],
                email=row[2],
                created_at=datetime.fromisoformat(row[3]) if row[3] else None
            ))

        return users

    def assign_role(self, user_id: str, role: str, namespace: Optional[str] = None) -> str:
        """
        Assign a role to a user.

        Args:
            user_id: UUID of the user
            role: Role name (admin, developer, reader, auditor)
            namespace: Optional namespace for namespace-specific roles

        Returns:
            user_role_id: UUID of the created user_role assignment

        Raises:
            ValueError: If user doesn't exist or role is invalid
        """
        # Validate role
        try:
            role_enum = Role.from_string(role)
        except ValueError as e:
            raise ValueError(str(e))

        # Check if user exists
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User with id '{user_id}' not found")

        user_role_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        role_id = role_enum.value  # Role IDs match role names

        with self._db_lock:
            try:
                self._conn.execute("""
                    INSERT INTO user_roles (id, user_id, role_id, namespace, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_role_id, user_id, role_id, namespace, now))
                self._conn.commit()
            except sqlite3.IntegrityError as e:
                if "FOREIGN KEY constraint failed" in str(e):
                    raise ValueError(f"Invalid user_id or role_id")
                raise

        return user_role_id

    def revoke_role(self, user_id: str, role: str, namespace: Optional[str] = None) -> bool:
        """
        Revoke a role from a user.

        Args:
            user_id: UUID of the user
            role: Role name to revoke
            namespace: Optional namespace (if None, revokes global role)

        Returns:
            True if role was revoked, False if assignment didn't exist

        Raises:
            ValueError: If role is invalid
        """
        # Validate role
        try:
            role_enum = Role.from_string(role)
        except ValueError as e:
            raise ValueError(str(e))

        role_id = role_enum.value

        with self._db_lock:
            if namespace:
                cursor = self._conn.execute("""
                    DELETE FROM user_roles
                    WHERE user_id = ? AND role_id = ? AND namespace = ?
                """, (user_id, role_id, namespace))
            else:
                cursor = self._conn.execute("""
                    DELETE FROM user_roles
                    WHERE user_id = ? AND role_id = ? AND namespace IS NULL
                """, (user_id, role_id))

            self._conn.commit()
            return cursor.rowcount > 0

    def get_user_roles(self, user_id: str, namespace: Optional[str] = None) -> List[Tuple[str, Optional[str]]]:
        """
        Get all roles assigned to a user.

        Args:
            user_id: UUID of the user
            namespace: Optional namespace filter

        Returns:
            List of (role_name, namespace) tuples
        """
        if namespace:
            cursor = self._conn.execute("""
                SELECT r.name, ur.namespace
                FROM user_roles ur
                JOIN roles r ON ur.role_id = r.id
                WHERE ur.user_id = ? AND ur.namespace = ?
                ORDER BY r.name
            """, (user_id, namespace))
        else:
            cursor = self._conn.execute("""
                SELECT r.name, ur.namespace
                FROM user_roles ur
                JOIN roles r ON ur.role_id = r.id
                WHERE ur.user_id = ?
                ORDER BY r.name
            """, (user_id,))

        return [(row[0], row[1]) for row in cursor]

    def get_user_permissions(self, user_id: str, namespace: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get all permissions for a user.

        Args:
            user_id: UUID of the user
            namespace: Optional namespace filter

        Returns:
            List of permission dictionaries with 'role', 'action', 'resource' keys
        """
        # Get user's roles
        roles = self.get_user_roles(user_id, namespace)

        # Import permission matrix from RBACManager
        from .rbac import RBACManager, Role

        permissions = []
        for role_name, role_namespace in roles:
            try:
                role_enum = Role.from_string(role_name)
                role_permissions = RBACManager.PERMISSION_MATRIX.get(role_enum, [])
                for action, resource in role_permissions:
                    permissions.append({
                        "role": role_name,
                        "action": action,
                        "resource": resource
                    })
            except ValueError:
                # Skip invalid roles
                pass

        return permissions

    def create_api_key(self, user_id: str) -> Tuple[str, str]:
        """
        Create an API key for a user.

        Args:
            user_id: UUID of the user

        Returns:
            Tuple of (key_id, api_key)
            - key_id: UUID of the API key record
            - api_key: The actual API key string (only shown once)

        Raises:
            ValueError: If user doesn't exist
        """
        # Check if user exists
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User with id '{user_id}' not found")

        # Generate secure random API key (32 bytes = 64 hex chars)
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._db_lock:
            self._conn.execute("""
                INSERT INTO api_keys (id, key_hash, user_id, created_at)
                VALUES (?, ?, ?, ?)
            """, (key_id, key_hash, user_id, now))
            self._conn.commit()

        return (key_id, api_key)

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: UUID of the API key to revoke

        Returns:
            True if key was revoked, False if key didn't exist
        """
        with self._db_lock:
            cursor = self._conn.execute("""
                DELETE FROM api_keys WHERE id = ?
            """, (key_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def get_api_keys(self, user_id: str) -> List[APIKey]:
        """
        Get all API keys for a user.

        Args:
            user_id: UUID of the user

        Returns:
            List of APIKey objects (without the actual key values)
        """
        cursor = self._conn.execute("""
            SELECT id, key_hash, user_id, created_at, last_used
            FROM api_keys
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,))

        api_keys = []
        for row in cursor:
            api_keys.append(APIKey(
                id=row[0],
                key_hash=row[1],
                user_id=row[2],
                created_at=datetime.fromisoformat(row[3]) if row[3] else None,
                last_used=datetime.fromisoformat(row[4]) if row[4] else None
            ))

        return api_keys

    def verify_api_key(self, api_key: str) -> Optional[User]:
        """
        Verify an API key and return the associated user.

        Args:
            api_key: The API key string to verify

        Returns:
            User object if key is valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Update last_used timestamp
        with self._db_lock:
            cursor = self._conn.execute("""
                SELECT user_id FROM api_keys WHERE key_hash = ?
            """, (key_hash,))

            row = cursor.fetchone()
            if not row:
                return None

            user_id = row[0]

            # Update last_used
            self._conn.execute("""
                UPDATE api_keys
                SET last_used = ?
                WHERE key_hash = ?
            """, (datetime.now().isoformat(), key_hash))
            self._conn.commit()

        return self.get_user(user_id)

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user and all associated data.

        Args:
            user_id: UUID of the user to delete

        Returns:
            True if user was deleted, False if user didn't exist

        Note:
            This will cascade delete all user_roles and api_keys due to
            foreign key constraints with ON DELETE CASCADE.
        """
        with self._db_lock:
            cursor = self._conn.execute("""
                DELETE FROM users WHERE id = ?
            """, (user_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection if owned by this instance."""
        if self._owns_connection and self._conn:
            self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
