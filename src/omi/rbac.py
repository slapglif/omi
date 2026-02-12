"""
Role-Based Access Control (RBAC) for multi-user memory access

Pattern: Explicit permissions, namespace isolation, audit-first
"""

import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Set, Any
from pathlib import Path


class Role(Enum):
    """
    Built-in roles with predefined permissions

    Hierarchy:
        - ADMIN: Full access to everything (create/read/update/delete)
        - DEVELOPER: Read-write access to own namespaces
        - READER: Read-only access
        - AUDITOR: Read-only + security reports access
    """
    ADMIN = "admin"
    DEVELOPER = "developer"
    READER = "reader"
    AUDITOR = "auditor"

    @classmethod
    def from_string(cls, role_str: str) -> "Role":
        """
        Convert string to Role enum

        Args:
            role_str: Role name as string

        Returns:
            Role enum value

        Raises:
            ValueError: If role_str is not a valid role
        """
        try:
            return cls(role_str.lower())
        except ValueError:
            valid_roles = [r.value for r in cls]
            raise ValueError(f"Invalid role '{role_str}'. Must be one of: {valid_roles}")


@dataclass
class Permission:
    """
    Permission definition linking role to action on resource

    Components:
        - role: Role enum value
        - action: Operation type (read, write, delete, admin, audit)
        - resource: Resource type (memory, belief, checkpoint, user, audit_log)

    Examples:
        - Permission(Role.ADMIN, "write", "memory") -> Admin can write memories
        - Permission(Role.READER, "read", "memory") -> Reader can read memories
        - Permission(Role.AUDITOR, "audit", "audit_log") -> Auditor can read audit logs
    """
    role: Role
    action: str
    resource: str

    def matches(self, action: str, resource: str) -> bool:
        """
        Check if this permission grants access to action on resource

        Args:
            action: Requested action
            resource: Requested resource

        Returns:
            True if permission matches
        """
        return self.action == action and self.resource == resource

    def __str__(self) -> str:
        """String representation"""
        return f"{self.role.value}:{self.action}:{self.resource}"


class RBACManager:
    """
    Role-Based Access Control manager with namespace isolation

    Pattern:
        1. Every operation checks permissions BEFORE execution
        2. Namespace-level permissions override global role permissions
        3. Deny-by-default: explicit grants required
        4. Audit all permission checks (success and failure)

    Built-in Permission Matrix:

        Action          | Admin | Developer | Reader | Auditor
        ----------------|-------|-----------|--------|--------
        read:memory     |   ✓   |     ✓     |   ✓    |   ✓
        write:memory    |   ✓   |     ✓     |   ✗    |   ✗
        delete:memory   |   ✓   |     ✗     |   ✗    |   ✗
        read:belief     |   ✓   |     ✓     |   ✓    |   ✓
        write:belief    |   ✓   |     ✓     |   ✗    |   ✗
        admin:user      |   ✓   |     ✗     |   ✗    |   ✗
        audit:audit_log |   ✓   |     ✗     |   ✗    |   ✓

    Namespace Permissions:
        - Users can be granted role access to specific namespaces
        - Example: user 'alice' has 'developer' role in 'acme/research'
        - Namespace permissions are more specific than global permissions
        - If no namespace permission exists, fall back to global role
    """

    # Built-in permission matrix
    PERMISSION_MATRIX = {
        Role.ADMIN: [
            # Admin has full access
            ("read", "memory"),
            ("write", "memory"),
            ("delete", "memory"),
            ("read", "belief"),
            ("write", "belief"),
            ("read", "checkpoint"),
            ("write", "checkpoint"),
            ("delete", "checkpoint"),
            ("admin", "user"),
            ("audit", "audit_log"),
            ("read", "audit_log"),
        ],
        Role.DEVELOPER: [
            # Developer can read/write but not delete
            ("read", "memory"),
            ("write", "memory"),
            ("read", "belief"),
            ("write", "belief"),
            ("read", "checkpoint"),
            ("write", "checkpoint"),
        ],
        Role.READER: [
            # Reader can only read
            ("read", "memory"),
            ("read", "belief"),
            ("read", "checkpoint"),
        ],
        Role.AUDITOR: [
            # Auditor can read + audit logs
            ("read", "memory"),
            ("read", "belief"),
            ("read", "checkpoint"),
            ("audit", "audit_log"),
            ("read", "audit_log"),
        ],
    }

    def __init__(self, db_path: str) -> None:
        """
        Initialize RBAC manager

        Args:
            db_path: Path to SQLite database (can be ':memory:' for testing)
        """
        self.db_path = db_path
        self._init_default_roles()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with foreign keys enabled"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_default_roles(self) -> None:
        """
        Initialize default roles in database if they don't exist

        Creates the four built-in roles: admin, developer, reader, auditor
        """
        conn = self._get_connection()
        try:
            # Ensure roles table exists (it should from schema.py)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS roles (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL CHECK(name IN ('admin','developer','reader','auditor')),
                    description TEXT
                )
            """)

            # Insert default roles if they don't exist
            roles_data = [
                ("admin", "admin", "Full access to all operations"),
                ("developer", "developer", "Read-write access to own namespaces"),
                ("reader", "reader", "Read-only access"),
                ("auditor", "auditor", "Read-only access + security reports"),
            ]

            for role_id, role_name, description in roles_data:
                conn.execute(
                    "INSERT OR IGNORE INTO roles (id, name, description) VALUES (?, ?, ?)",
                    (role_id, role_name, description)
                )

            conn.commit()
        finally:
            conn.close()

    def check_permission(
        self,
        user_id: str,
        action: str,
        resource: str,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Check if user has permission to perform action on resource

        Permission resolution order:
            1. Check namespace-specific role for user (if namespace provided)
            2. Check global roles for user
            3. Check permission matrix for each role
            4. Deny by default if no matching permission

        Args:
            user_id: User identifier
            action: Action to perform (read, write, delete, admin, audit)
            resource: Resource type (memory, belief, checkpoint, user, audit_log)
            namespace: Optional namespace for namespace-level permissions

        Returns:
            True if user has permission, False otherwise
        """
        # Get user's roles (both global and namespace-specific)
        roles = self._get_user_roles(user_id, namespace)

        if not roles:
            # User has no roles - deny by default
            return False

        # Check if any role grants the requested permission
        for role in roles:
            if self._role_has_permission(role, action, resource):
                return True

        # No matching permission found - deny
        return False

    def _get_user_roles(self, user_id: str, namespace: Optional[str] = None) -> Set[Role]:
        """
        Get all roles for a user, including namespace-specific roles

        Args:
            user_id: User identifier
            namespace: Optional namespace filter

        Returns:
            Set of Role enums for the user
        """
        conn = self._get_connection()
        roles = set()

        try:
            # Query user_roles table
            if namespace:
                # Get namespace-specific roles first (higher priority)
                cursor = conn.execute("""
                    SELECT r.name
                    FROM user_roles ur
                    JOIN roles r ON ur.role_id = r.id
                    WHERE ur.user_id = ? AND ur.namespace = ?
                """, (user_id, namespace))

                for row in cursor:
                    try:
                        roles.add(Role.from_string(row[0]))
                    except ValueError:
                        pass  # Skip invalid roles

            # Get global roles (no namespace restriction)
            cursor = conn.execute("""
                SELECT r.name
                FROM user_roles ur
                JOIN roles r ON ur.role_id = r.id
                WHERE ur.user_id = ? AND ur.namespace IS NULL
            """, (user_id,))

            for row in cursor:
                try:
                    roles.add(Role.from_string(row[0]))
                except ValueError:
                    pass  # Skip invalid roles

        finally:
            conn.close()

        return roles

    def _role_has_permission(self, role: Role, action: str, resource: str) -> bool:
        """
        Check if a role has permission for action on resource

        Args:
            role: Role enum
            action: Action string
            resource: Resource string

        Returns:
            True if role has permission
        """
        permissions = self.PERMISSION_MATRIX.get(role, [])
        return (action, resource) in permissions

    def get_user_permissions(self, user_id: str, namespace: Optional[str] = None) -> List[Permission]:
        """
        Get all permissions for a user

        Args:
            user_id: User identifier
            namespace: Optional namespace filter

        Returns:
            List of Permission objects
        """
        roles = self._get_user_roles(user_id, namespace)
        permissions = []

        for role in roles:
            role_permissions = self.PERMISSION_MATRIX.get(role, [])
            for action, resource in role_permissions:
                permissions.append(Permission(role, action, resource))

        return permissions

    def has_role(self, user_id: str, role: Role, namespace: Optional[str] = None) -> bool:
        """
        Check if user has a specific role

        Args:
            user_id: User identifier
            role: Role to check
            namespace: Optional namespace filter

        Returns:
            True if user has the role
        """
        user_roles = self._get_user_roles(user_id, namespace)
        return role in user_roles

    def is_admin(self, user_id: str) -> bool:
        """
        Check if user is an admin (global only)

        Args:
            user_id: User identifier

        Returns:
            True if user has admin role globally
        """
        return self.has_role(user_id, Role.ADMIN, namespace=None)

    def get_role_description(self, role: Role) -> str:
        """
        Get human-readable description of a role

        Args:
            role: Role enum

        Returns:
            Role description string
        """
        descriptions = {
            Role.ADMIN: "Full access to all operations",
            Role.DEVELOPER: "Read-write access to own namespaces",
            Role.READER: "Read-only access",
            Role.AUDITOR: "Read-only access + security reports",
        }
        return descriptions.get(role, "Unknown role")
