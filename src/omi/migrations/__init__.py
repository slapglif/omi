"""
OMI Migration System

Handles SQLite schema changes between OMI versions with automatic backup,
version tracking, and rollback support.

Key Features:
- Schema version tracking via PRAGMA user_version
- Automatic backup before migrations
- Migration history in _migrations table
- Dry-run support
- Rollback on failure
"""

from .migration_base import MigrationBase
from .registry import MigrationRegistry

__all__ = ["MigrationBase", "MigrationRegistry"]
