"""
Unit tests for Migration Manager and Backup Manager

Tests cover:
- Schema version tracking via PRAGMA user_version
- Migration history recording and retrieval
- Backup creation with metadata
- Restore operations
- Backup listing and cleanup
- Verification and error handling
"""

import unittest
import tempfile
import time
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omi.migrations.manager import MigrationManager, MigrationRecord
from omi.migrations.backup import BackupManager, BackupInfo


class TestMigrationManager(unittest.TestCase):
    """Test suite for MigrationManager."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_migrations.sqlite"
        self.manager = MigrationManager(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ==================== Schema Version Operations ====================

    def test_initial_schema_version(self):
        """Test that initial schema version is 0."""
        version = self.manager.get_schema_version()
        self.assertEqual(version, 0)

    def test_set_and_get_schema_version(self):
        """Test setting and retrieving schema version."""
        self.manager.set_schema_version(5)
        version = self.manager.get_schema_version()
        self.assertEqual(version, 5)

    def test_set_schema_version_negative(self):
        """Test that negative schema version raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.manager.set_schema_version(-1)
        self.assertIn("must be >= 0", str(context.exception))

    def test_schema_version_persists(self):
        """Test that schema version persists across manager instances."""
        self.manager.set_schema_version(10)

        # Create new manager instance with same database
        new_manager = MigrationManager(self.db_path)
        version = new_manager.get_schema_version()
        self.assertEqual(version, 10)

    # ==================== Migration Recording ====================

    def test_record_migration(self):
        """Test recording a migration."""
        record_id = self.manager.record_migration(
            version=1,
            description="Initial migration",
            duration_ms=150
        )

        self.assertIsNotNone(record_id)
        self.assertGreater(record_id, 0)

    def test_record_migration_with_metadata(self):
        """Test recording a migration with metadata."""
        metadata = {
            "author": "test_user",
            "tables_created": ["users", "posts"]
        }

        record_id = self.manager.record_migration(
            version=1,
            description="Migration with metadata",
            duration_ms=200,
            metadata=metadata
        )

        # Retrieve and verify metadata
        record = self.manager.get_migration_record(1)
        self.assertIsNotNone(record)
        self.assertEqual(record.metadata, metadata)

    def test_record_duplicate_migration_version(self):
        """Test that duplicate migration version raises IntegrityError."""
        self.manager.record_migration(1, "First migration")

        with self.assertRaises(sqlite3.IntegrityError):
            self.manager.record_migration(1, "Duplicate migration")

    # ==================== Migration Retrieval ====================

    def test_get_migration_record(self):
        """Test retrieving a specific migration record."""
        self.manager.record_migration(
            version=2,
            description="Test migration",
            duration_ms=100
        )

        record = self.manager.get_migration_record(2)
        self.assertIsNotNone(record)
        self.assertEqual(record.version, 2)
        self.assertEqual(record.description, "Test migration")
        self.assertEqual(record.duration_ms, 100)
        self.assertIsInstance(record.applied_at, datetime)

    def test_get_nonexistent_migration_record(self):
        """Test retrieving non-existent migration returns None."""
        record = self.manager.get_migration_record(999)
        self.assertIsNone(record)

    def test_get_applied_migrations(self):
        """Test retrieving all applied migrations in order."""
        # Record multiple migrations out of order
        self.manager.record_migration(3, "Third migration", duration_ms=300)
        self.manager.record_migration(1, "First migration", duration_ms=100)
        self.manager.record_migration(2, "Second migration", duration_ms=200)

        migrations = self.manager.get_applied_migrations()

        # Should be ordered by version ascending
        self.assertEqual(len(migrations), 3)
        self.assertEqual(migrations[0].version, 1)
        self.assertEqual(migrations[1].version, 2)
        self.assertEqual(migrations[2].version, 3)

    def test_get_applied_migrations_empty(self):
        """Test getting applied migrations when none exist."""
        migrations = self.manager.get_applied_migrations()
        self.assertEqual(len(migrations), 0)

    def test_get_last_migration(self):
        """Test retrieving the most recent migration."""
        self.manager.record_migration(1, "First", duration_ms=100)
        self.manager.record_migration(2, "Second", duration_ms=200)
        self.manager.record_migration(3, "Third", duration_ms=300)

        last = self.manager.get_last_migration()
        self.assertIsNotNone(last)
        self.assertEqual(last.version, 3)
        self.assertEqual(last.description, "Third")

    def test_get_last_migration_empty(self):
        """Test getting last migration when none exist."""
        last = self.manager.get_last_migration()
        self.assertIsNone(last)

    # ==================== Migration Status ====================

    def test_is_migration_applied(self):
        """Test checking if migration is applied."""
        self.assertFalse(self.manager.is_migration_applied(1))

        self.manager.record_migration(1, "Test migration")

        self.assertTrue(self.manager.is_migration_applied(1))
        self.assertFalse(self.manager.is_migration_applied(2))

    def test_get_migration_count(self):
        """Test getting total migration count."""
        self.assertEqual(self.manager.get_migration_count(), 0)

        self.manager.record_migration(1, "First")
        self.assertEqual(self.manager.get_migration_count(), 1)

        self.manager.record_migration(2, "Second")
        self.assertEqual(self.manager.get_migration_count(), 2)

    # ==================== History Management ====================

    def test_clear_history(self):
        """Test clearing migration history."""
        # Record some migrations
        self.manager.record_migration(1, "First")
        self.manager.record_migration(2, "Second")
        self.manager.set_schema_version(2)

        self.assertEqual(self.manager.get_migration_count(), 2)

        # Clear history
        self.manager.clear_history()

        # History should be empty, but schema version unchanged
        self.assertEqual(self.manager.get_migration_count(), 0)
        self.assertEqual(self.manager.get_schema_version(), 2)

    # ==================== MigrationRecord Serialization ====================

    def test_migration_record_to_dict(self):
        """Test MigrationRecord serialization to dictionary."""
        record_id = self.manager.record_migration(
            version=1,
            description="Test migration",
            duration_ms=150,
            metadata={"key": "value"}
        )

        record = self.manager.get_migration_record(1)
        record_dict = record.to_dict()

        self.assertEqual(record_dict["version"], 1)
        self.assertEqual(record_dict["description"], "Test migration")
        self.assertEqual(record_dict["duration_ms"], 150)
        self.assertEqual(record_dict["metadata"], {"key": "value"})
        self.assertIsInstance(record_dict["applied_at"], str)

    # ==================== WAL Mode ====================

    def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled by default."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

    def test_wal_mode_disabled(self):
        """Test that WAL mode can be disabled."""
        db_path = Path(self.temp_dir) / "no_wal.sqlite"
        manager = MigrationManager(db_path, enable_wal=False)

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            self.assertNotEqual(mode.lower(), "wal")


class TestBackupManager(unittest.TestCase):
    """Test suite for BackupManager."""

    def setUp(self):
        """Set up test database and backup manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db.sqlite"
        self.backup_dir = Path(self.temp_dir) / "backups"

        # Create a test database with some data
        self._create_test_database()

        self.backup_manager = BackupManager(self.db_path, self.backup_dir)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_database(self):
        """Create a test database with sample data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA user_version = 5")
            conn.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            """)
            conn.execute("INSERT INTO test_table (name) VALUES ('test_data')")
            conn.commit()

    # ==================== Backup Creation ====================

    def test_create_backup(self):
        """Test creating a basic backup."""
        backup_info = self.backup_manager.create_backup()

        self.assertIsNotNone(backup_info)
        self.assertTrue(backup_info.path.exists())
        self.assertEqual(backup_info.original_db, self.db_path)
        self.assertEqual(backup_info.schema_version, 5)
        self.assertGreater(backup_info.size_bytes, 0)
        self.assertIsInstance(backup_info.created_at, datetime)

    def test_create_backup_with_metadata(self):
        """Test creating a backup with custom metadata."""
        metadata = {
            "reason": "pre-migration backup",
            "version": "1.0.0"
        }

        backup_info = self.backup_manager.create_backup(metadata=metadata)

        self.assertEqual(backup_info.metadata, metadata)

    def test_create_backup_saves_metadata_file(self):
        """Test that backup creates a metadata sidecar file."""
        backup_info = self.backup_manager.create_backup()

        metadata_path = backup_info.path.with_suffix(
            backup_info.path.suffix + BackupManager.METADATA_SUFFIX
        )

        self.assertTrue(metadata_path.exists())

    def test_create_backup_nonexistent_database(self):
        """Test that backing up non-existent database raises error."""
        nonexistent_db = Path(self.temp_dir) / "nonexistent.db"
        manager = BackupManager(nonexistent_db)

        with self.assertRaises(FileNotFoundError):
            manager.create_backup()

    def test_create_backup_naming_pattern(self):
        """Test that backup files follow naming convention."""
        backup_info = self.backup_manager.create_backup()

        # Should contain {db_name}_backup_{timestamp}.db
        self.assertIn(self.db_path.stem, backup_info.path.name)
        self.assertIn(BackupManager.BACKUP_SUFFIX, backup_info.path.name)
        self.assertTrue(backup_info.path.name.endswith(".db"))

    def test_multiple_backups_unique_names(self):
        """Test that multiple backups get unique timestamped names."""
        backup1 = self.backup_manager.create_backup()
        time.sleep(1.1)  # Ensure different timestamp (1 second granularity)
        backup2 = self.backup_manager.create_backup()

        self.assertNotEqual(backup1.path, backup2.path)

    # ==================== Backup Verification ====================

    def test_verify_backup_valid(self):
        """Test verifying a valid backup."""
        backup_info = self.backup_manager.create_backup()

        is_valid = self.backup_manager.verify_backup(backup_info.path)
        self.assertTrue(is_valid)

    def test_verify_backup_nonexistent(self):
        """Test verifying non-existent backup returns False."""
        nonexistent = Path(self.temp_dir) / "nonexistent.db"
        is_valid = self.backup_manager.verify_backup(nonexistent)
        self.assertFalse(is_valid)

    def test_verify_backup_corrupted(self):
        """Test verifying corrupted backup returns False."""
        # Create a fake corrupted backup file
        corrupted_path = self.backup_dir / "corrupted.db"
        corrupted_path.write_text("This is not a valid SQLite database")

        is_valid = self.backup_manager.verify_backup(corrupted_path)
        self.assertFalse(is_valid)

    def test_backup_data_integrity(self):
        """Test that backup preserves database data."""
        backup_info = self.backup_manager.create_backup()

        # Verify backup contains the same data
        with sqlite3.connect(backup_info.path) as conn:
            cursor = conn.execute("SELECT name FROM test_table")
            row = cursor.fetchone()
            self.assertEqual(row[0], "test_data")

    def test_backup_schema_version(self):
        """Test that backup preserves schema version."""
        backup_info = self.backup_manager.create_backup()

        # Verify schema version in backup
        with sqlite3.connect(backup_info.path) as conn:
            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            self.assertEqual(version, 5)

    # ==================== Backup Restore ====================

    def test_restore_backup(self):
        """Test restoring a backup."""
        # Create backup
        backup_info = self.backup_manager.create_backup()

        # Modify database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM test_table")
            conn.execute("PRAGMA user_version = 10")
            conn.commit()

        # Restore backup
        self.backup_manager.restore_backup(backup_info.path)

        # Verify data restored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

            cursor = conn.execute("PRAGMA user_version")
            version = cursor.fetchone()[0]
            self.assertEqual(version, 5)

    def test_restore_nonexistent_backup(self):
        """Test that restoring non-existent backup raises error."""
        nonexistent = Path(self.temp_dir) / "nonexistent.db"

        with self.assertRaises(FileNotFoundError):
            self.backup_manager.restore_backup(nonexistent)

    # ==================== Backup Listing ====================

    def test_list_backups_empty(self):
        """Test listing backups when none exist."""
        backups = self.backup_manager.list_backups()
        self.assertEqual(len(backups), 0)

    def test_list_backups(self):
        """Test listing multiple backups."""
        # Create several backups
        backup1 = self.backup_manager.create_backup()
        time.sleep(1.1)
        backup2 = self.backup_manager.create_backup()
        time.sleep(1.1)
        backup3 = self.backup_manager.create_backup()

        backups = self.backup_manager.list_backups()

        # Should return all backups
        self.assertEqual(len(backups), 3)

        # Should be ordered newest first
        self.assertEqual(backups[0].path, backup3.path)
        self.assertEqual(backups[1].path, backup2.path)
        self.assertEqual(backups[2].path, backup1.path)

    def test_list_backups_without_metadata(self):
        """Test listing backups that don't have metadata files."""
        # Create a backup file manually without metadata
        manual_backup = self.backup_dir / f"{self.db_path.stem}_backup_20240101_120000.db"

        # Copy database to create a valid backup
        with sqlite3.connect(self.db_path) as source:
            with sqlite3.connect(manual_backup) as dest:
                source.backup(dest)

        backups = self.backup_manager.list_backups()

        # Should still list the backup
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].path, manual_backup)
        self.assertEqual(backups[0].schema_version, -1)  # Unknown

    def test_get_latest_backup(self):
        """Test getting the most recent backup."""
        self.assertIsNone(self.backup_manager.get_latest_backup())

        backup1 = self.backup_manager.create_backup()
        time.sleep(1.1)
        backup2 = self.backup_manager.create_backup()

        latest = self.backup_manager.get_latest_backup()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.path, backup2.path)

    # ==================== Backup Cleanup ====================

    def test_cleanup_old_backups(self):
        """Test cleaning up old backups."""
        # Create 7 backups
        for i in range(7):
            self.backup_manager.create_backup()
            if i < 6:  # Don't sleep after last one
                time.sleep(1.1)

        self.assertEqual(len(self.backup_manager.list_backups()), 7)

        # Keep only 3 most recent
        deleted = self.backup_manager.cleanup_old_backups(keep_count=3)

        self.assertEqual(deleted, 4)
        self.assertEqual(len(self.backup_manager.list_backups()), 3)

    def test_cleanup_removes_metadata_files(self):
        """Test that cleanup removes both backup and metadata files."""
        backup_info = self.backup_manager.create_backup()
        time.sleep(1.1)
        self.backup_manager.create_backup()

        metadata_path = backup_info.path.with_suffix(
            backup_info.path.suffix + BackupManager.METADATA_SUFFIX
        )
        self.assertTrue(metadata_path.exists())

        # Keep only 1, should delete the older one
        self.backup_manager.cleanup_old_backups(keep_count=1)

        self.assertFalse(backup_info.path.exists())
        self.assertFalse(metadata_path.exists())

    def test_cleanup_invalid_keep_count(self):
        """Test that invalid keep_count raises ValueError."""
        with self.assertRaises(ValueError):
            self.backup_manager.cleanup_old_backups(keep_count=0)

        with self.assertRaises(ValueError):
            self.backup_manager.cleanup_old_backups(keep_count=-1)

    def test_cleanup_when_fewer_backups_than_keep_count(self):
        """Test cleanup when there are fewer backups than keep_count."""
        self.backup_manager.create_backup()
        time.sleep(1.1)
        self.backup_manager.create_backup()

        deleted = self.backup_manager.cleanup_old_backups(keep_count=5)

        self.assertEqual(deleted, 0)
        self.assertEqual(len(self.backup_manager.list_backups()), 2)

    # ==================== Backup Info ====================

    def test_get_backup_info(self):
        """Test getting info about a specific backup."""
        backup_info = self.backup_manager.create_backup(
            metadata={"test": "value"}
        )

        retrieved_info = self.backup_manager.get_backup_info(backup_info.path)

        self.assertIsNotNone(retrieved_info)
        self.assertEqual(retrieved_info.path, backup_info.path)
        self.assertEqual(retrieved_info.schema_version, 5)
        self.assertEqual(retrieved_info.metadata, {"test": "value"})

    def test_get_backup_info_nonexistent(self):
        """Test getting info for non-existent backup."""
        nonexistent = Path(self.temp_dir) / "nonexistent.db"
        info = self.backup_manager.get_backup_info(nonexistent)
        self.assertIsNone(info)

    # ==================== BackupInfo Serialization ====================

    def test_backup_info_to_dict(self):
        """Test BackupInfo serialization to dictionary."""
        backup_info = self.backup_manager.create_backup(
            metadata={"key": "value"}
        )

        info_dict = backup_info.to_dict()

        self.assertIn("path", info_dict)
        self.assertIn("original_db", info_dict)
        self.assertIn("created_at", info_dict)
        self.assertIn("size_bytes", info_dict)
        self.assertIn("schema_version", info_dict)
        self.assertIn("metadata", info_dict)
        self.assertEqual(info_dict["metadata"], {"key": "value"})

    def test_backup_info_from_dict(self):
        """Test BackupInfo deserialization from dictionary."""
        original = self.backup_manager.create_backup()
        dict_data = original.to_dict()

        restored = BackupInfo.from_dict(dict_data)

        self.assertEqual(restored.path, original.path)
        self.assertEqual(restored.original_db, original.original_db)
        self.assertEqual(restored.size_bytes, original.size_bytes)
        self.assertEqual(restored.schema_version, original.schema_version)

    # ==================== Default Backup Directory ====================

    def test_default_backup_directory(self):
        """Test that default backup directory is created correctly."""
        manager = BackupManager(self.db_path)

        expected_dir = self.db_path.parent / "backups"
        self.assertEqual(manager.backup_dir, expected_dir)
        self.assertTrue(manager.backup_dir.exists())


# ==================== Performance Tests ====================

class TestMigrationPerformance(unittest.TestCase):
    """Performance and stress tests."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "perf_test.sqlite"
        self.manager = MigrationManager(self.db_path)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_record_many_migrations(self):
        """Test recording many migrations for performance."""
        start_time = time.time()

        # Record 100 migrations
        for i in range(1, 101):
            self.manager.record_migration(
                version=i,
                description=f"Migration {i}",
                duration_ms=i * 10
            )

        elapsed = time.time() - start_time

        # Should complete reasonably fast (< 1 second)
        self.assertLess(elapsed, 1.0)

        # Verify all recorded
        self.assertEqual(self.manager.get_migration_count(), 100)

    def test_query_many_migrations(self):
        """Test querying with many migration records."""
        # Record 100 migrations
        for i in range(1, 101):
            self.manager.record_migration(i, f"Migration {i}")

        start_time = time.time()

        # Query operations
        migrations = self.manager.get_applied_migrations()
        last = self.manager.get_last_migration()
        count = self.manager.get_migration_count()

        elapsed = time.time() - start_time

        # Should be fast
        self.assertLess(elapsed, 0.1)
        self.assertEqual(len(migrations), 100)
        self.assertEqual(last.version, 100)
        self.assertEqual(count, 100)


class TestMigrateCLI(unittest.TestCase):
    """Integration tests for CLI migrate commands."""

    def test_migrate_run_requires_init(self):
        """Test that 'omi migrate run' requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["migrate", "run"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("not initialized", result.output.lower())

    def test_migrate_run_requires_database(self):
        """Test that 'omi migrate run' requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base directory but no database
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "run"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("database not found", result.output.lower())

    def test_migrate_run_dry_run(self):
        """Test 'omi migrate run --dry-run' does not modify database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            base_path = Path(tmpdir) / "omi"
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            db_path = base_path / "palace.sqlite"
            self.assertTrue(db_path.exists())

            # Get initial database state
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            initial_version = cursor.fetchone()[0]
            conn.close()

            # Run dry-run migration
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "run", "--dry-run"])

            self.assertEqual(result.exit_code, 0)
            # Should show either "dry run" or "up to date" (if no migrations)
            self.assertTrue(
                "dry run" in result.output.lower() or
                "up to date" in result.output.lower()
            )

            # Verify database state unchanged
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            final_version = cursor.fetchone()[0]
            conn.close()

            self.assertEqual(initial_version, final_version)

    def test_migrate_run_success(self):
        """Test 'omi migrate run' completes successfully when up to date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            base_path = Path(tmpdir) / "omi"
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Run migration
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "run"])

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                "up to date" in result.output.lower() or
                "no migrations" in result.output.lower()
            )

    def test_migrate_status_requires_init(self):
        """Test that 'omi migrate status' requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["migrate", "status"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("not initialized", result.output.lower())

    def test_migrate_status_requires_database(self):
        """Test that 'omi migrate status' requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base directory but no database
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "status"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("database not found", result.output.lower())

    def test_migrate_status_shows_current_version(self):
        """Test 'omi migrate status' displays current schema version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            base_path = Path(tmpdir) / "omi"
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Check status
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "status"])

            self.assertEqual(result.exit_code, 0)
            self.assertIn("current schema version", result.output.lower())

    def test_migrate_status_shows_pending_migrations(self):
        """Test 'omi migrate status' shows pending migrations message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            base_path = Path(tmpdir) / "omi"
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            # Check status
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "status"])

            self.assertEqual(result.exit_code, 0)
            # Should show either "no pending migrations" or list of pending
            self.assertTrue(
                "no pending" in result.output.lower() or
                "pending migration" in result.output.lower()
            )

    def test_migrate_rollback_requires_init(self):
        """Test that 'omi migrate rollback' requires initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(Path(tmpdir) / "not_initialized")}):
                result = runner.invoke(cli, ["migrate", "rollback"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("not initialized", result.output.lower())

    def test_migrate_rollback_requires_database(self):
        """Test that 'omi migrate rollback' requires database to exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Create base directory but no database
            base_path = Path(tmpdir) / "omi"
            base_path.mkdir(parents=True)

            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "rollback"])

            self.assertEqual(result.exit_code, 1)
            self.assertIn("database not found", result.output.lower())

    def test_migrate_rollback_dry_run(self):
        """Test 'omi migrate rollback --dry-run' does not modify database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            base_path = Path(tmpdir) / "omi"
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            db_path = base_path / "palace.sqlite"
            self.assertTrue(db_path.exists())

            # Get initial database state
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            initial_version = cursor.fetchone()[0]
            conn.close()

            # Run dry-run rollback
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "rollback", "--dry-run"])

            self.assertEqual(result.exit_code, 0)
            # Should show dry run message or "nothing to roll back"
            self.assertTrue(
                "dry run" in result.output.lower() or
                "nothing to roll back" in result.output.lower()
            )

            # Verify database state unchanged
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA user_version")
            final_version = cursor.fetchone()[0]
            conn.close()

            self.assertEqual(initial_version, final_version)

    def test_migrate_rollback_at_version_zero(self):
        """Test 'omi migrate rollback' handles version 0 gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from click.testing import CliRunner
            runner = CliRunner()

            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from omi.cli import cli

            # Initialize OMI
            base_path = Path(tmpdir) / "omi"
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                runner.invoke(cli, ["init"])

            db_path = base_path / "palace.sqlite"

            # Set schema version to 0
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA user_version = 0")
            conn.close()

            # Try to rollback
            with patch.dict(os.environ, {"OMI_BASE_PATH": str(base_path)}):
                result = runner.invoke(cli, ["migrate", "rollback"])

            self.assertEqual(result.exit_code, 0)
            self.assertTrue(
                "nothing to roll back" in result.output.lower() or
                "version 0" in result.output.lower()
            )


class TestE2EMigration(unittest.TestCase):
    """End-to-end migration test with backup/restore workflow.

    Tests the complete migration flow:
    1. Create database with initial data
    2. Create backup before migration
    3. Run migration (simulate schema change)
    4. Verify migration succeeded
    5. Restore from backup
    6. Verify restore succeeded
    """

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_e2e.sqlite"
        self.backup_dir = Path(self.temp_dir) / "backups"

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_migration_workflow_with_backup_restore(self):
        """
        End-to-end verification:
        1. Create initial database with data (schema v0)
        2. Create backup before migration
        3. Run migration to v1 (add column)
        4. Verify migration applied correctly
        5. Verify backup integrity
        6. Restore from backup
        7. Verify restore brought back original state
        """
        # Step 1: Create initial database with data (schema v0)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA user_version = 0")
            conn.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL
                )
            """)
            conn.execute("INSERT INTO users (name) VALUES ('Alice')")
            conn.execute("INSERT INTO users (name) VALUES ('Bob')")
            conn.commit()

        # Verify initial state
        # Disable WAL mode for E2E tests with restore to avoid file conflicts
        migration_manager = MigrationManager(self.db_path, enable_wal=False)
        self.assertEqual(migration_manager.get_schema_version(), 0)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            initial_count = cursor.fetchone()[0]
            self.assertEqual(initial_count, 2)

        # Step 2: Create backup before migration
        backup_manager = BackupManager(self.db_path, self.backup_dir)
        backup_info = backup_manager.create_backup(
            metadata={
                "reason": "pre-migration backup",
                "from_version": 0,
                "to_version": 1
            }
        )

        # Verify backup was created
        self.assertIsNotNone(backup_info)
        self.assertTrue(backup_info.path.exists())
        self.assertEqual(backup_info.schema_version, 0)
        self.assertEqual(backup_info.metadata["reason"], "pre-migration backup")

        # Step 3: Run migration to v1 (simulate adding email column)
        start_time = time.time()

        with sqlite3.connect(self.db_path) as conn:
            # Add new column
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
            # Update schema version
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

        duration_ms = int((time.time() - start_time) * 1000)

        # Record migration
        migration_manager.record_migration(
            version=1,
            description="Add email column to users table",
            duration_ms=duration_ms,
            metadata={"column_added": "email"}
        )

        # Step 4: Verify migration applied correctly
        self.assertEqual(migration_manager.get_schema_version(), 1)
        self.assertTrue(migration_manager.is_migration_applied(1))

        # Verify new schema
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertIn("email", columns)

            # Verify data still intact
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor = conn.execute("SELECT name FROM users ORDER BY name")
            names = [row[0] for row in cursor.fetchall()]
            self.assertEqual(names, ["Alice", "Bob"])

        # Verify migration record
        migration_record = migration_manager.get_migration_record(1)
        self.assertIsNotNone(migration_record)
        self.assertEqual(migration_record.version, 1)
        self.assertEqual(migration_record.description, "Add email column to users table")
        self.assertEqual(migration_record.metadata["column_added"], "email")

        # Step 5: Verify backup integrity
        is_valid = backup_manager.verify_backup(backup_info.path)
        self.assertTrue(is_valid)

        # Verify backup contains original schema (no email column)
        with sqlite3.connect(backup_info.path) as conn:
            cursor = conn.execute("PRAGMA user_version")
            self.assertEqual(cursor.fetchone()[0], 0)

            cursor = conn.execute("PRAGMA table_info(users)")
            backup_columns = [row[1] for row in cursor.fetchall()]
            self.assertNotIn("email", backup_columns)

            cursor = conn.execute("SELECT COUNT(*) FROM users")
            self.assertEqual(cursor.fetchone()[0], 2)

        # Step 6: Restore from backup
        backup_manager.restore_backup(backup_info.path)

        # Step 7: Verify restore brought back original state
        # Create new manager instance after restore to avoid WAL file conflicts
        migration_manager = MigrationManager(self.db_path, enable_wal=False)
        self.assertEqual(migration_manager.get_schema_version(), 0)

        with sqlite3.connect(self.db_path) as conn:
            # Schema should be back to v0 (no email column)
            cursor = conn.execute("PRAGMA table_info(users)")
            restored_columns = [row[1] for row in cursor.fetchall()]
            self.assertNotIn("email", restored_columns)
            self.assertIn("id", restored_columns)
            self.assertIn("name", restored_columns)

            # Data should be intact
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor = conn.execute("SELECT name FROM users ORDER BY name")
            names = [row[0] for row in cursor.fetchall()]
            self.assertEqual(names, ["Alice", "Bob"])

    def test_migration_rollback_via_backup(self):
        """
        Test rollback scenario:
        1. Create database v0 with data
        2. Create backup
        3. Run failing migration to v1
        4. Detect failure
        5. Restore from backup
        6. Verify rollback successful
        """
        # Step 1: Create database v0 with data
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA user_version = 0")
            conn.execute("""
                CREATE TABLE products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    price REAL NOT NULL
                )
            """)
            conn.execute("INSERT INTO products (name, price) VALUES ('Widget', 9.99)")
            conn.execute("INSERT INTO products (name, price) VALUES ('Gadget', 19.99)")
            conn.commit()

        # Disable WAL mode for E2E tests with restore to avoid file conflicts
        migration_manager = MigrationManager(self.db_path, enable_wal=False)
        backup_manager = BackupManager(self.db_path, self.backup_dir)

        # Step 2: Create backup
        backup_info = backup_manager.create_backup(
            metadata={"reason": "pre-migration safety backup"}
        )

        self.assertTrue(backup_info.path.exists())
        original_version = migration_manager.get_schema_version()
        self.assertEqual(original_version, 0)

        # Step 3: Simulate a migration that needs rollback
        # (e.g., add column, realize it's wrong)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("ALTER TABLE products ADD COLUMN category TEXT")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

        migration_manager.record_migration(
            version=1,
            description="Add category column (will be rolled back)",
            duration_ms=50
        )

        # Step 4: Detect that migration needs rollback
        # In real scenario, this could be due to data validation failure, etc.
        self.assertEqual(migration_manager.get_schema_version(), 1)

        # Step 5: Restore from backup to rollback
        backup_manager.restore_backup(backup_info.path)

        # Step 6: Verify rollback successful
        # Create new manager instance after restore to avoid WAL file conflicts
        migration_manager = MigrationManager(self.db_path, enable_wal=False)
        self.assertEqual(migration_manager.get_schema_version(), 0)

        with sqlite3.connect(self.db_path) as conn:
            # Schema should not have category column
            cursor = conn.execute("PRAGMA table_info(products)")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertNotIn("category", columns)

            # Data should be preserved
            cursor = conn.execute("SELECT COUNT(*) FROM products")
            self.assertEqual(cursor.fetchone()[0], 2)

            cursor = conn.execute("SELECT name FROM products ORDER BY name")
            names = [row[0] for row in cursor.fetchall()]
            self.assertEqual(names, ["Gadget", "Widget"])

    def test_multiple_migrations_with_backups(self):
        """
        Test multiple migrations with backup at each step:
        1. Create database v0
        2. Backup, migrate to v1, verify
        3. Backup, migrate to v2, verify
        4. List all backups
        5. Verify backup chain integrity
        """
        # Step 1: Create initial database v0
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA user_version = 0")
            conn.execute("""
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    amount REAL NOT NULL
                )
            """)
            conn.execute("INSERT INTO orders (amount) VALUES (100.0)")
            conn.commit()

        # Disable WAL mode for E2E tests with restore to avoid file conflicts
        migration_manager = MigrationManager(self.db_path, enable_wal=False)
        backup_manager = BackupManager(self.db_path, self.backup_dir)

        # Step 2: Backup and migrate to v1
        backup_v0 = backup_manager.create_backup(
            metadata={"version": 0, "description": "Initial state"}
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("ALTER TABLE orders ADD COLUMN status TEXT DEFAULT 'pending'")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

        migration_manager.record_migration(
            version=1,
            description="Add status column",
            duration_ms=25
        )

        self.assertEqual(migration_manager.get_schema_version(), 1)
        time.sleep(1.1)  # Ensure different timestamps

        # Step 3: Backup and migrate to v2
        backup_v1 = backup_manager.create_backup(
            metadata={"version": 1, "description": "After status column"}
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("ALTER TABLE orders ADD COLUMN created_at TEXT")
            conn.execute("PRAGMA user_version = 2")
            conn.commit()

        migration_manager.record_migration(
            version=2,
            description="Add created_at column",
            duration_ms=30
        )

        self.assertEqual(migration_manager.get_schema_version(), 2)

        # Step 4: List all backups
        all_backups = backup_manager.list_backups()
        self.assertEqual(len(all_backups), 2)

        # Should be ordered newest first
        self.assertEqual(all_backups[0].path, backup_v1.path)
        self.assertEqual(all_backups[1].path, backup_v0.path)

        # Step 5: Verify backup chain integrity
        # Verify v0 backup
        with sqlite3.connect(backup_v0.path) as conn:
            cursor = conn.execute("PRAGMA user_version")
            self.assertEqual(cursor.fetchone()[0], 0)

            cursor = conn.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertNotIn("status", columns)
            self.assertNotIn("created_at", columns)

        # Verify v1 backup
        with sqlite3.connect(backup_v1.path) as conn:
            cursor = conn.execute("PRAGMA user_version")
            self.assertEqual(cursor.fetchone()[0], 1)

            cursor = conn.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertIn("status", columns)
            self.assertNotIn("created_at", columns)

        # Verify current database (v2)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(orders)")
            columns = [row[1] for row in cursor.fetchall()]
            self.assertIn("status", columns)
            self.assertIn("created_at", columns)

        # Verify migration history
        migrations = migration_manager.get_applied_migrations()
        self.assertEqual(len(migrations), 2)
        self.assertEqual(migrations[0].version, 1)
        self.assertEqual(migrations[1].version, 2)


if __name__ == "__main__":
    unittest.main()
