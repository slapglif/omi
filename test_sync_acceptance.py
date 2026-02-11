#!/usr/bin/env python3
"""
Test script for sync push/pull acceptance criteria.

This script validates:
1. Configure S3 backend
2. Run 'omi sync push'
3. Verify files in S3
4. Delete local files
5. Run 'omi sync pull'
6. Verify local files restored
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from omi.storage_backends import S3Backend, StorageError
from omi.moltvault import create_backend_from_config


def create_test_omi_directory(test_dir: Path):
    """Create a test OMI directory with sample files."""
    print(f"Creating test OMI directory at {test_dir}")

    # Create directory structure
    test_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = test_dir / "daily"
    daily_dir.mkdir(exist_ok=True)

    # Create sample files
    (test_dir / "NOW.md").write_text("# NOW\n\nCurrent focus: Testing cloud sync\n")
    (test_dir / "MEMORY.md").write_text("# MEMORY\n\nTest memory content\n")
    (daily_dir / "2024-01-01.md").write_text("# Daily Log 2024-01-01\n\nTest log\n")
    (daily_dir / "2024-01-02.md").write_text("# Daily Log 2024-01-02\n\nAnother test\n")

    # Create a mock palace.sqlite file
    (test_dir / "palace.sqlite").write_text("mock sqlite database")

    print("✓ Created test files:")
    print("  - NOW.md")
    print("  - MEMORY.md")
    print("  - daily/2024-01-01.md")
    print("  - daily/2024-01-02.md")
    print("  - palace.sqlite")


def create_test_config(test_dir: Path, backend: str, bucket: str, **kwargs):
    """Create a test config.yaml file."""
    config_path = test_dir / "config.yaml"

    config = {
        'backup': {
            'backend': backend,
            'bucket': bucket,
            **kwargs
        }
    }

    config_path.write_text(yaml.dump(config, default_flow_style=False))
    print(f"✓ Created config at {config_path}")
    print(f"  Backend: {backend}")
    print(f"  Bucket: {bucket}")

    return config


def test_sync_push(test_dir: Path, backend):
    """Test pushing files to cloud storage."""
    print("\n" + "="*60)
    print("STEP 2: Testing sync push")
    print("="*60)

    files_to_sync = [
        ('NOW.md', test_dir / "NOW.md"),
        ('MEMORY.md', test_dir / "MEMORY.md"),
        ('daily/2024-01-01.md', test_dir / "daily" / "2024-01-01.md"),
        ('daily/2024-01-02.md', test_dir / "daily" / "2024-01-02.md"),
        ('palace.sqlite', test_dir / "palace.sqlite"),
    ]

    uploaded = 0
    for remote_key, local_path in files_to_sync:
        try:
            backend.upload(local_path, remote_key)
            print(f"   ✓ Uploaded {remote_key}")
            uploaded += 1
        except StorageError as e:
            print(f"   ✗ Failed to upload {remote_key}: {e}")
            return False

    print(f"\n✓ Push complete: {uploaded}/{len(files_to_sync)} files uploaded")
    return uploaded == len(files_to_sync)


def test_verify_remote_files(backend):
    """Verify files are in cloud storage."""
    print("\n" + "="*60)
    print("STEP 3: Verifying files in cloud storage")
    print("="*60)

    try:
        remote_objects = backend.list(prefix="")
        print(f"Found {len(remote_objects)} file(s) in cloud storage:")

        for obj in remote_objects:
            size_kb = obj.size / 1024 if obj.size else 0
            print(f"   ✓ {obj.key} ({size_kb:.2f} KB)")

        expected_files = ['NOW.md', 'MEMORY.md', 'daily/2024-01-01.md',
                         'daily/2024-01-02.md', 'palace.sqlite']

        remote_keys = [obj.key for obj in remote_objects]
        missing = [f for f in expected_files if f not in remote_keys]

        if missing:
            print(f"\n✗ Missing files in cloud storage: {missing}")
            return False

        print("\n✓ All expected files found in cloud storage")
        return True

    except StorageError as e:
        print(f"✗ Failed to list remote files: {e}")
        return False


def test_delete_local_files(test_dir: Path):
    """Delete local files to simulate data loss."""
    print("\n" + "="*60)
    print("STEP 4: Deleting local files")
    print("="*60)

    files_to_delete = [
        test_dir / "NOW.md",
        test_dir / "MEMORY.md",
        test_dir / "daily" / "2024-01-01.md",
        test_dir / "daily" / "2024-01-02.md",
        test_dir / "palace.sqlite",
    ]

    for file_path in files_to_delete:
        if file_path.exists():
            file_path.unlink()
            print(f"   ✓ Deleted {file_path.name}")

    # Verify files are gone
    remaining = [f for f in files_to_delete if f.exists()]
    if remaining:
        print(f"\n✗ Some files still exist: {remaining}")
        return False

    print("\n✓ All local files deleted")
    return True


def test_sync_pull(test_dir: Path, backend):
    """Test pulling files from cloud storage."""
    print("\n" + "="*60)
    print("STEP 5: Testing sync pull")
    print("="*60)

    try:
        remote_objects = backend.list(prefix="")
        print(f"Found {len(remote_objects)} file(s) to download")

        downloaded = 0
        for obj in remote_objects:
            remote_key = obj.key
            local_path = test_dir / remote_key

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                backend.download(remote_key, local_path)
                print(f"   ✓ Downloaded {remote_key}")
                downloaded += 1
            except StorageError as e:
                print(f"   ✗ Failed to download {remote_key}: {e}")

        print(f"\n✓ Pull complete: {downloaded}/{len(remote_objects)} files downloaded")
        return downloaded == len(remote_objects)

    except StorageError as e:
        print(f"✗ Failed to pull files: {e}")
        return False


def test_verify_local_files(test_dir: Path):
    """Verify local files are restored correctly."""
    print("\n" + "="*60)
    print("STEP 6: Verifying local files restored")
    print("="*60)

    expected_files = [
        (test_dir / "NOW.md", "# NOW"),
        (test_dir / "MEMORY.md", "# MEMORY"),
        (test_dir / "daily" / "2024-01-01.md", "# Daily Log 2024-01-01"),
        (test_dir / "daily" / "2024-01-02.md", "# Daily Log 2024-01-02"),
        (test_dir / "palace.sqlite", "mock sqlite database"),
    ]

    all_ok = True
    for file_path, expected_content in expected_files:
        if not file_path.exists():
            print(f"   ✗ {file_path.name} does not exist")
            all_ok = False
            continue

        content = file_path.read_text()
        if expected_content in content:
            print(f"   ✓ {file_path.name} restored correctly")
        else:
            print(f"   ✗ {file_path.name} content mismatch")
            all_ok = False

    if all_ok:
        print("\n✓ All local files restored correctly")
    else:
        print("\n✗ Some files were not restored correctly")

    return all_ok


def cleanup_remote_files(backend):
    """Clean up remote test files."""
    print("\n" + "="*60)
    print("Cleaning up remote files")
    print("="*60)

    try:
        remote_objects = backend.list(prefix="")
        for obj in remote_objects:
            try:
                backend.delete(obj.key)
                print(f"   ✓ Deleted {obj.key}")
            except StorageError as e:
                print(f"   ✗ Failed to delete {obj.key}: {e}")
    except StorageError as e:
        print(f"✗ Failed to list files for cleanup: {e}")


def main():
    """Main test runner."""
    print("="*60)
    print("SYNC PUSH/PULL ACCEPTANCE TEST")
    print("="*60)

    # Check for required environment variables
    if 'S3_TEST_BUCKET' not in os.environ:
        print("\nError: S3_TEST_BUCKET environment variable not set")
        print("\nTo run this test, set the following environment variables:")
        print("  export S3_TEST_BUCKET=your-test-bucket")
        print("  export AWS_ACCESS_KEY_ID=your-access-key  # optional if using AWS credentials")
        print("  export AWS_SECRET_ACCESS_KEY=your-secret-key  # optional")
        print("  export S3_ENDPOINT_URL=https://...  # optional for R2/MinIO")
        sys.exit(1)

    bucket = os.environ['S3_TEST_BUCKET']
    endpoint = os.environ.get('S3_ENDPOINT_URL')
    region = os.environ.get('AWS_REGION', 'us-east-1')

    print(f"\nTest Configuration:")
    print(f"  Bucket: {bucket}")
    print(f"  Endpoint: {endpoint or 'default (AWS S3)'}")
    print(f"  Region: {region}")

    # Create temporary test directory
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp) / "omi_test"

        try:
            # Step 1: Create test OMI directory and configure backend
            print("\n" + "="*60)
            print("STEP 1: Creating test environment and configuring S3 backend")
            print("="*60)

            create_test_omi_directory(test_dir)

            config_kwargs = {'region': region}
            if endpoint:
                config_kwargs['endpoint'] = endpoint

            config = create_test_config(test_dir, 's3', bucket, **config_kwargs)

            # Create backend
            backend = create_backend_from_config({'backup': config['backup']})
            print("✓ Backend created successfully")

            # Run acceptance tests
            results = []

            # Test sync push
            results.append(("Sync Push", test_sync_push(test_dir, backend)))

            # Verify files in cloud
            results.append(("Verify Remote Files", test_verify_remote_files(backend)))

            # Delete local files
            results.append(("Delete Local Files", test_delete_local_files(test_dir)))

            # Test sync pull
            results.append(("Sync Pull", test_sync_pull(test_dir, backend)))

            # Verify local files restored
            results.append(("Verify Local Files", test_verify_local_files(test_dir)))

            # Clean up remote files
            cleanup_remote_files(backend)

            # Print summary
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)

            all_passed = True
            for test_name, passed in results:
                status = "✓ PASS" if passed else "✗ FAIL"
                color = '\033[92m' if passed else '\033[91m'
                reset = '\033[0m'
                print(f"{color}{status}{reset} - {test_name}")
                if not passed:
                    all_passed = False

            print("\n" + "="*60)
            if all_passed:
                print("✓ ALL ACCEPTANCE TESTS PASSED")
                print("="*60)
                return 0
            else:
                print("✗ SOME TESTS FAILED")
                print("="*60)
                return 1

        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    sys.exit(main())
