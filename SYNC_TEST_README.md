# Sync Push/Pull Acceptance Test

This document describes how to run the acceptance tests for the cloud sync push/pull functionality.

## Overview

The acceptance test validates the following workflow:
1. Configure S3 backend
2. Run sync push to upload files
3. Verify files exist in S3
4. Delete local files
5. Run sync pull to download files
6. Verify local files are restored

## Prerequisites

- Python 3.10+
- boto3 package installed
- AWS S3 bucket or S3-compatible storage (e.g., Cloudflare R2, MinIO)
- AWS credentials configured

## Setup

### Option 1: AWS S3

```bash
export S3_TEST_BUCKET=your-test-bucket-name
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION=us-east-1
```

### Option 2: Cloudflare R2

```bash
export S3_TEST_BUCKET=your-r2-bucket-name
export AWS_ACCESS_KEY_ID=your-r2-access-key
export AWS_SECRET_ACCESS_KEY=your-r2-secret-key
export S3_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
export AWS_REGION=auto
```

### Option 3: MinIO (Local Testing)

```bash
# Start MinIO locally
docker run -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"

# Configure environment
export S3_TEST_BUCKET=test-bucket
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export S3_ENDPOINT_URL=http://localhost:9000
export AWS_REGION=us-east-1

# Create bucket (using aws CLI or MinIO console)
aws --endpoint-url http://localhost:9000 s3 mb s3://test-bucket
```

## Running the Test

```bash
# Run the automated acceptance test
python3 test_sync_acceptance.py
```

## Manual Testing (using omi CLI)

If you prefer to test manually with the actual omi CLI:

```bash
# 1. Initialize OMI
omi init

# 2. Configure S3 backend
omi config set backup.backend s3
omi config set backup.bucket your-test-bucket
omi config set backup.region us-east-1
# Optional for R2/MinIO:
# omi config set backup.endpoint https://your-endpoint

# 3. Create some test content
echo "# NOW\nTesting sync" > ~/.openclaw/omi/NOW.md
echo "# Daily Log\nTest" > ~/.openclaw/omi/daily/$(date +%Y-%m-%d).md

# 4. Push to cloud
omi sync push

# 5. Verify files in cloud (using aws CLI)
aws s3 ls s3://your-test-bucket/ --recursive

# 6. Delete local files
rm ~/.openclaw/omi/NOW.md
rm ~/.openclaw/omi/daily/*.md

# 7. Pull from cloud
omi sync pull

# 8. Verify files restored
ls -la ~/.openclaw/omi/
cat ~/.openclaw/omi/NOW.md
```

## Expected Output

The test script should output:

```
============================================================
SYNC PUSH/PULL ACCEPTANCE TEST
============================================================

Test Configuration:
  Bucket: test-bucket
  Endpoint: default (AWS S3)
  Region: us-east-1

============================================================
STEP 1: Creating test environment and configuring S3 backend
============================================================
Creating test OMI directory at /tmp/.../omi_test
✓ Created test files:
  - NOW.md
  - MEMORY.md
  - daily/2024-01-01.md
  - daily/2024-01-02.md
  - palace.sqlite
✓ Created config at /tmp/.../omi_test/config.yaml
  Backend: s3
  Bucket: test-bucket
✓ Backend created successfully

============================================================
STEP 2: Testing sync push
============================================================
   ✓ Uploaded NOW.md
   ✓ Uploaded MEMORY.md
   ✓ Uploaded daily/2024-01-01.md
   ✓ Uploaded daily/2024-01-02.md
   ✓ Uploaded palace.sqlite

✓ Push complete: 5/5 files uploaded

============================================================
STEP 3: Verifying files in cloud storage
============================================================
Found 5 file(s) in cloud storage:
   ✓ NOW.md (0.05 KB)
   ✓ MEMORY.md (0.03 KB)
   ✓ daily/2024-01-01.md (0.04 KB)
   ✓ daily/2024-01-02.md (0.04 KB)
   ✓ palace.sqlite (0.02 KB)

✓ All expected files found in cloud storage

============================================================
STEP 4: Deleting local files
============================================================
   ✓ Deleted NOW.md
   ✓ Deleted MEMORY.md
   ✓ Deleted 2024-01-01.md
   ✓ Deleted 2024-01-02.md
   ✓ Deleted palace.sqlite

✓ All local files deleted

============================================================
STEP 5: Testing sync pull
============================================================
Found 5 file(s) to download
   ✓ Downloaded NOW.md
   ✓ Downloaded MEMORY.md
   ✓ Downloaded daily/2024-01-01.md
   ✓ Downloaded daily/2024-01-02.md
   ✓ Downloaded palace.sqlite

✓ Pull complete: 5/5 files downloaded

============================================================
STEP 6: Verifying local files restored
============================================================
   ✓ NOW.md restored correctly
   ✓ MEMORY.md restored correctly
   ✓ 2024-01-01.md restored correctly
   ✓ 2024-01-02.md restored correctly
   ✓ palace.sqlite restored correctly

✓ All local files restored correctly

============================================================
Cleaning up remote files
============================================================
   ✓ Deleted NOW.md
   ✓ Deleted MEMORY.md
   ✓ Deleted daily/2024-01-01.md
   ✓ Deleted daily/2024-01-02.md
   ✓ Deleted palace.sqlite

============================================================
TEST SUMMARY
============================================================
✓ PASS - Sync Push
✓ PASS - Verify Remote Files
✓ PASS - Delete Local Files
✓ PASS - Sync Pull
✓ PASS - Verify Local Files

============================================================
✓ ALL ACCEPTANCE TESTS PASSED
============================================================
```

## Troubleshooting

### Missing boto3 package
```bash
pip install boto3
```

### Access Denied errors
- Verify your AWS credentials are correct
- Ensure the bucket exists and you have read/write permissions
- For R2, make sure you're using R2-specific credentials (not Cloudflare API tokens)

### Connection errors
- Verify the endpoint URL is correct
- Check network connectivity
- For MinIO, ensure the server is running

## Security Notes

- The test uses temporary directories and cleans up remote files
- Never commit credentials to git
- Use environment variables for sensitive configuration
- Consider using IAM roles or instance profiles in production
