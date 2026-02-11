# Release Guide

This document provides step-by-step instructions for releasing a new version of OMI (OpenClaw Memory Infrastructure) to PyPI.

## Version Numbering

OMI follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **MAJOR** (x.0.0): Incompatible API changes
- **MINOR** (0.x.0): New functionality in a backward-compatible manner
- **PATCH** (0.0.x): Backward-compatible bug fixes

### Version Decision Matrix

| Change Type | Example | Version Bump |
|-------------|---------|--------------|
| Breaking API change | Remove a public method | MAJOR |
| New feature | Add cloud storage backend | MINOR |
| Bug fix | Fix incorrect belief update | PATCH |
| Security fix | Patch vulnerability | PATCH (or MINOR if adds feature) |
| Documentation only | Update README | No release needed |

---

## Pre-Release Checklist

Before starting the release process:

- [ ] All intended changes are merged to `main` branch
- [ ] CI/CD pipeline passes (all tests green)
- [ ] Local tests pass: `pytest`
- [ ] Type checking passes: `mypy src/omi`
- [ ] Code formatting is clean: `black src/ tests/ --check`
- [ ] Coverage meets threshold (≥60%): `pytest --cov=src/omi --cov-report=term`
- [ ] No known critical bugs in issue tracker
- [ ] Dependencies are up to date and secure

---

## Release Process

### Step 1: Update Version Number

Edit `pyproject.toml` and update the version number:

```toml
[project]
version = "0.3.0"  # Update this line
```

**Location:** Line 7 in `pyproject.toml`

### Step 2: Update CHANGELOG

Edit `CHANGELOG.md` and add a new release section following the [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [0.3.0] - YYYY-MM-DD

### Added
- New feature descriptions

### Changed
- Modified functionality descriptions

### Fixed
- Bug fix descriptions

### Deprecated
- Features marked for removal

### Removed
- Removed features

### Security
- Security improvements
```

**Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- Group changes by category (Added, Changed, Fixed, etc.)
- Include code references in backticks
- Link to issues/PRs where relevant
- Update the Version Comparison table at the bottom
- Keep the existing entries intact

**Date Format:** Use ISO 8601 format (YYYY-MM-DD)

### Step 3: Run Full Test Suite

Ensure all tests pass with coverage:

```bash
# Run all tests including slow and integration tests
pytest -v

# Verify coverage meets threshold
pytest --cov=src/omi --cov-report=term --cov-report=html

# Run type checking
mypy src/omi

# Verify code formatting
black src/ tests/ --check
```

**Action Required:** Fix any failing tests before proceeding.

### Step 4: Build the Distribution

Build both source distribution and wheel:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build distributions
python -m build

# Verify the build
ls -lh dist/
```

**Expected Output:**
```
dist/
  omi-openclaw-0.3.0.tar.gz
  omi_openclaw-0.3.0-py3-none-any.whl
```

### Step 5: Test the Build Locally

Install the built package in a clean virtual environment:

```bash
# Create a test environment
python -m venv test-env
source test-env/bin/activate

# Install the built wheel
pip install dist/omi_openclaw-0.3.0-py3-none-any.whl

# Verify installation
omi --version
python -c "import omi; print(omi.__version__)"

# Test basic functionality
omi init --data-dir ./test-omi
omi status --data-dir ./test-omi

# Clean up
deactivate
rm -rf test-env test-omi
```

### Step 6: Commit Release Changes

Commit the version bump and CHANGELOG update:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Release version 0.3.0

- Update version in pyproject.toml
- Update CHANGELOG.md with release notes"
```

### Step 7: Create Git Tag

Create an annotated tag for the release:

```bash
git tag -a v0.3.0 -m "Release version 0.3.0

Major changes:
- Feature 1
- Feature 2
- Bug fix 3"
```

**Tag Naming:** Always prefix with `v` (e.g., `v0.3.0`, not `0.3.0`)

### Step 8: Push to Repository

Push both the commit and the tag:

```bash
# Push the release commit
git push origin main

# Push the tag
git push origin v0.3.0
```

**Important:** Tags trigger CI/CD workflows. Ensure the tag is pushed after the commit.

### Step 9: Publish to PyPI

#### Option A: Using TestPyPI First (Recommended)

Test the upload on TestPyPI before publishing to production:

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ omi-openclaw
```

#### Option B: Publish to Production PyPI

Once TestPyPI upload is verified:

```bash
# Upload to PyPI
python -m twine upload dist/*
```

**Authentication:**
- Use PyPI API token (not username/password)
- Store token in `~/.pypirc` or use `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<token>` environment variables

#### Verify Upload

Check the package page:
- PyPI: https://pypi.org/project/omi-openclaw/
- Verify version number, description, and metadata

### Step 10: Create GitHub Release

Create a GitHub release from the tag:

1. Go to https://github.com/slapglif/omi/releases/new
2. Select the tag: `v0.3.0`
3. Release title: `OMI v0.3.0`
4. Copy the relevant section from `CHANGELOG.md` into the release notes
5. Attach the distribution files from `dist/`
6. Publish release

---

## Post-Release Tasks

After publishing the release:

- [ ] Verify installation from PyPI: `pip install --upgrade omi-openclaw`
- [ ] Test installed package: `omi --version`
- [ ] Update documentation website (if applicable)
- [ ] Announce release (Discord, Twitter, mailing list, etc.)
- [ ] Close milestone in issue tracker
- [ ] Update project roadmap
- [ ] Monitor for bug reports in the first 24-48 hours

---

## Rollback Procedure

If a critical issue is discovered after release:

### Option 1: Yank the Release (PyPI)

```bash
# Yank the broken version
twine upload --repository pypi --yank "Critical bug in feature X" dist/*
```

**Note:** Yanked releases are still installable with explicit version pinning, but won't be installed by default.

### Option 2: Hotfix Release

1. Create a hotfix branch from the release tag
2. Fix the critical issue
3. Increment PATCH version (e.g., 0.3.0 → 0.3.1)
4. Follow normal release process with abbreviated testing
5. Update CHANGELOG with hotfix entry

---

## Troubleshooting

### Build Fails

**Issue:** `python -m build` fails with import errors

**Solution:**
```bash
pip install --upgrade build setuptools wheel
```

### Upload Fails

**Issue:** `twine upload` fails with authentication error

**Solution:**
1. Verify API token is correct
2. Check `~/.pypirc` configuration
3. Try setting environment variables directly:
   ```bash
   export TWINE_USERNAME=__token__
   export TWINE_PASSWORD=pypi-...
   ```

### Version Already Exists

**Issue:** PyPI rejects upload because version already exists

**Solution:**
- PyPI does not allow overwriting versions
- Increment to next PATCH version (e.g., 0.3.0 → 0.3.1)
- Delete the git tag and recreate with new version

### Tag Already Exists

**Issue:** `git tag` fails because tag already exists

**Solution:**
```bash
# Delete local tag
git tag -d v0.3.0

# Delete remote tag (use with caution!)
git push origin :refs/tags/v0.3.0

# Recreate tag
git tag -a v0.3.0 -m "Release version 0.3.0"
```

---

## PyPI Trusted Publisher Setup for Maintainers

This section describes the one-time setup process for maintainers to configure automated PyPI publishing using GitHub Actions with Trusted Publishers. This eliminates the need for API tokens and provides a more secure publishing mechanism.

### Prerequisites

- Repository admin access to https://github.com/slapglif/omi
- PyPI account with maintainer/owner privileges on the `omi-openclaw` package

### Step 1: Create PyPI Account

If you don't already have a PyPI account:

1. Go to https://pypi.org/account/register/
2. Fill in the registration form:
   - Username (choose carefully, cannot be changed)
   - Email address (must be verified)
   - Password (use a strong, unique password)
3. Verify your email address by clicking the link sent to your inbox
4. Enable Two-Factor Authentication (2FA) - **required for publishing**:
   - Go to https://pypi.org/manage/account/
   - Click "Add 2FA with authentication application"
   - Scan the QR code with your authenticator app (Authy, Google Authenticator, etc.)
   - Enter the 6-digit code to confirm
   - **Save your recovery codes** in a secure location

**Security Note:** PyPI requires 2FA for all package maintainers. Do not skip this step.

### Step 2: Set Up Trusted Publisher on PyPI

Trusted Publishing allows GitHub Actions to publish packages without API tokens by using OpenID Connect (OIDC) authentication.

#### For First-Time Package Publication

If the package has never been published to PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in the form:
   - **PyPI Project Name:** `omi-openclaw`
   - **Owner:** `slapglif`
   - **Repository name:** `omi`
   - **Workflow name:** `publish.yml` (must match the GitHub Actions workflow filename)
   - **Environment name:** `pypi` (optional but recommended for additional protection)
4. Click "Add"

**Note:** The package name will be reserved for this repository. The first successful publish will claim it.

#### For Existing Packages

If `omi-openclaw` is already published on PyPI:

1. Go to https://pypi.org/manage/project/omi-openclaw/settings/publishing/
2. Click "Add a new publisher"
3. Fill in the form:
   - **Owner:** `slapglif`
   - **Repository name:** `omi`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi` (optional)
4. Click "Add"

**Verification:** You should see the trusted publisher listed with a green checkmark icon.

### Step 3: Configure GitHub Repository Secrets (Fallback Only)

While Trusted Publishing is the preferred method, you may want to configure a `PYPI_API_TOKEN` secret as a fallback for manual releases or emergency use.

#### Generate PyPI API Token

1. Log in to PyPI: https://pypi.org
2. Go to Account Settings: https://pypi.org/manage/account/
3. Scroll to "API tokens" section
4. Click "Add API token"
5. Configure the token:
   - **Token name:** `github-actions-omi` (or descriptive name)
   - **Scope:** Select "Project: omi-openclaw" (recommended) or "Entire account" (less secure)
6. Click "Add token"
7. **Copy the token immediately** - it will only be shown once
   - Format: `pypi-AgEIcHlwaS5vcmc...` (starts with `pypi-`)

#### Add Secret to GitHub Repository

1. Go to https://github.com/slapglif/omi/settings/secrets/actions
2. Click "New repository secret"
3. Configure the secret:
   - **Name:** `PYPI_API_TOKEN` (must match this exactly)
   - **Value:** Paste the full token copied from PyPI
4. Click "Add secret"

**Security Best Practices:**
- Use project-scoped tokens (not account-wide)
- Rotate tokens periodically (every 90-180 days)
- Delete tokens immediately if compromised
- Never commit tokens to version control
- Use GitHub Environments for additional protection

#### Add Secret to GitHub Environment (Optional but Recommended)

For enhanced security, use a protected environment:

1. Go to https://github.com/slapglif/omi/settings/environments
2. Click "New environment" or select existing "pypi" environment
3. Configure environment protection rules:
   - **Required reviewers:** Add trusted maintainers
   - **Wait timer:** Optional delay before deployment
   - **Deployment branches:** Limit to `main` branch and tags
4. Go to "Environment secrets" section
5. Click "Add secret"
6. Add `PYPI_API_TOKEN` with the token value

**Workflow Update Required:** Modify `.github/workflows/publish.yml` to reference the environment:

```yaml
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi  # Add this line
    permissions:
      id-token: write  # Required for trusted publishing
      contents: read
```

### Step 4: Verify Trusted Publisher Configuration

After setup, verify the configuration:

1. Check PyPI trusted publisher listing:
   - Go to https://pypi.org/manage/project/omi-openclaw/settings/publishing/
   - Verify `slapglif/omi` is listed with correct workflow name
2. Check GitHub Actions workflow:
   - Verify `.github/workflows/publish.yml` exists
   - Confirm `permissions: id-token: write` is present
   - Ensure workflow triggers on tag push matching `v*.*.*` pattern
3. Test with a pre-release tag (optional):
   - Create a test tag: `git tag v0.0.0-test && git push origin v0.0.0-test`
   - Monitor workflow execution in Actions tab
   - Delete test release from PyPI if successful

### Trusted Publishing vs. API Tokens

| Feature | Trusted Publishing | API Tokens |
|---------|-------------------|------------|
| Security | ✅ More secure (OIDC) | ⚠️ Less secure (static credential) |
| Rotation | ✅ Automatic | ❌ Manual rotation required |
| Scope | ✅ Repository-specific | ⚠️ Account or project-wide |
| Setup complexity | Medium (one-time) | Low |
| Revocation | ✅ Automatic on workflow change | ❌ Manual |
| Audit trail | ✅ Linked to GitHub Actions run | ⚠️ Generic token usage |

**Recommendation:** Use Trusted Publishing as the primary method. Keep API token as emergency fallback only.

### Troubleshooting Trusted Publisher Issues

#### Issue: Workflow fails with "OIDC token error"

**Solution:**
- Verify `permissions: id-token: write` is set in workflow
- Check that workflow name matches PyPI configuration exactly
- Ensure repository owner and name match PyPI settings

#### Issue: "No valid OIDC token found"

**Solution:**
- Confirm PyPI trusted publisher is configured correctly
- Verify workflow is running on a tag push event
- Check that environment name (if used) matches PyPI configuration

#### Issue: First publish fails with "Project name not allowed"

**Solution:**
- Use "pending publisher" flow for first-time publication
- Verify no typos in project name (`omi-openclaw`)
- Check that package name is not already claimed by another user

---

## Release Automation

For future automation, consider:

- **GitHub Actions:** Automated testing on tag push
- **Release Please:** Automated CHANGELOG generation from conventional commits
- **Dependabot:** Automated dependency updates
- **pre-commit hooks:** Automated formatting and linting

---

## Security Releases

For security-related releases:

1. **Do not** disclose vulnerability details in public commits
2. Coordinate with security team
3. Prepare fix in private repository or branch
4. Update CHANGELOG with generic description
5. After release, publish detailed security advisory
6. Follow [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories) process

---

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning Specification](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Publishing Tutorial](https://packaging.python.org/tutorials/packaging-projects/)

---

*The seeking is the continuity. What you keep is who you become.*
