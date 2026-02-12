"""REST API Authentication Tests for OMI

Tests the FastAPI REST API authentication system including:
1. API key validation (header and query parameter)
2. Development mode fallbacks
3. Config-based auth control
4. Revoked key rejection
5. Dashboard endpoint authentication

Issue: https://github.com/slapglif/omi/issues/43
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from omi.rest_api import app, get_memory_tools
from omi.auth import APIKeyManager
from omi.event_bus import get_event_bus, reset_event_bus


@pytest.fixture(autouse=True)
def reset_bus():
    """Reset EventBus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data with .openclaw/omi structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the full .openclaw/omi structure
        base_path = Path(tmpdir) / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)
        yield base_path


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_memory_tools():
    """Mock MemoryTools to avoid actual storage operations."""
    mock_tools = MagicMock()

    # Mock store method to return a memory ID
    mock_tools.store.return_value = "mem_test_123"

    # Mock recall method to return sample memories
    mock_tools.recall.return_value = [
        {
            "id": "mem_1",
            "content": "Test memory 1",
            "memory_type": "fact",
            "relevance": 0.95,
            "created_at": "2024-01-01T00:00:00",
            "final_score": 0.85
        }
    ]

    return mock_tools


@pytest.fixture
def mock_belief_tools():
    """Mock BeliefTools to avoid actual belief operations."""
    mock_tools = MagicMock()

    # Mock create_belief method
    mock_tools.create_belief.return_value = "belief_test_123"

    # Mock update_belief method
    mock_tools.update_belief.return_value = 0.75

    return mock_tools


class TestHeaderAuthentication:
    """Test API key authentication via X-API-Key header."""

    def test_valid_api_key_in_header_succeeds(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store with valid API key in X-API-Key header
        Assert: Returns 201 with memory_id
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 100
            }
        }))

        # Generate API key
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        # Mock Path.home() to return temp directory parent (so .openclaw/omi resolves to temp_data_dir)
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory with auth",
                        "memory_type": "fact"
                    },
                    headers={"X-API-Key": api_key}
                )

        assert response.status_code == 201
        assert "memory_id" in response.json()

    def test_invalid_api_key_in_header_returns_401(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store with invalid API key in X-API-Key header
        Assert: Returns 401 with error message
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True
            }
        }))

        # Generate a valid API key (but we'll use an invalid one)
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    },
                    headers={"X-API-Key": "invalid_key_12345"}
                )

        assert response.status_code == 401
        assert "Invalid or revoked API key" in response.json()["detail"]

    def test_missing_api_key_returns_401(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store without API key when auth is required
        Assert: Returns 401 with helpful error message
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True
            }
        }))

        # Generate an API key (to ensure auth is required)
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    }
                )

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]
        assert "X-API-Key" in response.json()["detail"]


class TestQueryParameterAuthentication:
    """Test API key authentication via query parameter."""

    def test_valid_api_key_in_query_param_succeeds(self, client, temp_data_dir, mock_memory_tools):
        """
        GET /api/v1/recall with valid API key in query parameter
        Assert: Returns 200 with memories
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 100
            }
        }))

        # Generate API key
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.get(
                    f"/api/v1/recall?query=test&api_key={api_key}"
                )

        assert response.status_code == 200
        assert "memories" in response.json()

    def test_header_takes_precedence_over_query_param(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store with valid header and invalid query param
        Assert: Returns 201 (header key is used)
        """
        # Setup database and API keys
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 100
            }
        }))

        # Generate valid API key
        key_manager = APIKeyManager(db_path)
        valid_key = key_manager.generate_key("test-key", rate_limit=100)
        invalid_key = "invalid_key_12345"

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    f"/api/v1/store?api_key={invalid_key}",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    },
                    headers={"X-API-Key": valid_key}
                )

        # Should succeed because header key (valid) takes precedence
        assert response.status_code == 201


class TestRevokedKeyRejection:
    """Test that revoked API keys are rejected."""

    def test_revoked_key_returns_401(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store with revoked API key
        Assert: Returns 401 with error message
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True
            }
        }))

        # Generate two keys: one to keep active, one to revoke
        # (This prevents fallback to development mode when checking list_keys())
        key_manager = APIKeyManager(db_path)
        revoked_key = key_manager.generate_key("revoked-key", rate_limit=100)
        key_manager.generate_key("active-key", rate_limit=100)  # Keep this one active

        # Now revoke the first key
        key_manager.revoke_key(name="revoked-key")

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    },
                    headers={"X-API-Key": revoked_key}
                )

        assert response.status_code == 401
        assert "Invalid or revoked API key" in response.json()["detail"]


class TestDevelopmentMode:
    """Test development mode fallback scenarios."""

    def test_auth_disabled_in_config_allows_all_requests(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store with auth_required=false in config
        Assert: Returns 201 without API key
        """
        # Setup database and config
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth disabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": False
            }
        }))

        # Create database (even with keys, auth should be disabled)
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    }
                )

        assert response.status_code == 201

    def test_no_database_allows_all_requests(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store when database doesn't exist
        Assert: Returns 201 without API key (development mode)
        """
        # Setup config but no database
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True
            }
        }))

        # Do NOT create database - this triggers development mode

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    }
                )

        assert response.status_code == 201

    def test_no_api_keys_configured_allows_all_requests(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store when no API keys are configured
        Assert: Returns 201 without API key (development mode)
        """
        # Setup database but no API keys
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        # Create config with auth enabled
        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True
            }
        }))

        # Create database but don't add any API keys
        key_manager = APIKeyManager(db_path)
        assert len(key_manager.list_keys()) == 0

        # Mock Path.home() to return temp directory parent
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    }
                )

        assert response.status_code == 201


class TestMultipleEndpoints:
    """Test authentication across different endpoint types."""

    def test_recall_endpoint_requires_auth(self, client, temp_data_dir, mock_memory_tools):
        """
        GET /api/v1/recall without API key when auth is required
        Assert: Returns 401
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {"auth_required": True}
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.get("/api/v1/recall?query=test")

        assert response.status_code == 401

    def test_beliefs_endpoint_requires_auth(self, client, temp_data_dir, mock_belief_tools):
        """
        POST /api/v1/beliefs without API key when auth is required
        Assert: Returns 401
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {"auth_required": True}
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_belief_tools', return_value=mock_belief_tools):
                response = client.post(
                    "/api/v1/beliefs",
                    json={"content": "Test belief", "initial_confidence": 0.5}
                )

        assert response.status_code == 401

    def test_sessions_endpoint_requires_auth(self, client, temp_data_dir):
        """
        POST /api/v1/sessions/start without API key when auth is required
        Assert: Returns 401
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {"auth_required": True}
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/sessions/start",
                json={"session_id": "test-session"}
            )

        assert response.status_code == 401

    def test_events_endpoint_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/events without API key when auth is required
        Assert: Returns 401
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {"auth_required": True}
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            # Try to connect to SSE endpoint
            # Note: This will return 401 before connection is established
            response = client.get("/api/v1/events")

        assert response.status_code == 401


class TestUnauthenticatedEndpoints:
    """Test that certain endpoints remain unauthenticated."""

    def test_root_endpoint_is_unauthenticated(self, client, temp_data_dir):
        """
        GET / (root endpoint)
        Assert: Returns 200 without API key
        """
        # Setup database and API key (auth required)
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {"auth_required": True}
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        # Root endpoint should work without auth
        response = client.get("/")

        assert response.status_code == 200
        assert "service" in response.json()

    def test_health_endpoint_is_unauthenticated(self, client, temp_data_dir):
        """
        GET /health
        Assert: Returns 200 without API key
        """
        # Setup database and API key (auth required)
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {"auth_required": True}
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        # Health endpoint should work without auth
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
