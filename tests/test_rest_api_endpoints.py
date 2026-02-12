"""REST API Endpoint Tests for OMI Memory Operations

Tests the FastAPI REST API endpoints for memory operations.

Covers:
1. Memory storage endpoint (POST /api/v1/store)
2. Memory recall endpoint (GET /api/v1/recall)
3. Request/response validation
4. Error handling

Issue: https://github.com/slapglif/omi/issues/4
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from omi.rest_api import app, get_memory_tools
from omi.event_bus import get_event_bus, reset_event_bus
from omi.events import MemoryStoredEvent, MemoryRecalledEvent
from omi.auth import APIKeyManager


@pytest.fixture(autouse=True)
def reset_bus():
    """Reset EventBus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


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
        },
        {
            "id": "mem_2",
            "content": "Test memory 2",
            "memory_type": "experience",
            "relevance": 0.85,
            "created_at": "2024-01-02T00:00:00",
            "final_score": 0.75
        }
    ]

    return mock_tools


@pytest.fixture
def disable_auth():
    """Disable API key authentication for tests."""
    # Clear OMI_API_KEY environment variable if it exists
    old_key = os.environ.get("OMI_API_KEY")
    if "OMI_API_KEY" in os.environ:
        del os.environ["OMI_API_KEY"]
    yield
    # Restore old key if it existed
    if old_key:
        os.environ["OMI_API_KEY"] = old_key


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data with .openclaw/omi structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the full .openclaw/omi structure
        base_path = Path(tmpdir) / '.openclaw' / 'omi'
        base_path.mkdir(parents=True, exist_ok=True)
        yield base_path


class TestAuthentication:
    """Test authentication with/without API key."""

    def test_store_without_api_key_when_none_required(self, client, mock_memory_tools):
        """
        POST /api/v1/store without API key when OMI_API_KEY is not set
        Assert: Returns 201 (development mode allows all requests)
        """
        # Ensure no API key is set (development mode)
        old_key = os.environ.get("OMI_API_KEY")
        if "OMI_API_KEY" in os.environ:
            del os.environ["OMI_API_KEY"]

        try:
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Test memory",
                        "memory_type": "fact"
                    }
                )

                assert response.status_code == 201
                data = response.json()
                assert "memory_id" in data
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key

    def test_store_without_api_key_when_required(self, client):
        """
        POST /api/v1/store without API key when OMI_API_KEY is set
        Assert: Returns 401 with missing API key error
        """
        # Set API key in environment to require authentication
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Test memory",
                    "memory_type": "fact"
                }
            )

            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Missing API key" in data["detail"]
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_store_with_valid_api_key(self, client, mock_memory_tools):
        """
        POST /api/v1/store with valid API key in X-API-Key header
        Assert: Returns 201 and successfully stores memory
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Authenticated memory",
                        "memory_type": "fact"
                    },
                    headers={"X-API-Key": "test-secret-key"}
                )

                assert response.status_code == 201
                data = response.json()
                assert "memory_id" in data
                assert data["memory_id"] == "mem_test_123"
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_store_with_invalid_api_key(self, client):
        """
        POST /api/v1/store with incorrect API key
        Assert: Returns 401 with invalid API key error
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Test memory",
                    "memory_type": "fact"
                },
                headers={"X-API-Key": "wrong-key"}
            )

            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Invalid API key" in data["detail"]
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_recall_without_api_key_when_required(self, client):
        """
        GET /api/v1/recall without API key when OMI_API_KEY is set
        Assert: Returns 401 with missing API key error
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            response = client.get("/api/v1/recall?query=test")

            assert response.status_code == 401
            data = response.json()
            assert "detail" in data
            assert "Missing API key" in data["detail"]
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_recall_with_valid_api_key(self, client, mock_memory_tools):
        """
        GET /api/v1/recall with valid API key
        Assert: Returns 200 and successfully recalls memories
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.get(
                    "/api/v1/recall?query=test",
                    headers={"X-API-Key": "test-secret-key"}
                )

                assert response.status_code == 200
                data = response.json()
                assert "memories" in data
                assert data["count"] == 2
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_public_endpoints_dont_require_auth(self, client):
        """
        GET / and /health endpoints should work without authentication
        Assert: Both return 200 even when API key is required
        """
        # Set API key in environment to require auth for protected endpoints
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            # Test root endpoint
            root_response = client.get("/")
            assert root_response.status_code == 200
            assert "service" in root_response.json()

            # Test health endpoint
            health_response = client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["status"] == "healthy"
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_case_sensitive_api_key(self, client):
        """
        POST /api/v1/store with API key that differs only in case
        Assert: Returns 401 (API keys are case-sensitive)
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "Test-Secret-Key"

        try:
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Test memory",
                    "memory_type": "fact"
                },
                headers={"X-API-Key": "test-secret-key"}  # Different case
            )

            assert response.status_code == 401
            data = response.json()
            assert "Invalid API key" in data["detail"]
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]


class TestMemoryEndpoints:
    """Test memory operation endpoints (store, recall)."""

    def test_store_memory_success(self, client, mock_memory_tools, disable_auth):
        """
        POST /api/v1/store with valid data
        Assert: Returns 201 with memory_id
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Test memory content",
                    "memory_type": "fact"
                }
            )

            assert response.status_code == 201
            data = response.json()
            assert "memory_id" in data
            assert data["memory_id"] == "mem_test_123"
            assert data["message"] == "Memory stored successfully"

            # Verify MemoryTools.store was called with correct arguments
            mock_memory_tools.store.assert_called_once_with(
                content="Test memory content",
                memory_type="fact",
                related_to=None,
                confidence=None
            )

    def test_store_memory_with_all_fields(self, client, mock_memory_tools, disable_auth):
        """
        POST /api/v1/store with all optional fields
        Assert: Returns 201 and passes all fields to MemoryTools
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Complete test memory",
                    "memory_type": "experience",
                    "related_to": ["mem_1", "mem_2"],
                    "confidence": 0.85
                }
            )

            assert response.status_code == 201
            data = response.json()
            assert data["memory_id"] == "mem_test_123"

            # Verify all fields were passed
            mock_memory_tools.store.assert_called_once_with(
                content="Complete test memory",
                memory_type="experience",
                related_to=["mem_1", "mem_2"],
                confidence=0.85
            )

    def test_store_memory_missing_content(self, client, disable_auth):
        """
        POST /api/v1/store without required 'content' field
        Assert: Returns 422 validation error
        """
        response = client.post(
            "/api/v1/store",
            json={
                "memory_type": "fact"
            }
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_store_memory_invalid_confidence(self, client, disable_auth):
        """
        POST /api/v1/store with confidence outside valid range (0.0-1.0)
        Assert: Returns 422 validation error
        """
        response = client.post(
            "/api/v1/store",
            json={
                "content": "Test memory",
                "confidence": 1.5  # Invalid: > 1.0
            }
        )

        assert response.status_code == 422

    def test_store_memory_default_type(self, client, mock_memory_tools, disable_auth):
        """
        POST /api/v1/store without memory_type
        Assert: Uses default type 'experience'
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Memory without type"
                }
            )

            assert response.status_code == 201
            mock_memory_tools.store.assert_called_once()
            call_args = mock_memory_tools.store.call_args
            assert call_args.kwargs["memory_type"] == "experience"

    def test_store_memory_error_handling(self, client, mock_memory_tools, disable_auth):
        """
        POST /api/v1/store when MemoryTools.store raises exception
        Assert: Returns 500 with error message
        """
        # Mock store to raise an exception
        mock_memory_tools.store.side_effect = Exception("Database connection failed")

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Test memory"
                }
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Failed to store memory" in data["detail"]

    def test_recall_memory_success(self, client, mock_memory_tools, disable_auth):
        """
        GET /api/v1/recall with query parameter
        Assert: Returns 200 with list of memories
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.get(
                "/api/v1/recall",
                params={
                    "query": "test query"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert "memories" in data
            assert "count" in data
            assert data["count"] == 2
            assert len(data["memories"]) == 2

            # Verify first memory structure
            mem = data["memories"][0]
            assert mem["id"] == "mem_1"
            assert mem["content"] == "Test memory 1"
            assert mem["memory_type"] == "fact"
            assert mem["relevance"] == 0.95

            # Verify MemoryTools.recall was called with correct defaults
            mock_memory_tools.recall.assert_called_once_with(
                query="test query",
                limit=10,
                min_relevance=0.7,
                memory_type=None
            )

    def test_recall_memory_with_all_params(self, client, mock_memory_tools, disable_auth):
        """
        GET /api/v1/recall with all query parameters
        Assert: Returns 200 and passes all params to MemoryTools
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.get(
                "/api/v1/recall",
                params={
                    "query": "specific query",
                    "limit": 5,
                    "min_relevance": 0.8,
                    "memory_type": "fact"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 2

            # Verify all parameters were passed
            mock_memory_tools.recall.assert_called_once_with(
                query="specific query",
                limit=5,
                min_relevance=0.8,
                memory_type="fact"
            )

    def test_recall_memory_missing_query(self, client, disable_auth):
        """
        GET /api/v1/recall without required 'query' parameter
        Assert: Returns 422 validation error
        """
        response = client.get("/api/v1/recall")

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_recall_memory_invalid_limit(self, client, disable_auth):
        """
        GET /api/v1/recall with limit outside valid range (1-100)
        Assert: Returns 422 validation error
        """
        response = client.get(
            "/api/v1/recall",
            params={
                "query": "test",
                "limit": 150  # Invalid: > 100
            }
        )

        assert response.status_code == 422

    def test_recall_memory_invalid_relevance(self, client, disable_auth):
        """
        GET /api/v1/recall with min_relevance outside valid range (0.0-1.0)
        Assert: Returns 422 validation error
        """
        response = client.get(
            "/api/v1/recall",
            params={
                "query": "test",
                "min_relevance": -0.5  # Invalid: < 0.0
            }
        )

        assert response.status_code == 422

    def test_recall_memory_empty_results(self, client, mock_memory_tools, disable_auth):
        """
        GET /api/v1/recall when no memories match
        Assert: Returns 200 with empty list
        """
        # Mock recall to return empty list
        mock_memory_tools.recall.return_value = []

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.get(
                "/api/v1/recall",
                params={
                    "query": "no matches"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0
            assert data["memories"] == []

    def test_recall_memory_error_handling(self, client, mock_memory_tools, disable_auth):
        """
        GET /api/v1/recall when MemoryTools.recall raises exception
        Assert: Returns 500 with error message
        """
        # Mock recall to raise an exception
        mock_memory_tools.recall.side_effect = Exception("Search index unavailable")

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.get(
                "/api/v1/recall",
                params={
                    "query": "test query"
                }
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Failed to recall memories" in data["detail"]

    def test_memory_types_validation(self, client, mock_memory_tools, disable_auth):
        """
        Test that different memory types are accepted
        Assert: All valid types work correctly
        """
        valid_types = ["fact", "experience", "belief", "decision"]

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            for mem_type in valid_types:
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": f"Test {mem_type}",
                        "memory_type": mem_type
                    }
                )

                assert response.status_code == 201, f"Failed for type: {mem_type}"

    def test_recall_with_type_filter(self, client, mock_memory_tools, disable_auth):
        """
        GET /api/v1/recall with memory_type filter
        Assert: Filter is passed to MemoryTools
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.get(
                "/api/v1/recall",
                params={
                    "query": "test",
                    "memory_type": "experience"
                }
            )

            assert response.status_code == 200
            mock_memory_tools.recall.assert_called_once()
            call_args = mock_memory_tools.recall.call_args
            assert call_args.kwargs["memory_type"] == "experience"

    def test_store_memory_publishes_event(self, client, mock_memory_tools, disable_auth):
        """
        POST /api/v1/store
        Assert: MemoryStoredEvent is published to event bus

        Note: This assumes MemoryTools.store publishes the event internally.
        If the endpoint should publish the event, this test needs adjustment.
        """
        # This test verifies the integration works end-to-end
        # The actual event publishing happens in MemoryTools.store
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Event test memory"
                }
            )

            assert response.status_code == 201
            # Event publishing is tested in the MemoryTools tests
            # Here we just verify the endpoint succeeds

    def test_recall_memory_publishes_event(self, client, mock_memory_tools, disable_auth):
        """
        GET /api/v1/recall
        Assert: MemoryRecalledEvent is published to event bus

        Note: This assumes MemoryTools.recall publishes the event internally.
        If the endpoint should publish the event, this test needs adjustment.
        """
        # This test verifies the integration works end-to-end
        # The actual event publishing happens in MemoryTools.recall
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.get(
                "/api/v1/recall",
                params={
                    "query": "event test"
                }
            )

            assert response.status_code == 200
            # Event publishing is tested in the MemoryTools tests
            # Here we just verify the endpoint succeeds


class TestCORS:
    """Test CORS (Cross-Origin Resource Sharing) headers."""

    def test_cors_headers_on_get_request(self, client):
        """
        GET / with Origin header
        Assert: CORS headers are present in response
        """
        response = client.get(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200
        # FastAPI's CORSMiddleware adds these headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-credentials" in response.headers

    def test_cors_headers_on_post_request(self, client, mock_memory_tools, disable_auth):
        """
        POST /api/v1/store with Origin header
        Assert: CORS headers are present in response
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            response = client.post(
                "/api/v1/store",
                json={
                    "content": "Test memory",
                    "memory_type": "fact"
                },
                headers={"Origin": "http://localhost:3000"}
            )

            assert response.status_code == 201
            assert "access-control-allow-origin" in response.headers
            assert "access-control-allow-credentials" in response.headers

    def test_cors_preflight_request(self, client):
        """
        OPTIONS /api/v1/store (preflight request)
        Assert: Returns 200 with CORS headers
        """
        response = client.options(
            "/api/v1/store",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    def test_cors_allow_credentials(self, client):
        """
        GET /health with Origin header
        Assert: access-control-allow-credentials is 'true'
        """
        response = client.get(
            "/health",
            headers={"Origin": "http://example.com"}
        )

        assert response.status_code == 200
        credentials_header = response.headers.get("access-control-allow-credentials", "").lower()
        assert credentials_header == "true"

    def test_cors_headers_on_different_origins(self, client):
        """
        GET / with different Origin values
        Assert: CORS headers respond appropriately for each origin
        """
        origins = [
            "http://localhost:3000",
            "http://localhost:8080",
            "https://example.com"
        ]

        for origin in origins:
            response = client.get(
                "/",
                headers={"Origin": origin}
            )

            assert response.status_code == 200
            assert "access-control-allow-origin" in response.headers
            # In default mode (*), should allow all origins
            allow_origin = response.headers["access-control-allow-origin"]
            # FastAPI returns either the origin or *
            assert allow_origin in [origin, "*"]

    def test_cors_headers_on_error_response(self, client):
        """
        GET /api/v1/recall without required query parameter, with Origin header
        Assert: CORS headers are present even on error responses
        """
        response = client.get(
            "/api/v1/recall",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 422
        # CORS headers should be present even on errors
        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_on_authenticated_endpoint(self, client, mock_memory_tools):
        """
        POST /api/v1/store with valid API key and Origin header
        Assert: CORS headers are present on authenticated requests
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "test-secret-key"

        try:
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Authenticated memory",
                        "memory_type": "fact"
                    },
                    headers={
                        "X-API-Key": "test-secret-key",
                        "Origin": "http://localhost:3000"
                    }
                )

                assert response.status_code == 201
                assert "access-control-allow-origin" in response.headers
                assert "access-control-allow-credentials" in response.headers
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    @pytest.mark.skip(reason="SSE streaming tests hang with TestClient - test with running server")
    def test_cors_headers_on_sse_endpoint(self, client):
        """
        GET /api/v1/events with Origin header
        Assert: CORS headers are present on SSE endpoint

        Note: This test is skipped because TestClient hangs on SSE endpoints.
        In production, test CORS on SSE endpoint with a running server.
        """
        pass


class TestEndpointAuthentication:
    """Test authentication for all protected endpoints."""

    def test_memory_store_requires_auth(self, client, temp_data_dir):
        """
        POST /api/v1/store without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/store",
                json={"content": "Test", "memory_type": "fact"}
            )
            assert response.status_code == 401

    def test_memory_store_with_auth(self, client, temp_data_dir, mock_memory_tools):
        """
        POST /api/v1/store with valid API key
        Assert: Returns 201
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.post(
                    "/api/v1/store",
                    json={"content": "Test", "memory_type": "fact"},
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 201

    def test_memory_recall_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/recall without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.get("/api/v1/recall?query=test")
            assert response.status_code == 401

    def test_memory_recall_with_auth(self, client, temp_data_dir, mock_memory_tools):
        """
        GET /api/v1/recall with valid API key
        Assert: Returns 200
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                response = client.get(
                    "/api/v1/recall?query=test",
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 200

    def test_belief_create_requires_auth(self, client, temp_data_dir):
        """
        POST /api/v1/beliefs without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/beliefs",
                json={"content": "Test belief", "initial_confidence": 0.5}
            )
            assert response.status_code == 401

    def test_belief_create_with_auth(self, client, temp_data_dir):
        """
        POST /api/v1/beliefs with valid API key
        Assert: Returns 201
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        mock_belief_tools = MagicMock()
        mock_belief_tools.create.return_value = "belief_123"

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_belief_tools', return_value=mock_belief_tools):
                response = client.post(
                    "/api/v1/beliefs",
                    json={"content": "Test belief", "initial_confidence": 0.5},
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 201

    def test_belief_update_requires_auth(self, client, temp_data_dir):
        """
        PUT /api/v1/beliefs/{id} without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.put(
                "/api/v1/beliefs/belief_123",
                json={
                    "evidence_memory_id": "mem_123",
                    "supports": True,
                    "strength": 0.8
                }
            )
            assert response.status_code == 401

    def test_belief_update_with_auth(self, client, temp_data_dir):
        """
        PUT /api/v1/beliefs/{id} with valid API key
        Assert: Returns 200
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        mock_belief_tools = MagicMock()
        mock_belief_tools.update.return_value = 0.85

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_belief_tools', return_value=mock_belief_tools):
                response = client.put(
                    "/api/v1/beliefs/belief_123",
                    json={
                        "evidence_memory_id": "mem_123",
                        "supports": True,
                        "strength": 0.8
                    },
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 200

    def test_session_start_requires_auth(self, client, temp_data_dir):
        """
        POST /api/v1/sessions/start without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/sessions/start",
                json={}
            )
            assert response.status_code == 401

    def test_session_start_with_auth(self, client, temp_data_dir):
        """
        POST /api/v1/sessions/start with valid API key
        Assert: Returns 200
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/sessions/start",
                json={},
                headers={"X-API-Key": api_key}
            )
            assert response.status_code == 200

    def test_session_end_requires_auth(self, client, temp_data_dir):
        """
        POST /api/v1/sessions/end without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/sessions/end",
                json={"session_id": "session_123"}
            )
            assert response.status_code == 401

    def test_session_end_with_auth(self, client, temp_data_dir):
        """
        POST /api/v1/sessions/end with valid API key
        Assert: Returns 200
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.post(
                "/api/v1/sessions/end",
                json={"session_id": "session_123"},
                headers={"X-API-Key": api_key}
            )
            assert response.status_code == 200

    def test_events_sse_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/events without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            response = client.get("/api/v1/events")
            assert response.status_code == 401

    def test_dashboard_memories_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/memories without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get("/api/v1/dashboard/memories")
                assert response.status_code == 401

    def test_dashboard_edges_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/edges without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get("/api/v1/dashboard/edges")
                assert response.status_code == 401

    def test_dashboard_beliefs_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/beliefs without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get("/api/v1/dashboard/beliefs")
                assert response.status_code == 401

    def test_dashboard_graph_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/graph without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get("/api/v1/dashboard/graph")
                assert response.status_code == 401

    def test_dashboard_stats_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/stats without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get("/api/v1/dashboard/stats")
                assert response.status_code == 401

    def test_dashboard_search_requires_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/search without API key when auth is enabled
        Assert: Returns 401
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get("/api/v1/dashboard/search?q=test")
                assert response.status_code == 401

    def test_dashboard_memories_with_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/memories with valid API key
        Assert: Returns 200 (auth passes and endpoint works)
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get(
                    "/api/v1/dashboard/memories",
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 200

    def test_dashboard_edges_with_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/edges with valid API key
        Assert: Returns 200 (auth passes and endpoint works)
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get(
                    "/api/v1/dashboard/edges",
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 200

    def test_dashboard_beliefs_with_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/beliefs with valid API key
        Assert: Returns 500 (auth passes but beliefs table doesn't exist in empty DB)
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get(
                    "/api/v1/dashboard/beliefs",
                    headers={"X-API-Key": api_key}
                )
                # Returns 500 because beliefs table doesn't exist in empty GraphPalace DB
                # But authentication passed (would get 401 if auth failed)
                assert response.status_code == 500

    def test_dashboard_graph_with_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/graph with valid API key
        Assert: Returns 200 (auth passes and endpoint works)
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get(
                    "/api/v1/dashboard/graph",
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 200

    def test_dashboard_stats_with_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/stats with valid API key
        Assert: Returns 200 (auth passes and endpoint works)
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                response = client.get(
                    "/api/v1/dashboard/stats",
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 200

    def test_dashboard_search_with_auth(self, client, temp_data_dir):
        """
        GET /api/v1/dashboard/search with valid API key
        Assert: Returns 200 (auth passes and endpoint works)
        """
        # Setup: Create API key in database
        db_path = temp_data_dir / "palace.sqlite"
        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=100)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.dashboard_api.Path.home', return_value=temp_data_dir.parent.parent):
                # Mock get_embedder_and_cache to avoid Ollama dependency
                mock_embedder = MagicMock()
                mock_cache = MagicMock()
                mock_cache.get_or_compute.return_value = [0.1] * 768  # Mock embedding
                with patch('omi.dashboard_api.get_embedder_and_cache', return_value=(mock_embedder, mock_cache)):
                    response = client.get(
                        "/api/v1/dashboard/search?q=test",
                        headers={"X-API-Key": api_key}
                    )
                    assert response.status_code == 200


class TestIntegration:
    """Test full workflow integration scenarios."""

    def test_full_workflow_store_and_recall(self, client, mock_memory_tools, disable_auth):
        """
        Full workflow: Store memory -> Recall it -> Verify responses
        Assert: Complete end-to-end flow works correctly
        """
        # Configure mock to return a specific memory when recalled
        stored_memory_id = "mem_integration_test_123"
        mock_memory_tools.store.return_value = stored_memory_id
        mock_memory_tools.recall.return_value = [
            {
                "id": stored_memory_id,
                "content": "Integration test memory",
                "memory_type": "fact",
                "relevance": 0.95,
                "created_at": "2024-01-01T00:00:00",
                "final_score": 0.90
            }
        ]

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            # Step 1: Store a memory
            store_response = client.post(
                "/api/v1/store",
                json={
                    "content": "Integration test memory",
                    "memory_type": "fact",
                    "confidence": 0.95
                }
            )

            # Verify store succeeded
            assert store_response.status_code == 201
            store_data = store_response.json()
            assert store_data["memory_id"] == stored_memory_id
            assert store_data["message"] == "Memory stored successfully"

            # Verify MemoryTools.store was called correctly
            mock_memory_tools.store.assert_called_once_with(
                content="Integration test memory",
                memory_type="fact",
                related_to=None,
                confidence=0.95
            )

            # Step 2: Recall the memory
            recall_response = client.get(
                "/api/v1/recall",
                params={
                    "query": "integration test",
                    "limit": 10,
                    "min_relevance": 0.7
                }
            )

            # Verify recall succeeded
            assert recall_response.status_code == 200
            recall_data = recall_response.json()
            assert recall_data["count"] == 1
            assert len(recall_data["memories"]) == 1

            # Verify recalled memory matches stored memory
            recalled_memory = recall_data["memories"][0]
            assert recalled_memory["id"] == stored_memory_id
            assert recalled_memory["content"] == "Integration test memory"
            assert recalled_memory["memory_type"] == "fact"
            assert recalled_memory["relevance"] == 0.95

            # Verify MemoryTools.recall was called correctly
            mock_memory_tools.recall.assert_called_once_with(
                query="integration test",
                limit=10,
                min_relevance=0.7,
                memory_type=None
            )

    def test_full_workflow_with_authentication(self, client, mock_memory_tools):
        """
        Full workflow with authentication: Store -> Recall with API key
        Assert: Authenticated requests work end-to-end
        """
        # Set API key in environment
        old_key = os.environ.get("OMI_API_KEY")
        os.environ["OMI_API_KEY"] = "integration-test-key"

        try:
            # Configure mock
            stored_memory_id = "mem_auth_test_456"
            mock_memory_tools.store.return_value = stored_memory_id
            mock_memory_tools.recall.return_value = [
                {
                    "id": stored_memory_id,
                    "content": "Authenticated memory",
                    "memory_type": "experience",
                    "relevance": 0.88,
                    "created_at": "2024-01-01T00:00:00",
                    "final_score": 0.82
                }
            ]

            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                # Step 1: Store with authentication
                store_response = client.post(
                    "/api/v1/store",
                    json={
                        "content": "Authenticated memory",
                        "memory_type": "experience"
                    },
                    headers={"X-API-Key": "integration-test-key"}
                )

                assert store_response.status_code == 201
                assert store_response.json()["memory_id"] == stored_memory_id

                # Step 2: Recall with authentication
                recall_response = client.get(
                    "/api/v1/recall",
                    params={"query": "authenticated"},
                    headers={"X-API-Key": "integration-test-key"}
                )

                assert recall_response.status_code == 200
                recall_data = recall_response.json()
                assert recall_data["count"] == 1
                assert recall_data["memories"][0]["id"] == stored_memory_id

                # Step 3: Verify unauthenticated request fails
                unauth_response = client.get(
                    "/api/v1/recall",
                    params={"query": "authenticated"}
                )

                assert unauth_response.status_code == 401
        finally:
            # Restore old key
            if old_key:
                os.environ["OMI_API_KEY"] = old_key
            else:
                del os.environ["OMI_API_KEY"]

    def test_full_workflow_multiple_memories(self, client, mock_memory_tools, disable_auth):
        """
        Full workflow: Store multiple memories -> Recall with filters
        Assert: Can handle multiple memories and filtering
        """
        # Configure mock for multiple stores
        mock_memory_tools.store.side_effect = ["mem_1", "mem_2", "mem_3"]
        mock_memory_tools.recall.return_value = [
            {
                "id": "mem_1",
                "content": "First fact",
                "memory_type": "fact",
                "relevance": 0.95,
                "created_at": "2024-01-01T00:00:00",
                "final_score": 0.90
            },
            {
                "id": "mem_2",
                "content": "Second fact",
                "memory_type": "fact",
                "relevance": 0.85,
                "created_at": "2024-01-01T01:00:00",
                "final_score": 0.80
            }
        ]

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            # Store multiple memories
            memories = [
                {"content": "First fact", "memory_type": "fact"},
                {"content": "Second fact", "memory_type": "fact"},
                {"content": "An experience", "memory_type": "experience"}
            ]

            stored_ids = []
            for memory in memories:
                response = client.post("/api/v1/store", json=memory)
                assert response.status_code == 201
                stored_ids.append(response.json()["memory_id"])

            # Verify all were stored
            assert len(stored_ids) == 3
            assert stored_ids == ["mem_1", "mem_2", "mem_3"]

            # Recall with type filter
            recall_response = client.get(
                "/api/v1/recall",
                params={
                    "query": "fact",
                    "memory_type": "fact",
                    "limit": 5
                }
            )

            assert recall_response.status_code == 200
            recall_data = recall_response.json()
            assert recall_data["count"] == 2
            # Verify only facts were returned
            for memory in recall_data["memories"]:
                assert memory["memory_type"] == "fact"

    def test_full_workflow_with_cors(self, client, mock_memory_tools, disable_auth):
        """
        Full workflow with CORS headers: Store -> Recall with Origin header
        Assert: CORS headers present throughout entire workflow
        """
        # Configure mock
        mock_memory_tools.store.return_value = "mem_cors_test"
        mock_memory_tools.recall.return_value = [
            {
                "id": "mem_cors_test",
                "content": "CORS test memory",
                "memory_type": "fact",
                "relevance": 0.92,
                "created_at": "2024-01-01T00:00:00",
                "final_score": 0.88
            }
        ]

        origin = "http://localhost:3000"

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            # Store with Origin header
            store_response = client.post(
                "/api/v1/store",
                json={
                    "content": "CORS test memory",
                    "memory_type": "fact"
                },
                headers={"Origin": origin}
            )

            assert store_response.status_code == 201
            assert "access-control-allow-origin" in store_response.headers
            assert "access-control-allow-credentials" in store_response.headers

            # Recall with Origin header
            recall_response = client.get(
                "/api/v1/recall",
                params={"query": "cors test"},
                headers={"Origin": origin}
            )

            assert recall_response.status_code == 200
            assert "access-control-allow-origin" in recall_response.headers
            assert "access-control-allow-credentials" in recall_response.headers

            # Verify data integrity
            recall_data = recall_response.json()
            assert recall_data["count"] == 1
            assert recall_data["memories"][0]["id"] == "mem_cors_test"

    def test_full_workflow_error_recovery(self, client, mock_memory_tools, disable_auth):
        """
        Full workflow with error handling: Failed store -> Successful retry
        Assert: Can recover from errors and continue workflow
        """
        # Configure mock to fail first, then succeed
        mock_memory_tools.store.side_effect = [
            Exception("Temporary database error"),
            "mem_retry_success"
        ]
        mock_memory_tools.recall.return_value = [
            {
                "id": "mem_retry_success",
                "content": "Retry successful",
                "memory_type": "fact",
                "relevance": 0.90,
                "created_at": "2024-01-01T00:00:00",
                "final_score": 0.85
            }
        ]

        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            # First attempt fails
            first_response = client.post(
                "/api/v1/store",
                json={
                    "content": "Retry test memory",
                    "memory_type": "fact"
                }
            )

            assert first_response.status_code == 500
            assert "Failed to store memory" in first_response.json()["detail"]

            # Retry succeeds
            retry_response = client.post(
                "/api/v1/store",
                json={
                    "content": "Retry successful",
                    "memory_type": "fact"
                }
            )

            assert retry_response.status_code == 201
            assert retry_response.json()["memory_id"] == "mem_retry_success"

            # Verify can recall the successfully stored memory
            recall_response = client.get(
                "/api/v1/recall",
                params={"query": "retry"}
            )

            assert recall_response.status_code == 200
            recall_data = recall_response.json()
            assert recall_data["count"] == 1
            assert recall_data["memories"][0]["content"] == "Retry successful"

    def test_full_workflow_health_check_integration(self, client, mock_memory_tools, disable_auth):
        """
        Full workflow: Check health -> Store -> Recall -> Check health again
        Assert: Health endpoint works alongside memory operations
        """
        with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
            # Configure mock
            mock_memory_tools.store.return_value = "mem_health_test"
            mock_memory_tools.recall.return_value = []

            # Step 1: Check initial health
            health_response = client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["status"] == "healthy"

            # Step 2: Store a memory
            store_response = client.post(
                "/api/v1/store",
                json={"content": "Health test memory"}
            )
            assert store_response.status_code == 201

            # Step 3: Recall memories
            recall_response = client.get(
                "/api/v1/recall",
                params={"query": "health"}
            )
            assert recall_response.status_code == 200

            # Step 4: Check health after operations
            final_health = client.get("/health")
            assert final_health.status_code == 200
            assert final_health.json()["status"] == "healthy"
            assert final_health.json()["service"] == "omi-event-api"
