"""REST API Rate Limiting Tests for OMI

Tests the FastAPI REST API rate limiting system including:
1. Requests under limit succeed
2. Request over limit returns 429 with Retry-After header
3. Rate limit resets after window expires
4. Different API keys have separate rate limits
5. Custom rate limits per key work correctly
6. Rate limiter sliding window behavior

Issue: https://github.com/slapglif/omi/issues/43
"""
import pytest
import time
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from omi.rest_api import app, rate_limiter
from omi.auth import APIKeyManager, RateLimiter
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
def reset_rate_limiter():
    """Reset the global rate limiter before and after each test."""
    rate_limiter.reset()
    yield
    rate_limiter.reset()


class TestRateLimiterUnit:
    """Test RateLimiter class directly (unit tests)."""

    def test_requests_under_limit_succeed(self):
        """
        Make requests under the rate limit
        Assert: All requests are allowed
        """
        limiter = RateLimiter(window_seconds=60)
        api_key = "test_key_123"
        limit = 10

        # Make 10 requests (all should succeed)
        for i in range(limit):
            allowed, retry_after = limiter.check_rate_limit(api_key, limit)
            assert allowed is True, f"Request {i+1} should be allowed"
            assert retry_after == 0, f"retry_after should be 0 for allowed request {i+1}"

    def test_request_over_limit_returns_false_with_retry_after(self):
        """
        Make requests exceeding the rate limit
        Assert: Excess requests are denied with retry_after > 0
        """
        limiter = RateLimiter(window_seconds=60)
        api_key = "test_key_123"
        limit = 5

        # Make 5 requests (all should succeed)
        for i in range(limit):
            allowed, retry_after = limiter.check_rate_limit(api_key, limit)
            assert allowed is True

        # 6th request should be denied
        allowed, retry_after = limiter.check_rate_limit(api_key, limit)
        assert allowed is False
        assert retry_after > 0
        assert retry_after <= 60  # Should be within the window

    def test_rate_limit_resets_after_window(self):
        """
        Make requests, wait for window to expire, make more requests
        Assert: Requests succeed after window expires
        """
        limiter = RateLimiter(window_seconds=2)  # Short window for testing
        api_key = "test_key_123"
        limit = 3

        # Make 3 requests (all should succeed)
        for i in range(limit):
            allowed, retry_after = limiter.check_rate_limit(api_key, limit)
            assert allowed is True

        # 4th request should be denied
        allowed, retry_after = limiter.check_rate_limit(api_key, limit)
        assert allowed is False

        # Wait for window to expire
        time.sleep(2.1)

        # New request should succeed (old timestamps expired)
        allowed, retry_after = limiter.check_rate_limit(api_key, limit)
        assert allowed is True
        assert retry_after == 0

    def test_different_api_keys_have_separate_limits(self):
        """
        Make requests with different API keys
        Assert: Each API key has its own rate limit tracking
        """
        limiter = RateLimiter(window_seconds=60)
        api_key_1 = "test_key_1"
        api_key_2 = "test_key_2"
        limit = 5

        # Exhaust limit for key 1
        for i in range(limit):
            allowed, _ = limiter.check_rate_limit(api_key_1, limit)
            assert allowed is True

        # Key 1 should be limited
        allowed, retry_after = limiter.check_rate_limit(api_key_1, limit)
        assert allowed is False
        assert retry_after > 0

        # Key 2 should still have full limit available
        for i in range(limit):
            allowed, _ = limiter.check_rate_limit(api_key_2, limit)
            assert allowed is True, f"Key 2 request {i+1} should be allowed"

    def test_custom_rate_limits_per_key_work_correctly(self):
        """
        Make requests with different rate limits for different keys
        Assert: Each key respects its configured limit
        """
        limiter = RateLimiter(window_seconds=60)
        api_key_low = "test_key_low_limit"
        api_key_high = "test_key_high_limit"
        low_limit = 3
        high_limit = 10

        # Low limit key should be limited after 3 requests
        for i in range(low_limit):
            allowed, _ = limiter.check_rate_limit(api_key_low, low_limit)
            assert allowed is True

        allowed, _ = limiter.check_rate_limit(api_key_low, low_limit)
        assert allowed is False

        # High limit key should still accept requests
        for i in range(high_limit):
            allowed, _ = limiter.check_rate_limit(api_key_high, high_limit)
            assert allowed is True

        allowed, _ = limiter.check_rate_limit(api_key_high, high_limit)
        assert allowed is False

    def test_get_remaining_returns_correct_count(self):
        """
        Make requests and check remaining count
        Assert: get_remaining() returns accurate remaining request count
        """
        limiter = RateLimiter(window_seconds=60)
        api_key = "test_key_123"
        limit = 10

        # Initial remaining should be limit
        remaining = limiter.get_remaining(api_key, limit)
        assert remaining == limit

        # Make 3 requests
        for i in range(3):
            limiter.check_rate_limit(api_key, limit)

        # Remaining should be 7
        remaining = limiter.get_remaining(api_key, limit)
        assert remaining == 7

        # Make 7 more requests
        for i in range(7):
            limiter.check_rate_limit(api_key, limit)

        # Remaining should be 0
        remaining = limiter.get_remaining(api_key, limit)
        assert remaining == 0

    def test_reset_clears_specific_key(self):
        """
        Make requests, reset specific key, make more requests
        Assert: reset() clears rate limit for specific key
        """
        limiter = RateLimiter(window_seconds=60)
        api_key_1 = "test_key_1"
        api_key_2 = "test_key_2"
        limit = 3

        # Exhaust both keys
        for i in range(limit):
            limiter.check_rate_limit(api_key_1, limit)
            limiter.check_rate_limit(api_key_2, limit)

        # Both should be limited
        allowed_1, _ = limiter.check_rate_limit(api_key_1, limit)
        allowed_2, _ = limiter.check_rate_limit(api_key_2, limit)
        assert allowed_1 is False
        assert allowed_2 is False

        # Reset only key 1
        limiter.reset(api_key_1)

        # Key 1 should work, key 2 should still be limited
        allowed_1, _ = limiter.check_rate_limit(api_key_1, limit)
        allowed_2, _ = limiter.check_rate_limit(api_key_2, limit)
        assert allowed_1 is True
        assert allowed_2 is False

    def test_reset_all_clears_all_keys(self):
        """
        Make requests with multiple keys, reset all, make more requests
        Assert: reset() with no argument clears all keys
        """
        limiter = RateLimiter(window_seconds=60)
        api_key_1 = "test_key_1"
        api_key_2 = "test_key_2"
        limit = 3

        # Exhaust both keys
        for i in range(limit):
            limiter.check_rate_limit(api_key_1, limit)
            limiter.check_rate_limit(api_key_2, limit)

        # Reset all
        limiter.reset()

        # Both should work
        allowed_1, _ = limiter.check_rate_limit(api_key_1, limit)
        allowed_2, _ = limiter.check_rate_limit(api_key_2, limit)
        assert allowed_1 is True
        assert allowed_2 is True


class TestRateLimitingEndToEnd:
    """Test rate limiting through FastAPI endpoints (end-to-end tests)."""

    def test_requests_under_limit_succeed(self, client, temp_data_dir, mock_memory_tools, reset_rate_limiter):
        """
        Make multiple requests under the rate limit
        Assert: All requests return 200/201
        """
        # Setup database and API key with low limit for testing
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 100
            }
        }))

        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=10)

        # Make 5 requests (under limit of 10)
        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                for i in range(5):
                    response = client.get(
                        f"/api/v1/recall?query=test_{i}",
                        headers={"X-API-Key": api_key}
                    )
                    assert response.status_code == 200, f"Request {i+1} should succeed"

    def test_request_over_limit_returns_429_with_retry_after(self, client, temp_data_dir, mock_memory_tools, reset_rate_limiter):
        """
        Make requests exceeding the rate limit
        Assert: Returns 429 with Retry-After header
        """
        # Setup database and API key with very low limit
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 3
            }
        }))

        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=3)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                # Make 3 requests (should all succeed)
                for i in range(3):
                    response = client.get(
                        f"/api/v1/recall?query=test_{i}",
                        headers={"X-API-Key": api_key}
                    )
                    assert response.status_code == 200

                # 4th request should be rate limited
                response = client.get(
                    "/api/v1/recall?query=test_over_limit",
                    headers={"X-API-Key": api_key}
                )

                assert response.status_code == 429
                assert "Retry-After" in response.headers
                retry_after = int(response.headers["Retry-After"])
                assert retry_after > 0
                assert retry_after <= 60
                assert "Rate limit exceeded" in response.json()["detail"]

    def test_different_endpoints_share_same_rate_limit(self, client, temp_data_dir, mock_memory_tools, reset_rate_limiter):
        """
        Make requests to different endpoints with same API key
        Assert: Rate limit is shared across all endpoints
        """
        # Setup database and API key with low limit
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 5
            }
        }))

        key_manager = APIKeyManager(db_path)
        api_key = key_manager.generate_key("test-key", rate_limit=5)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                # Make 2 recall requests
                for i in range(2):
                    response = client.get(
                        f"/api/v1/recall?query=test_{i}",
                        headers={"X-API-Key": api_key}
                    )
                    assert response.status_code == 200

                # Make 3 store requests
                for i in range(3):
                    response = client.post(
                        "/api/v1/store",
                        json={"content": f"Test memory {i}", "memory_type": "fact"},
                        headers={"X-API-Key": api_key}
                    )
                    assert response.status_code == 201

                # 6th request (any endpoint) should be rate limited
                response = client.get(
                    "/api/v1/recall?query=should_fail",
                    headers={"X-API-Key": api_key}
                )
                assert response.status_code == 429

    def test_different_api_keys_have_separate_rate_limits(self, client, temp_data_dir, mock_memory_tools, reset_rate_limiter):
        """
        Make requests with different API keys
        Assert: Each API key has independent rate limit
        """
        # Setup database and API keys
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 3
            }
        }))

        key_manager = APIKeyManager(db_path)
        api_key_1 = key_manager.generate_key("test-key-1", rate_limit=3)
        api_key_2 = key_manager.generate_key("test-key-2", rate_limit=3)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                # Exhaust key 1's limit
                for i in range(3):
                    response = client.get(
                        f"/api/v1/recall?query=test_{i}",
                        headers={"X-API-Key": api_key_1}
                    )
                    assert response.status_code == 200

                # Key 1 should be limited
                response = client.get(
                    "/api/v1/recall?query=should_fail",
                    headers={"X-API-Key": api_key_1}
                )
                assert response.status_code == 429

                # Key 2 should still work
                for i in range(3):
                    response = client.get(
                        f"/api/v1/recall?query=test_key2_{i}",
                        headers={"X-API-Key": api_key_2}
                    )
                    assert response.status_code == 200

    def test_custom_rate_limits_per_key_work_correctly(self, client, temp_data_dir, mock_memory_tools, reset_rate_limiter):
        """
        Create API keys with different rate limits
        Assert: Each key respects its configured limit
        """
        # Setup database and API keys with different limits
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True,
                "default_rate_limit": 100
            }
        }))

        key_manager = APIKeyManager(db_path)
        low_limit_key = key_manager.generate_key("low-limit-key", rate_limit=2)
        high_limit_key = key_manager.generate_key("high-limit-key", rate_limit=10)

        with patch('omi.rest_api.Path.home', return_value=temp_data_dir.parent.parent):
            with patch('omi.rest_api.get_memory_tools', return_value=mock_memory_tools):
                # Low limit key: make 2 requests (should succeed)
                for i in range(2):
                    response = client.get(
                        f"/api/v1/recall?query=low_{i}",
                        headers={"X-API-Key": low_limit_key}
                    )
                    assert response.status_code == 200

                # Low limit key: 3rd request should fail
                response = client.get(
                    "/api/v1/recall?query=low_fail",
                    headers={"X-API-Key": low_limit_key}
                )
                assert response.status_code == 429

                # High limit key: make 10 requests (should all succeed)
                for i in range(10):
                    response = client.get(
                        f"/api/v1/recall?query=high_{i}",
                        headers={"X-API-Key": high_limit_key}
                    )
                    assert response.status_code == 200

                # High limit key: 11th request should fail
                response = client.get(
                    "/api/v1/recall?query=high_fail",
                    headers={"X-API-Key": high_limit_key}
                )
                assert response.status_code == 429

    def test_rate_limit_does_not_apply_to_unauthenticated_endpoints(self, client, temp_data_dir, reset_rate_limiter):
        """
        Make many requests to unauthenticated endpoints
        Assert: No rate limiting on root and health endpoints
        """
        # Setup database and API key
        db_path = temp_data_dir / "palace.sqlite"
        config_path = temp_data_dir / "config.yaml"

        config_path.write_text(yaml.dump({
            "security": {
                "auth_required": True
            }
        }))

        key_manager = APIKeyManager(db_path)
        key_manager.generate_key("test-key", rate_limit=1)  # Very low limit

        # Make many requests to root and health (should all succeed)
        for i in range(10):
            response = client.get("/")
            assert response.status_code == 200

            response = client.get("/health")
            assert response.status_code == 200
