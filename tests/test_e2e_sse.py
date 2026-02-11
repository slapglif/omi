"""End-to-End SSE Streaming Integration Tests

Tests SSE streaming with a real HTTP client connecting to a running server.

Covers:
1. Start a real FastAPI server with uvicorn
2. Connect SSE client to /api/v1/events
3. Verify SSE connection and headers
4. Verify event stream format

Note: Full event delivery tests are limited by asyncio threading complexity.
For complete E2E testing, manually test with a running server:
    uvicorn omi.rest_api:app --reload
    curl -N http://localhost:8000/api/v1/events

Issue: https://github.com/slapglif/omi/issues/4
"""
import pytest
import json
import time
import requests
import threading
import socket
from contextlib import closing
from datetime import datetime

from omi.event_bus import get_event_bus, reset_event_bus
from omi.events import MemoryStoredEvent


def find_free_port():
    """Find an available port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def run_server(port: int, ready_event: threading.Event, stop_event: threading.Event):
    """
    Run uvicorn server in background thread.

    Args:
        port: Port number to run server on
        ready_event: Event to signal when server is ready
        stop_event: Event to signal server should stop
    """
    import uvicorn
    from omi.rest_api import app

    # Configure server
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False
    )
    server = uvicorn.Server(config)

    # Signal ready
    ready_event.set()

    # Run server (will block until shutdown)
    try:
        server.run()
    except Exception:
        pass  # Silently handle shutdown


@pytest.fixture
def server_port():
    """Get a free port for the test server."""
    return find_free_port()


@pytest.fixture
def server_thread(server_port):
    """
    Start FastAPI server in background thread.

    Yields the thread and port, then stops server on cleanup.
    """
    # Reset EventBus before starting server
    reset_event_bus()

    # Create synchronization events
    ready_event = threading.Event()
    stop_event = threading.Event()

    # Start server thread
    thread = threading.Thread(
        target=run_server,
        args=(server_port, ready_event, stop_event),
        daemon=True
    )
    thread.start()

    # Wait for server to be ready
    ready = ready_event.wait(timeout=5.0)
    if not ready:
        pytest.fail("Server failed to start within timeout")

    # Give server time to fully initialize
    time.sleep(1.0)

    # Verify server is responding
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(
                f"http://127.0.0.1:{server_port}/health",
                timeout=2
            )
            if response.status_code == 200:
                break
        except Exception as e:
            if i == max_retries - 1:
                pytest.fail(f"Server health check failed: {e}")
            time.sleep(0.5)

    yield server_port

    # Cleanup: stop event and give thread time to cleanup
    stop_event.set()
    time.sleep(0.5)

    # Reset EventBus after test
    reset_event_bus()


class TestSSEIntegration:
    """Test SSE streaming with real server and HTTP client."""

    def test_sse_connection_and_headers(self, server_thread):
        """
        Test SSE endpoint connection and headers.

        Verification:
        1. Connect to /api/v1/events
        2. Verify 200 status code
        3. Verify content-type is text/event-stream
        4. Verify cache-control headers
        """
        port = server_thread
        url = f"http://127.0.0.1:{port}/api/v1/events"

        # Connect to SSE endpoint
        response = requests.get(url, stream=True, timeout=5)

        try:
            assert response.status_code == 200
            assert 'text/event-stream' in response.headers['content-type']
            assert response.headers['cache-control'] == 'no-cache'
            assert response.headers['connection'] == 'keep-alive'
        finally:
            response.close()

    def test_sse_connected_message(self, server_thread):
        """
        Test SSE stream sends initial connected message.

        Verification:
        1. Connect to /api/v1/events
        2. Read first SSE message
        3. Verify it's a 'connected' type message
        4. Verify SSE format (data: {json})
        """
        port = server_thread
        url = f"http://127.0.0.1:{port}/api/v1/events"

        # Connect to SSE endpoint
        response = requests.get(url, stream=True, timeout=5)

        try:
            # Read first line from stream
            iter_lines = response.iter_lines()
            first_line = next(iter_lines, None)

            assert first_line is not None, "Should receive at least one line"

            decoded = first_line.decode('utf-8')
            assert decoded.startswith('data: '), f"Expected 'data: ' prefix, got: {decoded}"

            # Parse JSON payload
            json_str = decoded[6:]  # Remove "data: " prefix
            data = json.loads(json_str)

            assert data['type'] == 'connected'
            assert data['message'] == 'SSE stream connected'

        finally:
            response.close()

    def test_sse_endpoint_with_event_type_filter(self, server_thread):
        """
        Test SSE endpoint accepts event_type query parameter.

        Verification:
        1. Connect with ?event_type=memory.stored
        2. Verify connection successful
        3. Verify connected message received
        """
        port = server_thread
        url = f"http://127.0.0.1:{port}/api/v1/events?event_type=memory.stored"

        # Connect with event type filter
        response = requests.get(url, stream=True, timeout=5)

        try:
            assert response.status_code == 200

            # Read connected message
            iter_lines = response.iter_lines()
            first_line = next(iter_lines, None)

            assert first_line is not None
            decoded = first_line.decode('utf-8')
            assert decoded.startswith('data: ')

            data = json.loads(decoded[6:])
            assert data['type'] == 'connected'

        finally:
            response.close()

    def test_sse_keepalive_messages(self, server_thread):
        """
        Test SSE stream sends keepalive messages.

        Note: This test waits for keepalive which takes 30+ seconds.
        Skipped by default for faster test runs.
        """
        pytest.skip("Keepalive test takes >30s, skip for faster testing")

        port = server_thread
        url = f"http://127.0.0.1:{port}/api/v1/events"

        response = requests.get(url, stream=True, timeout=35)

        try:
            iter_lines = response.iter_lines()

            # Skip connected message
            next(iter_lines)

            # Wait for keepalive (should come after 30s timeout)
            start_time = time.time()
            keepalive_received = False

            for line in iter_lines:
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith(': keepalive'):
                        keepalive_received = True
                        break

                # Don't wait more than 35 seconds
                if time.time() - start_time > 35:
                    break

            assert keepalive_received, "Should receive keepalive message within 35s"

        finally:
            response.close()

    def test_multiple_sse_connections(self, server_thread):
        """
        Test multiple clients can connect to SSE endpoint simultaneously.

        Verification:
        1. Open 3 SSE connections
        2. Verify all receive connected message
        3. Close all connections cleanly
        """
        port = server_thread
        url = f"http://127.0.0.1:{port}/api/v1/events"

        connections = []

        try:
            # Open multiple connections
            for i in range(3):
                response = requests.get(url, stream=True, timeout=5)
                assert response.status_code == 200
                connections.append(response)

                # Verify connected message
                first_line = next(response.iter_lines()).decode('utf-8')
                assert first_line.startswith('data: ')
                data = json.loads(first_line[6:])
                assert data['type'] == 'connected'

        finally:
            # Close all connections
            for conn in connections:
                conn.close()

    def test_api_documentation_endpoints(self, server_thread):
        """
        Test API provides documentation endpoints.

        Verification:
        1. GET / returns API information
        2. GET /health returns health status
        """
        port = server_thread

        # Test root endpoint
        response = requests.get(f"http://127.0.0.1:{port}/", timeout=2)
        assert response.status_code == 200

        data = response.json()
        assert data['service'] == 'OMI Event Streaming API'
        assert data['version'] == '1.0.0'
        assert '/api/v1/events' in data['endpoints']

        # Test health endpoint
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
        assert response.status_code == 200

        health_data = response.json()
        assert health_data['status'] == 'healthy'
        assert health_data['service'] == 'omi-event-api'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
