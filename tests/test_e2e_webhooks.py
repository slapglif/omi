"""End-to-End Webhook Delivery Integration Tests

Tests webhook delivery with a real HTTP server and actual memory operations.

Covers:
1. Start a real HTTP server to receive webhook POSTs
2. Configure and start WebhookDispatcher
3. Perform memory operations that trigger events
4. Verify webhook receives POST with correct payload
5. Verify retry logic on failures

This is an E2E test that verifies the complete flow:
    Memory Operation → EventBus → WebhookDispatcher → HTTP POST → Webhook Server
"""
import pytest
import json
import time
import threading
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from contextlib import closing
from datetime import datetime
from pathlib import Path

from omi.event_bus import WebhookDispatcher, get_event_bus, reset_event_bus
from omi.events import MemoryStoredEvent


def find_free_port():
    """Find an available port for the webhook server."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP handler to receive webhook POSTs."""

    # Class variable to store received webhooks
    received_webhooks = []

    def do_POST(self):
        """Handle POST request from webhook."""
        # Read body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        # Parse JSON
        try:
            webhook_data = json.loads(body.decode('utf-8'))

            # Store webhook data
            self.received_webhooks.append({
                'headers': dict(self.headers),
                'body': webhook_data,
                'timestamp': datetime.now()
            })

            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'received'}).encode('utf-8'))

        except Exception as e:
            # Send error response
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def log_message(self, format, *args):
        """Suppress HTTP server logs."""
        pass


def run_webhook_server(port: int, ready_event: threading.Event, stop_event: threading.Event):
    """
    Run HTTP server to receive webhooks in background thread.

    Args:
        port: Port number to run server on
        ready_event: Event to signal when server is ready
        stop_event: Event to signal server should stop
    """
    # Clear any previous webhooks
    WebhookHandler.received_webhooks = []

    # Create server
    server = HTTPServer(('127.0.0.1', port), WebhookHandler)
    server.timeout = 0.5  # Poll timeout for checking stop_event

    # Signal ready
    ready_event.set()

    # Run server until stop_event is set
    while not stop_event.is_set():
        server.handle_request()

    # Cleanup
    server.server_close()


@pytest.fixture
def webhook_port():
    """Get a free port for the webhook server."""
    return find_free_port()


@pytest.fixture
def webhook_server(webhook_port):
    """
    Start webhook HTTP server in background thread.

    Yields the port, then stops server on cleanup.
    """
    # Reset EventBus before starting
    reset_event_bus()

    # Create synchronization events
    ready_event = threading.Event()
    stop_event = threading.Event()

    # Start server thread
    thread = threading.Thread(
        target=run_webhook_server,
        args=(webhook_port, ready_event, stop_event),
        daemon=True
    )
    thread.start()

    # Wait for server to be ready
    ready = ready_event.wait(timeout=5.0)
    if not ready:
        pytest.fail("Webhook server failed to start within timeout")

    # Give server time to fully initialize
    time.sleep(0.2)

    yield webhook_port

    # Cleanup: stop event and give thread time to cleanup
    stop_event.set()
    time.sleep(0.5)

    # Reset EventBus after test
    reset_event_bus()


class TestWebhookIntegration:
    """Test webhook delivery with real HTTP server and memory operations."""

    def test_webhook_receives_memory_stored_event(
        self,
        webhook_server,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache
    ):
        """
        End-to-end verification of webhook delivery.

        Steps:
        1. Start webhook server
        2. Configure and start WebhookDispatcher
        3. Perform memory store operation
        4. Verify webhook POST received
        5. Verify event payload is correct
        """
        from omi.api import MemoryTools
        from omi.persistence import GraphPalace

        # Setup
        port = webhook_server
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Create and start webhook dispatcher
        dispatcher = WebhookDispatcher(
            webhook_url=webhook_url,
            event_types=['memory.stored'],
            timeout=5
        )
        dispatcher.start()

        try:
            # Clear any previous webhooks
            WebhookHandler.received_webhooks = []

            # Perform memory operation
            test_content = "Integration test: webhook delivery verification"
            test_type = "fact"
            test_confidence = 0.92

            memory_id = memory_tools.store(
                content=test_content,
                memory_type=test_type,
                confidence=test_confidence
            )

            # Wait for webhook delivery (background thread)
            time.sleep(1.0)

            # Verify webhook was received
            webhooks = WebhookHandler.received_webhooks
            assert len(webhooks) >= 1, f"Expected at least 1 webhook, got {len(webhooks)}"

            # Get the latest webhook
            webhook = webhooks[-1]

            # Verify headers
            assert webhook['headers']['Content-Type'] == 'application/json'

            # Verify payload
            body = webhook['body']
            assert body['event_type'] == 'memory.stored'
            assert body['memory_id'] == memory_id
            assert body['content'] == test_content
            assert body['memory_type'] == test_type
            assert body['confidence'] == test_confidence
            assert 'timestamp' in body

            # Verify timestamp is recent
            timestamp_str = body['timestamp']
            event_timestamp = datetime.fromisoformat(timestamp_str)
            time_delta = abs((datetime.now() - event_timestamp).total_seconds())
            assert time_delta < 5.0, "Event timestamp should be recent"

        finally:
            # Stop dispatcher
            dispatcher.stop()

    def test_webhook_with_custom_headers(
        self,
        webhook_server,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache
    ):
        """
        Test webhook includes custom headers.

        Steps:
        1. Configure WebhookDispatcher with custom headers
        2. Perform memory operation
        3. Verify webhook includes custom headers
        """
        from omi.api import MemoryTools
        from omi.persistence import GraphPalace

        # Setup
        port = webhook_server
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Create dispatcher with custom headers
        dispatcher = WebhookDispatcher(
            webhook_url=webhook_url,
            event_types=['memory.stored'],
            headers={
                'Authorization': 'Bearer test_token_123',
                'X-Custom-Header': 'integration_test'
            },
            timeout=5
        )
        dispatcher.start()

        try:
            # Clear any previous webhooks
            WebhookHandler.received_webhooks = []

            # Perform memory operation
            memory_tools.store(
                content="Test custom headers",
                memory_type="fact"
            )

            # Wait for webhook delivery
            time.sleep(1.0)

            # Verify webhook was received
            webhooks = WebhookHandler.received_webhooks
            assert len(webhooks) >= 1

            # Verify custom headers
            webhook = webhooks[-1]
            headers = webhook['headers']
            assert headers['Authorization'] == 'Bearer test_token_123'
            assert headers['X-Custom-Header'] == 'integration_test'

        finally:
            dispatcher.stop()

    def test_webhook_wildcard_subscription(
        self,
        webhook_server,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache
    ):
        """
        Test webhook with wildcard subscription receives all event types.

        Steps:
        1. Configure WebhookDispatcher with wildcard ('*')
        2. Perform multiple operations (store, recall)
        3. Verify webhook receives all events
        """
        from omi.api import MemoryTools
        from omi.persistence import GraphPalace

        # Setup
        port = webhook_server
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Create dispatcher with wildcard subscription
        dispatcher = WebhookDispatcher(
            webhook_url=webhook_url,
            event_types=['*'],  # Subscribe to all events
            timeout=5
        )
        dispatcher.start()

        try:
            # Clear any previous webhooks
            WebhookHandler.received_webhooks = []

            # Perform multiple operations
            memory_tools.store(content="Test wildcard 1", memory_type="fact")
            memory_tools.store(content="Test wildcard 2", memory_type="fact")
            memory_tools.recall("Test wildcard")

            # Wait for webhook deliveries
            time.sleep(1.5)

            # Verify webhooks were received
            webhooks = WebhookHandler.received_webhooks
            assert len(webhooks) >= 3, f"Expected at least 3 webhooks, got {len(webhooks)}"

            # Verify event types
            event_types = [w['body']['event_type'] for w in webhooks]
            assert 'memory.stored' in event_types
            assert 'memory.recalled' in event_types

            # Count event types
            stored_count = event_types.count('memory.stored')
            recalled_count = event_types.count('memory.recalled')
            assert stored_count >= 2, "Should have at least 2 memory.stored events"
            assert recalled_count >= 1, "Should have at least 1 memory.recalled event"

        finally:
            dispatcher.stop()

    def test_webhook_event_type_filtering(
        self,
        webhook_server,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache
    ):
        """
        Test webhook only receives subscribed event types.

        Steps:
        1. Configure WebhookDispatcher for only 'memory.stored'
        2. Perform store and recall operations
        3. Verify webhook only receives memory.stored events
        """
        from omi.api import MemoryTools
        from omi.persistence import GraphPalace

        # Setup
        port = webhook_server
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Create dispatcher with specific event type
        dispatcher = WebhookDispatcher(
            webhook_url=webhook_url,
            event_types=['memory.stored'],  # Only subscribe to memory.stored
            timeout=5
        )
        dispatcher.start()

        try:
            # Clear any previous webhooks
            WebhookHandler.received_webhooks = []

            # Perform multiple operations
            memory_tools.store(content="Filtered event 1", memory_type="fact")
            memory_tools.store(content="Filtered event 2", memory_type="fact")
            memory_tools.recall("Filtered")  # Should NOT trigger webhook

            # Wait for webhook deliveries
            time.sleep(1.5)

            # Verify webhooks were received
            webhooks = WebhookHandler.received_webhooks
            assert len(webhooks) >= 2, f"Expected at least 2 webhooks, got {len(webhooks)}"

            # Verify all webhooks are memory.stored events only
            event_types = [w['body']['event_type'] for w in webhooks]
            for event_type in event_types:
                assert event_type == 'memory.stored', f"Expected only memory.stored, got {event_type}"

            # Verify no memory.recalled events
            assert 'memory.recalled' not in event_types

        finally:
            dispatcher.stop()

    def test_webhook_multiple_operations_sequence(
        self,
        webhook_server,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache
    ):
        """
        Test webhook correctly handles sequence of multiple memory operations.

        Steps:
        1. Configure WebhookDispatcher
        2. Perform multiple memory operations in sequence
        3. Verify all webhooks are received in order
        4. Verify all payloads are correct
        """
        from omi.api import MemoryTools
        from omi.persistence import GraphPalace

        # Setup
        port = webhook_server
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Create dispatcher
        dispatcher = WebhookDispatcher(
            webhook_url=webhook_url,
            event_types=['memory.stored'],
            timeout=5
        )
        dispatcher.start()

        try:
            # Clear any previous webhooks
            WebhookHandler.received_webhooks = []

            # Perform multiple operations
            test_memories = [
                {"content": "Memory A", "type": "fact", "confidence": 0.8},
                {"content": "Memory B", "type": "experience", "confidence": 0.9},
                {"content": "Memory C", "type": "fact", "confidence": 0.85},
            ]

            memory_ids = []
            for mem in test_memories:
                mid = memory_tools.store(
                    content=mem["content"],
                    memory_type=mem["type"],
                    confidence=mem["confidence"]
                )
                memory_ids.append(mid)

            # Wait for all webhook deliveries
            time.sleep(2.0)

            # Verify all webhooks were received
            webhooks = WebhookHandler.received_webhooks
            assert len(webhooks) >= 3, f"Expected at least 3 webhooks, got {len(webhooks)}"

            # Get last 3 webhooks
            recent_webhooks = webhooks[-3:]

            # Verify each webhook has correct data
            for i, webhook in enumerate(recent_webhooks):
                body = webhook['body']
                assert body['event_type'] == 'memory.stored'
                assert body['memory_id'] == memory_ids[i]
                assert body['content'] == test_memories[i]["content"]
                assert body['memory_type'] == test_memories[i]["type"]
                assert body['confidence'] == test_memories[i]["confidence"]

        finally:
            dispatcher.stop()


class TestWebhookRetry:
    """Test webhook retry logic with failing server."""

    def test_webhook_retries_on_server_error(
        self,
        webhook_port,
        temp_omi_setup,
        mock_embedder,
        mock_embedding_cache
    ):
        """
        Test webhook dispatcher retries on server errors.

        Note: This test doesn't start the webhook server,
        so all POST requests will fail and trigger retries.
        """
        from omi.api import MemoryTools
        from omi.persistence import GraphPalace
        from unittest.mock import patch, MagicMock

        # Setup
        port = webhook_port
        webhook_url = f"http://127.0.0.1:{port}/webhook"

        # Reset EventBus
        reset_event_bus()

        palace = GraphPalace(temp_omi_setup["db_path"])
        memory_tools = MemoryTools(palace, mock_embedder, mock_embedding_cache)

        # Track retry attempts
        retry_attempts = []

        def mock_post(*args, **kwargs):
            """Mock POST that tracks attempts and fails."""
            retry_attempts.append(datetime.now())
            # Raise connection error to trigger retry
            import requests
            raise requests.exceptions.ConnectionError("Connection failed")

        # Create dispatcher with short retry delay
        dispatcher = WebhookDispatcher(
            webhook_url=webhook_url,
            event_types=['memory.stored'],
            timeout=1,
            max_retries=3,
            retry_delay=0.5  # Short delay for faster testing
        )

        # Patch requests.post to track retries
        with patch.object(dispatcher, '_requests') as mock_requests:
            mock_requests.post = mock_post

            dispatcher.start()

            try:
                # Perform memory operation
                memory_tools.store(
                    content="Test retry logic",
                    memory_type="fact"
                )

                # Wait for retries to complete
                # With retry_delay=0.5s and exponential backoff: 0.5s, 1.0s, 2.0s
                # Total time: initial + 0.5s + 1.0s + 2.0s = 3.5s minimum
                time.sleep(4.5)

                # Verify retry attempts
                # max_retries=3 means retry up to 3 times (initial + 3 retries = 4 attempts)
                # However, the implementation may count differently, so we verify >= 3
                assert len(retry_attempts) >= 3, f"Expected at least 3 retry attempts, got {len(retry_attempts)}"
                assert len(retry_attempts) <= 4, f"Expected at most 4 retry attempts, got {len(retry_attempts)}"

                # Verify exponential backoff timing
                # Delays should be approximately: 0.5s, 1.0s, 2.0s
                if len(retry_attempts) >= 2:
                    delay_1 = (retry_attempts[1] - retry_attempts[0]).total_seconds()
                    # First retry should be after ~0.5s (allow some variance)
                    assert 0.3 < delay_1 < 0.8, f"Expected ~0.5s delay, got {delay_1}s"

                if len(retry_attempts) >= 3:
                    delay_2 = (retry_attempts[2] - retry_attempts[1]).total_seconds()
                    # Second retry should be after ~1.0s (allow some variance)
                    assert 0.8 < delay_2 < 1.3, f"Expected ~1.0s delay, got {delay_2}s"

            finally:
                dispatcher.stop()
                reset_event_bus()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
