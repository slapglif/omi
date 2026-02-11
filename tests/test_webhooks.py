"""
Tests for WebhookDispatcher module.

Covers:
- Webhook delivery to HTTP endpoints
- Retry logic with exponential backoff
- Event type filtering
- Custom headers and timeout
- Background thread execution
- Error handling
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from threading import Event as ThreadEvent

# Ensure omi modules are importable
import sys
from pathlib import Path

def ensure_omi_importable():
    """Add src to path if needed."""
    test_dir = Path(__file__).resolve().parent
    src_path = test_dir.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

ensure_omi_importable()


class TestWebhookDispatcherBasics:
    """Test basic WebhookDispatcher functionality."""

    def test_dispatcher_initialization(self):
        """Test WebhookDispatcher can be initialized."""
        from omi.event_bus import WebhookDispatcher

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored'],
            headers={'Authorization': 'Bearer token123'},
            timeout=5,
            max_retries=2
        )

        assert dispatcher.webhook_url == "https://example.com/webhook"
        assert dispatcher.event_types == ['memory.stored']
        assert dispatcher.headers == {'Authorization': 'Bearer token123'}
        assert dispatcher.timeout == 5
        assert dispatcher.max_retries == 2
        assert not dispatcher.is_active()

    def test_dispatcher_defaults(self):
        """Test WebhookDispatcher uses sensible defaults."""
        from omi.event_bus import WebhookDispatcher

        dispatcher = WebhookDispatcher(webhook_url="https://example.com/webhook")

        assert dispatcher.event_types == ['*']  # Subscribe to all events by default
        assert dispatcher.headers == {}
        assert dispatcher.timeout == 10
        assert dispatcher.max_retries == 3
        assert dispatcher.retry_delay == 1.0

    def test_dispatcher_requires_requests_library(self):
        """Test dispatcher fails gracefully if requests not available."""
        from omi.event_bus import WebhookDispatcher

        with patch.dict('sys.modules', {'requests': None}):
            with pytest.raises(ImportError, match="requests package required"):
                # Force reimport to trigger the check
                import importlib
                import omi.event_bus as eb
                importlib.reload(eb)
                # This should fail in __init__ when trying to import requests
                dispatcher = eb.WebhookDispatcher("https://example.com")


class TestWebhookDelivery:
    """Test webhook HTTP POST delivery."""

    @pytest.fixture
    def mock_requests(self):
        """Mock requests library."""
        mock = MagicMock()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock.post.return_value = mock_response

        return mock

    @pytest.fixture
    def dispatcher_with_mock_requests(self, mock_requests):
        """Create dispatcher with mocked requests."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus

        # Reset event bus BEFORE creating dispatcher
        reset_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored'],
            timeout=5
        )
        dispatcher._requests = mock_requests

        return dispatcher, mock_requests

    def test_successful_webhook_delivery(self, dispatcher_with_mock_requests):
        """Test successful webhook POST delivery."""
        from omi.events import MemoryStoredEvent
        from omi.event_bus import get_event_bus

        dispatcher, mock_requests = dispatcher_with_mock_requests

        # Get event bus (already reset in fixture)
        bus = get_event_bus()

        # Start dispatcher
        dispatcher.start()
        assert dispatcher.is_active()

        # Create and publish event
        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact",
            confidence=0.9
        )

        # Publish event
        bus.publish(event)

        # Give background thread time to complete
        time.sleep(0.2)

        # Verify POST was called
        assert mock_requests.post.called

        # Verify request details
        call_args = mock_requests.post.call_args
        assert call_args[1]['json']['event_type'] == 'memory.stored'
        assert call_args[1]['json']['memory_id'] == 'test123'
        assert call_args[1]['json']['content'] == 'Test memory'
        assert call_args[1]['timeout'] == 5

        # Cleanup
        dispatcher.stop()
        assert not dispatcher.is_active()

    def test_webhook_with_custom_headers(self, dispatcher_with_mock_requests):
        """Test webhook includes custom headers."""
        from omi.events import MemoryStoredEvent
        from omi.event_bus import get_event_bus

        dispatcher, mock_requests = dispatcher_with_mock_requests
        dispatcher.headers = {
            'Authorization': 'Bearer secret_token',
            'X-Custom-Header': 'custom_value'
        }

        # Get event bus (already reset in fixture)
        bus = get_event_bus()

        dispatcher.start()

        event = MemoryStoredEvent(
            memory_id="test456",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)

        # Wait for background thread
        time.sleep(0.2)

        # Verify headers
        call_args = mock_requests.post.call_args
        headers = call_args[1]['headers']
        assert headers['Content-Type'] == 'application/json'
        assert headers['Authorization'] == 'Bearer secret_token'
        assert headers['X-Custom-Header'] == 'custom_value'

        dispatcher.stop()

    def test_webhook_serializes_event_to_dict(self, dispatcher_with_mock_requests):
        """Test webhook serializes event using to_dict()."""
        from omi.events import BeliefUpdatedEvent
        from omi.event_bus import get_event_bus

        dispatcher, mock_requests = dispatcher_with_mock_requests
        dispatcher.event_types = ['belief.updated']

        bus = get_event_bus()

        dispatcher.start()

        event = BeliefUpdatedEvent(
            belief_id="belief789",
            old_confidence=0.7,
            new_confidence=0.9,
            evidence_id="evidence123"
        )
        bus.publish(event)

        time.sleep(0.2)

        # Verify serialization
        call_args = mock_requests.post.call_args
        payload = call_args[1]['json']
        assert payload['event_type'] == 'belief.updated'
        assert payload['belief_id'] == 'belief789'
        assert payload['old_confidence'] == 0.7
        assert payload['new_confidence'] == 0.9
        assert payload['evidence_id'] == 'evidence123'

        dispatcher.stop()


class TestWebhookRetry:
    """Test webhook retry logic."""

    @pytest.fixture
    def failing_requests(self):
        """Mock requests that fail."""
        mock = MagicMock()

        # Simulate failure
        mock.post.side_effect = Exception("Connection refused")

        return mock

    def test_retry_on_failure(self, failing_requests):
        """Test webhook retries on failure."""
        from omi.event_bus import WebhookDispatcher
        from omi.events import MemoryStoredEvent
        from omi.event_bus import reset_event_bus, get_event_bus

        reset_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored'],
            max_retries=2,
            retry_delay=0.05  # Short delay for testing
        )
        dispatcher._requests = failing_requests

        bus = get_event_bus()

        dispatcher.start()

        event = MemoryStoredEvent(
            memory_id="test_retry",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)

        # Wait for retries to complete
        time.sleep(0.5)

        # Should have tried 3 times (initial + 2 retries)
        assert failing_requests.post.call_count == 3

        dispatcher.stop()

    def test_exponential_backoff(self, failing_requests):
        """Test retry uses exponential backoff."""
        from omi.event_bus import WebhookDispatcher
        from omi.events import MemoryStoredEvent
        from omi.event_bus import reset_event_bus, get_event_bus

        # Track timing
        call_times = []

        def track_time(*args, **kwargs):
            call_times.append(time.time())
            raise Exception("Connection refused")

        failing_requests.post.side_effect = track_time

        reset_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['*'],
            max_retries=2,
            retry_delay=0.1  # 100ms base delay
        )
        dispatcher._requests = failing_requests

        bus = get_event_bus()

        dispatcher.start()

        event = MemoryStoredEvent(
            memory_id="test_backoff",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)

        # Wait for all retries
        time.sleep(1.0)

        # Verify exponential backoff timing
        # First retry after ~0.1s, second after ~0.2s
        assert len(call_times) == 3

        # Check delays are approximately exponential
        # (Allow some tolerance for thread scheduling)
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            assert 0.08 < delay1 < 0.15  # ~0.1s ± tolerance

        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            assert 0.15 < delay2 < 0.30  # ~0.2s ± tolerance

        dispatcher.stop()

    def test_success_after_retry(self):
        """Test webhook succeeds after initial failure."""
        from omi.event_bus import WebhookDispatcher
        from omi.events import MemoryStoredEvent
        from omi.event_bus import reset_event_bus, get_event_bus

        reset_event_bus()

        mock_requests = MagicMock()

        # Fail first time, succeed second time
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Temporary failure")
            return mock_response

        mock_requests.post.side_effect = side_effect

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            max_retries=2,
            retry_delay=0.05
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        event = MemoryStoredEvent(
            memory_id="test_success",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)

        # Wait for retry
        time.sleep(0.3)

        # Should have tried twice (failed once, succeeded once)
        assert mock_requests.post.call_count == 2

        dispatcher.stop()


class TestEventFiltering:
    """Test event type filtering."""

    @pytest.fixture
    def mock_requests(self):
        """Mock successful requests."""
        mock = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock.post.return_value = mock_response
        return mock

    def test_specific_event_type_filtering(self, mock_requests):
        """Test dispatcher only receives subscribed event types."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus
        from omi.events import MemoryStoredEvent, BeliefUpdatedEvent

        reset_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored']  # Only subscribe to memory.stored
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        # Publish memory.stored event (should be delivered)
        event1 = MemoryStoredEvent(
            memory_id="test1",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event1)

        time.sleep(0.1)

        # Publish belief.updated event (should NOT be delivered)
        event2 = BeliefUpdatedEvent(
            belief_id="belief1",
            old_confidence=0.5,
            new_confidence=0.8
        )
        bus.publish(event2)

        time.sleep(0.1)

        # Only memory.stored should have been posted
        assert mock_requests.post.call_count == 1
        call_args = mock_requests.post.call_args
        assert call_args[1]['json']['event_type'] == 'memory.stored'

        dispatcher.stop()

    def test_wildcard_receives_all_events(self, mock_requests):
        """Test wildcard subscription receives all event types."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus
        from omi.events import MemoryStoredEvent, BeliefUpdatedEvent, SessionStartedEvent

        reset_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['*']  # Wildcard - all events
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        # Publish different event types
        bus.publish(MemoryStoredEvent(
            memory_id="test1",
            content="Test",
            memory_type="fact"
        ))

        bus.publish(BeliefUpdatedEvent(
            belief_id="belief1",
            old_confidence=0.5,
            new_confidence=0.8
        ))

        bus.publish(SessionStartedEvent(session_id="session1"))

        # Wait for all deliveries
        time.sleep(0.3)

        # Should have received all 3 events
        assert mock_requests.post.call_count == 3

        dispatcher.stop()

    def test_multiple_event_types(self, mock_requests):
        """Test subscribing to multiple specific event types."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus
        from omi.events import MemoryStoredEvent, BeliefUpdatedEvent, SessionStartedEvent

        reset_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored', 'belief.updated']  # Two specific types
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        # Publish matching events
        bus.publish(MemoryStoredEvent(
            memory_id="test1",
            content="Test",
            memory_type="fact"
        ))

        bus.publish(BeliefUpdatedEvent(
            belief_id="belief1",
            old_confidence=0.5,
            new_confidence=0.8
        ))

        # Publish non-matching event
        bus.publish(SessionStartedEvent(session_id="session1"))

        time.sleep(0.3)

        # Should only receive 2 events (not session.started)
        assert mock_requests.post.call_count == 2

        dispatcher.stop()


class TestDispatcherLifecycle:
    """Test dispatcher start/stop lifecycle."""

    def test_start_subscribes_to_event_bus(self):
        """Test start() subscribes to EventBus."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus

        reset_event_bus()
        bus = get_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored', 'belief.updated']
        )

        # Initially no subscribers
        initial_count = bus.subscriber_count()

        # Start dispatcher
        dispatcher.start()

        # Should have added 2 subscriptions
        assert bus.subscriber_count() > initial_count
        assert dispatcher.is_active()

    def test_stop_unsubscribes_from_event_bus(self):
        """Test stop() unsubscribes from EventBus."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus

        reset_event_bus()
        bus = get_event_bus()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored']
        )

        dispatcher.start()
        count_after_start = bus.subscriber_count()

        dispatcher.stop()
        count_after_stop = bus.subscriber_count()

        # Subscriptions should be removed
        assert count_after_stop < count_after_start
        assert not dispatcher.is_active()

    def test_double_start_warning(self):
        """Test starting dispatcher twice logs warning."""
        from omi.event_bus import WebhookDispatcher
        import logging

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook"
        )

        dispatcher.start()

        # Try starting again
        with patch('omi.event_bus.logger') as mock_logger:
            dispatcher.start()
            mock_logger.warning.assert_called()

        dispatcher.stop()

    def test_stop_when_not_started(self):
        """Test stopping dispatcher that was never started."""
        from omi.event_bus import WebhookDispatcher

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook"
        )

        # Should not raise error
        dispatcher.stop()
        assert not dispatcher.is_active()


class TestErrorHandling:
    """Test error handling in webhook delivery."""

    def test_event_serialization_error(self):
        """Test handling of event that can't be serialized."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus

        reset_event_bus()

        # Create mock event without to_dict method
        mock_event = MagicMock()
        mock_event.event_type = 'test.event'
        del mock_event.to_dict  # Remove to_dict method

        mock_requests = MagicMock()

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook"
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        # Publish problematic event
        bus.publish(mock_event)

        time.sleep(0.2)

        # Should still send with minimal payload
        assert mock_requests.post.called

        dispatcher.stop()

    def test_http_timeout_error(self):
        """Test handling of HTTP timeout."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus
        from omi.events import MemoryStoredEvent
        import requests

        reset_event_bus()

        mock_requests = MagicMock()
        mock_requests.post.side_effect = requests.exceptions.Timeout("Request timed out")

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            max_retries=1,
            retry_delay=0.05
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        event = MemoryStoredEvent(
            memory_id="test_timeout",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)

        time.sleep(0.3)

        # Should have retried
        assert mock_requests.post.call_count == 2  # Initial + 1 retry

        dispatcher.stop()

    def test_http_error_status(self):
        """Test handling of HTTP error status codes."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus
        from omi.events import MemoryStoredEvent
        import requests

        reset_event_bus()

        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_requests.post.return_value = mock_response

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            max_retries=1,
            retry_delay=0.05
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        event = MemoryStoredEvent(
            memory_id="test_error",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)

        time.sleep(0.3)

        # Should have retried on error
        assert mock_requests.post.call_count == 2

        dispatcher.stop()


class TestBackgroundExecution:
    """Test background thread execution."""

    def test_webhook_does_not_block_event_bus(self):
        """Test webhook delivery runs in background and doesn't block EventBus."""
        from omi.event_bus import WebhookDispatcher, reset_event_bus, get_event_bus
        from omi.events import MemoryStoredEvent

        reset_event_bus()

        # Create slow webhook
        mock_requests = MagicMock()

        def slow_post(*args, **kwargs):
            time.sleep(0.5)  # Slow webhook
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_requests.post.side_effect = slow_post

        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook"
        )
        dispatcher._requests = mock_requests

        bus = get_event_bus()

        dispatcher.start()

        # Track event publishing
        events_published = []

        def track_event(event):
            events_published.append(event)

        bus.subscribe('*', track_event)

        # Publish event
        start_time = time.time()
        event = MemoryStoredEvent(
            memory_id="test_async",
            content="Test",
            memory_type="fact"
        )
        bus.publish(event)
        publish_time = time.time() - start_time

        # Publishing should be fast (not blocked by slow webhook)
        assert publish_time < 0.1
        assert len(events_published) == 1

        # Wait for webhook to complete
        time.sleep(0.6)

        # Webhook should have completed
        assert mock_requests.post.called

        dispatcher.stop()
