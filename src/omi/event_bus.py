"""
EventBus for in-process pub/sub event streaming.

Provides thread-safe event subscription and publishing for memory operations.
Orchestrators can subscribe to specific event types to react to memory changes,
contradictions, and session boundaries.

Usage:
    bus = EventBus()

    # Subscribe to specific event types
    bus.subscribe('memory.stored', lambda event: print(f"Stored: {event.memory_id}"))
    bus.subscribe('belief.contradiction_detected', handle_contradiction)

    # Subscribe to all events
    bus.subscribe('*', lambda event: log_event(event))

    # Publish events
    from omi.events import MemoryStoredEvent
    event = MemoryStoredEvent(memory_id="123", content="test", memory_type="fact")
    bus.publish(event)
"""

from typing import Callable, Dict, List, Any
from threading import Lock
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """
    Thread-safe in-process event bus for pub/sub.

    Supports:
    - subscribe(event_type, callback): Register callbacks for specific event types
    - publish(event): Emit events to all matching subscribers
    - Wildcard subscription: subscribe('*', callback) receives all events
    """

    def __init__(self):
        """Initialize empty event bus with thread safety."""
        # Dictionary mapping event_type -> list of callbacks
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._lock = Lock()

    def subscribe(self, event_type: str, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to listen for (e.g., 'memory.stored')
                       Use '*' to subscribe to all event types
            callback: Function called with event object when event occurs
                     Signature: callback(event: Event) -> None

        Example:
            def on_memory_stored(event):
                print(f"Memory {event.memory_id} stored")

            bus.subscribe('memory.stored', on_memory_stored)
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}: {callback.__name__ if hasattr(callback, '__name__') else 'lambda'}")

    def unsubscribe(self, event_type: str, callback: Callable[[Any], None]) -> bool:
        """
        Unsubscribe a callback from an event type.

        Args:
            event_type: Event type to unsubscribe from
            callback: The callback function to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    if not self._subscribers[event_type]:
                        # Clean up empty subscriber lists
                        del self._subscribers[event_type]
                    logger.debug(f"Unsubscribed from {event_type}")
                    return True
                except ValueError:
                    pass
        return False

    def publish(self, event: Any) -> None:
        """
        Publish an event to all matching subscribers.

        Notifies:
        1. Subscribers to the specific event type
        2. Wildcard subscribers ('*')

        Args:
            event: Event object (must have 'event_type' attribute)

        Example:
            from omi.events import MemoryStoredEvent
            event = MemoryStoredEvent(
                memory_id="abc123",
                content="Important fact",
                memory_type="fact"
            )
            bus.publish(event)
        """
        if not hasattr(event, 'event_type'):
            logger.warning(f"Event missing 'event_type' attribute: {type(event).__name__}")
            return

        event_type = event.event_type

        # Get subscribers (make a copy to avoid holding lock during callbacks)
        with self._lock:
            specific_subscribers = self._subscribers.get(event_type, []).copy()
            wildcard_subscribers = self._subscribers.get('*', []).copy()

        # Notify specific subscribers
        for callback in specific_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber callback for {event_type}: {e}", exc_info=True)

        # Notify wildcard subscribers
        for callback in wildcard_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in wildcard subscriber callback for {event_type}: {e}", exc_info=True)

        logger.debug(f"Published {event_type} to {len(specific_subscribers) + len(wildcard_subscribers)} subscribers")

    def clear(self) -> None:
        """
        Clear all subscriptions.

        Useful for testing and cleanup.
        """
        with self._lock:
            self._subscribers.clear()
            logger.debug("Cleared all subscriptions")

    def subscriber_count(self, event_type: str = None) -> int:
        """
        Get count of subscribers.

        Args:
            event_type: Optional event type to count. If None, returns total.

        Returns:
            Number of subscribers
        """
        with self._lock:
            if event_type:
                return len(self._subscribers.get(event_type, []))
            else:
                return sum(len(subs) for subs in self._subscribers.values())


# Global event bus instance
# Services can import and use this shared bus
_global_bus = None


def get_event_bus() -> EventBus:
    """
    Get or create global EventBus instance.

    Returns:
        Global EventBus singleton
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """
    Reset global event bus (mainly for testing).

    Creates a fresh EventBus instance.
    """
    global _global_bus
    _global_bus = EventBus()


class WebhookDispatcher:
    """
    Webhook dispatcher that subscribes to EventBus and sends HTTP POST notifications.

    Features:
    - Subscribes to specific event types or all events ('*')
    - Sends POST requests to configured webhook URLs
    - Automatic retry with exponential backoff
    - Timeout handling
    - Background thread execution to avoid blocking EventBus

    Usage:
        dispatcher = WebhookDispatcher(
            webhook_url="https://example.com/webhook",
            event_types=['memory.stored', 'belief.contradiction_detected'],
            headers={'Authorization': 'Bearer token'}
        )
        dispatcher.start()  # Start listening and dispatching
        dispatcher.stop()   # Stop and cleanup
    """

    def __init__(
        self,
        webhook_url: str,
        event_types: List[str] = None,
        headers: Dict[str, str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize webhook dispatcher.

        Args:
            webhook_url: Target URL for webhook POST requests
            event_types: List of event types to subscribe to (None = all events)
            headers: Optional HTTP headers to include in requests
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.webhook_url = webhook_url
        self.event_types = event_types or ['*']
        self.headers = headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._event_bus = get_event_bus()
        self._active = False
        self._callbacks_registered = []

        # Validate requests library is available
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError(
                "requests package required for webhooks. Install with: pip install requests"
            )

    def _send_webhook(self, event: Any) -> None:
        """
        Send webhook POST request for an event.

        Executes in background thread to avoid blocking EventBus.
        Implements retry logic with exponential backoff.

        Args:
            event: Event object to send
        """
        import time

        # Serialize event
        try:
            if hasattr(event, 'to_dict'):
                payload = event.to_dict()
            else:
                payload = {'event_type': event.event_type}
        except Exception as e:
            logger.error(f"Failed to serialize event for webhook: {e}")
            return

        # Prepare request
        headers = {
            'Content-Type': 'application/json',
            **self.headers
        }

        # Retry loop
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._requests.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                logger.debug(
                    f"Webhook delivered successfully: {event.event_type} "
                    f"(status={response.status_code})"
                )
                return  # Success
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Webhook delivery failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Webhook delivery failed after {self.max_retries + 1} attempts: {e}"
                    )

    def _handle_event(self, event: Any) -> None:
        """
        Event handler callback for EventBus.

        Spawns background thread to send webhook without blocking.

        Args:
            event: Event received from EventBus
        """
        from threading import Thread

        # Execute webhook in background thread
        thread = Thread(target=self._send_webhook, args=(event,), daemon=True)
        thread.start()

    def start(self) -> None:
        """
        Start webhook dispatcher.

        Subscribes to EventBus for configured event types.
        """
        if self._active:
            logger.warning("WebhookDispatcher already started")
            return

        # Subscribe to event types
        for event_type in self.event_types:
            self._event_bus.subscribe(event_type, self._handle_event)
            self._callbacks_registered.append(event_type)
            logger.info(f"WebhookDispatcher subscribed to {event_type} -> {self.webhook_url}")

        self._active = True

    def stop(self) -> None:
        """
        Stop webhook dispatcher.

        Unsubscribes from EventBus and cleans up.
        """
        if not self._active:
            return

        # Unsubscribe from all registered event types
        for event_type in self._callbacks_registered:
            self._event_bus.unsubscribe(event_type, self._handle_event)
            logger.debug(f"WebhookDispatcher unsubscribed from {event_type}")

        self._callbacks_registered.clear()
        self._active = False
        logger.info("WebhookDispatcher stopped")

    def is_active(self) -> bool:
        """
        Check if dispatcher is active.

        Returns:
            True if dispatcher is running, False otherwise
        """
        return self._active


__all__ = ['EventBus', 'get_event_bus', 'reset_event_bus', 'WebhookDispatcher']
