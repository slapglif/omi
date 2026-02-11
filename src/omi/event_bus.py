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


__all__ = ['EventBus', 'get_event_bus', 'reset_event_bus']
