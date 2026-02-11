"""Unit Tests for EventBus

Tests: EventBus subscribe/publish, thread safety, wildcard subscriptions
"""
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock


class TestEventBusBasics:
    """Tests for core EventBus functionality."""

    def test_instantiation(self):
        """Can instantiate EventBus."""
        from omi.event_bus import EventBus

        bus = EventBus()
        assert bus is not None
        assert bus._subscribers == {}

    def test_subscribe_creates_subscription(self):
        """Subscribe adds callback to subscription list."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback = Mock()

        bus.subscribe('memory.stored', callback)

        assert 'memory.stored' in bus._subscribers
        assert callback in bus._subscribers['memory.stored']

    def test_subscribe_multiple_callbacks_same_type(self):
        """Multiple callbacks can subscribe to same event type."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback1 = Mock()
        callback2 = Mock()

        bus.subscribe('memory.stored', callback1)
        bus.subscribe('memory.stored', callback2)

        assert len(bus._subscribers['memory.stored']) == 2
        assert callback1 in bus._subscribers['memory.stored']
        assert callback2 in bus._subscribers['memory.stored']

    def test_subscribe_different_event_types(self):
        """Can subscribe to different event types."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback1 = Mock()
        callback2 = Mock()

        bus.subscribe('memory.stored', callback1)
        bus.subscribe('belief.updated', callback2)

        assert 'memory.stored' in bus._subscribers
        assert 'belief.updated' in bus._subscribers

    def test_subscriber_count_single_type(self):
        """subscriber_count returns count for specific type."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback1 = Mock()
        callback2 = Mock()

        bus.subscribe('memory.stored', callback1)
        bus.subscribe('memory.stored', callback2)

        assert bus.subscriber_count('memory.stored') == 2
        assert bus.subscriber_count('belief.updated') == 0

    def test_subscriber_count_total(self):
        """subscriber_count returns total when no type specified."""
        from omi.event_bus import EventBus

        bus = EventBus()

        bus.subscribe('memory.stored', Mock())
        bus.subscribe('memory.stored', Mock())
        bus.subscribe('belief.updated', Mock())

        assert bus.subscriber_count() == 3


class TestEventBusPublish:
    """Tests for event publishing."""

    def test_publish_calls_subscriber(self):
        """Publishing event calls subscribed callback."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        callback = Mock()
        bus.subscribe('memory.stored', callback)

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        bus.publish(event)

        callback.assert_called_once_with(event)

    def test_publish_calls_multiple_subscribers(self):
        """Publishing calls all subscribers for that event type."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        bus.subscribe('memory.stored', callback1)
        bus.subscribe('memory.stored', callback2)
        bus.subscribe('belief.updated', callback3)

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        bus.publish(event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)
        callback3.assert_not_called()

    def test_publish_without_subscribers(self):
        """Publishing without subscribers doesn't raise error."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        # Should not raise
        bus.publish(event)

    def test_publish_missing_event_type_attribute(self):
        """Publishing object without event_type attribute is handled gracefully."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback = Mock()
        bus.subscribe('test.event', callback)

        # Object without event_type
        invalid_event = {"data": "test"}

        bus.publish(invalid_event)

        # Callback should not be called
        callback.assert_not_called()

    def test_publish_handles_callback_exception(self):
        """Publishing continues if a callback raises exception."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()

        def failing_callback(event):
            raise ValueError("Callback error")

        callback2 = Mock()

        bus.subscribe('memory.stored', failing_callback)
        bus.subscribe('memory.stored', callback2)

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        # Should not raise, should call callback2
        bus.publish(event)

        callback2.assert_called_once_with(event)


class TestEventBusWildcard:
    """Tests for wildcard subscriptions."""

    def test_wildcard_receives_all_events(self):
        """Wildcard subscription ('*') receives all event types."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent, BeliefUpdatedEvent

        bus = EventBus()
        wildcard_callback = Mock()

        bus.subscribe('*', wildcard_callback)

        event1 = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )
        event2 = BeliefUpdatedEvent(
            belief_id="belief123",
            old_confidence=0.5,
            new_confidence=0.8
        )

        bus.publish(event1)
        bus.publish(event2)

        assert wildcard_callback.call_count == 2
        wildcard_callback.assert_any_call(event1)
        wildcard_callback.assert_any_call(event2)

    def test_wildcard_and_specific_both_called(self):
        """Both wildcard and specific subscribers receive events."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        wildcard_callback = Mock()
        specific_callback = Mock()

        bus.subscribe('*', wildcard_callback)
        bus.subscribe('memory.stored', specific_callback)

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        bus.publish(event)

        wildcard_callback.assert_called_once_with(event)
        specific_callback.assert_called_once_with(event)


class TestEventBusUnsubscribe:
    """Tests for unsubscribing from events."""

    def test_unsubscribe_removes_callback(self):
        """Unsubscribe removes callback from subscription list."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback = Mock()

        bus.subscribe('memory.stored', callback)
        result = bus.unsubscribe('memory.stored', callback)

        assert result is True
        assert 'memory.stored' not in bus._subscribers

    def test_unsubscribe_nonexistent_returns_false(self):
        """Unsubscribing nonexistent callback returns False."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback = Mock()

        result = bus.unsubscribe('memory.stored', callback)

        assert result is False

    def test_unsubscribe_cleans_up_empty_lists(self):
        """Unsubscribe removes event type when no subscribers left."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callback1 = Mock()
        callback2 = Mock()

        bus.subscribe('memory.stored', callback1)
        bus.subscribe('memory.stored', callback2)

        bus.unsubscribe('memory.stored', callback1)
        assert 'memory.stored' in bus._subscribers

        bus.unsubscribe('memory.stored', callback2)
        assert 'memory.stored' not in bus._subscribers

    def test_unsubscribed_callback_not_called(self):
        """Unsubscribed callback doesn't receive events."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        callback = Mock()

        bus.subscribe('memory.stored', callback)
        bus.unsubscribe('memory.stored', callback)

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        bus.publish(event)

        callback.assert_not_called()


class TestEventBusClear:
    """Tests for clearing subscriptions."""

    def test_clear_removes_all_subscriptions(self):
        """Clear removes all subscriptions."""
        from omi.event_bus import EventBus

        bus = EventBus()

        bus.subscribe('memory.stored', Mock())
        bus.subscribe('belief.updated', Mock())
        bus.subscribe('*', Mock())

        assert bus.subscriber_count() == 3

        bus.clear()

        assert bus.subscriber_count() == 0
        assert bus._subscribers == {}

    def test_cleared_bus_no_callbacks(self):
        """Cleared bus doesn't call any callbacks."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        callback = Mock()

        bus.subscribe('memory.stored', callback)
        bus.clear()

        event = MemoryStoredEvent(
            memory_id="test123",
            content="Test memory",
            memory_type="fact"
        )

        bus.publish(event)

        callback.assert_not_called()


class TestEventBusThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_subscriptions(self):
        """Multiple threads can subscribe concurrently."""
        from omi.event_bus import EventBus

        bus = EventBus()
        callbacks = []
        threads = []

        def subscribe_callback():
            callback = Mock()
            callbacks.append(callback)
            bus.subscribe('memory.stored', callback)

        # Create 10 threads subscribing concurrently
        for _ in range(10):
            thread = threading.Thread(target=subscribe_callback)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        assert bus.subscriber_count('memory.stored') == 10

    def test_concurrent_publish(self):
        """Multiple threads can publish concurrently."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        callback = Mock()
        bus.subscribe('memory.stored', callback)

        threads = []

        def publish_event(i):
            event = MemoryStoredEvent(
                memory_id=f"test{i}",
                content=f"Memory {i}",
                memory_type="fact"
            )
            bus.publish(event)

        # Create 10 threads publishing concurrently
        for i in range(10):
            thread = threading.Thread(target=publish_event, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All events should be received
        assert callback.call_count == 10


class TestGlobalEventBus:
    """Tests for global event bus singleton."""

    def test_get_event_bus_returns_singleton(self):
        """get_event_bus returns same instance."""
        from omi.event_bus import get_event_bus

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

    def test_reset_event_bus_creates_new_instance(self):
        """reset_event_bus creates fresh instance."""
        from omi.event_bus import get_event_bus, reset_event_bus

        bus1 = get_event_bus()
        bus1.subscribe('memory.stored', Mock())

        reset_event_bus()

        bus2 = get_event_bus()

        # New instance should have no subscribers
        assert bus2.subscriber_count() == 0
        assert bus1 is not bus2


class TestEventBusIntegration:
    """Integration tests with real event types."""

    def test_memory_stored_event_flow(self):
        """Full flow: subscribe -> publish MemoryStoredEvent -> callback receives event."""
        from omi.event_bus import EventBus
        from omi.events import MemoryStoredEvent

        bus = EventBus()
        received_events = []

        def on_memory_stored(event):
            received_events.append(event)

        bus.subscribe('memory.stored', on_memory_stored)

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Important fact",
            memory_type="fact",
            confidence=0.9
        )

        bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].memory_id == "mem123"
        assert received_events[0].content == "Important fact"
        assert received_events[0].confidence == 0.9

    def test_multiple_event_types(self):
        """Can handle multiple event types in sequence."""
        from omi.event_bus import EventBus
        from omi.events import (
            MemoryStoredEvent,
            BeliefUpdatedEvent,
            SessionStartedEvent
        )

        bus = EventBus()
        memory_callback = Mock()
        belief_callback = Mock()
        session_callback = Mock()

        bus.subscribe('memory.stored', memory_callback)
        bus.subscribe('belief.updated', belief_callback)
        bus.subscribe('session.started', session_callback)

        event1 = MemoryStoredEvent(
            memory_id="mem123",
            content="Test",
            memory_type="fact"
        )
        event2 = BeliefUpdatedEvent(
            belief_id="belief123",
            old_confidence=0.5,
            new_confidence=0.8
        )
        event3 = SessionStartedEvent(session_id="session123")

        bus.publish(event1)
        bus.publish(event2)
        bus.publish(event3)

        memory_callback.assert_called_once()
        belief_callback.assert_called_once()
        session_callback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
