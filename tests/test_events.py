"""Unit Tests for Event Types

Tests: Event dataclass creation, serialization, and attributes
"""
import pytest
from datetime import datetime
from unittest.mock import Mock


class TestMemoryStoredEvent:
    """Tests for MemoryStoredEvent."""

    def test_creation_with_required_fields(self):
        """Can create MemoryStoredEvent with required fields."""
        from omi.events import MemoryStoredEvent

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test memory",
            memory_type="fact"
        )

        assert event.memory_id == "mem123"
        assert event.content == "Test memory"
        assert event.memory_type == "fact"
        assert event.event_type == "memory.stored"

    def test_creation_with_optional_fields(self):
        """Can create MemoryStoredEvent with optional fields."""
        from omi.events import MemoryStoredEvent

        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        metadata = {"source": "test", "tags": ["important"]}

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test memory",
            memory_type="fact",
            confidence=0.95,
            timestamp=timestamp,
            metadata=metadata
        )

        assert event.confidence == 0.95
        assert event.timestamp == timestamp
        assert event.metadata == metadata

    def test_timestamp_defaults_to_now(self):
        """Timestamp defaults to current time."""
        from omi.events import MemoryStoredEvent

        before = datetime.now()
        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test memory",
            memory_type="fact"
        )
        after = datetime.now()

        assert before <= event.timestamp <= after

    def test_to_dict_serialization(self):
        """to_dict returns proper dictionary format."""
        from omi.events import MemoryStoredEvent

        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test memory",
            memory_type="fact",
            confidence=0.9,
            timestamp=timestamp,
            metadata={"key": "value"}
        )

        result = event.to_dict()

        assert result["event_type"] == "memory.stored"
        assert result["memory_id"] == "mem123"
        assert result["content"] == "Test memory"
        assert result["memory_type"] == "fact"
        assert result["confidence"] == 0.9
        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["metadata"] == {"key": "value"}

    def test_to_dict_handles_none_metadata(self):
        """to_dict handles None metadata gracefully."""
        from omi.events import MemoryStoredEvent

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test memory",
            memory_type="fact"
        )

        result = event.to_dict()

        assert result["metadata"] == {}

    def test_memory_types(self):
        """Supports different memory types."""
        from omi.events import MemoryStoredEvent

        types = ["fact", "experience", "belief", "decision"]

        for mem_type in types:
            event = MemoryStoredEvent(
                memory_id="mem123",
                content="Test",
                memory_type=mem_type
            )
            assert event.memory_type == mem_type


class TestMemoryRecalledEvent:
    """Tests for MemoryRecalledEvent."""

    def test_creation_with_required_fields(self):
        """Can create MemoryRecalledEvent with required fields."""
        from omi.events import MemoryRecalledEvent

        event = MemoryRecalledEvent(
            query="test query",
            result_count=5
        )

        assert event.query == "test query"
        assert event.result_count == 5
        assert event.event_type == "memory.recalled"
        assert event.top_results == []

    def test_creation_with_top_results(self):
        """Can create MemoryRecalledEvent with top results."""
        from omi.events import MemoryRecalledEvent

        top_results = [
            {"memory_id": "mem1", "content": "Result 1", "score": 0.95},
            {"memory_id": "mem2", "content": "Result 2", "score": 0.85}
        ]

        event = MemoryRecalledEvent(
            query="test query",
            result_count=2,
            top_results=top_results
        )

        assert event.top_results == top_results
        assert len(event.top_results) == 2

    def test_to_dict_serialization(self):
        """to_dict returns proper dictionary format."""
        from omi.events import MemoryRecalledEvent

        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        top_results = [{"memory_id": "mem1", "score": 0.9}]

        event = MemoryRecalledEvent(
            query="test query",
            result_count=1,
            top_results=top_results,
            timestamp=timestamp,
            metadata={"context": "test"}
        )

        result = event.to_dict()

        assert result["event_type"] == "memory.recalled"
        assert result["query"] == "test query"
        assert result["result_count"] == 1
        assert result["top_results"] == top_results
        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["metadata"] == {"context": "test"}


class TestBeliefUpdatedEvent:
    """Tests for BeliefUpdatedEvent."""

    def test_creation_with_required_fields(self):
        """Can create BeliefUpdatedEvent with required fields."""
        from omi.events import BeliefUpdatedEvent

        event = BeliefUpdatedEvent(
            belief_id="belief123",
            old_confidence=0.5,
            new_confidence=0.8
        )

        assert event.belief_id == "belief123"
        assert event.old_confidence == 0.5
        assert event.new_confidence == 0.8
        assert event.event_type == "belief.updated"
        assert event.evidence_id is None

    def test_creation_with_evidence(self):
        """Can create BeliefUpdatedEvent with evidence ID."""
        from omi.events import BeliefUpdatedEvent

        event = BeliefUpdatedEvent(
            belief_id="belief123",
            old_confidence=0.5,
            new_confidence=0.8,
            evidence_id="evidence456"
        )

        assert event.evidence_id == "evidence456"

    def test_to_dict_serialization(self):
        """to_dict returns proper dictionary format."""
        from omi.events import BeliefUpdatedEvent

        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        event = BeliefUpdatedEvent(
            belief_id="belief123",
            old_confidence=0.5,
            new_confidence=0.8,
            evidence_id="evidence456",
            timestamp=timestamp,
            metadata={"reason": "new evidence"}
        )

        result = event.to_dict()

        assert result["event_type"] == "belief.updated"
        assert result["belief_id"] == "belief123"
        assert result["old_confidence"] == 0.5
        assert result["new_confidence"] == 0.8
        assert result["evidence_id"] == "evidence456"
        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["metadata"] == {"reason": "new evidence"}

    def test_confidence_changes(self):
        """Handles various confidence change scenarios."""
        from omi.events import BeliefUpdatedEvent

        # Confidence increase
        event1 = BeliefUpdatedEvent(
            belief_id="belief1",
            old_confidence=0.3,
            new_confidence=0.7
        )
        assert event1.new_confidence > event1.old_confidence

        # Confidence decrease
        event2 = BeliefUpdatedEvent(
            belief_id="belief2",
            old_confidence=0.9,
            new_confidence=0.4
        )
        assert event2.new_confidence < event2.old_confidence

        # No change (edge case)
        event3 = BeliefUpdatedEvent(
            belief_id="belief3",
            old_confidence=0.5,
            new_confidence=0.5
        )
        assert event3.new_confidence == event3.old_confidence


class TestContradictionDetectedEvent:
    """Tests for ContradictionDetectedEvent."""

    def test_creation_with_required_fields(self):
        """Can create ContradictionDetectedEvent with required fields."""
        from omi.events import ContradictionDetectedEvent

        event = ContradictionDetectedEvent(
            memory_id_1="mem1",
            memory_id_2="mem2",
            contradiction_pattern="negation"
        )

        assert event.memory_id_1 == "mem1"
        assert event.memory_id_2 == "mem2"
        assert event.contradiction_pattern == "negation"
        assert event.event_type == "belief.contradiction_detected"
        assert event.confidence is None

    def test_creation_with_confidence(self):
        """Can create ContradictionDetectedEvent with confidence."""
        from omi.events import ContradictionDetectedEvent

        event = ContradictionDetectedEvent(
            memory_id_1="mem1",
            memory_id_2="mem2",
            contradiction_pattern="negation",
            confidence=0.85
        )

        assert event.confidence == 0.85

    def test_to_dict_serialization(self):
        """to_dict returns proper dictionary format."""
        from omi.events import ContradictionDetectedEvent

        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        event = ContradictionDetectedEvent(
            memory_id_1="mem1",
            memory_id_2="mem2",
            contradiction_pattern="negation",
            confidence=0.9,
            timestamp=timestamp,
            metadata={"severity": "high"}
        )

        result = event.to_dict()

        assert result["event_type"] == "belief.contradiction_detected"
        assert result["memory_id_1"] == "mem1"
        assert result["memory_id_2"] == "mem2"
        assert result["contradiction_pattern"] == "negation"
        assert result["confidence"] == 0.9
        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["metadata"] == {"severity": "high"}

    def test_contradiction_patterns(self):
        """Supports different contradiction patterns."""
        from omi.events import ContradictionDetectedEvent

        patterns = [
            "negation",
            "temporal_conflict",
            "value_conflict",
            "logical_inconsistency"
        ]

        for pattern in patterns:
            event = ContradictionDetectedEvent(
                memory_id_1="mem1",
                memory_id_2="mem2",
                contradiction_pattern=pattern
            )
            assert event.contradiction_pattern == pattern


class TestSessionStartedEvent:
    """Tests for SessionStartedEvent."""

    def test_creation_minimal(self):
        """Can create SessionStartedEvent with minimal fields."""
        from omi.events import SessionStartedEvent

        event = SessionStartedEvent()

        assert event.event_type == "session.started"
        assert event.session_id is None
        assert isinstance(event.timestamp, datetime)

    def test_creation_with_session_id(self):
        """Can create SessionStartedEvent with session ID."""
        from omi.events import SessionStartedEvent

        event = SessionStartedEvent(session_id="session123")

        assert event.session_id == "session123"

    def test_to_dict_serialization(self):
        """to_dict returns proper dictionary format."""
        from omi.events import SessionStartedEvent

        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        event = SessionStartedEvent(
            session_id="session123",
            timestamp=timestamp,
            metadata={"user": "test_user"}
        )

        result = event.to_dict()

        assert result["event_type"] == "session.started"
        assert result["session_id"] == "session123"
        assert result["timestamp"] == "2024-01-15T10:30:00"
        assert result["metadata"] == {"user": "test_user"}


class TestSessionEndedEvent:
    """Tests for SessionEndedEvent."""

    def test_creation_minimal(self):
        """Can create SessionEndedEvent with minimal fields."""
        from omi.events import SessionEndedEvent

        event = SessionEndedEvent()

        assert event.event_type == "session.ended"
        assert event.session_id is None
        assert event.duration_seconds is None
        assert isinstance(event.timestamp, datetime)

    def test_creation_with_all_fields(self):
        """Can create SessionEndedEvent with all fields."""
        from omi.events import SessionEndedEvent

        event = SessionEndedEvent(
            session_id="session123",
            duration_seconds=3600.5
        )

        assert event.session_id == "session123"
        assert event.duration_seconds == 3600.5

    def test_to_dict_serialization(self):
        """to_dict returns proper dictionary format."""
        from omi.events import SessionEndedEvent

        timestamp = datetime(2024, 1, 15, 11, 30, 0)
        event = SessionEndedEvent(
            session_id="session123",
            duration_seconds=3600.0,
            timestamp=timestamp,
            metadata={"user": "test_user", "exit_reason": "normal"}
        )

        result = event.to_dict()

        assert result["event_type"] == "session.ended"
        assert result["session_id"] == "session123"
        assert result["duration_seconds"] == 3600.0
        assert result["timestamp"] == "2024-01-15T11:30:00"
        assert result["metadata"] == {"user": "test_user", "exit_reason": "normal"}


class TestEventTypeAttributes:
    """Tests for common event attributes."""

    def test_all_events_have_event_type(self):
        """All event types have event_type attribute."""
        from omi.events import (
            MemoryStoredEvent,
            MemoryRecalledEvent,
            BeliefUpdatedEvent,
            ContradictionDetectedEvent,
            SessionStartedEvent,
            SessionEndedEvent
        )

        events = [
            MemoryStoredEvent(memory_id="1", content="test", memory_type="fact"),
            MemoryRecalledEvent(query="test", result_count=0),
            BeliefUpdatedEvent(belief_id="1", old_confidence=0.5, new_confidence=0.8),
            ContradictionDetectedEvent(memory_id_1="1", memory_id_2="2", contradiction_pattern="test"),
            SessionStartedEvent(),
            SessionEndedEvent()
        ]

        for event in events:
            assert hasattr(event, 'event_type')
            assert isinstance(event.event_type, str)
            assert len(event.event_type) > 0

    def test_all_events_have_timestamp(self):
        """All event types have timestamp attribute."""
        from omi.events import (
            MemoryStoredEvent,
            MemoryRecalledEvent,
            BeliefUpdatedEvent,
            ContradictionDetectedEvent,
            SessionStartedEvent,
            SessionEndedEvent
        )

        events = [
            MemoryStoredEvent(memory_id="1", content="test", memory_type="fact"),
            MemoryRecalledEvent(query="test", result_count=0),
            BeliefUpdatedEvent(belief_id="1", old_confidence=0.5, new_confidence=0.8),
            ContradictionDetectedEvent(memory_id_1="1", memory_id_2="2", contradiction_pattern="test"),
            SessionStartedEvent(),
            SessionEndedEvent()
        ]

        for event in events:
            assert hasattr(event, 'timestamp')
            assert isinstance(event.timestamp, datetime)

    def test_all_events_have_to_dict(self):
        """All event types have to_dict method."""
        from omi.events import (
            MemoryStoredEvent,
            MemoryRecalledEvent,
            BeliefUpdatedEvent,
            ContradictionDetectedEvent,
            SessionStartedEvent,
            SessionEndedEvent
        )

        events = [
            MemoryStoredEvent(memory_id="1", content="test", memory_type="fact"),
            MemoryRecalledEvent(query="test", result_count=0),
            BeliefUpdatedEvent(belief_id="1", old_confidence=0.5, new_confidence=0.8),
            ContradictionDetectedEvent(memory_id_1="1", memory_id_2="2", contradiction_pattern="test"),
            SessionStartedEvent(),
            SessionEndedEvent()
        ]

        for event in events:
            assert hasattr(event, 'to_dict')
            result = event.to_dict()
            assert isinstance(result, dict)
            assert 'event_type' in result
            assert 'timestamp' in result

    def test_event_types_are_unique(self):
        """Each event type has unique event_type string."""
        from omi.events import (
            MemoryStoredEvent,
            MemoryRecalledEvent,
            BeliefUpdatedEvent,
            ContradictionDetectedEvent,
            SessionStartedEvent,
            SessionEndedEvent
        )

        event_types = {
            MemoryStoredEvent(memory_id="1", content="test", memory_type="fact").event_type,
            MemoryRecalledEvent(query="test", result_count=0).event_type,
            BeliefUpdatedEvent(belief_id="1", old_confidence=0.5, new_confidence=0.8).event_type,
            ContradictionDetectedEvent(memory_id_1="1", memory_id_2="2", contradiction_pattern="test").event_type,
            SessionStartedEvent().event_type,
            SessionEndedEvent().event_type
        }

        # All should be unique
        assert len(event_types) == 6


class TestEventSerialization:
    """Tests for event serialization edge cases."""

    def test_none_timestamp_serialization(self):
        """Handles None timestamp in serialization."""
        from omi.events import SessionStartedEvent

        event = SessionStartedEvent()
        event.timestamp = None

        result = event.to_dict()

        assert result["timestamp"] is None

    def test_special_characters_in_content(self):
        """Handles special characters in event content."""
        from omi.events import MemoryStoredEvent

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test with <special> & 'chars' \"quotes\" 你好",
            memory_type="fact"
        )

        result = event.to_dict()

        assert result["content"] == "Test with <special> & 'chars' \"quotes\" 你好"

    def test_empty_metadata_serialization(self):
        """Empty metadata serializes as empty dict."""
        from omi.events import MemoryStoredEvent

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test",
            memory_type="fact",
            metadata={}
        )

        result = event.to_dict()

        assert result["metadata"] == {}

    def test_complex_metadata_serialization(self):
        """Complex metadata structures serialize correctly."""
        from omi.events import MemoryStoredEvent

        complex_metadata = {
            "nested": {
                "key": "value",
                "list": [1, 2, 3],
                "bool": True
            },
            "tags": ["tag1", "tag2"],
            "count": 42
        }

        event = MemoryStoredEvent(
            memory_id="mem123",
            content="Test",
            memory_type="fact",
            metadata=complex_metadata
        )

        result = event.to_dict()

        assert result["metadata"] == complex_metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
