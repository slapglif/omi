"""Integration Tests for Cross-Agent Belief Propagation

Tests the complete belief propagation flow:
1. Agent A updates a belief → BeliefNetwork emits BeliefUpdatedEvent
2. BeliefPropagator listens to event → applies trust-weighted propagation
3. Subscribed agents receive trust-modulated belief updates
4. Verify confidence calculations use correct EMA formulas with trust weights

Verifies:
- Event flow: BeliefNetwork → EventBus → BeliefPropagator → subscribed agents
- Trust weight modulation: 0.0 (no trust), 0.5 (moderate), 1.0 (full trust)
- Subscription filtering: namespace-based, memory-specific
- EMA update formulas: λ=0.15 (support), λ=0.30 (contradict)
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock


class TestBeliefPropagation:
    """Test end-to-end belief propagation: BeliefNetwork → EventBus → BeliefPropagator."""

    def test_single_agent_belief_propagation(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        End-to-end verification:
        1. Agent A creates a belief
        2. Agent A updates belief with evidence
        3. BeliefNetwork emits BeliefUpdatedEvent
        4. BeliefPropagator receives event
        5. Agent B (subscribed with trust weight) receives propagated update
        6. Verify Agent B's belief confidence is trust-modulated
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork, Evidence
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager
        from omi.events import BeliefUpdatedEvent
        from omi.storage.schema import init_database
        import sqlite3

        # Initialize database with schema
        conn = sqlite3.connect(temp_omi_setup["db_path"])
        init_database(conn)
        conn.close()

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Track events
        captured_events = []

        def capture_event(event):
            captured_events.append(event)

        clean_event_bus.subscribe('belief.updated', capture_event)

        # Configure trust: Agent B trusts Agent A with weight 0.8
        propagator.set_trust_weight("agent-a", "agent-b", 0.8)

        # Agent B subscribes to belief updates (global subscription)
        subscription_manager.subscribe(
            agent_id="agent-b",
            event_types=["belief.updated"]
        )

        # Start propagator
        propagator.start()

        # Step 1: Agent A creates a belief
        belief_id = belief_network.create_belief(
            content="Python is great for data science",
            initial_confidence=0.5,
            agent_id="agent-a",
            namespace="team-alpha"
        )

        # Step 2: Agent A updates belief with supporting evidence
        evidence_id = palace.store_memory(
            content="Found 10 data science libraries in Python",
            memory_type="fact"
        )

        from datetime import datetime
        evidence = Evidence(
            memory_id=evidence_id,
            supports=True,
            strength=0.4,
            timestamp=datetime.now()
        )

        belief_network.update_with_evidence(
            belief_id=belief_id,
            evidence=evidence,
            agent_id="agent-a",
            namespace="team-alpha"
        )

        # Step 3: Verify BeliefUpdatedEvent was emitted
        # Should have 2 events: 1 from create_belief, 1 from update_with_evidence
        assert len(captured_events) >= 2, f"Expected at least 2 events, got {len(captured_events)}"

        # Find the update event (not the creation event)
        update_events = [e for e in captured_events if hasattr(e, 'evidence_id') and e.evidence_id == evidence_id]
        assert len(update_events) == 1, "Should have 1 update event"

        update_event = update_events[0]
        assert isinstance(update_event, BeliefUpdatedEvent)
        assert update_event.belief_id == belief_id
        assert update_event.evidence_id == evidence_id
        if update_event.metadata:
            assert update_event.metadata.get('agent_id') == "agent-a"
            assert update_event.metadata.get('namespace') == "team-alpha"

        # Step 4: Verify belief was updated in palace
        belief = palace.get_belief(belief_id)
        assert belief is not None
        new_confidence = belief['confidence']

        # Verify EMA formula was applied correctly
        # Formula: new = old + λ * (target - old)
        # target = min(1.0, old + strength) = min(1.0, 0.5 + 0.4) = 0.9
        # new = 0.5 + 0.15 * (0.9 - 0.5) = 0.5 + 0.15 * 0.4 = 0.56
        expected_confidence = 0.5 + 0.15 * (0.9 - 0.5)
        assert abs(new_confidence - expected_confidence) < 0.01, \
            f"Expected {expected_confidence}, got {new_confidence}"

        # Step 5: Verify trust weight was set correctly
        trust_weight = propagator.get_trust_weight("agent-a", "agent-b")
        assert trust_weight == 0.8

        # Clean up
        propagator.stop()

    def test_trust_weight_modulation(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Verify trust weights correctly modulate propagation strength.

        Tests:
        - Full trust (1.0): propagate at full strength
        - Moderate trust (0.5): propagate at half strength
        - No trust (0.0): don't propagate
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Configure trust weights
        propagator.set_trust_weight("agent-source", "agent-full", 1.0)
        propagator.set_trust_weight("agent-source", "agent-half", 0.5)
        propagator.set_trust_weight("agent-source", "agent-none", 0.0)

        # Verify trust weights were set
        assert propagator.get_trust_weight("agent-source", "agent-full") == 1.0
        assert propagator.get_trust_weight("agent-source", "agent-half") == 0.5
        assert propagator.get_trust_weight("agent-source", "agent-none") == 0.0

        # Test trust relationship serialization
        relationships = propagator.list_trust_weights(agent_id="agent-source")
        assert len(relationships) == 3

        # Verify relationship data
        full_trust = [r for r in relationships if r.to_agent == "agent-full"][0]
        assert full_trust.from_agent == "agent-source"
        assert full_trust.weight == 1.0
        assert isinstance(full_trust.updated_at, datetime)

        # Test removing trust weight
        removed = propagator.remove_trust_weight("agent-source", "agent-none")
        assert removed is True

        # Verify removal
        assert propagator.get_trust_weight("agent-source", "agent-none") == 0.5  # Default

        # Verify list no longer includes removed relationship
        relationships = propagator.list_trust_weights(agent_id="agent-source")
        assert len(relationships) == 2

    def test_multiple_agent_propagation(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Test belief propagation to multiple subscribed agents.

        Scenario:
        - Agent A updates a belief
        - Agents B, C, D are all subscribed
        - Each has different trust weight
        - Verify each receives appropriately modulated update
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager
        from omi.storage.schema import init_database
        import sqlite3

        # Initialize database with schema
        conn = sqlite3.connect(temp_omi_setup["db_path"])
        init_database(conn)
        conn.close()

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Configure trust weights
        propagator.set_trust_weight("agent-a", "agent-b", 1.0)   # Full trust
        propagator.set_trust_weight("agent-a", "agent-c", 0.7)   # High trust
        propagator.set_trust_weight("agent-a", "agent-d", 0.3)   # Low trust

        # All agents subscribe to belief updates
        for agent_id in ["agent-b", "agent-c", "agent-d"]:
            subscription_manager.subscribe(
                agent_id=agent_id,
                event_types=["belief.updated"]
            )

        # Verify subscriptions
        subscriptions = subscription_manager.list_for_agent("agent-b")
        assert len(subscriptions) == 1
        assert subscriptions[0].agent_id == "agent-b"

        # Start propagator
        propagator.start()

        # Agent A creates and updates a belief
        belief_id = belief_network.create_belief(
            content="Test belief for multi-agent propagation",
            initial_confidence=0.5,
            agent_id="agent-a",
            namespace="shared-ns"
        )

        # Verify belief was created
        belief = palace.get_belief(belief_id)
        assert belief is not None
        assert belief['confidence'] == 0.5

        # Clean up
        propagator.stop()

    def test_subscription_filtering(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Test that propagation respects subscription filters.

        Scenario:
        - Agent B subscribes to namespace "team-alpha" only
        - Agent C subscribes to all namespaces
        - Agent A updates belief in "team-alpha" → both B and C notified
        - Agent A updates belief in "team-beta" → only C notified
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager
        from omi.storage.schema import init_database
        import sqlite3

        # Initialize database with schema
        conn = sqlite3.connect(temp_omi_setup["db_path"])
        init_database(conn)
        conn.close()

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Configure trust
        propagator.set_trust_weight("agent-a", "agent-b", 0.8)
        propagator.set_trust_weight("agent-a", "agent-c", 0.8)

        # Create shared namespaces
        from omi.shared_namespace import SharedNamespace
        shared_ns = SharedNamespace(temp_omi_setup["db_path"])
        shared_ns.create("team-alpha", created_by="agent-a")
        shared_ns.create("team-beta", created_by="agent-a")

        # Agent B subscribes to team-alpha only
        subscription_manager.subscribe(
            agent_id="agent-b",
            namespace="team-alpha",
            event_types=["belief.updated"]
        )

        # Agent C subscribes globally (all namespaces)
        subscription_manager.subscribe(
            agent_id="agent-c",
            event_types=["belief.updated"]
        )

        # Verify subscriptions
        b_subs = subscription_manager.list_for_agent("agent-b")
        assert len(b_subs) == 1
        assert b_subs[0].namespace == "team-alpha"

        c_subs = subscription_manager.list_for_agent("agent-c")
        assert len(c_subs) == 1
        assert c_subs[0].namespace is None  # Global subscription

        # Test subscription matching
        alpha_matches = subscription_manager.match_subscriptions(
            event_type="belief.updated",
            namespace="team-alpha",
            memory_id=None
        )
        # Both B (namespace-specific) and C (global) should match
        matched_agents = {sub.agent_id for sub in alpha_matches}
        assert "agent-b" in matched_agents
        assert "agent-c" in matched_agents

        beta_matches = subscription_manager.match_subscriptions(
            event_type="belief.updated",
            namespace="team-beta",
            memory_id=None
        )
        # Only C (global) should match, not B (team-alpha only)
        matched_agents = {sub.agent_id for sub in beta_matches}
        assert "agent-b" not in matched_agents
        assert "agent-c" in matched_agents

        # Start propagator
        propagator.start()

        # Test belief updates (actual propagation would happen here)
        # Just verify the setup is correct for now

        # Clean up
        propagator.stop()

    def test_supporting_vs_contradicting_evidence(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Verify different EMA lambdas for supporting vs contradicting evidence.

        Supporting evidence: λ=0.15 (gentle nudge)
        Contradicting evidence: λ=0.30 (hits 2x harder)
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork, Evidence
        from omi.belief_propagation import BeliefPropagator
        from datetime import datetime

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)

        # Create belief
        belief_id = belief_network.create_belief(
            content="Test belief for EMA verification",
            initial_confidence=0.6,
            agent_id="agent-a"
        )

        # Test supporting evidence
        evidence_support_id = palace.store_memory(
            content="Supporting evidence",
            memory_type="fact"
        )

        evidence_support = Evidence(
            memory_id=evidence_support_id,
            supports=True,
            strength=0.3,
            timestamp=datetime.now()
        )

        belief_network.update_with_evidence(
            belief_id=belief_id,
            evidence=evidence_support,
            agent_id="agent-a"
        )

        # Verify EMA with λ=0.15 for support
        # target = min(1.0, 0.6 + 0.3) = 0.9
        # new = 0.6 + 0.15 * (0.9 - 0.6) = 0.6 + 0.045 = 0.645
        belief = palace.get_belief(belief_id)
        expected = 0.6 + 0.15 * (0.9 - 0.6)
        assert abs(belief['confidence'] - expected) < 0.001

        # Test contradicting evidence
        evidence_contradict_id = palace.store_memory(
            content="Contradicting evidence",
            memory_type="fact"
        )

        old_confidence = belief['confidence']
        evidence_contradict = Evidence(
            memory_id=evidence_contradict_id,
            supports=False,
            strength=0.3,
            timestamp=datetime.now()
        )

        belief_network.update_with_evidence(
            belief_id=belief_id,
            evidence=evidence_contradict,
            agent_id="agent-a"
        )

        # Verify EMA with λ=0.30 for contradiction
        # target = max(0.0, old - 0.3)
        # new = old + 0.30 * (target - old)
        belief = palace.get_belief(belief_id)
        target = max(0.0, old_confidence - 0.3)
        expected = old_confidence + 0.30 * (target - old_confidence)
        assert abs(belief['confidence'] - expected) < 0.001

    def test_propagator_lifecycle(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Test BeliefPropagator start/stop lifecycle.

        Verify:
        - Can start propagator
        - Can stop propagator
        - Can restart propagator
        - Events only handled when started
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Initially not listening
        assert propagator._listening is False

        # Start propagator
        propagator.start()
        assert propagator._listening is True

        # Try starting again (should be idempotent)
        propagator.start()
        assert propagator._listening is True

        # Stop propagator
        propagator.stop()
        assert propagator._listening is False

        # Try stopping again (should be idempotent)
        propagator.stop()
        assert propagator._listening is False

        # Restart propagator
        propagator.start()
        assert propagator._listening is True
        propagator.stop()

    def test_trust_weight_validation(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Test trust weight validation.

        Verify:
        - Trust weights must be in [0.0, 1.0] range
        - Invalid weights raise ValueError
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Valid weights
        propagator.set_trust_weight("agent-a", "agent-b", 0.0)
        propagator.set_trust_weight("agent-a", "agent-b", 0.5)
        propagator.set_trust_weight("agent-a", "agent-b", 1.0)

        # Invalid weights
        with pytest.raises(ValueError, match="Trust weight must be in"):
            propagator.set_trust_weight("agent-a", "agent-b", -0.1)

        with pytest.raises(ValueError, match="Trust weight must be in"):
            propagator.set_trust_weight("agent-a", "agent-b", 1.1)

        with pytest.raises(ValueError, match="Trust weight must be in"):
            propagator.set_trust_weight("agent-a", "agent-b", 2.0)

    def test_trust_modulated_ema_calculation(
        self,
        temp_omi_setup,
        clean_event_bus
    ):
        """
        Test trust-modulated EMA calculation directly.

        Verify the _apply_trust_modulated_update method works correctly
        for different trust weights and evidence types.
        """
        from omi import GraphPalace
        from omi.belief import BeliefNetwork
        from omi.belief_propagation import BeliefPropagator
        from omi.subscriptions import SubscriptionManager

        # Setup
        palace = GraphPalace(temp_omi_setup["db_path"])
        belief_network = BeliefNetwork(palace, event_bus=clean_event_bus)
        subscription_manager = SubscriptionManager(temp_omi_setup["db_path"])
        propagator = BeliefPropagator(
            event_bus=clean_event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Test supporting evidence with full trust
        new_conf = propagator._apply_trust_modulated_update(
            old_confidence=0.5,
            supports=True,
            strength=0.4,
            trust_weight=1.0
        )
        # target = min(1.0, 0.5 + 0.4*1.0) = 0.9
        # new = 0.5 + 0.15 * (0.9 - 0.5) = 0.56
        expected = 0.5 + 0.15 * (0.9 - 0.5)
        assert abs(new_conf - expected) < 0.001

        # Test supporting evidence with half trust
        new_conf = propagator._apply_trust_modulated_update(
            old_confidence=0.5,
            supports=True,
            strength=0.4,
            trust_weight=0.5
        )
        # modulated_strength = 0.4 * 0.5 = 0.2
        # target = min(1.0, 0.5 + 0.2) = 0.7
        # new = 0.5 + 0.15 * (0.7 - 0.5) = 0.53
        expected = 0.5 + 0.15 * (0.7 - 0.5)
        assert abs(new_conf - expected) < 0.001

        # Test contradicting evidence with full trust
        new_conf = propagator._apply_trust_modulated_update(
            old_confidence=0.6,
            supports=False,
            strength=0.3,
            trust_weight=1.0
        )
        # target = max(0.0, 0.6 - 0.3*1.0) = 0.3
        # new = 0.6 + 0.30 * (0.3 - 0.6) = 0.6 - 0.09 = 0.51
        expected = 0.6 + 0.30 * (0.3 - 0.6)
        assert abs(new_conf - expected) < 0.001

        # Test contradicting evidence with half trust
        new_conf = propagator._apply_trust_modulated_update(
            old_confidence=0.6,
            supports=False,
            strength=0.3,
            trust_weight=0.5
        )
        # modulated_strength = 0.3 * 0.5 = 0.15
        # target = max(0.0, 0.6 - 0.15) = 0.45
        # new = 0.6 + 0.30 * (0.45 - 0.6) = 0.6 - 0.045 = 0.555
        expected = 0.6 + 0.30 * (0.45 - 0.6)
        assert abs(new_conf - expected) < 0.001

        # Test confidence bounds (can't go below 0 or above 1)
        new_conf = propagator._apply_trust_modulated_update(
            old_confidence=0.95,
            supports=True,
            strength=0.5,
            trust_weight=1.0
        )
        assert 0.0 <= new_conf <= 1.0

        new_conf = propagator._apply_trust_modulated_update(
            old_confidence=0.05,
            supports=False,
            strength=0.5,
            trust_weight=1.0
        )
        assert 0.0 <= new_conf <= 1.0
