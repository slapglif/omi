"""
Cross-agent belief propagation with configurable trust weights

Enables belief updates to propagate across agents in shared namespaces.
When one agent updates a belief, subscribed agents receive the update
modulated by configurable trust weights.

Pattern: EventBus-based propagation with trust-weighted evidence
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrustRelationship:
    """Trust relationship between two agents"""
    from_agent: str
    to_agent: str
    weight: float  # 0.0 to 1.0
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "weight": self.weight,
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class PropagationResult:
    """Result of a belief propagation operation"""
    source_agent: str
    target_agent: str
    belief_id: str
    old_confidence: float
    new_confidence: float
    trust_weight: float
    propagated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "belief_id": self.belief_id,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "trust_weight": self.trust_weight,
            "propagated_at": self.propagated_at.isoformat()
        }


class BeliefPropagator:
    """
    Cross-agent belief propagation with configurable trust weights

    Features:
    - Subscribe to belief update events via EventBus
    - Propagate updates to subscribed agents in shared namespaces
    - Apply configurable trust weights to modulate propagation strength
    - Use same EMA update logic as BeliefNetwork (λ=0.15 for support, λ=0.30 for contradict)
    - Thread-safe trust weight management
    - Emit propagation events for audit trail

    Trust weight interpretation:
    - 1.0 = Full trust (propagate update at full strength)
    - 0.5 = Moderate trust (propagate at half strength)
    - 0.0 = No trust (don't propagate)

    Pattern follows BeliefNetwork and EventBus architecture:
    - Event-driven propagation
    - EMA-based confidence updates
    - Thread-safe operations
    - Comprehensive logging

    Usage:
        propagator = BeliefPropagator(
            event_bus=event_bus,
            subscription_manager=subscription_manager,
            belief_network=belief_network
        )

        # Configure trust relationships
        propagator.set_trust_weight("agent-1", "agent-2", 0.8)
        propagator.set_trust_weight("agent-2", "agent-1", 0.9)

        # Start listening for belief updates
        propagator.start()

        # When agent-1 updates a belief, agent-2 will receive
        # a trust-weighted (0.8x) version of the update
    """

    # EMA lambdas from BeliefNetwork
    SUPPORT_LAMBDA = 0.15
    CONTRADICT_LAMBDA = 0.30

    def __init__(
        self,
        event_bus: Any,
        subscription_manager: Any,
        belief_network: Any
    ):
        """
        Initialize BeliefPropagator.

        Args:
            event_bus: EventBus instance for subscribing to events
            subscription_manager: SubscriptionManager for finding subscribed agents
            belief_network: BeliefNetwork for applying updates
        """
        self.event_bus = event_bus
        self.subscription_manager = subscription_manager
        self.belief_network = belief_network

        # Trust weights: (from_agent, to_agent) -> weight
        self._trust_weights: Dict[Tuple[str, str], float] = {}
        self._trust_lock = threading.Lock()

        # Track whether we're actively listening
        self._listening = False

        logger.info("BeliefPropagator initialized")

    def set_trust_weight(
        self,
        from_agent: str,
        to_agent: str,
        weight: float
    ) -> TrustRelationship:
        """
        Set trust weight from one agent to another.

        Trust weights modulate how much belief updates from one agent
        affect another agent's beliefs. Higher trust = stronger influence.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID
            weight: Trust weight (0.0 to 1.0)

        Returns:
            TrustRelationship object

        Raises:
            ValueError: If weight is outside [0.0, 1.0] range

        Examples:
            # Full trust
            propagator.set_trust_weight("expert-agent", "learner-agent", 1.0)

            # Moderate trust
            propagator.set_trust_weight("peer-1", "peer-2", 0.6)

            # No trust (disable propagation)
            propagator.set_trust_weight("untrusted", "agent", 0.0)
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Trust weight must be in [0.0, 1.0], got {weight}")

        with self._trust_lock:
            key = (from_agent, to_agent)
            self._trust_weights[key] = weight

        relationship = TrustRelationship(
            from_agent=from_agent,
            to_agent=to_agent,
            weight=weight,
            updated_at=datetime.now()
        )

        logger.info(
            f"Set trust weight: {from_agent} -> {to_agent} = {weight:.2f}"
        )

        return relationship

    def get_trust_weight(
        self,
        from_agent: str,
        to_agent: str
    ) -> float:
        """
        Get trust weight from one agent to another.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID

        Returns:
            Trust weight (defaults to 0.5 if not set)
        """
        with self._trust_lock:
            key = (from_agent, to_agent)
            return self._trust_weights.get(key, 0.5)  # Default: moderate trust

    def remove_trust_weight(
        self,
        from_agent: str,
        to_agent: str
    ) -> bool:
        """
        Remove trust weight between two agents.

        Args:
            from_agent: Source agent ID
            to_agent: Target agent ID

        Returns:
            True if removed, False if didn't exist
        """
        with self._trust_lock:
            key = (from_agent, to_agent)
            if key in self._trust_weights:
                del self._trust_weights[key]
                logger.info(f"Removed trust weight: {from_agent} -> {to_agent}")
                return True
        return False

    def list_trust_weights(
        self,
        agent_id: Optional[str] = None
    ) -> List[TrustRelationship]:
        """
        List trust weights, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter by (as source or target)

        Returns:
            List of TrustRelationship objects
        """
        with self._trust_lock:
            relationships = []
            for (from_agent, to_agent), weight in self._trust_weights.items():
                if agent_id is None or agent_id in (from_agent, to_agent):
                    relationships.append(
                        TrustRelationship(
                            from_agent=from_agent,
                            to_agent=to_agent,
                            weight=weight,
                            updated_at=datetime.now()  # We don't track update time
                        )
                    )
            return relationships

    def start(self) -> None:
        """
        Start listening for belief update events.

        Subscribes to 'belief.updated' events from EventBus and
        begins propagating updates to subscribed agents.
        """
        if self._listening:
            logger.warning("BeliefPropagator already listening")
            return

        self.event_bus.subscribe('belief.updated', self._handle_belief_update)
        self._listening = True
        logger.info("BeliefPropagator started listening for belief updates")

    def stop(self) -> None:
        """
        Stop listening for belief update events.

        Unsubscribes from EventBus.
        """
        if not self._listening:
            return

        self.event_bus.unsubscribe('belief.updated', self._handle_belief_update)
        self._listening = False
        logger.info("BeliefPropagator stopped listening")

    def _handle_belief_update(self, event: Any) -> None:
        """
        Handle belief.updated event from EventBus.

        Args:
            event: BeliefUpdatedEvent with belief_id, old/new confidence, etc.
        """
        try:
            # Extract event data
            belief_id = event.belief_id
            old_confidence = event.old_confidence
            new_confidence = event.new_confidence

            # Get source agent from metadata
            metadata = getattr(event, 'metadata', {}) or {}
            source_agent = metadata.get('agent_id')
            namespace = metadata.get('namespace')

            if not source_agent:
                logger.debug(
                    f"Belief update for {belief_id} has no source agent, skipping propagation"
                )
                return

            # Calculate evidence strength from confidence change
            confidence_delta = new_confidence - old_confidence
            supports = confidence_delta > 0
            strength = abs(confidence_delta)

            # Find subscribed agents
            subscribed_agents = self._find_subscribed_agents(
                belief_id=belief_id,
                namespace=namespace,
                source_agent=source_agent
            )

            if not subscribed_agents:
                logger.debug(
                    f"No subscribed agents for belief {belief_id}, skipping propagation"
                )
                return

            # Propagate to each subscribed agent
            results = []
            for target_agent in subscribed_agents:
                result = self._propagate_to_agent(
                    belief_id=belief_id,
                    source_agent=source_agent,
                    target_agent=target_agent,
                    supports=supports,
                    strength=strength,
                    namespace=namespace
                )
                if result:
                    results.append(result)

            logger.info(
                f"Propagated belief {belief_id} update from {source_agent} "
                f"to {len(results)} agents"
            )

            # Emit propagation events
            self._emit_propagation_events(results)

        except Exception as e:
            logger.error(f"Error handling belief update: {e}", exc_info=True)

    def _find_subscribed_agents(
        self,
        belief_id: str,
        namespace: Optional[str],
        source_agent: str
    ) -> List[str]:
        """
        Find agents subscribed to this belief or namespace.

        Args:
            belief_id: Belief ID
            namespace: Optional namespace
            source_agent: Source agent (excluded from results)

        Returns:
            List of subscribed agent IDs (excluding source_agent)
        """
        try:
            # Find subscriptions matching this belief or namespace
            subscriptions = self.subscription_manager.match_subscriptions(
                event_type='belief.updated',
                namespace=namespace,
                memory_id=belief_id
            )

            # Extract unique agent IDs, excluding source
            agent_ids = set()
            for sub in subscriptions:
                agent_id = sub.agent_id
                if agent_id != source_agent:
                    agent_ids.add(agent_id)

            return list(agent_ids)

        except Exception as e:
            logger.error(f"Error finding subscribed agents: {e}", exc_info=True)
            return []

    def _propagate_to_agent(
        self,
        belief_id: str,
        source_agent: str,
        target_agent: str,
        supports: bool,
        strength: float,
        namespace: Optional[str]
    ) -> Optional[PropagationResult]:
        """
        Propagate belief update to a single target agent.

        Args:
            belief_id: Belief ID
            source_agent: Source agent ID
            target_agent: Target agent ID
            supports: Whether this is supporting evidence
            strength: Evidence strength (0.0 to 1.0)
            namespace: Optional namespace

        Returns:
            PropagationResult if successful, None otherwise
        """
        try:
            # Get trust weight
            trust_weight = self.get_trust_weight(source_agent, target_agent)

            # If no trust, skip propagation
            if trust_weight <= 0.0:
                logger.debug(
                    f"Zero trust from {source_agent} to {target_agent}, "
                    f"skipping propagation"
                )
                return None

            # Get current belief state for target agent
            # Note: In a multi-agent system, each agent would have their own
            # belief instance or namespace-scoped belief
            current_belief = self._get_agent_belief(target_agent, belief_id)
            if not current_belief:
                logger.debug(
                    f"Target agent {target_agent} doesn't have belief {belief_id}, "
                    f"skipping propagation"
                )
                return None

            old_confidence = current_belief.get('confidence', 0.5)

            # Apply trust-modulated EMA update
            new_confidence = self._apply_trust_modulated_update(
                old_confidence=old_confidence,
                supports=supports,
                strength=strength,
                trust_weight=trust_weight
            )

            # Update belief confidence
            self._update_agent_belief(
                target_agent=target_agent,
                belief_id=belief_id,
                new_confidence=new_confidence,
                source_agent=source_agent
            )

            result = PropagationResult(
                source_agent=source_agent,
                target_agent=target_agent,
                belief_id=belief_id,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                trust_weight=trust_weight,
                propagated_at=datetime.now()
            )

            logger.debug(
                f"Propagated to {target_agent}: {belief_id} "
                f"{old_confidence:.3f} -> {new_confidence:.3f} "
                f"(trust={trust_weight:.2f})"
            )

            return result

        except Exception as e:
            logger.error(
                f"Error propagating to {target_agent}: {e}",
                exc_info=True
            )
            return None

    def _apply_trust_modulated_update(
        self,
        old_confidence: float,
        supports: bool,
        strength: float,
        trust_weight: float
    ) -> float:
        """
        Apply trust-modulated EMA update to confidence.

        Uses same EMA logic as BeliefNetwork, but modulates the
        evidence strength by trust weight.

        Args:
            old_confidence: Current confidence value
            supports: Whether this is supporting evidence
            strength: Raw evidence strength
            trust_weight: Trust weight (0.0 to 1.0)

        Returns:
            New confidence value
        """
        # Modulate strength by trust weight
        modulated_strength = strength * trust_weight

        # Select lambda based on support/contradict
        lambda_val = self.SUPPORT_LAMBDA if supports else self.CONTRADICT_LAMBDA

        # Calculate target confidence
        if supports:
            target = min(1.0, old_confidence + modulated_strength)
        else:
            target = max(0.0, old_confidence - modulated_strength)

        # Apply EMA update
        new_confidence = old_confidence + lambda_val * (target - old_confidence)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, new_confidence))

    def _get_agent_belief(
        self,
        agent_id: str,
        belief_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get belief state for a specific agent.

        In a real multi-agent system, this would query the agent's
        own BeliefNetwork or namespace-scoped beliefs.

        For now, we use the shared BeliefNetwork.

        Args:
            agent_id: Agent ID
            belief_id: Belief ID

        Returns:
            Belief dict if found, None otherwise
        """
        try:
            belief = self.belief_network.palace.get_belief(belief_id)
            return belief
        except Exception as e:
            logger.debug(f"Error getting belief {belief_id}: {e}")
            return None

    def _update_agent_belief(
        self,
        target_agent: str,
        belief_id: str,
        new_confidence: float,
        source_agent: str
    ) -> None:
        """
        Update belief confidence for target agent.

        Args:
            target_agent: Target agent ID
            belief_id: Belief ID
            new_confidence: New confidence value
            source_agent: Source agent ID (for metadata)
        """
        try:
            self.belief_network.palace.update_belief_confidence(
                belief_id,
                new_confidence
            )
        except Exception as e:
            logger.error(
                f"Error updating belief {belief_id} for {target_agent}: {e}"
            )

    def _emit_propagation_events(
        self,
        results: List[PropagationResult]
    ) -> None:
        """
        Emit propagation events to EventBus.

        This allows other components to react to belief propagation.
        Note: The actual event types will be defined in events.py

        Args:
            results: List of PropagationResult objects
        """
        # Import here to avoid circular dependency
        # The actual BeliefPropagatedEvent will be defined in events.py
        # For now, we log the results
        for result in results:
            logger.debug(
                f"Propagation event: {result.source_agent} -> {result.target_agent} "
                f"for belief {result.belief_id}"
            )
