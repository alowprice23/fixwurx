"""
agents/agent_handoff.py
───────────────────────
Secure, structured protocol for coordinated **agent-to-agent communication**.

The handoff protocol ensures:
1. Reliable transfer of context between agents
2. Cryptographic verification of message integrity
3. Structured format for key information exchange
4. Metrics tracking for handoff success rates
5. Support for both synchronous and asynchronous handoffs

This system acts as the nervous system connecting the Planner Agent (root)
with its specialized children (Observer, Analyst, Verifier) and enables
the implementation of complex multi-agent workflows.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar, Generic, Union

# Import Compressor for large context windows
from compress import Compressor

# Type variables for generics
T = TypeVar('T')
HandoffCallback = Callable[[str, 'HandoffPayload'], None]


class HandoffStatus(Enum):
    """Status of a handoff operation."""
    PENDING = "pending"       # Handoff created but not yet sent
    SENT = "sent"             # Handoff sent but not yet acknowledged
    RECEIVED = "received"     # Handoff received but not yet processed
    ACCEPTED = "accepted"     # Handoff accepted and being processed
    COMPLETED = "completed"   # Handoff processing completed successfully
    REJECTED = "rejected"     # Handoff rejected by recipient
    FAILED = "failed"         # Handoff processing failed
    TIMEOUT = "timeout"       # Handoff timed out waiting for response


class HandoffType(Enum):
    """Type of handoff operation."""
    PLAN_TO_OBSERVE = "plan_to_observe"    # Planner → Observer
    OBSERVE_TO_PLAN = "observe_to_plan"    # Observer → Planner
    OBSERVE_TO_ANALYST = "observe_to_analyst"  # Observer → Analyst
    ANALYST_TO_PLAN = "analyst_to_plan"    # Analyst → Planner
    ANALYST_TO_VERIFY = "analyst_to_verify"  # Analyst → Verifier
    VERIFY_TO_PLAN = "verify_to_plan"      # Verifier → Planner
    VERIFY_TO_ANALYST = "verify_to_analyst"  # Verifier → Analyst
    PLAN_TO_ANALYST = "plan_to_analyst"    # Planner → Analyst
    PLAN_TO_VERIFY = "plan_to_verify"      # Planner → Verifier


@dataclass
class HandoffPayload:
    """
    Core data structure for agent handoffs.
    
    Contains the payload data being transferred between agents along with
    metadata for tracking and verification.
    """
    # Unique identifier for this handoff
    handoff_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Agent identifiers
    sender_id: str = ""
    recipient_id: str = ""
    
    # Type of handoff
    handoff_type: HandoffType = HandoffType.PLAN_TO_OBSERVE
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    received_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Current status
    status: HandoffStatus = HandoffStatus.PENDING
    
    # Payload data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Compressed large context (if needed)
    compressed_context: Optional[str] = None
    compression_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity verification
    checksum: Optional[str] = None
    
    # Error information (if any)
    error: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate a checksum for the payload data for integrity verification."""
        # Create a deterministic representation of the data
        serialized = json.dumps(self.data, sort_keys=True)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def update_checksum(self) -> None:
        """Update the checksum based on the current data."""
        self.checksum = self.calculate_checksum()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the payload data using the checksum."""
        if not self.checksum:
            return False
            
        current = self.calculate_checksum()
        return current == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        result = asdict(self)
        
        # Convert enum values to strings
        result["handoff_type"] = self.handoff_type.value
        result["status"] = self.status.value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HandoffPayload:
        """Create a HandoffPayload from a dictionary."""
        # Make a copy to avoid modifying the input
        data_copy = data.copy()
        
        # Convert string values back to enums
        data_copy["handoff_type"] = HandoffType(data_copy["handoff_type"])
        data_copy["status"] = HandoffStatus(data_copy["status"])
        
        return cls(**data_copy)
    
    def compress_context(self, context: str, max_tokens: int = 4096) -> None:
        """
        Compress a large context string and store it in the payload.
        
        Args:
            context: The large context string to compress
            max_tokens: Maximum tokens for compression
        """
        compressor = Compressor(max_tokens)
        
        # Compress the context
        compressed, bits_saved = compressor.compress(context)
        
        # Store the compressed context
        self.compressed_context = compressed
        
        # Store compression stats
        original_size = len(context)
        compressed_size = len(compressed)
        self.compression_stats = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compressed_size / original_size if original_size > 0 else 1.0,
            "bits_saved": bits_saved
        }
    
    def decompress_context(self) -> Optional[str]:
        """
        Decompress the stored context.
        
        Returns:
            The decompressed context string or None if no compressed context exists
        """
        if not self.compressed_context:
            return None
            
        # For this implementation, we directly return the compressed context
        # since we're using the system's compression module elsewhere
        return self.compressed_context


class HandoffRegistry:
    """
    Central registry for tracking and coordinating all agent handoffs.
    
    Features:
    - Tracks all handoffs in the system
    - Provides status updates and metrics
    - Handles routing of handoffs to appropriate agents
    - Maintains history for debugging and auditing
    """
    
    def __init__(self, metric_bus: Optional[Any] = None):
        """
        Initialize the HandoffRegistry.
        
        Args:
            metric_bus: Optional metric bus for sending metrics
        """
        self.handoffs: Dict[str, HandoffPayload] = {}
        self.history: List[Dict[str, Any]] = []
        self.callbacks: Dict[str, List[HandoffCallback]] = {}
        self.metric_bus = metric_bus
        self.max_history_size = 1000
    
    def register_handoff(self, payload: HandoffPayload) -> str:
        """
        Register a new handoff in the registry.
        
        Args:
            payload: The handoff payload
            
        Returns:
            The handoff ID
        """
        # Ensure the payload has a valid checksum
        if not payload.checksum:
            payload.update_checksum()
            
        # Store the handoff
        self.handoffs[payload.handoff_id] = payload
        
        # Emit metrics if a metric bus is available
        if self.metric_bus:
            self.metric_bus.send(
                "triangulum.handoff_registered",
                1.0,
                tags={
                    "sender": payload.sender_id,
                    "recipient": payload.recipient_id,
                    "type": payload.handoff_type.value
                }
            )
        
        return payload.handoff_id
    
    def get_handoff(self, handoff_id: str) -> Optional[HandoffPayload]:
        """
        Get a handoff by ID.
        
        Args:
            handoff_id: The handoff ID
            
        Returns:
            The handoff payload or None if not found
        """
        return self.handoffs.get(handoff_id)
    
    def update_status(self, handoff_id: str, status: HandoffStatus, 
                     error: Optional[str] = None) -> bool:
        """
        Update the status of a handoff.
        
        Args:
            handoff_id: The handoff ID
            status: The new status
            error: Optional error message
            
        Returns:
            True if the update was successful, False otherwise
        """
        handoff = self.handoffs.get(handoff_id)
        if not handoff:
            return False
            
        # Update the handoff status
        handoff.status = status
        
        # Update timestamps based on status
        if status == HandoffStatus.SENT:
            handoff.sent_at = time.time()
        elif status == HandoffStatus.RECEIVED:
            handoff.received_at = time.time()
        elif status in (HandoffStatus.COMPLETED, HandoffStatus.FAILED):
            handoff.completed_at = time.time()
            
            # Archive completed or failed handoffs
            self._archive_handoff(handoff_id)
            
        # Set error message if provided
        if error:
            handoff.error = error
            
        # Emit metrics if a metric bus is available
        if self.metric_bus:
            self.metric_bus.send(
                "triangulum.handoff_status_update",
                1.0,
                tags={
                    "handoff_id": handoff_id,
                    "status": status.value,
                    "sender": handoff.sender_id,
                    "recipient": handoff.recipient_id,
                    "type": handoff.handoff_type.value
                }
            )
            
        # Trigger callbacks
        self._trigger_callbacks(handoff_id, handoff)
            
        return True
    
    def register_callback(self, handoff_id: str, 
                         callback: HandoffCallback) -> None:
        """
        Register a callback for a specific handoff.
        
        Args:
            handoff_id: The handoff ID
            callback: The callback function
        """
        if handoff_id not in self.callbacks:
            self.callbacks[handoff_id] = []
            
        self.callbacks[handoff_id].append(callback)
    
    def _trigger_callbacks(self, handoff_id: str, 
                          payload: HandoffPayload) -> None:
        """
        Trigger all callbacks registered for a handoff.
        
        Args:
            handoff_id: The handoff ID
            payload: The handoff payload
        """
        if handoff_id not in self.callbacks:
            return
            
        for callback in self.callbacks[handoff_id]:
            try:
                callback(handoff_id, payload)
            except Exception as e:
                print(f"Error in handoff callback: {e}")
    
    def _archive_handoff(self, handoff_id: str) -> None:
        """
        Archive a handoff to the history.
        
        Args:
            handoff_id: The handoff ID
        """
        handoff = self.handoffs.get(handoff_id)
        if not handoff:
            return
            
        # Add to history
        self.history.append(handoff.to_dict())
        
        # Trim history if it exceeds the maximum size
        while len(self.history) > self.max_history_size:
            self.history.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the handoffs.
        
        Returns:
            Dictionary of metrics
        """
        # Count handoffs by status
        status_counts = {}
        for status in HandoffStatus:
            status_counts[status.value] = 0
            
        for handoff in self.handoffs.values():
            status_counts[handoff.status.value] += 1
            
        # Count handoffs by type
        type_counts = {}
        for handoff_type in HandoffType:
            type_counts[handoff_type.value] = 0
            
        for handoff in self.handoffs.values():
            type_counts[handoff.handoff_type.value] += 1
            
        return {
            "active_handoffs": len(self.handoffs),
            "history_size": len(self.history),
            "status_counts": status_counts,
            "type_counts": type_counts
        }
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the handoff history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of handoff history entries
        """
        return self.history[-limit:] if limit > 0 else self.history.copy()
    
    def clear(self) -> None:
        """Clear all handoffs and history."""
        self.handoffs.clear()
        self.history.clear()
        self.callbacks.clear()


# Simplified artifacts structure for component interaction
@dataclass
class _Artefacts:
    """Container for artifacts passed between agents"""
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None

# Global instance
handoff_registry = HandoffRegistry()


class HandoffProtocol:
    """
    Implementation of the agent handoff protocol.
    
    Provides methods for sending and receiving handoffs between agents.
    """
    
    def __init__(self, agent_id: str, registry: HandoffRegistry = handoff_registry):
        """
        Initialize the HandoffProtocol.
        
        Args:
            agent_id: The ID of the agent using this protocol
            registry: The handoff registry to use
        """
        self.agent_id = agent_id
        self.registry = registry
        
    def create_handoff(self, recipient_id: str, handoff_type: HandoffType, 
                      data: Dict[str, Any], context: Optional[str] = None) -> str:
        """
        Create a new handoff.
        
        Args:
            recipient_id: The ID of the recipient agent
            handoff_type: The type of handoff
            data: The payload data
            context: Optional large context string to compress
            
        Returns:
            The handoff ID
        """
        # Create the payload
        payload = HandoffPayload(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            handoff_type=handoff_type,
            data=data
        )
        
        # Compress context if provided
        if context:
            payload.compress_context(context)
            
        # Update checksum
        payload.update_checksum()
        
        # Register the handoff
        return self.registry.register_handoff(payload)
    
    def send_handoff(self, handoff_id: str) -> bool:
        """
        Send a handoff to its recipient.
        
        Args:
            handoff_id: The handoff ID
            
        Returns:
            True if the handoff was sent successfully, False otherwise
        """
        # Get the handoff
        handoff = self.registry.get_handoff(handoff_id)
        if not handoff:
            return False
            
        # Verify that this agent is the sender
        if handoff.sender_id != self.agent_id:
            return False
            
        # Update the handoff status
        return self.registry.update_status(handoff_id, HandoffStatus.SENT)
    
    def receive_handoff(self, handoff_id: str) -> Optional[HandoffPayload]:
        """
        Receive a handoff.
        
        Args:
            handoff_id: The handoff ID
            
        Returns:
            The handoff payload or None if not found or this agent is not the recipient
        """
        # Get the handoff
        handoff = self.registry.get_handoff(handoff_id)
        if not handoff:
            return None
            
        # Verify that this agent is the recipient
        if handoff.recipient_id != self.agent_id:
            return None
            
        # Verify the handoff integrity
        if not handoff.verify_integrity():
            self.registry.update_status(
                handoff_id, 
                HandoffStatus.FAILED, 
                "Integrity check failed"
            )
            return None
            
        # Update the handoff status
        self.registry.update_status(handoff_id, HandoffStatus.RECEIVED)
        
        return handoff
    
    def accept_handoff(self, handoff_id: str) -> bool:
        """
        Accept a handoff for processing.
        
        Args:
            handoff_id: The handoff ID
            
        Returns:
            True if the handoff was accepted successfully, False otherwise
        """
        # Get the handoff
        handoff = self.registry.get_handoff(handoff_id)
        if not handoff:
            return False
            
        # Verify that this agent is the recipient
        if handoff.recipient_id != self.agent_id:
            return False
            
        # Update the handoff status
        return self.registry.update_status(handoff_id, HandoffStatus.ACCEPTED)
    
    def complete_handoff(self, handoff_id: str) -> bool:
        """
        Mark a handoff as completed.
        
        Args:
            handoff_id: The handoff ID
            
        Returns:
            True if the handoff was completed successfully, False otherwise
        """
        # Get the handoff
        handoff = self.registry.get_handoff(handoff_id)
        if not handoff:
            return False
            
        # Verify that this agent is the recipient
        if handoff.recipient_id != self.agent_id:
            return False
            
        # Update the handoff status
        return self.registry.update_status(handoff_id, HandoffStatus.COMPLETED)
    
    def reject_handoff(self, handoff_id: str, reason: str) -> bool:
        """
        Reject a handoff.
        
        Args:
            handoff_id: The handoff ID
            reason: The reason for rejection
            
        Returns:
            True if the handoff was rejected successfully, False otherwise
        """
        # Get the handoff
        handoff = self.registry.get_handoff(handoff_id)
        if not handoff:
            return False
            
        # Verify that this agent is the recipient
        if handoff.recipient_id != self.agent_id:
            return False
            
        # Update the handoff status
        return self.registry.update_status(handoff_id, HandoffStatus.REJECTED, reason)
    
    def fail_handoff(self, handoff_id: str, error: str) -> bool:
        """
        Mark a handoff as failed.
        
        Args:
            handoff_id: The handoff ID
            error: The error message
            
        Returns:
            True if the handoff was marked as failed successfully, False otherwise
        """
        # Get the handoff
        handoff = self.registry.get_handoff(handoff_id)
        if not handoff:
            return False
            
        # Verify that this agent is the recipient
        if handoff.recipient_id != self.agent_id:
            return False
            
        # Update the handoff status
        return self.registry.update_status(handoff_id, HandoffStatus.FAILED, error)
    
    def wait_for_handoff(self, handoff_id: str, 
                        timeout: float = 60.0) -> Optional[HandoffPayload]:
        """
        Wait for a handoff to be completed.
        
        Args:
            handoff_id: The handoff ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            The handoff payload or None if the handoff timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            handoff = self.registry.get_handoff(handoff_id)
            if not handoff:
                return None
                
            if handoff.status in (HandoffStatus.COMPLETED, HandoffStatus.REJECTED, HandoffStatus.FAILED):
                return handoff
                
            time.sleep(0.1)
            
        # Timeout
        self.registry.update_status(handoff_id, HandoffStatus.TIMEOUT, "Timeout waiting for completion")
        return None
    
    async def wait_for_handoff_async(self, handoff_id: str, 
                                   timeout: float = 60.0) -> Optional[HandoffPayload]:
        """
        Wait for a handoff to be completed asynchronously.
        
        Args:
            handoff_id: The handoff ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            The handoff payload or None if the handoff timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            handoff = self.registry.get_handoff(handoff_id)
            if not handoff:
                return None
                
            if handoff.status in (HandoffStatus.COMPLETED, HandoffStatus.REJECTED, HandoffStatus.FAILED):
                return handoff
                
            await asyncio.sleep(0.1)
            
        # Timeout
        self.registry.update_status(handoff_id, HandoffStatus.TIMEOUT, "Timeout waiting for completion")
        return None
    
    def list_pending_handoffs(self) -> List[HandoffPayload]:
        """
        List all pending handoffs for this agent.
        
        Returns:
            List of pending handoff payloads
        """
        return [
            handoff for handoff in self.registry.handoffs.values()
            if handoff.recipient_id == self.agent_id
            and handoff.status in (HandoffStatus.SENT, HandoffStatus.RECEIVED)
        ]
    
    def list_sent_handoffs(self) -> List[HandoffPayload]:
        """
        List all handoffs sent by this agent.
        
        Returns:
            List of sent handoff payloads
        """
        return [
            handoff for handoff in self.registry.handoffs.values()
            if handoff.sender_id == self.agent_id
        ]


# ---------------------------------------------------------------------------—
# Quick demo
# ---------------------------------------------------------------------------—
class AgentHandoff:
    """
    High-level agent handoff coordinator.

    Provides a simplified interface for handling agent-to-agent communications,
    managing the handoff protocol details and providing callbacks for async operations.
    """
    
    def __init__(self, agent_id: str, registry: Optional[HandoffRegistry] = None):
        """
        Initialize the AgentHandoff.
        
        Args:
            agent_id: ID of the agent using this handoff coordinator
            registry: Optional registry to use (uses global registry by default)
        """
        self.agent_id = agent_id
        self.registry = registry or handoff_registry
        self.protocol = HandoffProtocol(agent_id, self.registry)
        self.callbacks: Dict[str, Callable] = {}
    
    def send_to(self, recipient_id: str, handoff_type: HandoffType, 
                data: Dict[str, Any], context: Optional[str] = None) -> str:
        """
        Send data to another agent.
        
        Args:
            recipient_id: ID of the recipient agent
            handoff_type: Type of handoff
            data: Data to send
            context: Optional large context
            
        Returns:
            Handoff ID
        """
        # Create and send handoff
        handoff_id = self.protocol.create_handoff(recipient_id, handoff_type, data, context)
        self.protocol.send_handoff(handoff_id)
        return handoff_id
    
    def receive(self) -> List[HandoffPayload]:
        """
        Receive all pending handoffs.
        
        Returns:
            List of received handoff payloads
        """
        # Get all pending handoffs
        pending = self.protocol.list_pending_handoffs()
        
        # Receive each handoff
        received = []
        for handoff in pending:
            result = self.protocol.receive_handoff(handoff.handoff_id)
            if result:
                received.append(result)
                
        return received
    
    def get_pending_count(self) -> int:
        """
        Get count of pending handoffs.
        
        Returns:
            Number of pending handoffs
        """
        return len(self.protocol.list_pending_handoffs())
    
    def accept(self, handoff_id: str) -> bool:
        """
        Accept a handoff for processing.
        
        Args:
            handoff_id: ID of the handoff to accept
            
        Returns:
            True if accepted, False otherwise
        """
        return self.protocol.accept_handoff(handoff_id)
    
    def complete(self, handoff_id: str) -> bool:
        """
        Mark a handoff as completed.
        
        Args:
            handoff_id: ID of the handoff to complete
            
        Returns:
            True if completed, False otherwise
        """
        return self.protocol.complete_handoff(handoff_id)
    
    def reject(self, handoff_id: str, reason: str) -> bool:
        """
        Reject a handoff.
        
        Args:
            handoff_id: ID of the handoff to reject
            reason: Reason for rejection
            
        Returns:
            True if rejected, False otherwise
        """
        return self.protocol.reject_handoff(handoff_id, reason)
    
    def fail(self, handoff_id: str, error: str) -> bool:
        """
        Mark a handoff as failed.
        
        Args:
            handoff_id: ID of the handoff to fail
            error: Error message
            
        Returns:
            True if marked as failed, False otherwise
        """
        return self.protocol.fail_handoff(handoff_id, error)
    
    def wait_for(self, handoff_id: str, timeout: float = 60.0) -> Optional[HandoffPayload]:
        """
        Wait for a handoff to complete.
        
        Args:
            handoff_id: ID of the handoff to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Handoff payload or None if timed out
        """
        return self.protocol.wait_for_handoff(handoff_id, timeout)
    
    async def wait_for_async(self, handoff_id: str, timeout: float = 60.0) -> Optional[HandoffPayload]:
        """
        Wait for a handoff to complete asynchronously.
        
        Args:
            handoff_id: ID of the handoff to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Handoff payload or None if timed out
        """
        return await self.protocol.wait_for_handoff_async(handoff_id, timeout)
    
    def register_callback(self, handoff_id: str, callback: Callable) -> None:
        """
        Register a callback for a handoff.
        
        Args:
            handoff_id: ID of the handoff
            callback: Callback function
        """
        self.callbacks[handoff_id] = callback
        
        # Register with registry too
        def wrapper(hid: str, payload: HandoffPayload) -> None:
            try:
                self.callbacks[hid](payload)
            except Exception as e:
                print(f"Error in handoff callback: {e}")
                
        self.registry.register_callback(handoff_id, wrapper)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get handoff metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self.registry.get_metrics()
        
        # Add agent-specific metrics
        sent = len(self.protocol.list_sent_handoffs())
        pending = len(self.protocol.list_pending_handoffs())
        
        metrics["agent"] = {
            "id": self.agent_id,
            "sent_handoffs": sent,
            "pending_handoffs": pending
        }
        
        return metrics
    
    def extract_data(self, handoff: HandoffPayload) -> _Artefacts:
        """
        Extract data from a handoff payload into a simplified structure.
        
        Args:
            handoff: Handoff payload
            
        Returns:
            Simplified artifacts structure
        """
        context = None
        if handoff.compressed_context:
            context = handoff.decompress_context()
            
        return _Artefacts(
            data=handoff.data,
            metadata={
                "handoff_id": handoff.handoff_id,
                "sender_id": handoff.sender_id,
                "handoff_type": handoff.handoff_type.value,
                "created_at": handoff.created_at,
                "status": handoff.status.value
            },
            context=context
        )

if __name__ == "__main__":  # pragma: no cover
    # Create a HandoffProtocol for each agent
    planner_protocol = HandoffProtocol("planner")
    observer_protocol = HandoffProtocol("observer")
    analyst_protocol = HandoffProtocol("analyst")
    
    # Create a handoff from the planner to the observer
    handoff_id = planner_protocol.create_handoff(
        "observer",
        HandoffType.PLAN_TO_OBSERVE,
        {
            "bug_id": "BUG-42",
            "description": "Fix import path typo",
            "priority": "high"
        },
        context="This is a large context that would be compressed in a real scenario."
    )
    
    print(f"Created handoff {handoff_id} from planner to observer")
    
    # Send the handoff
    planner_protocol.send_handoff(handoff_id)
    print("Handoff sent")
    
    # Observer receives the handoff
    handoff = observer_protocol.receive_handoff(handoff_id)
    if handoff:
        print(f"Observer received handoff: {handoff.data}")
        
        # Observer accepts and processes the handoff
        observer_protocol.accept_handoff(handoff_id)
        print("Observer accepted handoff")
        
        # Observer completes the handoff
        observer_protocol.complete_handoff(handoff_id)
        print("Observer completed handoff")
    
    # Get metrics
    metrics = handoff_registry.get_metrics()
    print("\nHandoff metrics:")
    print(f"  Active handoffs: {metrics['active_handoffs']}")
    print(f"  History size: {metrics['history_size']}")
    print(f"  Status counts: {metrics['status_counts']}")
