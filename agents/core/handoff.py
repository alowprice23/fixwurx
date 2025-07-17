"""
agents/core/handoff.py
────────────────────
Agent handoff protocol implementation.

This module provides the AgentHandoff class, which is responsible for
managing handoffs between agents in the FixWurx system.
"""

import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("agent_handoff")

class HandoffInstance:
    """
    Agent handoff instance implementation.
    
    The HandoffInstance class represents a single handoff between two agents,
    tracking the status and result of the handoff.
    """
    
    def __init__(self, from_agent: str, to_agent: str, context: Dict[str, Any] = None):
        """
        Initialize a handoff instance.
        
        Args:
            from_agent: ID of the source agent
            to_agent: ID of the target agent
            context: Optional context for the handoff
        """
        self.handoff_id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.context = context or {}
        self.created_at = time.time()
        self.completed_at = None
        self.status = "pending"
        self.result = None
        self.metadata = {}
        
        self.logger = logging.getLogger(f"handoff.{self.handoff_id}")
        self.logger.info(f"Created handoff from {from_agent} to {to_agent}")
    
    def complete(self, status: str, result: Any = None) -> None:
        """
        Complete the handoff.
        
        Args:
            status: Status of the handoff (success, failure, etc.)
            result: Result of the handoff
        """
        self.status = status
        self.result = result
        self.completed_at = time.time()
        self.logger.info(f"Completed handoff with status {status}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "handoff_id": self.handoff_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "context": self.context,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "result": self.result,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffInstance":
        """Create from dictionary representation."""
        handoff = cls(
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            context=data.get("context", {})
        )
        handoff.handoff_id = data["handoff_id"]
        handoff.created_at = data["created_at"]
        handoff.completed_at = data.get("completed_at")
        handoff.status = data["status"]
        handoff.result = data.get("result")
        handoff.metadata = data.get("metadata", {})
        return handoff
    
    @classmethod
    def from_json(cls, json_str: str) -> "HandoffInstance":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class AgentHandoff:
    """
    Agent handoff protocol implementation.
    
    The AgentHandoff class manages handoffs between agents, ensuring that
    information is properly transferred and actions are coordinated.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the agent handoff system.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.handoffs = {}
        self.logger = logging.getLogger("agent_handoff")
        self.logger.info("Agent handoff system initialized")
    
    def create_handoff(self, from_agent: str, to_agent: str, context: Dict[str, Any] = None) -> HandoffInstance:
        """
        Create a handoff between two agents.
        
        Args:
            from_agent: ID of the source agent
            to_agent: ID of the target agent
            context: Optional context for the handoff
            
        Returns:
            The created handoff
        """
        handoff = HandoffInstance(from_agent, to_agent, context)
        self.handoffs[handoff.handoff_id] = handoff
        self.logger.info(f"Created handoff {handoff.handoff_id} from {from_agent} to {to_agent}")
        return handoff
    
    def get_handoff(self, handoff_id: str) -> Optional[HandoffInstance]:
        """
        Get a handoff by ID.
        
        Args:
            handoff_id: ID of the handoff to get
            
        Returns:
            The handoff, or None if not found
        """
        return self.handoffs.get(handoff_id)
    
    def get_handoffs_from(self, agent_id: str) -> List[HandoffInstance]:
        """
        Get handoffs from an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of handoffs from the agent
        """
        return [h for h in self.handoffs.values() if h.from_agent == agent_id]
    
    def get_handoffs_to(self, agent_id: str) -> List[HandoffInstance]:
        """
        Get handoffs to an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of handoffs to the agent
        """
        return [h for h in self.handoffs.values() if h.to_agent == agent_id]
    
    def get_pending_handoffs(self) -> List[HandoffInstance]:
        """
        Get pending handoffs.
        
        Returns:
            List of pending handoffs
        """
        return [h for h in self.handoffs.values() if h.status == "pending"]
    
    def complete_handoff(self, handoff_id: str, status: str, result: Any = None) -> Optional[HandoffInstance]:
        """
        Complete a handoff.
        
        Args:
            handoff_id: ID of the handoff to complete
            status: Status of the handoff (success, failure, etc.)
            result: Result of the handoff
            
        Returns:
            The completed handoff, or None if not found
        """
        handoff = self.get_handoff(handoff_id)
        if handoff:
            handoff.complete(status, result)
            self.logger.info(f"Completed handoff {handoff_id} with status {status}")
        return handoff


class HandoffProtocol:
    """
    Handoff protocol implementation.
    
    The HandoffProtocol class provides methods for creating and tracking
    handoffs between agents in the system.
    """
    
    def __init__(self):
        """Initialize the handoff protocol."""
        self.handoffs = {}
        self.logger = logging.getLogger("handoff_protocol")
    
    def create_handoff(self, from_agent: str, to_agent: str, context: Dict[str, Any] = None) -> HandoffInstance:
        """
        Create a handoff between two agents.
        
        Args:
            from_agent: ID of the source agent
            to_agent: ID of the target agent
            context: Optional context for the handoff
            
        Returns:
            The created handoff
        """
        handoff = HandoffInstance(from_agent, to_agent, context)
        self.handoffs[handoff.handoff_id] = handoff
        self.logger.info(f"Created handoff {handoff.handoff_id} from {from_agent} to {to_agent}")
        return handoff
    
    def get_handoff(self, handoff_id: str) -> Optional[HandoffInstance]:
        """
        Get a handoff by ID.
        
        Args:
            handoff_id: ID of the handoff to get
            
        Returns:
            The handoff, or None if not found
        """
        return self.handoffs.get(handoff_id)
    
    def get_handoffs_from(self, agent_id: str) -> List[HandoffInstance]:
        """
        Get handoffs from an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of handoffs from the agent
        """
        return [h for h in self.handoffs.values() if h.from_agent == agent_id]
    
    def get_handoffs_to(self, agent_id: str) -> List[HandoffInstance]:
        """
        Get handoffs to an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of handoffs to the agent
        """
        return [h for h in self.handoffs.values() if h.to_agent == agent_id]
    
    def get_pending_handoffs(self) -> List[HandoffInstance]:
        """
        Get pending handoffs.
        
        Returns:
            List of pending handoffs
        """
        return [h for h in self.handoffs.values() if h.status == "pending"]
    
    def complete_handoff(self, handoff_id: str, status: str, result: Any = None) -> Optional[HandoffInstance]:
        """
        Complete a handoff.
        
        Args:
            handoff_id: ID of the handoff to complete
            status: Status of the handoff (success, failure, etc.)
            result: Result of the handoff
            
        Returns:
            The completed handoff, or None if not found
        """
        handoff = self.get_handoff(handoff_id)
        if handoff:
            handoff.complete(status, result)
            self.logger.info(f"Completed handoff {handoff_id} with status {status}")
        return handoff


# Create a singleton instance of the handoff protocol
_protocol_instance = None

def get_protocol() -> HandoffProtocol:
    """
    Get the singleton instance of the handoff protocol.
    
    Returns:
        The handoff protocol instance
    """
    global _protocol_instance
    if _protocol_instance is None:
        _protocol_instance = HandoffProtocol()
    return _protocol_instance
