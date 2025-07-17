#!/usr/bin/env python3
"""
Agent System Core Module

This module serves as the core of the agent system, providing a unified API
for interacting with all agents within the FixWurx system.
"""

import logging
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union

# Import core agents
from agents.core.meta_agent import MetaAgent
from agents.core.planner_agent import PlannerAgent

# Import specialized agents
from agents.specialized.specialized_agents import ObserverAgent, AnalystAgent, VerifierAgent

# Import utility modules
from agents.utils.memory import AgentMemory

logger = logging.getLogger("AgentSystem")

class AgentRegistry:
    """Registry for managing agent instances."""
    
    def __init__(self):
        """Initialize the agent registry."""
        self.agents = {}
        self.logger = logging.getLogger("AgentRegistry")
    
    def register_agent(self, name: str, agent: Any):
        """Register an agent."""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    def get_agent(self, name: str) -> Any:
        """Get an agent by name."""
        if name in self.agents:
            return self.agents[name]
        self.logger.warning(f"Agent {name} not found")
        return None
    
    def list_agents(self) -> List[str]:
        """List all registered agents."""
        return list(self.agents.keys())
    
    def broadcast(self, message: Dict[str, Any], exclude: List[str] = None):
        """Broadcast a message to all agents."""
        exclude = exclude or []
        for name, agent in self.agents.items():
            if name not in exclude and hasattr(agent, "receive_message"):
                agent.receive_message(message)

class AgentMessage:
    """Message for inter-agent communication."""
    
    def __init__(self, sender: str, message_type: str, content: Any):
        """Initialize a message."""
        self.sender = sender
        self.message_type = message_type
        self.content = content
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        message = cls(data["sender"], data["message_type"], data["content"])
        message.timestamp = data["timestamp"]
        message.id = data["id"]
        return message

class AgentSystem:
    """
    Agent System class that serves as the central hub for all agent activities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent system.
        
        Args:
            config: Configuration dictionary for the agent system
        """
        self.config = config or {}
        self.initialized = False
        self.agent_registry = AgentRegistry()
        self.memory = None
        self._lock = threading.RLock()
    
    def initialize(self) -> bool:
        """
        Initialize the agent system, creating all necessary agents.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.initialized:
            logger.warning("Agent system already initialized")
            return True
        
        try:
            with self._lock:
                # Initialize memory system
                self.memory = AgentMemory()
                
                # Initialize memory system
                self.memory = AgentMemory()

                # Initialize agents and register them
                meta_agent = MetaAgent(self.config.get("meta_agent", {}))
                self.agent_registry.register_agent("meta", meta_agent)

                planner_agent = PlannerAgent(self.config.get("planner_agent", {}))
                self.agent_registry.register_agent("planner", planner_agent)

                observer_agent = ObserverAgent(self.config.get("observer_agent", {}))
                self.agent_registry.register_agent("observer", observer_agent)

                analyst_agent = AnalystAgent(self.config.get("analyst_agent", {}))
                self.agent_registry.register_agent("analyst", analyst_agent)

                verifier_agent = VerifierAgent(self.config.get("verifier_agent", {}))
                self.agent_registry.register_agent("verifier", verifier_agent)

                # Pass registry to meta agent for orchestration
                meta_agent.agent_registry = self.agent_registry

                # Start oversight thread
                meta_agent.start_oversight()
                
                self.initialized = True
                logger.info("Agent system initialized successfully")
                return True
        except Exception as e:
            logger.error(f"Error initializing agent system: {e}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the agent system, stopping all agents.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return True
        
        try:
            with self._lock:
                # Stop oversight thread
                meta_agent = self.agent_registry.get_agent("meta")
                if meta_agent:
                    meta_agent.stop_oversight()
                
                # Clear agents
                self.agent_registry.agents.clear()
                
                self.initialized = False
                logger.info("Agent system shutdown successfully")
                return True
        except Exception as e:
            logger.error(f"Error shutting down agent system: {e}")
            return False
    
    def get_agent(self, agent_type: str) -> Any:
        """
        Get an agent of the specified type.
        
        Args:
            agent_type: Type of agent to get (meta, planner, observer, analyst, verifier)
            
        Returns:
            Agent instance or None if not found
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        return self.agent_registry.get_agent(agent_type)
    
    def create_bug(self, bug_id: str, title: str, description: Optional[str] = None, 
                  severity: str = "medium", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new bug entry in the system.
        
        Args:
            bug_id: Unique identifier for the bug
            title: Title of the bug
            description: Description of the bug
            severity: Severity of the bug (low, medium, high, critical)
            tags: List of tags for the bug
            
        Returns:
            Dict containing the bug information or None if creation failed
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        try:
            return self.planner_agent.create_bug(bug_id, title, description, severity, tags)
        except Exception as e:
            logger.error(f"Error creating bug: {e}")
            return None
    
    def generate_solution_paths(self, bug_id: str) -> List[Dict[str, Any]]:
        """
        Generate solution paths for a bug.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            List of solution paths
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return []
        
        try:
            return self.planner_agent.generate_solution_paths(bug_id)
        except Exception as e:
            logger.error(f"Error generating solution paths: {e}")
            return []
    
    def select_solution_path(self, bug_id: str, path_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Select a solution path for a bug.
        
        Args:
            bug_id: ID of the bug
            path_id: ID of the path to select (if None, the best path will be selected)
            
        Returns:
            Selected solution path
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        try:
            return self.planner_agent.select_solution_path(bug_id, path_id)
        except Exception as e:
            logger.error(f"Error selecting solution path: {e}")
            return None
    
    def analyze_bug(self, bug_id: str) -> Dict[str, Any]:
        """
        Analyze a bug using the observer agent.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            Analysis results
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        try:
            observer = self.specialized_agents.get("observer")
            if observer:
                return observer.analyze_bug(bug_id)
            else:
                logger.warning("Observer agent not found")
                return None
        except Exception as e:
            logger.error(f"Error analyzing bug: {e}")
            return None
    
    def generate_patch(self, bug_id: str) -> Dict[str, Any]:
        """
        Generate a patch for a bug using the analyst agent.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            Generated patch
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        try:
            analyst = self.specialized_agents.get("analyst")
            if analyst:
                return analyst.generate_patch(bug_id)
            else:
                logger.warning("Analyst agent not found")
                return None
        except Exception as e:
            logger.error(f"Error generating patch: {e}")
            return None
    
    def verify_patch(self, bug_id: str, patch_id: str) -> Dict[str, Any]:
        """
        Verify a patch using the verifier agent.
        
        Args:
            bug_id: ID of the bug
            patch_id: ID of the patch to verify
            
        Returns:
            Verification results
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        try:
            verifier = self.specialized_agents.get("verifier")
            if verifier:
                return verifier.verify_patch(bug_id, patch_id)
            else:
                logger.warning("Verifier agent not found")
                return None
        except Exception as e:
            logger.error(f"Error verifying patch: {e}")
            return None
    
    def fix_bug(self, bug_id: str) -> Dict[str, Any]:
        """
        Fix a bug using the full agent system workflow.
        
        Args:
            bug_id: ID of the bug
            
        Returns:
            Results of the bug fixing process
        """
        if not self.initialized:
            logger.warning("Agent system not initialized")
            return None
        
        try:
            # Use the planner agent to coordinate the bug fixing process
            return self.planner_agent.fix_bug(bug_id)
        except Exception as e:
            logger.error(f"Error fixing bug: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the agent system.
        
        Returns:
            Status information
        """
        if not self.initialized:
            return {
                "initialized": False,
                "agents": {}
            }
        
        try:
            return {
                "initialized": True,
                "agents": {
                    "meta": self.meta_agent.get_status() if self.meta_agent else None,
                    "planner": self.planner_agent.get_status() if self.planner_agent else None,
                    "specialized": {
                        agent_type: agent.get_status()
                        for agent_type, agent in self.specialized_agents.items()
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting agent system status: {e}")
            return {
                "initialized": True,
                "error": str(e)
            }

# Create a singleton instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> AgentSystem:
    """
    Get the singleton instance of the agent system.
    
    Args:
        config: Configuration dictionary for the agent system
        
    Returns:
        AgentSystem instance
    """
    global _instance
    if _instance is None:
        _instance = AgentSystem(config)
    return _instance
