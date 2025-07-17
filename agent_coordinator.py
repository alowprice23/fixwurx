#!/usr/bin/env python3
"""
agent_coordinator.py
────────────────────
This is an import wrapper for the AgentCoordinator class.
The implementation is provided in agents/core/coordinator.py.
"""

import time
import json
import logging
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from agents.core.coordinator import AgentCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentCoordinator")

class HandoffStatus(Enum):
    """Status codes for agent handoffs."""
    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    REJECTED = auto()


class _Artefacts:
    """
    Container for artefacts produced during agent coordination.
    Used to track the state of the coordinator across ticks.
    """
    
    def __init__(self):
        """Initialize artefacts container."""
        # Context is a general purpose dict for storing data
        self.context = {}
        
        # Time tracking
        self.start_time = time.time()
        self.action_times = {}
        self.total_duration = 0.0
        
        # Solution paths
        self.solution_paths = []
        self.current_path_id = None
        self.current_action_index = 0
        
        # Agent reports
        self.observer_report = None
        self.analyst_report = None
        self.verifier_report = None
        
        # Metrics
        self.handoff_counts = {}
        self.error_counts = {}
        self.fallbacks_used = 0
    
    def add_to_context(self, key: str, value: Any):
        """Add an item to the context."""
        self.context[key] = value
    
    def get_from_context(self, key: str, default: Any = None) -> Any:
        """Get an item from the context."""
        return self.context.get(key, default)
    
    def record_action_time(self, action: str, duration: float):
        """Record the time taken for an action."""
        if action not in self.action_times:
            self.action_times[action] = []
        self.action_times[action].append(duration)
    
    def record_handoff(self, from_agent: str, to_agent: str):
        """Record a handoff between agents."""
        key = f"{from_agent}_to_{to_agent}"
        self.handoff_counts[key] = self.handoff_counts.get(key, 0) + 1
    
    def record_error(self, component: str, error_type: str):
        """Record an error."""
        key = f"{component}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_current_action(self) -> Optional[Dict[str, Any]]:
        """Get the current action from the current path."""
        if not self.solution_paths or self.current_path_id is None:
            return None
        
        # Find the current path
        for path in self.solution_paths:
            if path.get("path_id") == self.current_path_id:
                actions = path.get("actions", [])
                if 0 <= self.current_action_index < len(actions):
                    return actions[self.current_action_index]
                return None
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics data."""
        self.total_duration = time.time() - self.start_time
        
        return {
            "total_duration": self.total_duration,
            "action_times": self.action_times,
            "handoff_counts": self.handoff_counts,
            "error_counts": self.error_counts,
            "fallbacks_used": self.fallbacks_used,
            "solution_paths_count": len(self.solution_paths)
        }


__all__ = ["AgentCoordinator", "HandoffStatus", "_Artefacts"]
