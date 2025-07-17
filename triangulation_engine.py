"""
triangulation_engine.py
──────────────────────
Triangulation engine for the FixWurx system.

This module provides the TriangulationEngine class, which manages
the bug resolution process by triangulating multiple solution approaches.
"""

import logging
import time
import uuid
import json
import os
import requests
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from state_machine import Phase

# Setup logging
logger = logging.getLogger("triangulation_engine")

# Global engine instance for singleton access
_engine_instance = None

def get_engine() -> 'TriangulationEngine':
    """
    Get the global triangulation engine instance.
    
    Returns:
        The TriangulationEngine instance
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TriangulationEngine()
    return _engine_instance

def register_bug(bug_id=None, title="", description="", severity="medium"):
    """
    Register a bug with the triangulation engine.
    
    Args:
        bug_id: Optional bug ID (generated if not provided)
        title: Bug title
        description: Bug description
        severity: Bug severity
        
    Returns:
        Dictionary with registration result
    """
    engine = get_engine()
    bug = engine.add_bug(
        bug_id=bug_id,
        metadata={
            "title": title,
            "description": description,
            "severity": severity
        }
    )
    
    return {
        "success": True,
        "bug_id": bug.id,
        "status": "registered"
    }

def start_bug_fix(bug_id):
    """
    Start fixing a bug.
    
    Args:
        bug_id: The bug ID to fix
        
    Returns:
        Dictionary with operation result
    """
    engine = get_engine()
    bug = engine.get_bug(bug_id)
    
    if not bug:
        return {
            "success": False,
            "error": f"Bug {bug_id} not found"
        }
    
    # Create an execution ID
    execution_id = f"exec-{bug_id}-{int(time.time())}"
    
    # Store execution info in bug metadata
    if "executions" not in bug.metadata:
        bug.metadata["executions"] = []
    
    bug.metadata["executions"].append({
        "execution_id": execution_id,
        "start_time": time.time(),
        "status": "started"
    })
    
    # Update phase to ANALYZE
    engine.update_bug_phase(bug_id, Phase.ANALYZE)
    
    return {
        "success": True,
        "execution_id": execution_id,
        "bug_id": bug_id,
        "status": "started"
    }

def get_bug_status(bug_id):
    """
    Get the status of a bug.
    
    Args:
        bug_id: The bug ID to check
        
    Returns:
        Dictionary with bug status
    """
    engine = get_engine()
    bug = engine.get_bug(bug_id)
    
    if not bug:
        return {
            "success": False,
            "error": f"Bug {bug_id} not found"
        }
    
    return {
        "success": True,
        "status": {
            "bug_id": bug.id,
            "status": bug.metadata.get("status", "unknown"),
            "phase": bug.phase.name,
            "timer": bug.timer,
            "attempts": bug.attempts,
            "max_attempts": bug.max_attempts
        }
    }

def get_execution_status(execution_id):
    """
    Get the status of an execution.
    
    Args:
        execution_id: The execution ID to check
        
    Returns:
        Dictionary with execution status
    """
    engine = get_engine()
    
    # Search for the execution in all bugs
    for bug in engine.bugs:
        executions = bug.metadata.get("executions", [])
        for execution in executions:
            if execution.get("execution_id") == execution_id:
                return {
                    "success": True,
                    "status": {
                        "execution_id": execution_id,
                        "bug_id": bug.id,
                        "status": execution.get("status", "unknown"),
                        "start_time": execution.get("start_time"),
                        "end_time": execution.get("end_time", None)
                    }
                }
    
    return {
        "success": False,
        "error": f"Execution {execution_id} not found"
    }

def get_engine_stats():
    """
    Get statistics about the triangulation engine.
    
    Returns:
        Dictionary with engine statistics
    """
    engine = get_engine()
    metrics = engine.get_metrics()
    
    # Format stats for API response
    return {
        "success": True,
        "stats": {
            "bugs": {
                "total": metrics["total_bugs"],
                "by_phase": metrics["phase_counts"]
            },
            "paths": {
                "total": 0  # Placeholder, would track solution paths in a real implementation
            },
            "executions": {
                "total": 0,  # Placeholder
                "active": 0  # Placeholder
            }
        }
    }

# For compatibility with integration test
FixPhase = Phase

@dataclass
class BugState:
    """
    State information for a bug being processed by the triangulation engine.
    
    This class represents the state of a bug in the system, including its
    current phase, timer, and other metadata.
    """
    id: str
    phase: Phase = Phase.TRIAGE
    timer: float = 0.0
    start_time: float = time.time()
    attempts: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        if self.metadata is None:
            self.metadata = {}
        # Initialize phase history
        if "phase_history" not in self.metadata:
            self.metadata["phase_history"] = []
    
    def add_phase(self, phase_name: str, details: Dict[str, Any] = None) -> None:
        """
        Add a phase to the bug's history.
        
        Args:
            phase_name: Name of the phase
            details: Optional details about the phase
        """
        if "phase_history" not in self.metadata:
            self.metadata["phase_history"] = []
        
        phase_entry = {
            "name": phase_name,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        self.metadata["phase_history"].append(phase_entry)
        
        # For compatibility with agent_commands.py
        self.phase_history = self.metadata["phase_history"]

class PlannerPath:
    """
    Represents a solution path generated by the planner.
    
    A planner path is a sequence of actions that can be taken to resolve a bug.
    It includes an approach type, a success rate estimate, and a list of actions.
    """
    
    def __init__(self, path_id: str, approach: str, actions: List[Dict[str, Any]], estimated_success_rate: float = 0.5):
        """
        Initialize a planner path.
        
        Args:
            path_id: Unique identifier for the path
            approach: Approach type (e.g., "standard", "fallback")
            actions: List of actions in the path
            estimated_success_rate: Estimated success rate (0.0 to 1.0)
        """
        self.path_id = path_id
        self.approach = approach
        self.actions = actions
        self.estimated_success_rate = estimated_success_rate
        self.created_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path_id": self.path_id,
            "approach": self.approach,
            "actions": self.actions,
            "estimated_success_rate": self.estimated_success_rate,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannerPath":
        """Create from dictionary representation."""
        return cls(
            path_id=data["path_id"],
            approach=data["approach"],
            actions=data["actions"],
            estimated_success_rate=data.get("estimated_success_rate", 0.5)
        )

def cancel_execution(execution_id: str) -> Dict[str, Any]:
    """
    Cancel an execution.
    
    Args:
        execution_id: The execution ID to cancel
        
    Returns:
        Dictionary with operation result
    """
    engine = get_engine()
    
    # Search for the execution in all bugs
    for bug in engine.bugs:
        executions = bug.metadata.get("executions", [])
        for i, execution in enumerate(executions):
            if execution.get("execution_id") == execution_id:
                # Update execution status
                execution["status"] = "cancelled"
                execution["end_time"] = time.time()
                bug.metadata["executions"][i] = execution
                
                # Update bug status
                bug.metadata["status"] = "cancelled"
                
                return {
                    "success": True,
                    "execution_id": execution_id,
                    "bug_id": bug.id,
                    "status": "cancelled"
                }
    
    return {
        "success": False,
        "error": f"Execution {execution_id} not found"
    }

class TriangulationEngine:
    """
    Engine for triangulating multiple solution approaches.
    
    The TriangulationEngine manages the bug resolution process by coordinating
    the execution of multiple solution paths and selecting the best approach.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the triangulation engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.bugs: List[BugState] = []
        self.logger = logging.getLogger("triangulation_engine")
        
        # Configuration options
        self.max_bugs = self.config.get("max_bugs", 10)
        self.timeout_seconds = self.config.get("timeout_seconds", 3600)  # 1 hour default
        self.auto_escalate = self.config.get("auto_escalate", True)
        
        # Neural matrix integration
        self.neural_enabled = self.config.get("neural_matrix", {}).get("enabled", False)
        self.neural_weights_path = self.config.get("neural_matrix", {}).get("weights_path", None)
        self.neural_hub_url = self.config.get("neural_matrix", {}).get("hub_url", "http://localhost:8001")
        
        # Cache for neural weights
        self.neural_weights_cache = {}
        self.neural_weight_update_time = 0
        
        # Load neural weights if enabled
        if self.neural_enabled:
            self._load_neural_weights()
    
    def add_bug(self, bug_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> BugState:
        """
        Add a new bug to the engine.
        
        Args:
            bug_id: Optional bug ID (generated if not provided)
            metadata: Optional metadata for the bug
            
        Returns:
            The newly created BugState
        """
        # Generate bug ID if not provided
        if bug_id is None:
            bug_id = str(uuid.uuid4())
        
        # Create bug state
        bug = BugState(
            id=bug_id,
            phase=Phase.TRIAGE,
            timer=0.0,
            metadata=metadata or {}
        )
        
        # Add to bugs list
        self.bugs.append(bug)
        self.logger.info(f"Added bug {bug_id} to triangulation engine")
        
        return bug
    
    def get_bug(self, bug_id: str) -> Optional[BugState]:
        """
        Get a bug by ID.
        
        Args:
            bug_id: The bug ID to look for
            
        Returns:
            The BugState if found, None otherwise
        """
        for bug in self.bugs:
            if bug.id == bug_id:
                return bug
        return None
    
    def update_bug_phase(self, bug_id: str, phase: Phase) -> bool:
        """
        Update the phase of a bug.
        
        Args:
            bug_id: The bug ID to update
            phase: The new phase
            
        Returns:
            True if the bug was updated, False otherwise
        """
        bug = self.get_bug(bug_id)
        if bug:
            bug.phase = phase
            self.logger.info(f"Updated bug {bug_id} phase to {phase.name}")
            return True
        return False
    
    def remove_bug(self, bug_id: str) -> bool:
        """
        Remove a bug from the engine.
        
        Args:
            bug_id: The bug ID to remove
            
        Returns:
            True if the bug was removed, False otherwise
        """
        for i, bug in enumerate(self.bugs):
            if bug.id == bug_id:
                self.bugs.pop(i)
                self.logger.info(f"Removed bug {bug_id} from triangulation engine")
                return True
        return False
    
    def tick(self) -> None:
        """
        Advance the engine by one tick.
        
        This method updates the timer for each bug and checks for timeouts.
        It should be called regularly by the scheduler.
        """
        now = time.time()
        
        for bug in self.bugs:
            # Update timer
            bug.timer = now - bug.start_time
            
            # Check for timeout
            if self.auto_escalate and bug.timer > self.timeout_seconds:
                if bug.phase != Phase.DONE and bug.phase != Phase.ESCALATE:
                    self.logger.warning(f"Bug {bug.id} timed out after {bug.timer:.1f} seconds, escalating")
                    bug.phase = Phase.ESCALATE
    
    def get_bugs_in_phase(self, phase: Phase) -> List[BugState]:
        """
        Get all bugs in a specific phase.
        
        Args:
            phase: The phase to filter by
            
        Returns:
            List of bugs in the specified phase
        """
        return [bug for bug in self.bugs if bug.phase == phase]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the triangulation engine.
        
        Returns:
            Dictionary with metrics
        """
        phase_counts = {}
        for phase in Phase:
            phase_counts[phase.name] = len(self.get_bugs_in_phase(phase))
        
        metrics = {
            "total_bugs": len(self.bugs),
            "phase_counts": phase_counts,
            "oldest_bug_timer": max([bug.timer for bug in self.bugs]) if self.bugs else 0.0,
            "newest_bug_timer": min([bug.timer for bug in self.bugs]) if self.bugs else 0.0
        }
        
        # Add neural matrix metrics if enabled
        if self.neural_enabled:
            metrics["neural_matrix"] = {
                "enabled": True,
                "weights_loaded": bool(self.neural_weights_cache),
                "weights_update_time": self.neural_weight_update_time,
                "hub_url": self.neural_hub_url
            }
        
        return metrics
    
    def _load_neural_weights(self) -> None:
        """
        Load neural weights from file or API.
        
        This method loads neural weights from the specified file
        or from the neural hub API if a file is not provided.
        """
        try:
            # If a weights file is specified, load from file
            if self.neural_weights_path:
                self.logger.info(f"Loading neural weights from file: {self.neural_weights_path}")
                with open(self.neural_weights_path, 'r') as f:
                    weights_data = json.load(f)
                
                # Extract agent weights
                for agent, weight in weights_data.get("agent_weights", {}).items():
                    self.neural_weights_cache[agent] = weight
                
                # Extract feature weights
                for feature, weight in weights_data.get("feature_weights", {}).items():
                    self.neural_weights_cache[f"feature_{feature}"] = weight
                
                self.neural_weight_update_time = time.time()
                self.logger.info(f"Loaded {len(self.neural_weights_cache)} neural weights from file")
            
            # Otherwise, try to load from the neural hub API
            else:
                self.logger.info(f"Loading neural weights from API: {self.neural_hub_url}")
                response = requests.get(f"{self.neural_hub_url}/neural/weights?category=agent")
                
                if response.status_code == 200:
                    weights_data = response.json()
                    
                    # Process weights data
                    for weight in weights_data:
                        self.neural_weights_cache[weight["weight_key"]] = weight["weight_value"]
                    
                    self.neural_weight_update_time = time.time()
                    self.logger.info(f"Loaded {len(self.neural_weights_cache)} neural weights from API")
                else:
                    self.logger.warning(f"Failed to load neural weights from API: {response.status_code}")
        
        except Exception as e:
            self.logger.error(f"Error loading neural weights: {e}")
    
    def apply_neural_weights(self, bug_id: str, path_id: str = None) -> Dict[str, float]:
        """
        Apply neural weights to a bug resolution path.
        
        Args:
            bug_id: ID of the bug
            path_id: Optional ID of the path
            
        Returns:
            Dictionary of weighted scores
        """
        # If neural is not enabled or no weights are loaded, return default weights
        if not self.neural_enabled or not self.neural_weights_cache:
            return {"observer": 1.0, "analyst": 1.0, "verifier": 1.0, "planner": 1.0}
        
        # Get the bug
        bug = self.get_bug(bug_id)
        if not bug:
            self.logger.warning(f"Bug {bug_id} not found for neural weighting")
            return {"observer": 1.0, "analyst": 1.0, "verifier": 1.0, "planner": 1.0}
        
        # Apply agent weights
        weights = {}
        for agent in ["observer", "analyst", "verifier", "planner"]:
            weights[agent] = self.neural_weights_cache.get(agent, 1.0)
        
        # Adjust weights based on bug metadata
        bug_type = bug.metadata.get("bug_type", "unknown")
        if bug_type == "memory-leak":
            weights["analyst"] *= 1.2  # Boost analyst for memory leak bugs
        elif bug_type == "security-vulnerability":
            weights["verifier"] *= 1.3  # Boost verifier for security bugs
        
        # Log the applied weights
        self.logger.info(f"Applied neural weights for bug {bug_id}: {weights}")
        
        return weights
