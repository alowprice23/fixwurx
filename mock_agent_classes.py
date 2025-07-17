#!/usr/bin/env python3
"""
mock_agent_classes.py
────────────────────
Mock implementations of agent classes for testing and development.

These mock classes provide simplified implementations of the core agent
functionality without all the dependencies, making it easier to test
the shell integration without requiring the full agent system.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

# Configure logging
logger = logging.getLogger("MockAgentClasses")

class BugState:
    """
    Simplified mock of the BugState class.
    """
    def __init__(
        self,
        bug_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        severity: str = "medium",
        status: str = "new",
        metadata: Dict[str, Any] = None,
        tags: Set[str] = None
    ):
        self.bug_id = bug_id
        self.title = title
        self.description = description
        self.severity = severity
        self.status = status
        self.metadata = metadata or {}
        self.phase_history = []
        self.planner_solutions = []
        self.tags = tags or set()
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def add_phase(self, phase_name: str, details: Dict[str, Any] = None) -> None:
        """Add a phase to the history."""
        if "phase_history" not in self.metadata:
            self.metadata["phase_history"] = []
        
        phase_entry = {
            "name": phase_name,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        # Add to both class attribute and metadata for compatibility
        self.phase_history.append(phase_entry)
        self.metadata["phase_history"] = self.phase_history
        self.updated_at = time.time()
    
    def add_planner_solution(self, solution: Dict[str, Any]) -> None:
        """Add a solution from the planner."""
        self.planner_solutions.append(solution)
        self.updated_at = time.time()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the bug."""
        self.tags.add(tag)
        self.updated_at = time.time()
    
    def calculate_entropy(self) -> float:
        """Calculate an entropy value (simplified version)."""
        # Simple approximation
        return 0.5
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON."""
        return {
            "bug_id": self.bug_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "status": self.status,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class PlannerPath:
    """
    Simplified mock of the PlannerPath class.
    """
    def __init__(
        self,
        path_id: str,
        bug_id: str,
        actions: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None,
        fallbacks: List[Dict[str, Any]] = None
    ):
        self.path_id = path_id
        self.bug_id = bug_id
        self.actions = actions
        self.fallbacks = fallbacks or []
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.updated_at = time.time()
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON."""
        return {
            "path_id": self.path_id,
            "bug_id": self.bug_id,
            "actions": self.actions,
            "fallbacks": self.fallbacks,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class FamilyTree:
    """
    Simplified mock of the FamilyTree class.
    """
    def __init__(self):
        self.relationships = {
            "planner": {
                "children": [],
                "metadata": {"type": "root", "created_at": time.time()}
            }
        }
    
    def add_agent(self, agent_id: str, parent_id: str, metadata: Dict[str, Any] = None) -> bool:
        """Add an agent to the family tree."""
        if agent_id in self.relationships:
            return False
        
        if parent_id not in self.relationships:
            return False
        
        self.relationships[agent_id] = {
            "parent": parent_id,
            "children": [],
            "metadata": metadata or {"created_at": time.time()}
        }
        
        self.relationships[parent_id]["children"].append(agent_id)
        return True
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON."""
        return {
            "relationships": self.relationships,
            "updated_at": time.time()
        }


class AgentMemory:
    """
    Simplified mock of the AgentMemory class.
    """
    def __init__(self, **kwargs):
        self.memory = {}
    
    def store(self, key: str, value: Any) -> bool:
        """Store a value in memory."""
        self.memory[key] = value
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory."""
        return self.memory.get(key)


class PlannerAgent:
    """
    Simplified mock of the PlannerAgent class.
    """
    def __init__(self, config: Dict[str, Any], agent_memory: AgentMemory):
        self.config = config
        self.agent_memory = agent_memory
        self.enabled = True
        self.family_tree = FamilyTree()
        self.active_bugs = {}
        self.active_paths = {}
        self.metrics = {
            "paths_generated": 0,
            "fallbacks_used": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "avg_path_length": 0,
            "pattern_recognitions": 0,
            "neural_adjustments": 0,
            "path_optimizations": 0,
            "active_bugs": 0,
            "active_paths": 0,
            "family_tree_size": 1,
            "enabled": True
        }
        self.neural_weights = {
            "observer": 1.0,
            "analyst": 1.0,
            "verifier": 1.0,
            "fallback": 0.5,
            "pattern_match": 1.5,
            "entropy": 0.8
        }
    
    def generate_solution_paths(self, bug: BugState) -> List[PlannerPath]:
        """Generate solution paths for a bug."""
        # Add bug to active bugs
        self.active_bugs[bug.bug_id] = bug
        self.metrics["active_bugs"] = len(self.active_bugs)
        
        # Generate some mock paths
        paths = []
        for i in range(3):
            path_id = f"{bug.bug_id}-path-{i}-{uuid.uuid4().hex[:8]}"
            
            # Create mock actions
            actions = [
                {
                    "type": "analyze",
                    "agent": "observer",
                    "description": f"Analyze bug {bug.bug_id}",
                    "parameters": {"bug_id": bug.bug_id, "depth": "full"}
                },
                {
                    "type": "patch",
                    "agent": "analyst",
                    "description": f"Generate patch for bug {bug.bug_id}",
                    "parameters": {"bug_id": bug.bug_id}
                },
                {
                    "type": "verify",
                    "agent": "verifier",
                    "description": f"Verify fix for bug {bug.bug_id}",
                    "parameters": {"bug_id": bug.bug_id, "comprehensive": True}
                }
            ]
            
            # Create fallbacks for some paths
            fallbacks = []
            if i == 1:
                fallbacks = [{
                    "type": "simplify",
                    "description": f"Simplified approach for bug {bug.bug_id}",
                    "actions": [
                        {
                            "type": "analyze",
                            "agent": "observer",
                            "description": f"Quick analysis of bug {bug.bug_id}",
                            "parameters": {"bug_id": bug.bug_id, "depth": "basic"}
                        }
                    ]
                }]
            
            # Create a path with varying priorities
            path = PlannerPath(
                path_id=path_id,
                bug_id=bug.bug_id,
                actions=actions,
                fallbacks=fallbacks,
                metadata={
                    "priority": 0.9 - (i * 0.2),
                    "entropy": 0.7 - (i * 0.1),
                    "estimated_time": 10 - i,
                    "created_by": "planner"
                }
            )
            
            paths.append(path)
            self.active_paths[path_id] = path
            
            # Add solution to bug
            bug.add_planner_solution(path.to_json())
        
        # Update metrics
        self.metrics["paths_generated"] += len(paths)
        self.metrics["active_paths"] = len(self.active_paths)
        self.metrics["avg_path_length"] = 3.0  # Mock value
        
        return paths
    
    def select_best_path(self, bug_id: str) -> Optional[PlannerPath]:
        """Select the best solution path for a bug."""
        if bug_id not in self.active_bugs:
            return None
        
        # Get all paths for this bug
        bug_paths = [p for p in self.active_paths.values() if p.bug_id == bug_id]
        if not bug_paths:
            return None
        
        # Sort by priority (higher is better)
        bug_paths.sort(key=lambda p: p.metadata.get("priority", 0), reverse=True)
        return bug_paths[0]
    
    def register_agent(self, agent_id: str, agent_type: str) -> bool:
        """Register a new agent in the family tree."""
        result = self.family_tree.add_agent(
            agent_id=agent_id,
            parent_id="planner",
            metadata={
                "type": agent_type,
                "created_at": time.time()
            }
        )
        
        if result:
            self.metrics["family_tree_size"] = len(self.family_tree.relationships)
        
        return result
    
    def get_family_relationships(self, agent_id: str = None) -> Dict[str, Any]:
        """Get family relationships."""
        return self.family_tree.to_json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        metrics = self.metrics.copy()
        metrics["enabled"] = self.enabled
        return metrics
    
    def record_path_result(self, path_id: str, success: bool, metrics: Dict[str, Any]) -> None:
        """Record a path result."""
        if path_id not in self.active_paths:
            return
        
        path = self.active_paths[path_id]
        bug_id = path.bug_id
        
        # Update bug if it exists
        if bug_id in self.active_bugs:
            bug = self.active_bugs[bug_id]
            bug.metadata["last_path_result"] = {
                "path_id": path_id,
                "success": success,
                "metrics": metrics,
                "timestamp": time.time()
            }
            
            # Add tags based on result
            if success:
                bug.add_tag("fixed")
                bug.add_tag(f"fixed_by_path_{path_id}")
            else:
                bug.add_tag("failed_attempt")
        
        # Update path metadata
        path.metadata["executed"] = True
        path.metadata["success"] = success
        path.metadata["execution_metrics"] = metrics
        path.metadata["executed_at"] = time.time()
        
        # Update metrics
        if success:
            self.metrics["successful_fixes"] += 1
        else:
            self.metrics["failed_fixes"] += 1
    
    def activate_fallback(self, path_id: str) -> Optional[Dict[str, Any]]:
        """Activate a fallback strategy for a failing path."""
        if path_id not in self.active_paths:
            return None
        
        path = self.active_paths[path_id]
        
        # Check if fallbacks exist
        if not path.fallbacks:
            return None
        
        # Get the first fallback
        fallback = path.fallbacks[0]
        
        # Remove it from the list
        path.fallbacks.pop(0)
        
        # Update metrics
        self.metrics["fallbacks_used"] += 1
        
        return fallback


class ObserverAgent:
    """Mock ObserverAgent."""
    def __init__(self):
        pass


class AnalystAgent:
    """Mock AnalystAgent."""
    def __init__(self):
        pass


class VerifierAgent:
    """Mock VerifierAgent."""
    def __init__(self):
        pass


class MetaAgent:
    """
    Mock implementation of the Meta Agent for testing.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Meta Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.agent_states = {}
        self.agent_activities = {}
        self.agent_conflicts = {}
        self.coordination_matrix = {}
        self.metrics = {
            "coordination_events": 0,
            "conflict_resolutions": 0,
            "optimizations": 0,
            "oversight_cycles": 0,
            "agent_activities_tracked": 0,
            "resources_allocated": 0,
            "meta_insights_generated": 0
        }
    
    def start_oversight(self) -> bool:
        """Start oversight thread."""
        return True
    
    def stop_oversight(self) -> bool:
        """Stop oversight thread."""
        return True
    
    def register_agent(self, agent_id: str, agent_type: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Register an agent with the Meta Agent.
        
        Args:
            agent_id: ID of the agent
            agent_type: Type of the agent
            metadata: Additional agent metadata
            
        Returns:
            True if registered successfully, False otherwise
        """
        if agent_id in self.agent_states:
            return False
        
        # Create agent state
        self.agent_states[agent_id] = {
            "id": agent_id,
            "type": agent_type,
            "registered_at": time.time(),
            "last_activity": time.time(),
            "status": "active",
            "metadata": metadata or {}
        }
        
        # Update coordination matrix
        for existing_agent_id in self.agent_states:
            if existing_agent_id != agent_id:
                coord_key = f"{agent_id}:{existing_agent_id}"
                reverse_key = f"{existing_agent_id}:{agent_id}"
                
                self.coordination_matrix[coord_key] = 0.0
                self.coordination_matrix[reverse_key] = 0.0
        
        return True
    
    def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent."""
        if agent_id not in self.agent_states:
            return False
        
        del self.agent_states[agent_id]
        return True
    
    def coordinate_agents(self, agent_ids: List[str], task_id: str, task_type: str, task_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Coordinate agents."""
        valid_agents = [agent_id for agent_id in agent_ids if agent_id in self.agent_states]
        if not valid_agents:
            return {"success": False, "error": "No valid agents"}
        
        self.metrics["coordination_events"] += 1
        
        return {
            "success": True,
            "plan_id": task_id,
            "agent_count": len(valid_agents),
            "step_count": len(valid_agents)
        }
    
    def get_agent_network(self) -> Dict[str, Any]:
        """Get agent network."""
        # Create nodes
        nodes = []
        for agent_id, state in self.agent_states.items():
            nodes.append({
                "id": agent_id,
                "type": state["type"],
                "status": state["status"],
                "last_activity": state["last_activity"]
            })
        
        # Create edges
        edges = []
        for coord_key, strength in self.coordination_matrix.items():
            if strength > 0.1:  # Only include significant connections
                agent1, agent2 = coord_key.split(":")
                edges.append({
                    "source": agent1,
                    "target": agent2,
                    "strength": strength
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "timestamp": time.time()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        metrics = self.metrics.copy()
        metrics["agent_count"] = len(self.agent_states)
        metrics["conflict_count"] = len(self.agent_conflicts)
        metrics["coordination_matrix_size"] = len(self.coordination_matrix)
        
        return metrics
    
    def generate_insight(self) -> Dict[str, Any]:
        """Generate insight."""
        insight_types = ["resource_optimization", "conflict_pattern", "coordination_improvement", "activity_trend"]
        insight_type = insight_types[int(time.time()) % len(insight_types)]
        
        self.metrics["meta_insights_generated"] += 1
        
        return {
            "id": f"insight-{int(time.time())}",
            "type": insight_type,
            "timestamp": time.time(),
            "confidence": 0.7 + (0.3 * (int(time.time()) % 10) / 10),
            "description": f"Simulated {insight_type} insight",
            "details": {
                "agent_count": len(self.agent_states),
                "activity_count": 0,
                "coordination_events": self.metrics["coordination_events"],
                "conflict_resolutions": self.metrics["conflict_resolutions"]
            },
            "recommendations": [
                {
                    "action": "optimize_resources",
                    "confidence": 0.8,
                    "expected_impact": 0.6
                },
                {
                    "action": "increase_coordination",
                    "confidence": 0.7,
                    "expected_impact": 0.5
                }
            ]
        }
    
    def visualize_agent_network(self, output_path: Optional[str] = None) -> str:
        """Visualize agent network."""
        return f".triangulum/meta/agent_network_{int(time.time())}.json"
