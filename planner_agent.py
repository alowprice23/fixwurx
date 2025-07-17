#!/usr/bin/env python3
"""
planner_agent.py
────────────────
Planner Agent for orchestrating bug resolution and managing solution paths.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional

from data_structures import PlannerPath, FamilyTree

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlannerAgent")

class PlannerAgent:
    """
    Planner Agent for orchestrating bug resolution.
    
    The PlannerAgent coordinates the execution of solution paths and
    manages the overall bug resolution process.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Planner Agent."""
        self.config = config or {}
        self.active_paths: Dict[str, PlannerPath] = {}
        self.completed_paths: Dict[str, PlannerPath] = {}
        self.failed_paths: Dict[str, PlannerPath] = {}
        self.family_tree = FamilyTree()
        self._metrics = {
            "paths_generated": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "fallbacks_used": 0
        }
        self.logger = logging.getLogger("PlannerAgent")

    async def ask(self, prompt: str) -> str:
        """
        Send a prompt to the planner agent.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Response from the planner agent
        """
        # Mock implementation for testing
        if "Initialize the family tree" in prompt:
            return json.dumps({
                "status": "success",
                "message": "Family tree initialized successfully",
                "agents": ["planner", "observer", "analyst", "verifier"]
            })
        elif "Generate solution paths" in prompt:
            # Generate a mock solution path
            path_id = f"path-{int(time.time())}"
            path = PlannerPath(
                path_id=path_id,
                bug_id="bug-123",
                actions=[
                    {"type": "analyze", "agent": "observer"},
                    {"type": "patch", "agent": "analyst"},
                    {"type": "verify", "agent": "verifier"}
                ]
            )
            self.active_paths[path_id] = path
            self._metrics["paths_generated"] += 1
            return json.dumps({"status": "success", "paths": [path.to_dict()]})
        return json.dumps({"status": "unknown_command"})

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the planner."""
        self._metrics["active_paths"] = len(self.active_paths)
        self._metrics["completed_paths"] = len(self.completed_paths)
        self._metrics["failed_paths"] = len(self.failed_paths)
        return self._metrics

    def record_path_result(self, path_id: str, success: bool, details: Dict[str, Any]) -> None:
        """
        Record the result of a solution path execution.
        
        Args:
            path_id: The ID of the path
            success: Whether the path was successful
            details: Additional details about the execution
        """
        if path_id in self.active_paths:
            path = self.active_paths.pop(path_id)
            if success:
                self.completed_paths[path_id] = path
                self._metrics["successful_fixes"] += 1
                self.logger.info(f"Path {path_id} completed successfully.")
            else:
                self.failed_paths[path_id] = path
                self._metrics["failed_fixes"] += 1
                self.logger.warning(f"Path {path_id} failed.")
        else:
            self.logger.error(f"Path {path_id} not found in active paths.")

    def generate_solution_paths(self, bug_id: str, context: Dict[str, Any]) -> List[PlannerPath]:
        """
        Generate one or more solution paths for a given bug.

        Args:
            bug_id: The ID of the bug to generate paths for.
            context: The context surrounding the bug.

        Returns:
            A list of PlannerPath objects.
        """
        self.logger.info(f"Generating solution paths for bug {bug_id}...")
        # This is a mock implementation. A real implementation would use an LLM or a rule engine.
        path_id = f"path-{bug_id}-{int(time.time())}"
        
        # Simple path for demonstration
        actions = [
            {"type": "analyze", "agent": "observer", "parameters": {"depth": "full"}},
            {"type": "patch", "agent": "analyst", "parameters": {"conservative": True}},
            {"type": "verify", "agent": "verifier", "parameters": {"retries": 2}},
        ]
        
        path = PlannerPath(path_id=path_id, bug_id=bug_id, actions=actions)
        self.active_paths[path_id] = path
        self._metrics["paths_generated"] += 1
        
        self.logger.info(f"Generated path {path_id} for bug {bug_id}.")
        return [path]

    def select_best_path(self, bug_id: str) -> Optional[PlannerPath]:
        """
        Select the best path to execute for a given bug.

        Args:
            bug_id: The ID of the bug.

        Returns:
            The best PlannerPath to execute, or None if no paths are available.
        """
        paths_for_bug = [p for p in self.active_paths.values() if p.bug_id == bug_id]
        if not paths_for_bug:
            return None
        
        # Simple selection strategy: choose the one with the highest priority metadata, if present.
        paths_for_bug.sort(key=lambda p: p.metadata.get("priority", 0), reverse=True)
        
        best_path = paths_for_bug[0]
        self.logger.info(f"Selected path {best_path.path_id} for bug {bug_id}.")
        return best_path

    def activate_fallback(self, path_id: str) -> Optional[PlannerPath]:
        """
        Activate a fallback path if the primary path fails.

        Args:
            path_id: The ID of the failed path.

        Returns:
            A new fallback PlannerPath, or None if no fallbacks are available.
        """
        original_path = self.failed_paths.get(path_id)
        if not original_path or not original_path.fallbacks:
            return None

        self.logger.info(f"Activating fallback for path {path_id}.")
        self._metrics["fallbacks_used"] += 1
        
        # Create a new path from the first fallback strategy
        fallback_strategy = original_path.fallbacks[0]
        fallback_path_id = f"fallback-{path_id}-{int(time.time())}"
        
        fallback_actions = [
             {"type": "analyze", "agent": "observer", "parameters": {"depth": "shallow"}},
             {"type": "patch", "agent": "analyst", "parameters": {"conservative": False}},
             {"type": "verify", "agent": "verifier", "parameters": {"retries": 1}},
        ]

        fallback_path = PlannerPath(
            path_id=fallback_path_id,
            bug_id=original_path.bug_id,
            actions=fallback_actions,
            metadata={"original_path": path_id, "fallback_strategy": fallback_strategy}
        )
        
        self.active_paths[fallback_path_id] = fallback_path
        self._metrics["paths_generated"] += 1
        
        return fallback_path
