#!/usr/bin/env python3
"""
Decision Flow

This module implements the decision flow logic for bug fixing.
"""

import os
import sys
import json
import time
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(".triangulum/decision_flow.log", mode='a')
    ]
)

logger = logging.getLogger("DecisionFlow")

class FlowNode:
    """
    Represents a node in the decision flow.
    """
    
    def __init__(
        self,
        node_id: str,
        name: str,
        description: str,
        action: Callable = None,
        next_nodes: List[str] = None
    ):
        """
        Initialize a flow node.
        
        Args:
            node_id: ID of the node
            name: Name of the node
            description: Description of the node
            action: Action to perform
            next_nodes: List of next node IDs
        """
        self.node_id = node_id
        self.name = name
        self.description = description
        self.action = action
        self.next_nodes = next_nodes or []
    
    def execute(
        self, 
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Execute the node's action.
        
        Args:
            context: Context information
            
        Returns:
            str: ID of the next node to execute, or None
        """
        if self.action:
            result = self.action(context)
            
            if isinstance(result, str) and result in self.next_nodes:
                return result
            else:
                return self.next_nodes[0] if self.next_nodes else None
        else:
            return self.next_nodes[0] if self.next_nodes else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "next_nodes": self.next_nodes
        }

class Decision:
    """
    Represents a decision in the flow.
    """
    
    def __init__(
        self,
        decision_id: str,
        name: str,
        description: str,
        options: List[str],
        default_option: str = None
    ):
        """
        Initialize a decision.
        
        Args:
            decision_id: ID of the decision
            name: Name of the decision
            description: Description of the decision
            options: List of options
            default_option: Default option
        """
        self.decision_id = decision_id
        self.name = name
        self.description = description
        self.options = options
        self.default_option = default_option or options[0] if options else None
    
    def decide(
        self, 
        context: Dict[str, Any]
    ) -> str:
        """
        Make a decision.
        
        Args:
            context: Context information
            
        Returns:
            str: Selected option
        """
        # In a real implementation, this would use the context to make a decision.
        # For demonstration purposes, we'll just return the default option.
        return self.default_option
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision_id": self.decision_id,
            "name": self.name,
            "description": self.description,
            "options": self.options,
            "default_option": self.default_option
        }

class DecisionFlow:
    """
    Represents a decision flow for bug fixing.
    """
    
    def __init__(
        self,
        flow_id: str,
        name: str,
        description: str,
        nodes: Dict[str, FlowNode] = None,
        decisions: Dict[str, Decision] = None,
        start_node: str = None
    ):
        """
        Initialize a decision flow.
        
        Args:
            flow_id: ID of the flow
            name: Name of the flow
            description: Description of the flow
            nodes: Dictionary of nodes
            decisions: Dictionary of decisions
            start_node: ID of the start node
        """
        self.flow_id = flow_id
        self.name = name
        self.description = description
        self.nodes = nodes or {}
        self.decisions = decisions or {}
        self.start_node = start_node
    
    def execute(
        self, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the decision flow.
        
        Args:
            context: Context information
            
        Returns:
            Dict[str, Any]: Result of the execution
        """
        if not self.start_node:
            logger.error("No start node defined")
            return {
                "success": False,
                "error": "No start node defined"
            }
        
        # Initialize execution context
        execution_context = context.copy()
        execution_context["flow_id"] = self.flow_id
        execution_context["current_node"] = self.start_node
        execution_context["visited_nodes"] = []
        execution_context["decisions"] = {}
        
        # Execute nodes
        current_node_id = self.start_node
        max_iterations = 100  # Prevent infinite loops
        
        for i in range(max_iterations):
            if not current_node_id:
                logger.info("Flow completed")
                break
            
            current_node = self.nodes.get(current_node_id)
            
            if not current_node:
                logger.error(f"Node {current_node_id} not found")
                return {
                    "success": False,
                    "error": f"Node {current_node_id} not found"
                }
            
            logger.info(f"Executing node {current_node_id}: {current_node.name}")
            
            # Record visited node
            execution_context["visited_nodes"].append(current_node_id)
            execution_context["current_node"] = current_node_id
            
            # Execute node
            next_node_id = current_node.execute(execution_context)
            
            # Check if node returned a decision
            if next_node_id and next_node_id.startswith("decision:"):
                decision_id = next_node_id[9:]  # Remove "decision:" prefix
                decision = self.decisions.get(decision_id)
                
                if not decision:
                    logger.error(f"Decision {decision_id} not found")
                    return {
                        "success": False,
                        "error": f"Decision {decision_id} not found"
                    }
                
                logger.info(f"Making decision {decision_id}: {decision.name}")
                
                # Make decision
                option = decision.decide(execution_context)
                
                # Record decision
                execution_context["decisions"][decision_id] = option
                
                # Set next node based on decision
                next_node_id = option
            
            current_node_id = next_node_id
        
        if i == max_iterations - 1:
            logger.warning("Maximum iterations reached, flow may be stuck in a loop")
        
        return {
            "success": True,
            "context": execution_context
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "description": self.description,
            "start_node": self.start_node,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "decisions": {decision_id: decision.to_dict() for decision_id, decision in self.decisions.items()}
        }

# Default decision flow
def get_decision_flow(config: Dict[str, Any] = None) -> DecisionFlow:
    """
    Get the default decision flow.
    
    Args:
        config: Configuration options
        
    Returns:
        DecisionFlow: Default decision flow
    """
    logger.info("Creating default decision flow")
    
    # Define nodes
    nodes = {
        "start": FlowNode(
            node_id="start",
            name="Start",
            description="Start of the flow",
            next_nodes=["identify_bug"]
        ),
        "identify_bug": FlowNode(
            node_id="identify_bug",
            name="Identify Bug",
            description="Identify the bug",
            action=lambda context: "generate_paths",
            next_nodes=["generate_paths"]
        ),
        "generate_paths": FlowNode(
            node_id="generate_paths",
            name="Generate Solution Paths",
            description="Generate solution paths",
            action=lambda context: "select_path",
            next_nodes=["select_path"]
        ),
        "select_path": FlowNode(
            node_id="select_path",
            name="Select Path",
            description="Select the best solution path",
            action=lambda context: "generate_patch",
            next_nodes=["generate_patch"]
        ),
        "generate_patch": FlowNode(
            node_id="generate_patch",
            name="Generate Patch",
            description="Generate a patch",
            action=lambda context: "apply_patch",
            next_nodes=["apply_patch"]
        ),
        "apply_patch": FlowNode(
            node_id="apply_patch",
            name="Apply Patch",
            description="Apply the patch",
            action=lambda context: "verify_fix",
            next_nodes=["verify_fix"]
        ),
        "verify_fix": FlowNode(
            node_id="verify_fix",
            name="Verify Fix",
            description="Verify the fix",
            action=lambda context: "decision:fix_successful",
            next_nodes=["decision:fix_successful"]
        ),
        "end_success": FlowNode(
            node_id="end_success",
            name="End (Success)",
            description="End of the flow (success)",
            next_nodes=[]
        ),
        "end_failure": FlowNode(
            node_id="end_failure",
            name="End (Failure)",
            description="End of the flow (failure)",
            next_nodes=[]
        )
    }
    
    # Define decisions
    decisions = {
        "fix_successful": Decision(
            decision_id="fix_successful",
            name="Fix Successful?",
            description="Was the fix successful?",
            options=["end_success", "end_failure"],
            default_option="end_success"
        )
    }
    
    # Create flow
    flow = DecisionFlow(
        flow_id="bug_fix_flow",
        name="Bug Fix Flow",
        description="Flow for fixing bugs",
        nodes=nodes,
        decisions=decisions,
        start_node="start"
    )
    
    logger.info("Default decision flow created")
    return flow

# API Functions

def process_bug(
    bug_id: str,
    context: Dict[str, Any] = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process a bug through the decision flow.
    
    Args:
        bug_id: ID of the bug to process
        context: Additional context information
        config: Configuration options
        
    Returns:
        Dict[str, Any]: Result of the processing
    """
    # Initialize context
    context = context or {}
    context["bug_id"] = bug_id
    
    # Get decision flow
    flow = get_decision_flow(config)
    
    # Execute flow
    result = flow.execute(context)
    
    if result.get("success", False):
        # Extract relevant information
        executed_context = result.get("context", {})
        visited_nodes = executed_context.get("visited_nodes", [])
        decisions = executed_context.get("decisions", {})
        
        # Determine current phase
        current_node = executed_context.get("current_node")
        
        # Determine next phase
        next_phase = None
        current_idx = visited_nodes.index(current_node) if current_node in visited_nodes else -1
        
        if current_idx < len(visited_nodes) - 1:
            next_phase = visited_nodes[current_idx + 1]
        
        return {
            "success": True,
            "bug_id": bug_id,
            "flow_id": flow.flow_id,
            "visited_nodes": visited_nodes,
            "decisions": decisions,
            "current_phase": current_node,
            "next_phase": next_phase
        }
    else:
        return result
