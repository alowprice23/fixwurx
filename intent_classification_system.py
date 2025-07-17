#!/usr/bin/env python3
"""
Advanced Intent Classification System for FixWurx

This module implements the core intent classification functionality for the FixWurx ecosystem,
with full integration into the existing infrastructure including Launchpad, Triangulum,
Neural Matrix, and the agent system.
"""

import os
import sys
import json
import logging
import importlib
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_classification.log"), logging.StreamHandler()]
)
logger = logging.getLogger("IntentClassificationSystem")

class Intent:
    """Class representing a classified user intent."""
    
    def __init__(
        self,
        query: str,
        type: str,
        execution_path: str,
        parameters: Dict[str, Any],
        required_agents: Optional[List[str]] = None,
        confidence: float = 0.0
    ):
        """Initialize an intent object."""
        self.query = query
        self.type = type
        self.execution_path = execution_path
        self.parameters = parameters
        self.required_agents = required_agents or []
        self.confidence = confidence
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the intent to a dictionary."""
        return {
            "query": self.query,
            "type": self.type,
            "execution_path": self.execution_path,
            "parameters": self.parameters,
            "required_agents": self.required_agents,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """Create an intent from a dictionary."""
        return cls(
            query=data.get("query", ""),
            type=data.get("type", "generic"),
            execution_path=data.get("execution_path", "planning"),
            parameters=data.get("parameters", {}),
            required_agents=data.get("required_agents", []),
            confidence=data.get("confidence", 0.0)
        )

class IntentClassificationSystem:
    """Advanced intent classification system for FixWurx."""
    
    def __init__(self, registry: Any):
        """Initialize the intent classification system."""
        self.registry = registry
        self.pattern_matchers = self._initialize_pattern_matchers()
        self.agent_capabilities = self._initialize_agent_capabilities()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize neural integration if available
        self.neural_matrix = registry.get_component("neural_matrix")
        if self.neural_matrix:
            logger.info("Neural Matrix integration available")
        else:
            logger.warning("Neural Matrix not available - using basic classification only")
        
        # Initialize triangulum integration if available
        self.triangulum_client = registry.get_component("triangulum_client")
        if self.triangulum_client:
            logger.info("Triangulum integration available for distributed processing")
        else:
            logger.warning("Triangulum not available - using local processing only")
        
        logger.info("Intent Classification System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the intent classification configuration."""
        config_path = "config/intent_system_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Default configuration
        return {
            "confidence_threshold": 0.7,
            "distribution_threshold": 0.8,
            "intent_patterns": {
                "file_access": [
                    r"(?i)(show|list|display|open|read|view).*(?:file|folder|directory)",
                    r"(?i)(create|make|add).*(?:file|folder|directory)",
                    r"(?i)(modify|update|edit|change).*(?:file|folder|directory)",
                    r"(?i)(delete|remove|drop).*(?:file|folder|directory)"
                ],
                "command_execution": [
                    r"(?i)(run|execute|launch|start).*(?:command|script|program|app|application)",
                    r"(?i)(stop|kill|end|terminate).*(?:command|script|program|process)",
                    r"(?i)(restart|relaunch).*(?:command|script|program|service|app)"
                ],
                "system_debugging": [
                    r"(?i)(debug|diagnose|analyze|check|test).*(?:issue|problem|bug|error|warning|code)",
                    r"(?i)(fix|solve|resolve|repair).*(?:issue|problem|bug|error)",
                    r"(?i)(find|locate|identify).*(?:bug|error|issue|problem|warning)",
                    r"(?i)(optimize|improve|enhance).*(?:performance|speed|memory|code)"
                ],
                "data_analysis": [
                    r"(?i)(analyze|examine|study|investigate).*(?:data|dataset|information|results)",
                    r"(?i)(visualize|plot|chart|graph).*(?:data|results|output)",
                    r"(?i)(calculate|compute|find).*(?:average|mean|median|total|sum)"
                ],
                "deployment": [
                    r"(?i)(deploy|publish|release|ship).*(?:app|application|service|update)",
                    r"(?i)(rollback|revert).*(?:deployment|update|release)",
                    r"(?i)(configure|setup|install).*(?:service|app|system|tool|library)"
                ]
            },
            "semantic_keywords": {
                "file_access": ["file", "folder", "directory", "path", "open", "read", "write", "create", "list", "show"],
                "command_execution": ["run", "execute", "launch", "start", "command", "script", "program", "process"],
                "system_debugging": ["debug", "diagnose", "analyze", "fix", "solve", "bug", "error", "issue", "problem"],
                "data_analysis": ["analyze", "data", "statistics", "metrics", "average", "total", "graph", "plot"],
                "deployment": ["deploy", "publish", "release", "install", "setup", "configure", "update"]
            },
            "specialist_intents": {
                "system_debugging": ["analyzer", "debugger", "fixer"],
                "file_access": ["file_handler", "storage_manager"],
                "command_execution": ["executor", "process_manager"],
                "data_analysis": ["analyzer", "visualizer"],
                "deployment": ["deployer", "configurator"]
            },
            "fallback_mapping": {
                "analyzer": ["auditor", "debugger"],
                "debugger": ["developer", "analyzer"],
                "fixer": ["developer", "tester"],
                "file_handler": ["storage_manager", "executor"],
                "executor": ["command_handler", "process_manager"],
                "process_manager": ["executor", "system_manager"],
                "deployer": ["configurator", "developer"],
                "configurator": ["deployer", "system_manager"]
            },
            "execution_paths": {
                "direct": ["file_access", "command_execution"],
                "agent_collaboration": ["system_debugging", "data_analysis"],
                "planning": ["deployment", "generic"]
            }
        }
    
    def _initialize_pattern_matchers(self) -> Dict[str, List[Any]]:
        """Initialize the pattern matchers for intent classification."""
        # In production this would load more sophisticated pattern matchers
        # For now, we'll use simple regex patterns loaded from config
        return {}
    
    def _initialize_agent_capabilities(self) -> Dict[str, Set[str]]:
        """Initialize the agent capabilities mapping."""
        # This would normally query the agent system for agent capabilities
        # For now, we'll use a simple mapping
        return {
            "analyzer": {"analyze", "diagnose", "inspect", "examine"},
            "debugger": {"debug", "fix", "solve", "repair"},
            "developer": {"code", "implement", "modify", "write"},
            "tester": {"test", "verify", "validate", "check"},
            "file_handler": {"read", "write", "copy", "move", "delete"},
            "storage_manager": {"store", "retrieve", "backup", "archive"},
            "executor": {"run", "execute", "launch", "stop", "restart"},
            "process_manager": {"monitor", "track", "kill", "prioritize"},
            "planner": {"plan", "organize", "schedule", "coordinate"},
            "deployer": {"deploy", "publish", "release", "rollback"},
            "configurator": {"configure", "setup", "install", "customize"}
        }
    
    def classify_intent(self, query: str, context: Dict[str, Any] = None) -> Intent:
        """
        Classify the user's intent based on the query and context.
        
        Args:
            query: The user's query
            context: Additional context information
            
        Returns:
            An Intent object representing the classified intent
        """
        # Log the classification attempt
        logger.debug(f"Classifying intent for query: {query}")
        
        # Initialize context if not provided
        context = context or {}
        
        # Check if neural matrix is available and system load is appropriate for neural enhancement
        if self.neural_matrix and hasattr(self.neural_matrix, "process_intent"):
            try:
                # Attempt neural classification
                neural_result = self.neural_matrix.process_intent(query, context)
                
                # If successful and confident, use the neural result
                if (neural_result.get("success", False) and 
                    neural_result.get("confidence", 0) >= self.config["confidence_threshold"]):
                    
                    logger.info(f"Using neural classification for query: {query}")
                    return Intent(
                        query=query,
                        type=neural_result.get("type", "generic"),
                        execution_path=neural_result.get("execution_path", "planning"),
                        parameters=neural_result.get("parameters", {}),
                        required_agents=neural_result.get("required_agents", []),
                        confidence=neural_result.get("confidence", 0.0)
                    )
            except Exception as e:
                logger.warning(f"Neural classification failed: {e}, falling back to pattern matching")
        
        # Check if triangulum is available and system load is high
        if (self.triangulum_client and 
            context.get("system_load", 0) > self.config["distribution_threshold"] and
            hasattr(self.triangulum_client, "submit_job")):
            try:
                # Attempt distributed classification
                logger.info(f"Using distributed classification for query: {query}")
                job_id = self.triangulum_client.submit_job(
                    "intent_classification",
                    {"query": query, "context": context}
                )
                
                # Wait for the result
                result = self.triangulum_client.get_job_result(job_id)
                
                if result.get("success", False):
                    # Use the distributed result
                    return Intent.from_dict(result.get("result", {}))
            except Exception as e:
                logger.warning(f"Distributed classification failed: {e}, falling back to local classification")
        
        # Fall back to local pattern-based and semantic classification
        return self._local_classification(query, context)
    
    def _local_classification(self, query: str, context: Dict[str, Any]) -> Intent:
        """Perform local pattern-based and semantic classification."""
        # Check each intent type using regex patterns
        max_confidence = 0.0
        intent_type = "generic"
        
        # Pattern matching
        for type_name, patterns in self.config["intent_patterns"].items():
            for pattern in patterns:
                if re.search(pattern, query):
                    confidence = 0.8  # Base confidence for pattern match
                    
                    # Adjust confidence based on context
                    if context.get("previous_intent", {}).get("type") == type_name:
                        confidence += 0.1  # Boost confidence for consistent intent types
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        intent_type = type_name
        
        # Semantic analysis as backup
        if intent_type == "generic":
            # Analyze the query for keywords
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            
            for type_name, keywords in self.config["semantic_keywords"].items():
                # Count the number of matching keywords
                matches = query_words.intersection(set(keywords))
                
                if matches:
                    confidence = 0.5 + (len(matches) / len(keywords)) * 0.3
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        intent_type = type_name
        
        # Determine execution path
        execution_path = "planning"  # Default
        for path, intent_types in self.config["execution_paths"].items():
            if intent_type in intent_types:
                execution_path = path
                break
        
        # Extract parameters based on intent type
        parameters = self._extract_parameters(query, intent_type)
        
        # Determine required agents
        required_agents = self.determine_required_agents_for_type(intent_type, parameters)
        
        return Intent(
            query=query,
            type=intent_type,
            execution_path=execution_path,
            parameters=parameters,
            required_agents=required_agents,
            confidence=max_confidence
        )
    
    def _extract_parameters(self, query: str, intent_type: str) -> Dict[str, Any]:
        """Extract parameters from the query based on the intent type."""
        parameters = {}
        
        # Extract file/folder paths
        if intent_type == "file_access" or intent_type == "system_debugging":
            # Look for quoted paths
            path_match = re.search(r'[\'"](.*?)[\'"]', query)
            if path_match:
                parameters["path"] = path_match.group(1)
            else:
                # Look for words that might be file or folder names
                words = re.findall(r'\b(\w+\.\w+|\w+(?=\s+folder|\s+file|\s+directory))\b', query)
                if words:
                    parameters["path"] = words[0]
        
        # Extract command parameters
        if intent_type == "command_execution":
            # Look for the command name
            cmd_match = re.search(r'(run|execute|launch|start)\s+(\w+)', query, re.IGNORECASE)
            if cmd_match:
                parameters["command"] = cmd_match.group(2)
            
            # Look for arguments
            arg_match = re.search(r'with\s+(?:args|arguments)\s+(.*?)(?:$|and|then)', query, re.IGNORECASE)
            if arg_match:
                parameters["arguments"] = arg_match.group(1)
        
        # Extract debugging targets
        if intent_type == "system_debugging":
            # Look for specific bug types
            bug_types = ["syntax", "logic", "performance", "security", "memory"]
            for bug_type in bug_types:
                if bug_type in query.lower():
                    parameters["bug_type"] = bug_type
                    break
            
            # Look for specific files or components
            component_match = re.search(r'(?:in|of|for)\s+(?:the\s+)?(\w+)(?:\s+component|\s+module|\s+class|\s+file)?', query, re.IGNORECASE)
            if component_match:
                parameters["component"] = component_match.group(1)
        
        return parameters
    
    def determine_required_agents(self, intent: Intent) -> List[str]:
        """
        Determine the agents required to handle an intent.
        
        Args:
            intent: The classified intent
            
        Returns:
            A list of required agent types
        """
        # Start with the specialists for this intent type if available
        required_agents = self.determine_required_agents_for_type(intent.type, intent.parameters)
        
        # Check if any additional agents are needed based on the execution path
        if intent.execution_path == "planning" and "planner" not in required_agents:
            required_agents.append("planner")
            
        elif intent.execution_path == "agent_collaboration":
            # For collaboration, ensure we have at least a coordinator
            if "coordinator" not in required_agents and len(required_agents) < 2:
                required_agents.append("coordinator")
        
        # If there are still no required agents, add a default executor
        if not required_agents:
            required_agents.append("executor")
        
        return required_agents
    
    def determine_required_agents_for_type(self, intent_type: str, parameters: Dict[str, Any]) -> List[str]:
        """Determine the required agents based on the intent type and parameters."""
        # Get specialists for this intent type
        specialists = self.config["specialist_intents"].get(intent_type, [])
        
        # Add additional specialists based on parameters
        if intent_type == "system_debugging":
            bug_type = parameters.get("bug_type")
            if bug_type == "performance":
                specialists.append("optimizer")
            elif bug_type == "security":
                specialists.append("security_auditor")
        
        return specialists
    
    def handle_agent_failure(self, intent: Intent, failed_agents: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Handle agent failures by finding appropriate replacements.
        
        Args:
            intent: The intent being processed
            failed_agents: List of agents that failed
            
        Returns:
            A tuple of (updated agent list, fallback mapping)
        """
        # Copy the original required agents
        updated_agents = intent.required_agents.copy()
        fallback_mapping = {}
        
        # Process each failed agent
        for failed_agent in failed_agents:
            # Check if we have fallbacks for this agent type
            fallbacks = self.config["fallback_mapping"].get(failed_agent, [])
            
            if fallbacks:
                # Find the first available fallback that's not already in use
                for fallback in fallbacks:
                    if fallback not in updated_agents and fallback not in failed_agents:
                        # Replace the failed agent with the fallback
                        updated_agents = [fallback if a == failed_agent else a for a in updated_agents]
                        fallback_mapping[failed_agent] = fallback
                        logger.info(f"Replacing failed agent {failed_agent} with {fallback}")
                        break
            
            # If no fallback was found, remove the failed agent
            if failed_agent not in fallback_mapping:
                updated_agents = [a for a in updated_agents if a != failed_agent]
                logger.warning(f"No fallback found for failed agent {failed_agent}, removing from required agents")
        
        return updated_agents, fallback_mapping

def initialize_system(registry: Any) -> IntentClassificationSystem:
    """
    Initialize the intent classification system.
    
    Args:
        registry: The component registry to use for dependency injection
        
    Returns:
        An initialized IntentClassificationSystem instance
    """
    # Create the intent classification system
    intent_system = IntentClassificationSystem(registry)
    
    # Register it with the registry
    if hasattr(registry, "register_component"):
        registry.register_component("intent_classification_system", intent_system)
    
    return intent_system

if __name__ == "__main__":
    print("Intent Classification System - This module should be imported, not run directly.")
