#!/usr/bin/env python3
"""
Advanced Intent Classification System for FixWurx

This module implements the core intent classification functionality for the FixWurx ecosystem,
with full integration into the existing infrastructure including Launchpad, Triangulum,
Neural Matrix, and the agent system.
"""

import os
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union

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
        type: str = "generic",
        execution_path: str = "planning",
        parameters: Optional[Dict[str, Any]] = None,
        required_agents: Optional[List[str]] = None,
        confidence: float = 0.0
    ):
        """Initialize an intent object."""
        self.query = query
        self.type = type
        self.execution_path = execution_path
        self.parameters = parameters or {}
        self.required_agents = required_agents or []
        self.confidence = confidence
        self.timestamp = time.time()
        self.context_references = {}  # References to context elements

    def to_dict(self) -> Dict[str, Any]:
        """Convert the intent to a dictionary."""
        return {
            "query": self.query,
            "type": self.type,
            "execution_path": self.execution_path,
            "parameters": self.parameters,
            "required_agents": self.required_agents,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "context_references": self.context_references
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """Create an intent from a dictionary."""
        intent = cls(
            query=data.get("query", ""),
            type=data.get("type", "generic"),
            execution_path=data.get("execution_path", "planning"),
            parameters=data.get("parameters", {}),
            required_agents=data.get("required_agents", []),
            confidence=data.get("confidence", 0.0)
        )
        intent.timestamp = data.get("timestamp", time.time())
        intent.context_references = data.get("context_references", {})
        return intent


class IntentClassificationSystem:
    """Advanced intent classification system for FixWurx."""

    def __init__(self, registry: Any):
        """Initialize the intent classification system."""
        self.registry = registry
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.pattern_matchers = self._initialize_pattern_matchers()
        self.semantic_keywords = self._initialize_semantic_keywords()
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.execution_paths = self._initialize_execution_paths()
        
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
                    r"(?i)what(?:'s| is) in (?:the )?(?:file|folder|directory)"
                ],
                "file_modification": [
                    r"(?i)(create|make|add).*(?:file|folder|directory)",
                    r"(?i)(modify|update|edit|change).*(?:file|folder|directory|code)",
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
                    r"(?i)(find|locate|identify).*(?:bug|error|issue|problem|warning)"
                ],
                "performance_optimization": [
                    r"(?i)(optimize|improve|enhance).*(?:performance|speed|memory|code|efficiency)",
                    r"(?i)(faster|quicker|speedier|snappier)",
                    r"(?i)(reduce|lower|decrease).*(?:latency|lag|resource usage|memory consumption)"
                ],
                "decision_tree": [
                    r"(?i)(use|run|execute).*(?:decision tree)",
                    r"(?i)(decide|determine).*(?:using|with).*(?:decision tree)"
                ],
                "script_execution": [
                    r"(?i)(run|execute|launch).*(?:script|\.fx|fx script)",
                    r"(?i)(create|generate).*(?:script|\.fx file) (?:for|to)"
                ],
                "agent_introspection": [
                    r"(?i)(how are you (doing|performing)|what('s| is) your (status|state))",
                    r"(?i)(check (your|agent) (health|status))",
                    r"(?i)(optimize|improve) (yourself|your performance)",
                    r"(?i)(analyze|diagnose) (yourself|your (state|performance))"
                ],
                "rotate_credentials": [
                    r"(?i)(rotate|update|change|refresh).*(?:credentials|keys|secrets|passwords)",
                    r"(?i)(secure|protect).*(?:credentials|keys|secrets|passwords)"
                ],
                "security_audit": [
                    r"(?i)(security audit|audit security)",
                    r"(?i)(check|verify|assess).*(?:security|vulnerabilities|threats)",
                    r"(?i)(secure|harden).*(?:system|application|code)"
                ],
                "bug_fix": [
                    r"(?i)(fix|repair|solve).*(?:bug|issue|problem|error)",
                    r"(?i)(found|identified|discovered).*(?:bug|issue|problem|error)"
                ]
            },
            "semantic_keywords": {
                "file_access": ["file", "folder", "directory", "path", "open", "read", "view", "list", "show", "display"],
                "file_modification": ["create", "modify", "update", "edit", "change", "delete", "remove", "write", "save"],
                "command_execution": ["run", "execute", "launch", "start", "command", "script", "program", "process", "stop", "kill", "end"],
                "system_debugging": ["debug", "diagnose", "analyze", "fix", "solve", "bug", "error", "issue", "problem", "warning"],
                "performance_optimization": ["optimize", "improve", "enhance", "performance", "speed", "memory", "efficiency", "faster", "quicker"],
                "security_audit": ["security", "audit", "vulnerability", "threat", "secure", "harden", "protect", "risk", "scan"],
                "bug_fix": ["bug", "fix", "issue", "problem", "error", "repair", "solve", "resolve"]
            },
            "specialist_intents": {
                "system_debugging": ["auditor", "analyst", "verifier"],
                "file_access": ["executor"],
                "file_modification": ["executor"],
                "command_execution": ["executor"],
                "performance_optimization": ["analyst", "optimizer"],
                "decision_tree": ["planner", "coordinator"],
                "script_execution": ["executor"],
                "agent_introspection": ["meta"],
                "rotate_credentials": ["auditor"],
                "security_audit": ["auditor", "security_specialist"],
                "bug_fix": ["analyst", "verifier"]
            },
            "fallback_mapping": {
                "analyst": ["debugger", "auditor"],
                "auditor": ["security_specialist", "meta"],
                "verifier": ["tester", "auditor"],
                "executor": ["command_handler", "process_manager"],
                "optimizer": ["analyst", "executor"],
                "planner": ["coordinator", "meta"],
                "coordinator": ["planner", "meta"],
                "security_specialist": ["auditor", "verifier"]
            },
            "execution_paths": {
                "direct": ["file_access", "file_modification", "command_execution", "agent_introspection", "rotate_credentials"],
                "agent_collaboration": ["system_debugging", "performance_optimization", "bug_fix", "security_audit"],
                "decision_tree": ["decision_tree"],
                "script_execution": ["script_execution"],
                "planning": ["generic"]
            }
        }

    def _initialize_pattern_matchers(self) -> Dict[str, re.Pattern]:
        """Initialize the pattern matchers for intent classification."""
        patterns = {}
        
        # Load patterns from config
        for intent_type, pattern_list in self.config["intent_patterns"].items():
            # Combine patterns with OR for each intent type
            combined_pattern = "|".join(f"({pattern})" for pattern in pattern_list)
            try:
                patterns[intent_type] = re.compile(combined_pattern)
            except re.error as e:
                logger.error(f"Error compiling pattern for {intent_type}: {e}")
                # Fallback to a simple pattern
                patterns[intent_type] = re.compile(r"(" + intent_type.replace("_", " ") + r")")
        
        return patterns

    def _initialize_semantic_keywords(self) -> Dict[str, List[str]]:
        """Initialize the semantic keywords for intent classification."""
        return self.config["semantic_keywords"]

    def _initialize_agent_capabilities(self) -> Dict[str, Set[str]]:
        """Initialize the agent capabilities mapping."""
        # This would normally query the agent system for agent capabilities
        # For now, we'll use a simple mapping
        return {
            "analyst": {"analyze", "diagnose", "inspect", "examine", "debug"},
            "auditor": {"audit", "monitor", "verify", "secure", "check"},
            "verifier": {"test", "verify", "validate", "check"},
            "executor": {"run", "execute", "launch", "stop", "restart"},
            "optimizer": {"optimize", "improve", "enhance", "speed up"},
            "planner": {"plan", "organize", "schedule", "coordinate"},
            "coordinator": {"coordinate", "orchestrate", "manage", "handle"},
            "meta": {"introspect", "self-analyze", "report", "status"},
            "security_specialist": {"audit", "secure", "protect", "harden"},
            "tester": {"test", "validate", "verify", "check"}
        }

    def _initialize_execution_paths(self) -> Dict[str, List[str]]:
        """Initialize the execution paths for different intent types."""
        return self.config["execution_paths"]

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

        # Check if neural matrix is available for advanced classification
        if self.neural_matrix and hasattr(self.neural_matrix, "process_intent"):
            try:
                # Attempt neural classification
                neural_result = self.neural_matrix.process_intent(query, context)

                # If successful and confident, use the neural result
                if (neural_result.get("success", False) and
                    neural_result.get("confidence", 0) >= self.config["confidence_threshold"]):

                    logger.info(f"Using neural classification for query: {query}")
                    intent = Intent(
                        query=query,
                        type=neural_result.get("type", "generic"),
                        execution_path=neural_result.get("execution_path", "planning"),
                        parameters=neural_result.get("parameters", {}),
                        required_agents=neural_result.get("required_agents", []),
                        confidence=neural_result.get("confidence", 0.0)
                    )
                    
                    # Process context references
                    if "context_references" in neural_result:
                        intent.context_references = neural_result["context_references"]
                    
                    return intent
            except Exception as e:
                logger.warning(f"Neural classification failed: {e}, falling back to pattern matching")

        # Check if triangulum is available for distributed classification under high load
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
        intent = self._local_classification(query, context)
        
        # Enhance intent with context awareness
        self._enhance_with_context_awareness(intent, context)
        
        # Determine required agents
        if not intent.required_agents:
            intent.required_agents = self.determine_required_agents(intent)
            
        return intent

    def _local_classification(self, query: str, context: Dict[str, Any]) -> Intent:
        """Perform local pattern-based and semantic classification."""
        # Pattern matching first (faster and more precise for clear intents)
        for intent_type, pattern in self.pattern_matchers.items():
            if pattern.search(query):
                # Pattern match found, extract parameters
                parameters = self._extract_parameters(query, intent_type)
                
                # Determine execution path
                execution_path = self._determine_execution_path(intent_type)
                
                # Create and return the intent
                return Intent(
                    query=query,
                    type=intent_type,
                    execution_path=execution_path,
                    parameters=parameters,
                    confidence=0.9  # High confidence for pattern matches
                )
        
        # If no pattern match, try semantic analysis
        intent_type, confidence = self._semantic_analysis(query)
        
        # Create the intent with semantic analysis results
        parameters = self._extract_parameters(query, intent_type)
        execution_path = self._determine_execution_path(intent_type)
        
        return Intent(
            query=query,
            type=intent_type,
            execution_path=execution_path,
            parameters=parameters,
            confidence=confidence
        )

    def _semantic_analysis(self, query: str) -> Tuple[str, float]:
        """
        Perform semantic analysis to determine intent type.
        
        Returns:
            Tuple of (intent_type, confidence)
        """
        # Convert query to lowercase and tokenize
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Track matches for each intent type
        intent_matches = {}
        
        # Count keyword matches for each intent type
        for intent_type, keywords in self.semantic_keywords.items():
            keywords_set = set(keywords)
            matches = query_words.intersection(keywords_set)
            
            if matches:
                # Calculate confidence based on number of matches and keyword set size
                match_ratio = len(matches) / len(keywords_set)
                coverage_ratio = len(matches) / max(1, len(query_words))
                
                # Weighted confidence score
                confidence = 0.4 + (match_ratio * 0.3) + (coverage_ratio * 0.3)
                
                intent_matches[intent_type] = confidence
        
        # Return the intent type with highest confidence, or generic if none
        if intent_matches:
            best_intent = max(intent_matches.items(), key=lambda x: x[1])
            return best_intent[0], best_intent[1]
        
        return "generic", 0.3  # Default to generic intent with low confidence

    def _extract_parameters(self, query: str, intent_type: str) -> Dict[str, Any]:
        """Extract parameters from the query based on the intent type."""
        parameters = {}

        # Handle file operations
        if intent_type in ["file_access", "file_modification"]:
            # Look for quoted paths
            path_match = re.search(r'[\'"](.*?)[\'"]', query)
            if path_match:
                parameters["path"] = path_match.group(1)
            else:
                # Look for paths in typical formats
                path_match = re.search(r'(?:in|at|to|from|path)\s+(?:/\S+|\w+(?:\.\w+)?|(?:/\w+)+)', query)
                if path_match:
                    parameters["path"] = path_match.group(0).split()[-1]
                else:
                    # Last resort - look for words that might be filenames
                    filename_match = re.search(r'\b(\w+\.\w+)\b', query)
                    if filename_match:
                        parameters["path"] = filename_match.group(1)

        # Handle command execution
        if intent_type == "command_execution":
            # Look for commands in backticks
            cmd_match = re.search(r'`(.*?)`', query)
            if cmd_match:
                cmd_parts = cmd_match.group(1).split()
                parameters["command"] = cmd_parts[0]
                if len(cmd_parts) > 1:
                    parameters["args"] = cmd_parts[1:]
            else:
                # Look for the command name
                cmd_match = re.search(r'(?:run|execute|launch|start)\s+(\w+)', query, re.IGNORECASE)
                if cmd_match:
                    parameters["command"] = cmd_match.group(1)
                
                # Look for arguments
                arg_match = re.search(r'with\s+(?:args|arguments)\s+(.*?)(?:$|and|then)', query, re.IGNORECASE)
                if arg_match:
                    args_str = arg_match.group(1)
                    # Split args but respect quoted strings
                    parameters["args"] = re.findall(r'[^\s"\']+|"([^"]*)"|\'([^\']*)\'', args_str)

        # Handle system debugging
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

        # Handle performance optimization
        if intent_type == "performance_optimization":
            # Look for specific optimization targets
            targets = ["memory", "speed", "cpu", "disk", "network", "latency", "throughput"]
            for target in targets:
                if target in query.lower():
                    parameters["optimization_target"] = target
                    break

        # Handle security audit
        if intent_type == "security_audit":
            # Look for specific security focus areas
            focus_areas = ["authentication", "authorization", "encryption", "input validation", "api", "database", "network"]
            for area in focus_areas:
                if area in query.lower():
                    parameters["security_focus"] = area
                    break

        return parameters

    def _determine_execution_path(self, intent_type: str) -> str:
        """Determine the execution path for an intent type."""
        # Check each execution path
        for path, intent_types in self.execution_paths.items():
            if intent_type in intent_types:
                return path
        
        # Default to planning for unknown intent types
        return "planning"

    def _enhance_with_context_awareness(self, intent: Intent, context: Dict[str, Any]) -> None:
        """
        Enhance intent with context awareness, including reference resolution
        and parameter extraction from context.
        """
        # Skip if no context or context has no history
        if not context or "history" not in context or not context["history"]:
            return
        
        # Extract conversation history
        history = context["history"]
        
        # Initialize context references
        intent.context_references = {}
        
        # Look for file references in conversation history
        file_references = []
        for item in reversed(history):  # Start with most recent
            if item.get("role") != "user":  # Only look at system messages
                continue
                
            content = item.get("content", "")
            
            # Look for file paths in the content
            path_matches = re.findall(r'(?:/[\w/.-]+|[\w.-]+\.\w+)', content)
            file_references.extend(path_matches)
        
        # Store file references if found
        if file_references:
            intent.context_references["file_references"] = file_references
            
            # If the intent is file related but has no path parameter,
            # use the most recent file reference
            if (intent.type in ["file_access", "file_modification"] and 
                "path" not in intent.parameters and file_references):
                intent.parameters["path"] = file_references[0]
        
        # Look for command references in conversation history
        command_references = []
        for item in reversed(history):
            if item.get("role") != "system":
                continue
                
            content = item.get("content", "")
            
            # Look for command execution results
            cmd_matches = re.findall(r'Executed command: `(.*?)`', content)
            command_references.extend(cmd_matches)
        
        # Store command references if found
        if command_references:
            intent.context_references["command_references"] = command_references
            
            # If the intent is command related but has no command parameter,
            # use the most recent command reference
            if (intent.type == "command_execution" and 
                "command" not in intent.parameters and command_references):
                # Parse the command and args
                cmd_parts = command_references[0].split()
                intent.parameters["command"] = cmd_parts[0]
                if len(cmd_parts) > 1:
                    intent.parameters["args"] = cmd_parts[1:]

    def determine_required_agents(self, intent: Intent) -> List[str]:
        """
        Determine the agents required to handle an intent.

        Args:
            intent: The classified intent

        Returns:
            A list of required agent types
        """
        # Start with the specialists for this intent type if available
        required_agents = self.config["specialist_intents"].get(intent.type, []).copy()

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
