#!/usr/bin/env python3
"""
FixWurx Intent Classification Integration

This module provides the integration layer between the intent classification system
and the FixWurx ecosystem, including the shell environment, agent system, and other
core components.
"""

import os
import sys
import json
import logging
import importlib
import time
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_integration.log"), logging.StreamHandler()]
)
logger = logging.getLogger("FixWurxIntentIntegration")

# Import intent system components
from intent_classification_system import IntentClassificationSystem, Intent, initialize_system
from components.intent_caching_system import IntentOptimizationSystem, create_context_hash

class IntentIntegrationSystem:
    """
    Integration system for the intent classification functionality with FixWurx.
    
    This class serves as the bridge between the intent classification system and
    the rest of the FixWurx ecosystem, including the shell environment, agent system,
    neural matrix, and triangulum.
    """
    
    def __init__(self, registry: Any, config_path: Optional[str] = None):
        """
        Initialize the intent integration system.
        
        Args:
            registry: The component registry to use for dependency injection
            config_path: Optional path to a configuration file
        """
        self.registry = registry
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize intent classification system
        self.intent_system = initialize_system(registry)
        
        # Initialize intent optimization system
        self.optimization_system = IntentOptimizationSystem(
            cache_capacity=self.config.get("cache_capacity", 100),
            history_size=self.config.get("history_size", 50),
            window_size=self.config.get("window_size", 10)
        )
        
        # Register optimization system with registry
        if hasattr(registry, "register_component"):
            registry.register_component("intent_optimization_system", self.optimization_system)
        
        # Track conversation history for context
        self.conversation_history = []
        
        # Store the last classified intent for context
        self.last_intent = None
        
        # Initialize connected flag
        self.connected_to_shell = False
        self.connected_to_agents = False
        
        logger.info("Intent Integration System initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load the intent integration configuration."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Default configuration path
        default_path = "config/intent_system_config.json"
        if os.path.exists(default_path):
            try:
                with open(default_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded configuration from {default_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Fallback to default configuration
        logger.warning("Using default configuration")
        return {
            "cache_capacity": 100,
            "history_size": 50,
            "window_size": 10,
            "shell_integration": {
                "enabled": True,
                "command_prefix": "!intent",
                "intercept_all": True
            },
            "agent_integration": {
                "enabled": True,
                "fallback_enabled": True
            },
            "neural_integration": {
                "enabled": True,
                "confidence_threshold": 0.7
            },
            "triangulum_integration": {
                "enabled": True,
                "distribution_threshold": 0.8
            }
        }
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the intent classification system.
        
        Args:
            query: The user's query
            context: Additional context information
            
        Returns:
            The processed result
        """
        # Initialize context if not provided
        context = context or {}
        
        # Add conversation history to context if available
        if self.conversation_history and "history" not in context:
            context["history"] = self.conversation_history
        
        # Add last intent to context if available
        if self.last_intent and "previous_intent" not in context:
            context["previous_intent"] = self.last_intent.to_dict()
        
        # Create context hash for caching
        context_hash = create_context_hash(context)
        
        # Check cache first
        cached_intent = self.optimization_system.get_cached_intent(query, context_hash)
        if cached_intent:
            logger.info(f"Using cached intent for query: {query}")
            intent = Intent.from_dict(cached_intent)
        else:
            # Classify the intent
            intent = self.intent_system.classify_intent(query, context)
            
            # Cache the result for future use
            self.optimization_system.cache_intent(query, intent.to_dict(), context_hash)
        
        # Record this intent in the history
        self.optimization_system.record_intent_sequence(intent.type)
        
        # Store as the last intent for future context
        self.last_intent = intent
        
        # Execute the intent through the appropriate system
        result = self._execute_intent(intent, context)
        
        # Record the interaction in conversation history
        self._record_interaction(query, intent, result)
        
        return result
    
    def _execute_intent(self, intent: Intent, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an intent through the appropriate system based on its type and execution path.
        
        Args:
            intent: The intent to execute
            context: The context for execution
            
        Returns:
            The execution result
        """
        # Different execution paths based on the intent type
        if intent.execution_path == "direct":
            # Direct execution for simple intents
            return self._execute_direct(intent, context)
            
        elif intent.execution_path == "agent_collaboration":
            # Collaborative execution using multiple agents
            return self._execute_collaborative(intent, context)
            
        elif intent.execution_path == "planning":
            # Complex execution requiring planning
            return self._execute_with_planning(intent, context)
            
        else:
            # Default fallback for unknown execution paths
            logger.warning(f"Unknown execution path: {intent.execution_path}, falling back to direct execution")
            return self._execute_direct(intent, context)
    
    def _execute_direct(self, intent: Intent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an intent directly without complex agent interactions."""
        # For file access intents
        if intent.type == "file_access":
            return self._handle_file_access(intent, context)
            
        # For command execution intents
        elif intent.type == "command_execution":
            return self._handle_command_execution(intent, context)
            
        # For other direct execution intents
        else:
            # Try to find a handler in the registry
            handler = self.registry.get_component(f"{intent.type}_handler")
            if handler and hasattr(handler, "handle_intent"):
                return handler.handle_intent(intent, context)
            
            # Fallback to generic response
            return {
                "success": False,
                "error": f"No handler found for intent type: {intent.type}",
                "intent": intent.to_dict()
            }
    
    def _execute_collaborative(self, intent: Intent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an intent using multiple collaborating agents."""
        # Get the required agents
        required_agents = self.intent_system.determine_required_agents(intent)
        
        # Get the agent system
        agent_system = self.registry.get_component("agent_system")
        if not agent_system:
            logger.error("Agent system not available for collaborative execution")
            return {
                "success": False,
                "error": "Agent system not available",
                "intent": intent.to_dict()
            }
        
        # Request agent collaboration
        if hasattr(agent_system, "collaborate"):
            result = agent_system.collaborate(
                intent.query,
                intent.to_dict(),
                required_agents,
                context
            )
            return result
        
        # Fallback to coordinator if collaborate method not available
        coordinator = self.registry.get_component("agent_coordinator")
        if coordinator and hasattr(coordinator, "coordinate_task"):
            result = coordinator.coordinate_task(
                intent.query,
                intent.to_dict(),
                required_agents,
                context
            )
            return result
        
        # Last fallback to direct execution
        logger.warning("No agent collaboration capability found, falling back to direct execution")
        return self._execute_direct(intent, context)
    
    def _execute_with_planning(self, intent: Intent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an intent that requires planning before execution."""
        # Get the planning engine
        planning_engine = self.registry.get_component("planning_engine")
        if not planning_engine:
            logger.error("Planning engine not available for planning execution")
            return {
                "success": False,
                "error": "Planning engine not available",
                "intent": intent.to_dict()
            }
        
        # Generate a plan
        if hasattr(planning_engine, "generate_plan"):
            plan = planning_engine.generate_plan(intent.query, context)
            
            # Return the plan without execution if requested
            if context.get("plan_only", False):
                return {
                    "success": True,
                    "plan": plan,
                    "intent": intent.to_dict(),
                    "executed": False
                }
            
            # Execute the plan
            if hasattr(planning_engine, "execute_plan"):
                result = planning_engine.execute_plan(plan, context)
                result["intent"] = intent.to_dict()
                return result
            
            # Fallback to agent system if execute_plan not available
            agent_system = self.registry.get_component("agent_system")
            if agent_system and hasattr(agent_system, "execute_plan"):
                result = agent_system.execute_plan(plan, context)
                result["intent"] = intent.to_dict()
                return result
            
            # If no execution is possible, just return the plan
            return {
                "success": True,
                "plan": plan,
                "intent": intent.to_dict(),
                "executed": False,
                "warning": "Plan generated but no execution method available"
            }
        
        # Fallback to collaborative execution if planning not available
        logger.warning("Planning engine doesn't support plan generation, falling back to collaborative execution")
        return self._execute_collaborative(intent, context)
    
    def _handle_file_access(self, intent: Intent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file access intents."""
        # Get the file access utility
        file_access = self.registry.get_component("file_access_utility")
        if not file_access:
            logger.error("File access utility not available")
            return {
                "success": False,
                "error": "File access utility not available",
                "intent": intent.to_dict()
            }
        
        # Extract parameters
        path = intent.parameters.get("path", "")
        
        # Determine the file operation based on keywords in the query
        query_lower = intent.query.lower()
        
        if any(op in query_lower for op in ["read", "show", "display", "view", "open"]):
            # Read file
            if hasattr(file_access, "read_file"):
                return file_access.read_file(path)
            
        elif any(op in query_lower for op in ["write", "create", "add", "make"]):
            # Write file
            content = intent.parameters.get("content", "")
            if hasattr(file_access, "write_file"):
                return file_access.write_file(path, content)
            
        elif any(op in query_lower for op in ["delete", "remove", "drop"]):
            # Delete file
            if hasattr(file_access, "delete_file"):
                return file_access.delete_file(path)
            
        elif any(op in query_lower for op in ["list", "dir", "ls", "directory"]):
            # List directory
            recursive = "recursive" in query_lower or "all" in query_lower
            if hasattr(file_access, "list_directory"):
                return file_access.list_directory(path, recursive)
        
        # Fallback for unknown file operation
        return {
            "success": False,
            "error": f"Unknown file operation for path: {path}",
            "intent": intent.to_dict()
        }
    
    def _handle_command_execution(self, intent: Intent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command execution intents."""
        # Get the command executor
        command_executor = self.registry.get_component("command_executor")
        if not command_executor:
            logger.error("Command executor not available")
            return {
                "success": False,
                "error": "Command executor not available",
                "intent": intent.to_dict()
            }
        
        # Extract parameters
        command = intent.parameters.get("command", "")
        arguments = intent.parameters.get("arguments", "")
        
        # Construct the full command
        full_command = command
        if arguments:
            full_command = f"{command} {arguments}"
        
        # Execute the command
        if hasattr(command_executor, "execute"):
            # Get user ID from context
            user_id = context.get("user_id", "system")
            
            return command_executor.execute(full_command, user_id)
        
        # Fallback for unknown command execution method
        return {
            "success": False,
            "error": f"Command execution not supported: {full_command}",
            "intent": intent.to_dict()
        }
    
    def _record_interaction(self, query: str, intent: Intent, result: Dict[str, Any]) -> None:
        """Record the interaction in the conversation history."""
        # Add user query
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": time.time()
        })
        
        # Add system response
        response_content = result.get("response", str(result))
        self.conversation_history.append({
            "role": "system",
            "content": response_content,
            "intent": intent.to_dict(),
            "timestamp": time.time()
        })
        
        # Limit history size
        max_history = self.config.get("history_size", 50)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def connect_to_shell(self, shell_env: Any) -> bool:
        """
        Connect the intent system to the shell environment.
        
        Args:
            shell_env: The shell environment to connect to
            
        Returns:
            True if connection was successful, False otherwise
        """
        if not self.config.get("shell_integration", {}).get("enabled", True):
            logger.info("Shell integration disabled in configuration")
            return False
        
        if not shell_env:
            logger.error("Cannot connect to shell: shell environment not provided")
            return False
        
        try:
            # Check if the shell has a register_processor method
            if hasattr(shell_env, "register_processor"):
                # Register our processor with the shell
                shell_env.register_processor(
                    "intent_processor",
                    self.process_query,
                    {
                        "intercept_all": self.config.get("shell_integration", {}).get("intercept_all", True),
                        "command_prefix": self.config.get("shell_integration", {}).get("command_prefix", "!intent")
                    }
                )
                self.connected_to_shell = True
                logger.info("Intent system connected to shell environment")
                return True
            
            # Alternative connection method if register_processor not available
            if hasattr(shell_env, "add_command_handler"):
                # Register our command handler with the shell
                command_prefix = self.config.get("shell_integration", {}).get("command_prefix", "!intent")
                shell_env.add_command_handler(
                    command_prefix,
                    lambda args, ctx: self.process_query(" ".join(args), ctx)
                )
                self.connected_to_shell = True
                logger.info(f"Intent system connected to shell environment with command prefix: {command_prefix}")
                return True
            
            logger.error("Cannot connect to shell: no compatible registration method found")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to shell: {e}")
            return False
    
    def connect_to_agents(self, agent_system: Any) -> bool:
        """
        Connect the intent system to the agent system.
        
        Args:
            agent_system: The agent system to connect to
            
        Returns:
            True if connection was successful, False otherwise
        """
        if not self.config.get("agent_integration", {}).get("enabled", True):
            logger.info("Agent integration disabled in configuration")
            return False
        
        if not agent_system:
            logger.error("Cannot connect to agents: agent system not provided")
            return False
        
        try:
            # Check if the agent system has a register_intent_handler method
            if hasattr(agent_system, "register_intent_handler"):
                # Register our handler with the agent system
                agent_system.register_intent_handler(
                    self.intent_system,
                    {
                        "fallback_enabled": self.config.get("agent_integration", {}).get("fallback_enabled", True)
                    }
                )
                self.connected_to_agents = True
                logger.info("Intent system connected to agent system")
                return True
            
            # Alternative connection method if register_intent_handler not available
            if hasattr(agent_system, "register_component"):
                # Register our system as a component of the agent system
                agent_system.register_component(
                    "intent_classification_system",
                    self.intent_system
                )
                self.connected_to_agents = True
                logger.info("Intent system registered with agent system")
                return True
            
            logger.error("Cannot connect to agents: no compatible registration method found")
            return False
            
        except Exception as e:
            logger.error(f"Error connecting to agents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the intent system.
        
        Returns:
            Dictionary with statistics
        """
        # Get stats from the optimization system
        optimization_stats = self.optimization_system.get_stats()
        
        # Add our own stats
        stats = {
            "connected_to_shell": self.connected_to_shell,
            "connected_to_agents": self.connected_to_agents,
            "conversation_history_size": len(self.conversation_history),
            "last_intent": self.last_intent.to_dict() if self.last_intent else None
        }
        
        # Combine the stats
        stats.update(optimization_stats)
        
        return stats

def get_instance(registry: Any = None) -> IntentIntegrationSystem:
    """
    Get the singleton instance of the IntentIntegrationSystem.
    
    Args:
        registry: The component registry to use for dependency injection
        
    Returns:
        The IntentIntegrationSystem instance
    """
    # Check if we already have an instance in the registry
    if registry and hasattr(registry, "get_component"):
        instance = registry.get_component("intent_integration_system")
        if instance:
            return instance
    
    # Create a new instance
    instance = IntentIntegrationSystem(registry)
    
    # Register it with the registry if possible
    if registry and hasattr(registry, "register_component"):
        registry.register_component("intent_integration_system", instance)
    
    return instance

def integrate_with_fixwurx(shell_env: Any = None, launchpad: Any = None) -> IntentIntegrationSystem:
    """
    Integrate the intent classification system with the FixWurx ecosystem.
    
    Args:
        shell_env: The shell environment to connect to
        launchpad: The launchpad to connect to
        
    Returns:
        The IntentIntegrationSystem instance
    """
    # Get or create the registry
    registry = None
    
    # Try to get the registry from the shell environment
    if shell_env and hasattr(shell_env, "registry"):
        registry = shell_env.registry
    
    # Try to get the registry from the launchpad
    elif launchpad and hasattr(launchpad, "registry"):
        registry = launchpad.registry
    
    # Create a new registry if needed
    if not registry:
        # Try to import the component registry
        try:
            from components.component_registry import ComponentRegistry
            registry = ComponentRegistry()
        except ImportError:
            # Create a simple registry
            class SimpleRegistry:
                def __init__(self):
                    self.components = {}
                
                def register_component(self, name, component):
                    self.components[name] = component
                
                def get_component(self, name):
                    return self.components.get(name)
            
            registry = SimpleRegistry()
    
    # Get the integration system instance
    integration = get_instance(registry)
    
    # Connect to the shell environment if provided
    if shell_env:
        integration.connect_to_shell(shell_env)
    
    # Connect to the agent system if available
    agent_system = registry.get_component("agent_system")
    if agent_system:
        integration.connect_to_agents(agent_system)
    
    # Connect to the launchpad if provided and agent system not available
    elif launchpad and hasattr(launchpad, "agent_system"):
        integration.connect_to_agents(launchpad.agent_system)
    
    return integration

if __name__ == "__main__":
    print("FixWurx Intent Integration - This module should be imported, not run directly.")
