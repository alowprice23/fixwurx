#!/usr/bin/env python3
"""
Launchpad Agent

This module provides the Launchpad Agent, which is responsible for bootstrapping 
and initializing the entire FixWurx system. It's the starting point for all system
operations and ensures proper initialization of other agents and components.
"""

import logging
import os
import sys
import time
import json
import openai
from typing import Dict, List, Optional, Any, Tuple

# Import core components
from agents.core.agent_system import get_instance as get_agent_system
from agents.core.coordinator import AgentCoordinator
from agents.core.handoff import AgentHandoff

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o"

logger = logging.getLogger("LaunchpadAgent")

class LaunchpadAgent:
    """
    Launchpad Agent class that bootstraps the entire FixWurx system.
    This agent is responsible for starting up all other agent components
    and ensuring they are properly initialized. It uses LLM capabilities
    to intelligently optimize the initialization process and provide
    recommendations for system configuration.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Launchpad Agent.
        
        Args:
            config: Configuration dictionary for the Launchpad Agent
        """
        self.config = config or {}
        self.initialized = False
        self.agent_system = None
        self.coordinator = None
        self.handoff = None
        self.startup_time = None
        
        # Initialize LLM configuration
        self.llm_config = {
            "model": self.config.get("llm_model", DEFAULT_MODEL),
            "temperature": self.config.get("llm_temperature", 0.2),
            "max_tokens": self.config.get("llm_max_tokens", 1000),
            "top_p": self.config.get("llm_top_p", 1.0),
            "presence_penalty": self.config.get("llm_presence_penalty", 0.0),
            "frequency_penalty": self.config.get("llm_frequency_penalty", 0.0)
        }
        
        # Initialize OpenAI client if key is available
        self.openai_client = None
        if OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        # Initialize metrics for LLM usage
        self.llm_metrics = {
            "llm_calls": 0,
            "llm_tokens_used": 0,
            "llm_successful_calls": 0,
            "llm_failed_calls": 0
        }
        
    def initialize(self) -> bool:
        """
        Initialize the Launchpad Agent and bootstrap the system,
        using LLM to optimize the initialization process.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.initialized:
            logger.warning("Launchpad Agent already initialized")
            return True
        
        try:
            logger.info("Initializing Launchpad Agent")
            self.startup_time = time.time()
            
            # Use LLM to suggest optimal initialization sequence
            initialization_plan = self._generate_initialization_plan()
            
            # Initialize components based on plan or fall back to default order
            if initialization_plan and isinstance(initialization_plan, list):
                # Follow LLM-suggested initialization order
                for component in initialization_plan:
                    success = self._initialize_component(component)
                    if not success:
                        logger.error(f"Failed to initialize {component}")
                        return False
            else:
                # Default initialization order
                
                # Initialize the agent system
                self.agent_system = get_agent_system(self.config.get("agent_system", {}))
                if not self.agent_system.initialize():
                    logger.error("Failed to initialize agent system")
                    return False
                
                # Initialize the agent coordinator
                self.coordinator = AgentCoordinator(self.config.get("coordinator", {}))
                
                # Initialize the agent handoff mechanism
                self.handoff = AgentHandoff(self.config.get("handoff", {}))
            
            # Register with event system
            self._register_events()
            
            self.initialized = True
            logger.info(f"Launchpad Agent initialized successfully in {time.time() - self.startup_time:.2f} seconds")
            return True
        except Exception as e:
            logger.error(f"Error initializing Launchpad Agent: {e}")
            return False
    
    def shutdown(self) -> bool:
        """
        Shutdown the Launchpad Agent and all bootstrapped components.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        if not self.initialized:
            logger.warning("Launchpad Agent not initialized")
            return True
        
        try:
            logger.info("Shutting down Launchpad Agent")
            
            # Shutdown the agent system
            if self.agent_system:
                self.agent_system.shutdown()
            
            # Shutdown the auditor agent if it exists
            if hasattr(self, 'auditor_agent') and self.auditor_agent:
                try:
                    logger.info("Shutting down Auditor Agent")
                    self.auditor_agent.stop_auditing()
                except Exception as e:
                    logger.error(f"Error shutting down Auditor Agent: {e}")
            
            self.initialized = False
            logger.info("Launchpad Agent shut down successfully")
            return True
        except Exception as e:
            logger.error(f"Error shutting down Launchpad Agent: {e}")
            return False
    
    def _register_events(self) -> None:
        """
        Register event handlers for system events.
        """
        # This would connect to the event system and register handlers
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the Launchpad Agent.
        
        Returns:
            Dict containing status information
        """
        if not self.initialized:
            return {
                "initialized": False,
                "uptime": 0,
                "llm_status": "not_used"
            }
        
        return {
            "initialized": True,
            "uptime": time.time() - self.startup_time if self.startup_time else 0,
            "agent_system": self.agent_system.get_status() if self.agent_system else None,
            "llm_metrics": self.llm_metrics,
            "llm_status": "active" if self.openai_client else "inactive"
        }
    
    def restart_component(self, component_name: str) -> bool:
        """
        Restart a specific component with LLM-guided optimization.
        
        Args:
            component_name: Name of the component to restart
            
        Returns:
            bool: True if restart was successful, False otherwise
        """
        if not self.initialized:
            logger.warning("Launchpad Agent not initialized")
            return False
        
        try:
            logger.info(f"Restarting component: {component_name}")
            
            # Get LLM recommendations for restart
            restart_recommendations = self._get_restart_recommendations(component_name)
            
            if component_name == "agent_system":
                # Apply any LLM recommendations before restart
                if restart_recommendations:
                    logger.info(f"Applying LLM recommendations for {component_name} restart")
                
                self.agent_system.shutdown()
                return self.agent_system.initialize()
            elif component_name == "auditor":
                # Restart the Auditor Agent
                try:
                    # First stop the current auditor if it exists
                    if hasattr(self, 'auditor_agent') and self.auditor_agent:
                        logger.info("Stopping existing Auditor Agent")
                        self.auditor_agent.stop_auditing()
                    
                    # Re-initialize the auditor component
                    return self._initialize_component("auditor")
                except Exception as e:
                    logger.error(f"Error restarting Auditor Agent: {e}")
                    return False
            elif component_name == "coordinator":
                # Restart the coordinator
                try:
                    self.coordinator = AgentCoordinator(self.config.get("coordinator", {}))
                    logger.info("Coordinator restarted successfully")
                    return True
                except Exception as e:
                    logger.error(f"Error restarting Coordinator: {e}")
                    return False
            elif component_name == "handoff":
                # Restart the handoff mechanism
                try:
                    self.handoff = AgentHandoff(self.config.get("handoff", {}))
                    logger.info("Handoff mechanism restarted successfully")
                    return True
                except Exception as e:
                    logger.error(f"Error restarting Handoff mechanism: {e}")
                    return False
            else:
                logger.warning(f"Unknown component: {component_name}")
                return False
        except Exception as e:
            logger.error(f"Error restarting component {component_name}: {e}")
            return False
    
    def _initialize_component(self, component_name: str) -> bool:
        """
        Initialize a specific component.
        
        Args:
            component_name: Name of the component to initialize
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if component_name == "agent_system":
                self.agent_system = get_agent_system(self.config.get("agent_system", {}))
                return self.agent_system.initialize()
            elif component_name == "coordinator":
                self.coordinator = AgentCoordinator(self.config.get("coordinator", {}))
                return True
            elif component_name == "handoff":
                self.handoff = AgentHandoff(self.config.get("handoff", {}))
                return True
            elif component_name == "auditor":
                # Task AUD-1: Start the Auditor Agent in a separate thread
                from agents.auditor.auditor_agent import get_instance as get_auditor_agent
                
                try:
                    auditor_agent = get_auditor_agent(self.config.get("auditor_agent", {}))
                    # Start auditing in a separate thread
                    if auditor_agent.start_auditing():
                        logger.info("Auditor Agent started successfully")
                        # Register the auditor agent so it can be referenced later
                        self.auditor_agent = auditor_agent
                        return True
                    else:
                        logger.error("Failed to start Auditor Agent")
                        return False
                except Exception as e:
                    logger.error(f"Error starting Auditor Agent: {e}")
                    return False
            else:
                logger.warning(f"Unknown component: {component_name}")
                return False
        except Exception as e:
            logger.error(f"Error initializing component {component_name}: {e}")
            return False
    
    def _generate_initialization_plan(self) -> List[str]:
        """
        Use LLM to generate an optimal initialization plan.
        
        Returns:
            List of component names in optimal initialization order
        """
        if not self.openai_client:
            # Include auditor agent in the default initialization plan
            return ["agent_system", "coordinator", "handoff", "auditor"]
        
        try:
            # Update metrics
            self.llm_metrics["llm_calls"] += 1
            
            # Create prompt
            prompt = f"""
            As a Launchpad Agent, I need to initialize the FixWurx system components in the optimal order.
            
            Available components:
            1. agent_system - Core agent system that manages all agents
            2. coordinator - Agent coordinator for managing agent interactions
            3. handoff - Agent handoff mechanism for transferring control between agents
            
            System configuration:
            {json.dumps(self.config, indent=2)}
            
            Please suggest the optimal initialization order considering dependencies and efficiency.
            Format your response as a JSON array of component names in the order they should be initialized.
            """
            
            # Make the API call
            response = self.openai_client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for a Launchpad Agent that bootstraps an agent system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"]
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Update metrics
            self.llm_metrics["llm_successful_calls"] += 1
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                self.llm_metrics["llm_tokens_used"] += response.usage.total_tokens
            
            # Parse response
            try:
                # Look for JSON array in the response
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end]
                    components = json.loads(json_content)
                    
                    # Validate that all required components are included
                    required_components = {"agent_system", "coordinator", "handoff"}
                    if set(components) >= required_components:
                        return components
                
                # Try to parse the entire response as JSON
                components = json.loads(response_text)
                if isinstance(components, list):
                    return components
            except:
                logger.warning("Failed to parse LLM initialization plan")
            
            # Default initialization order
            return ["agent_system", "coordinator", "handoff"]
        except Exception as e:
            logger.error(f"Error generating initialization plan: {e}")
            self.llm_metrics["llm_failed_calls"] += 1
            return ["agent_system", "coordinator", "handoff"]
    
    def _get_restart_recommendations(self, component_name: str) -> Dict[str, Any]:
        """
        Use LLM to get recommendations for restarting a component.
        
        Args:
            component_name: Name of the component to restart
            
        Returns:
            Recommendations for restarting the component
        """
        if not self.openai_client:
            return {}
        
        try:
            # Update metrics
            self.llm_metrics["llm_calls"] += 1
            
            # Create prompt
            prompt = f"""
            As a Launchpad Agent, I need to restart the '{component_name}' component in the FixWurx system.
            
            Current system status:
            - Initialized: {self.initialized}
            - Uptime: {time.time() - self.startup_time if self.startup_time else 0} seconds
            - Agent System: {'Initialized' if self.agent_system else 'Not initialized'}
            - Coordinator: {'Initialized' if self.coordinator else 'Not initialized'}
            - Handoff: {'Initialized' if self.handoff else 'Not initialized'}
            
            Please provide recommendations for optimizing the restart process for the '{component_name}' component.
            Format your response as a JSON object with the following keys:
            - pre_restart_actions: array of strings for actions to take before restart
            - post_restart_actions: array of strings for actions to take after restart
            - optimization_suggestions: array of strings for general optimization suggestions
            """
            
            # Make the API call
            response = self.openai_client.chat.completions.create(
                model=self.llm_config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for a Launchpad Agent that bootstraps an agent system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"]
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Update metrics
            self.llm_metrics["llm_successful_calls"] += 1
            if hasattr(response, 'usage') and hasattr(response.usage, 'total_tokens'):
                self.llm_metrics["llm_tokens_used"] += response.usage.total_tokens
            
            # Parse response
            try:
                # Try to find and parse JSON content
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end]
                    recommendations = json.loads(json_content)
                    return recommendations
                
                # Try to parse the entire response as JSON
                recommendations = json.loads(response_text)
                return recommendations
            except:
                logger.warning("Failed to parse LLM restart recommendations")
                return {}
        except Exception as e:
            logger.error(f"Error getting restart recommendations: {e}")
            self.llm_metrics["llm_failed_calls"] += 1
            return {}

# Create a singleton instance
_instance = None

def get_instance(config: Optional[Dict[str, Any]] = None) -> LaunchpadAgent:
    """
    Get the singleton instance of the Launchpad Agent.
    
    Args:
        config: Configuration dictionary for the Launchpad Agent
        
    Returns:
        LaunchpadAgent instance
    """
    global _instance
    if _instance is None:
        _instance = LaunchpadAgent(config)
    return _instance
