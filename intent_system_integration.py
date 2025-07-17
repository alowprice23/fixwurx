#!/usr/bin/env python3
"""
Intent System Integration

This module provides seamless integration between the Intent Classification System
and other core FixWurx components including Neural Matrix, Triangulum, and the Agent System.
It implements a multi-layered approach that allows for distributed intent processing,
neural-enhanced classification, and dynamic agent selection.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, List, Any, Tuple, Optional, Union, Set

# Import core components
from components.intent_classification_system import IntentClassificationSystem, Intent
from components.intent_caching_system import IntentOptimizationSystem
from components.state_manager import StateManager
from components.conversational_interface import ConversationalInterface

# Import integration components
from neural_matrix.core.neural_matrix import NeuralMatrix
from triangulum.client import TriangulumClient
from agents.core.agent_system import AgentSystem
from components.decision_flow import DecisionFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_integration.log"), logging.StreamHandler()]
)
logger = logging.getLogger("IntentSystemIntegration")

class NeuralIntentClassifier:
    """
    Neural-enhanced intent classifier that leverages the Neural Matrix
    for more accurate intent classification.
    """
    
    def __init__(self, neural_matrix: NeuralMatrix, standard_classifier: IntentClassificationSystem):
        """Initialize the neural intent classifier."""
        self.neural_matrix = neural_matrix
        self.standard_classifier = standard_classifier
        self.confidence_threshold = 0.75  # Threshold for neural classification
        self.training_data = []
        self.pattern_weights = {}
        self.load_pattern_weights()
        
    def load_pattern_weights(self):
        """Load pattern weights from Neural Matrix."""
        try:
            weights_path = os.path.join(".triangulum", "neural_matrix", "weights", "default_weights.json")
            if os.path.exists(weights_path):
                with open(weights_path, 'r') as f:
                    self.pattern_weights = json.load(f)
                logger.info(f"Loaded {len(self.pattern_weights)} pattern weights from Neural Matrix")
        except Exception as e:
            logger.error(f"Error loading pattern weights: {e}")
    
    def classify_intent(self, query: str, context: Dict[str, Any]) -> Intent:
        """
        Classify intent using both standard pattern matching and neural enhancements.
        
        This uses a hybrid approach:
        1. First try standard classification
        2. If confidence is high, use that result
        3. If confidence is low, enhance with neural classification
        4. Record the result for future training
        """
        # First, try standard classification
        intent = self.standard_classifier.classify_intent(query, context)
        
        # Check if we should enhance with neural classification
        if hasattr(intent, 'confidence') and intent.confidence < self.confidence_threshold:
            # Use Neural Matrix to enhance classification
            enhanced_intent = self._enhance_with_neural_matrix(query, intent, context)
            
            # Record for training
            self._record_for_training(query, enhanced_intent, context)
            
            return enhanced_intent
        
        # Record for training
        self._record_for_training(query, intent, context)
        
        return intent
    
    def _enhance_with_neural_matrix(self, query: str, initial_intent: Intent, context: Dict[str, Any]) -> Intent:
        """Enhance intent classification with Neural Matrix."""
        try:
            # Prepare input for Neural Matrix
            input_data = {
                "query": query,
                "initial_type": initial_intent.type,
                "context_keys": list(context.keys() if context else []),
                "history_length": len(context.get("history", [])) if context else 0
            }
            
            # Get enhanced classification from Neural Matrix
            result = self.neural_matrix.process(
                "intent_classification",
                input_data,
                self.pattern_weights
            )
            
            if result and result.get("success", False):
                # Update intent with neural results if confidence is higher
                neural_confidence = result.get("confidence", 0)
                if neural_confidence > getattr(initial_intent, 'confidence', 0):
                    # Create a new intent with updated values
                    enhanced_type = result.get("intent_type", initial_intent.type)
                    enhanced_path = result.get("execution_path", initial_intent.execution_path)
                    
                    # Copy the initial intent and update fields
                    enhanced_intent = Intent(
                        query=initial_intent.query,
                        type=enhanced_type,
                        execution_path=enhanced_path,
                        parameters=initial_intent.parameters.copy() if hasattr(initial_intent, "parameters") else {},
                        required_agents=result.get("required_agents", initial_intent.required_agents) if hasattr(initial_intent, "required_agents") else None
                    )
                    
                    # Add confidence attribute
                    enhanced_intent.confidence = neural_confidence
                    
                    logger.info(f"Neural enhancement improved classification: {initial_intent.type} -> {enhanced_type}")
                    return enhanced_intent
            
            # Default to initial intent if enhancement failed
            return initial_intent
            
        except Exception as e:
            logger.error(f"Error in neural enhancement: {e}")
            return initial_intent
    
    def _record_for_training(self, query: str, intent: Intent, context: Dict[str, Any]):
        """Record classification result for future training."""
        try:
            # Create training record
            training_record = {
                "query": query,
                "intent_type": intent.type,
                "execution_path": intent.execution_path,
                "context_snapshot": {
                    "state": context.get("state", "unknown"),
                    "history_length": len(context.get("history", [])),
                    "has_file_context": "file" in str(context).lower()
                },
                "timestamp": time.time()
            }
            
            # Add to training data
            self.training_data.append(training_record)
            
            # If we have enough training data, schedule a background training task
            if len(self.training_data) >= 50:
                threading.Thread(target=self._train_neural_matrix).start()
        except Exception as e:
            logger.error(f"Error recording training data: {e}")
    
    def _train_neural_matrix(self):
        """Train Neural Matrix with collected data."""
        try:
            if not self.training_data:
                return
                
            # Copy and clear training data
            training_data = self.training_data.copy()
            self.training_data = []
            
            logger.info(f"Training Neural Matrix with {len(training_data)} samples")
            
            # Train Neural Matrix
            result = self.neural_matrix.train(
                "intent_classification",
                training_data
            )
            
            if result and result.get("success", False):
                logger.info(f"Neural Matrix training completed successfully. New patterns: {result.get('new_patterns', 0)}")
                
                # Update pattern weights
                self.load_pattern_weights()
            else:
                logger.error(f"Neural Matrix training failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error training Neural Matrix: {e}")


class DistributedIntentProcessor:
    """
    Distributes intent processing across Triangulum nodes for high-performance
    classification and execution.
    """
    
    def __init__(self, triangulum_client: TriangulumClient, local_processor: NeuralIntentClassifier):
        """Initialize the distributed intent processor."""
        self.triangulum = triangulum_client
        self.local_processor = local_processor
        self.distribution_threshold = 0.8  # CPU load threshold for distribution
        self.current_load = 0.0
        self.update_load()
        
    def update_load(self):
        """Update current CPU load."""
        try:
            # Get CPU load from Triangulum
            load_info = self.triangulum.get_system_status().get("cpu_load", 0.0)
            self.current_load = load_info
        except Exception as e:
            logger.error(f"Error updating CPU load: {e}")
    
    def process_intent(self, query: str, context: Dict[str, Any]) -> Intent:
        """
        Process intent locally or distribute to Triangulum based on load.
        
        This implements an adaptive load balancing strategy:
        1. If local load is high, distribute to Triangulum nodes
        2. If local load is low, process locally
        3. If distributed processing fails, fall back to local
        """
        # Update current load
        self.update_load()
        
        # Check if we should distribute
        if self.current_load > self.distribution_threshold:
            try:
                # Distribute intent processing to Triangulum
                distributed_result = self._distribute_intent_processing(query, context)
                
                if distributed_result:
                    logger.info(f"Distributed intent processing successful: {distributed_result.type}")
                    return distributed_result
            except Exception as e:
                logger.error(f"Error in distributed intent processing: {e}")
        
        # Fall back to local processing
        return self.local_processor.classify_intent(query, context)
    
    def _distribute_intent_processing(self, query: str, context: Dict[str, Any]) -> Optional[Intent]:
        """Distribute intent processing to Triangulum nodes."""
        try:
            # Prepare data for distribution
            intent_data = {
                "query": query,
                "context": {
                    "state": context.get("state", "unknown"),
                    "history": context.get("history", [])[:5],  # Last 5 history items
                    "current_context": context.get("current_context", {})
                }
            }
            
            # Submit job to Triangulum
            job_id = self.triangulum.submit_job(
                "intent_classification",
                intent_data,
                priority=1  # High priority
            )
            
            # Wait for result (with timeout)
            result = self.triangulum.get_job_result(job_id, timeout=2.0)
            
            if result and result.get("success", False):
                # Convert result to Intent object
                intent_result = result.get("result", {})
                
                intent = Intent(
                    query=query,
                    type=intent_result.get("type", "generic"),
                    execution_path=intent_result.get("execution_path", "planning"),
                    parameters=intent_result.get("parameters", {}),
                    required_agents=intent_result.get("required_agents")
                )
                
                return intent
                
            return None
            
        except Exception as e:
            logger.error(f"Error distributing intent processing: {e}")
            return None


class AdaptiveAgentSelector:
    """
    Adaptively selects and coordinates agents based on intent classification
    and system state.
    """
    
    def __init__(self, agent_system: AgentSystem, decision_flow: DecisionFlow):
        """Initialize the adaptive agent selector."""
        self.agent_system = agent_system
        self.decision_flow = decision_flow
        self.capability_map = self._build_capability_map()
        self.specialist_intents = {
            "code_analysis": ["bug_fix", "refactoring", "code_optimization"],
            "system_maintenance": ["performance_monitoring", "resource_optimization"],
            "security": ["security_audit", "vulnerability_assessment"]
        }
        
    def _build_capability_map(self) -> Dict[str, Set[str]]:
        """Build a map of agent capabilities."""
        capability_map = {}
        
        # Get all registered agents
        agents = self.agent_system.get_registered_agents()
        
        # Build capability map
        for agent_name, agent_info in agents.items():
            capabilities = agent_info.get("capabilities", [])
            for capability in capabilities:
                if capability not in capability_map:
                    capability_map[capability] = set()
                capability_map[capability].add(agent_name)
                
        return capability_map
    
    def select_agents_for_intent(self, intent: Intent) -> List[str]:
        """
        Select the optimal agents for handling the given intent.
        
        This uses a sophisticated agent selection strategy:
        1. Check if intent needs specialist agents
        2. Use decision flow to select agents based on intent type and parameters
        3. Optimize selection based on current system state
        """
        selected_agents = []
        
        # If intent already has required agents, use those
        if hasattr(intent, "required_agents") and intent.required_agents:
            return intent.required_agents
        
        # Check if this intent requires specialist agents
        for specialty, intent_types in self.specialist_intents.items():
            if intent.type in intent_types:
                specialist_agents = self._get_agents_by_capability(specialty)
                if specialist_agents:
                    selected_agents.extend(specialist_agents)
        
        # If we don't have any specialists, use decision flow
        if not selected_agents:
            decision_input = {
                "intent_type": intent.type,
                "execution_path": intent.execution_path,
                "parameters": intent.parameters,
                "has_file_operation": "file" in intent.type or "path" in intent.parameters
            }
            
            decision_result = self.decision_flow.evaluate("agent_selection", decision_input)
            
            if decision_result and decision_result.get("success", False):
                selected_agents = decision_result.get("selected_agents", [])
        
        # If we still don't have agents, fall back to default selection
        if not selected_agents:
            # Default selection based on execution path
            if intent.execution_path == "direct":
                selected_agents = ["executor", "file_handler"]
            elif intent.execution_path == "agent_collaboration":
                selected_agents = ["coordinator", "analyzer", "executor"]
            else:  # planning
                selected_agents = ["planner", "executor"]
        
        # Optimize selection based on current system state
        optimized_agents = self._optimize_agent_selection(selected_agents, intent)
        
        return optimized_agents
    
    def _get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agents with the specified capability."""
        return list(self.capability_map.get(capability, set()))
    
    def _optimize_agent_selection(self, agents: List[str], intent: Intent) -> List[str]:
        """Optimize agent selection based on system state."""
        # Check if any selected agents are overloaded
        optimized_agents = []
        
        for agent in agents:
            # Check if agent is available
            agent_status = self.agent_system.get_agent_status(agent)
            
            if not agent_status or agent_status.get("status") == "unavailable":
                # Find alternative agent with similar capabilities
                alternative = self._find_alternative_agent(agent, intent)
                if alternative:
                    optimized_agents.append(alternative)
                    logger.info(f"Replaced unavailable agent {agent} with {alternative}")
                else:
                    # Keep the original if no alternative found
                    optimized_agents.append(agent)
            else:
                optimized_agents.append(agent)
        
        return optimized_agents
    
    def _find_alternative_agent(self, agent: str, intent: Intent) -> Optional[str]:
        """Find an alternative agent with similar capabilities."""
        # Get agent capabilities
        agent_info = self.agent_system.get_agent_info(agent)
        if not agent_info:
            return None
            
        capabilities = agent_info.get("capabilities", [])
        
        # Find agents with similar capabilities
        candidates = set()
        for capability in capabilities:
            agents_with_capability = self.capability_map.get(capability, set())
            candidates.update(agents_with_capability)
        
        # Remove the original agent
        if agent in candidates:
            candidates.remove(agent)
        
        # Find the most suitable alternative
        if candidates:
            # Sort by availability
            available_candidates = []
            for candidate in candidates:
                status = self.agent_system.get_agent_status(candidate)
                if status and status.get("status") == "available":
                    available_candidates.append((candidate, status.get("load", 1.0)))
            
            if available_candidates:
                # Sort by load (lowest first)
                available_candidates.sort(key=lambda x: x[1])
                return available_candidates[0][0]
        
        return None


class IntentIntegrationSystem:
    """
    Main integration system that connects the intent classification system
    with other FixWurx components.
    """
    
    def __init__(self, registry):
        """Initialize the intent integration system."""
        self.registry = registry
        
        # Create state manager
        self.state_manager = registry.get_component("state_manager")
        if not self.state_manager:
            self.state_manager = StateManager(registry)
            registry.register_component("state_manager", self.state_manager)
        
        # Set up standard intent components
        self.intent_classification_system = registry.get_component("intent_classification_system")
        if not self.intent_classification_system:
            self.intent_classification_system = IntentClassificationSystem(registry)
            registry.register_component("intent_classification_system", self.intent_classification_system)
        
        self.intent_optimization_system = registry.get_component("intent_optimization_system")
        if not self.intent_optimization_system:
            self.intent_optimization_system = IntentOptimizationSystem(
                cache_capacity=100,
                history_size=50,
                window_size=20
            )
            registry.register_component("intent_optimization_system", self.intent_optimization_system)
        
        # Set up integration components
        self._setup_integration_components()
        
        # Create conversational interface
        self.conversational_interface = registry.get_component("conversational_interface")
        if not self.conversational_interface:
            self.conversational_interface = ConversationalInterface(registry)
            registry.register_component("conversational_interface", self.conversational_interface)
        
        logger.info("Intent Integration System initialized")
    
    def _setup_integration_components(self):
        """Set up integration components."""
        try:
            # Set up Neural Matrix integration
            neural_matrix = self.registry.get_component("neural_matrix")
            if neural_matrix:
                self.neural_classifier = NeuralIntentClassifier(neural_matrix, self.intent_classification_system)
                self.registry.register_component("neural_intent_classifier", self.neural_classifier)
            else:
                logger.warning("Neural Matrix not available, falling back to standard classification")
                self.neural_classifier = None
            
            # Set up Triangulum integration
            triangulum_client = self.registry.get_component("triangulum_client")
            if triangulum_client and self.neural_classifier:
                self.distributed_processor = DistributedIntentProcessor(triangulum_client, self.neural_classifier)
                self.registry.register_component("distributed_intent_processor", self.distributed_processor)
            else:
                logger.warning("Triangulum client not available, distributed processing disabled")
                self.distributed_processor = None
            
            # Set up agent system integration
            agent_system = self.registry.get_component("agent_system")
            decision_flow = self.registry.get_component("decision_flow")
            if agent_system and decision_flow:
                self.agent_selector = AdaptiveAgentSelector(agent_system, decision_flow)
                self.registry.register_component("agent_selector", self.agent_selector)
            else:
                logger.warning("Agent system or decision flow not available, agent selection disabled")
                self.agent_selector = None
        
        except Exception as e:
            logger.error(f"Error setting up integration components: {e}")
    
    def process_intent(self, query: str, context: Dict[str, Any]) -> Tuple[Intent, List[str]]:
        """
        Process an intent through the full integration pipeline.
        
        Steps:
        1. Check cache for previous classification
        2. If not in cache, use distributed processing if available
        3. If distributed processing not available, use neural classification
        4. If neural classification not available, use standard classification
        5. Select appropriate agents for the intent
        6. Cache the result
        7. Return the intent and selected agents
        """
        # Check if we have this in cache
        cache_key = self.intent_optimization_system.generate_context_hash(query, context)
        cached_result = self.intent_optimization_system.get_from_cache(cache_key)
        
        if cached_result:
            intent = cached_result.get("intent")
            agents = cached_result.get("agents", [])
            
            # Update cache stats
            self.intent_optimization_system.update_stats("hits")
            
            logger.info(f"Cache hit for intent: {intent.type}")
            return intent, agents
        
        # Update cache stats for miss
        self.intent_optimization_system.update_stats("misses")
        
        # Process intent through integration pipeline
        if self.distributed_processor:
            # Use distributed processing
            intent = self.distributed_processor.process_intent(query, context)
        elif self.neural_classifier:
            # Use neural classification
            intent = self.neural_classifier.classify_intent(query, context)
        else:
            # Fall back to standard classification
            intent = self.intent_classification_system.classify_intent(query, context)
        
        # Select agents
        agents = []
        if self.agent_selector:
            agents = self.agent_selector.select_agents_for_intent(intent)
        
        # Cache the result
        cache_entry = {
            "intent": intent,
            "agents": agents,
            "timestamp": time.time()
        }
        self.intent_optimization_system.add_to_cache(cache_key, cache_entry)
        
        # Update predictive model
        self.intent_optimization_system.update_sequence_model(query, intent.type)
        
        return intent, agents
    
    def get_predicted_intents(self, current_intent: str) -> List[str]:
        """Get predicted next intents based on current intent."""
        return self.intent_optimization_system.predict_next_intents(current_intent)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for the intent system."""
        return self.intent_optimization_system.generate_performance_report()
    
    def handle_agent_failure(self, intent: Intent, failed_agents: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Handle agent failures by finding suitable replacements."""
        if self.agent_selector:
            # Find replacements for failed agents
            replacements = {}
            updated_agents = []
            
            for agent in intent.required_agents:
                if agent in failed_agents:
                    # Find replacement
                    replacement = self.agent_selector._find_alternative_agent(agent, intent)
                    if replacement:
                        replacements[agent] = replacement
                        updated_agents.append(replacement)
                    else:
                        # If no replacement found, keep the original
                        updated_agents.append(agent)
                else:
                    updated_agents.append(agent)
            
            return updated_agents, replacements
        
        # Fall back to standard handler
        return self.intent_classification_system.handle_agent_failure(intent, failed_agents)


def get_instance(registry):
    """Get an instance of the Intent Integration System."""
    return IntentIntegrationSystem(registry)
