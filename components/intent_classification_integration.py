#!/usr/bin/env python3
"""
Intent Classification Integration Module

This module integrates the IntentClassificationSystem with the
IntentOptimizationSystem to provide a high-performance, optimized
intent classification experience.

It implements:
- Caching of frequent queries
- Performance monitoring
- Predictive intent classification
- Prioritization of operations
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Tuple

from components.intent_classification_system import IntentClassificationSystem, Intent
from components.intent_caching_system import (
    IntentOptimizationSystem,
    create_context_hash
)

# Configure logging
logger = logging.getLogger(__name__)

class OptimizedIntentClassifier:
    """
    A wrapper around IntentClassificationSystem that adds performance
    optimization features like caching, prediction, and performance tracking.
    """
    
    def __init__(self, registry):
        """
        Initialize the optimized intent classifier.
        
        Args:
            registry: Component registry for accessing other system components
        """
        self.registry = registry
        self.intent_classifier = IntentClassificationSystem(registry)
        self.optimization_system = IntentOptimizationSystem(
            cache_capacity=1000,  # Store up to 1000 classified intents
            history_size=10000,   # Remember patterns from the last 10000 intents
            window_size=1000      # Keep performance metrics for the last 1000 operations
        )
        
        # Load patterns, models etc. from the base classifier
        # This is already done in the IntentClassificationSystem constructor
        
    def classify_intent(self, query: str, context: Dict[str, Any]) -> Intent:
        """
        Classify the intent of a user query with optimization.
        
        This method first checks the cache for previously classified similar queries,
        then falls back to the standard classification if needed. It also tracks
        performance metrics and predicts likely follow-up intents.
        
        Args:
            query: The user's query string
            context: Contextual information including conversation history
            
        Returns:
            Intent: An Intent object with the classified intent type and extracted parameters
        """
        # Start timing for performance tracking
        start_time = time.time()
        
        # Create a hash of the context for cache lookup
        context_hash = create_context_hash(context)
        
        # Try to get from cache first
        cached_result = self.optimization_system.get_cached_intent(query, context_hash)
        
        if cached_result:
            # Convert cached dict back to Intent object
            intent = Intent(cached_result["type"])
            intent.query = cached_result.get("query", query)
            intent.context = context
            intent.parameters = cached_result.get("parameters", {})
            intent.required_agents = cached_result.get("required_agents", [])
            intent.execution_path = cached_result.get("execution_path", "planning")
            intent.context_references = cached_result.get("context_references", {})
            intent.confidence = cached_result.get("confidence", 0.0)
            
            # Record cache hit timing
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            self.optimization_system.track_classification_performance(duration_ms)
            
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return intent
        
        # Cache miss, perform full classification
        logger.debug(f"Cache miss for query: {query[:50]}...")
        intent = self.intent_classifier.classify_intent(query, context)
        
        # Record the classification time
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        self.optimization_system.track_classification_performance(duration_ms)
        
        # Cache the result for future use
        self.optimization_system.cache_intent_result(
            query, 
            context_hash, 
            {
                "type": intent.type,
                "query": intent.query,
                "parameters": intent.parameters,
                "required_agents": intent.required_agents,
                "execution_path": intent.execution_path,
                "context_references": intent.context_references,
                "confidence": intent.confidence
            }
        )
        
        return intent
    
    def predict_next_intents(self, current_intent: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict the most likely next intents based on historical patterns.
        
        Args:
            current_intent: The current intent type (optional)
            
        Returns:
            List of predicted intents with probability
        """
        raw_predictions = self.optimization_system.get_intent_predictions(current_intent)
        
        # Format the predictions for easier consumption
        return [
            {
                "intent_type": intent_type,
                "probability": probability,
                "suggested_query": self._generate_suggested_query(intent_type)
            }
            for intent_type, probability in raw_predictions
            if probability > 0.1  # Only include reasonably likely predictions
        ]
    
    def _generate_suggested_query(self, intent_type: str) -> str:
        """
        Generate a suggested query template for a given intent type.
        
        Args:
            intent_type: The intent type to generate a query for
            
        Returns:
            A template query string for the intent type
        """
        # This could be enhanced with ML-generated suggestions based on
        # the actual queries users have used for this intent type
        suggestions = {
            "file_access": "show me the content of [file]",
            "file_modification": "change [file] to [content]",
            "command_execution": "run the command `[command]`",
            "script_execution": "execute the script [script_name]",
            "bug_fix": "fix the bug in [file/component]",
            "system_debugging": "debug the [component/system]",
            "performance_optimization": "optimize [component] for better performance",
            "security_audit": "perform a security audit on [component]",
            "rotate_credentials": "rotate my credentials",
            "decision_tree": "run the decision tree for [problem]"
        }
        
        return suggestions.get(intent_type, f"[{intent_type} operation]")
    
    def track_execution_performance(self, intent_type: str, duration_ms: float) -> None:
        """
        Track the performance of executing an intent.
        
        Args:
            intent_type: The type of intent that was executed
            duration_ms: The duration in milliseconds
        """
        self.optimization_system.track_execution_performance(intent_type, duration_ms)
    
    def prioritize_operation(self, operation: Dict[str, Any], priority: str = "normal") -> None:
        """
        Add an operation to the priority queue.
        
        Args:
            operation: The operation details
            priority: Priority level (critical, high, normal, low, background)
        """
        self.optimization_system.enqueue_operation(operation, priority)
    
    def get_next_prioritized_operation(self) -> Optional[Dict[str, Any]]:
        """
        Get the next operation from the priority queue.
        
        Returns:
            The next operation or None if queue is empty
        """
        return self.optimization_system.get_next_operation()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive performance report.
        
        Returns:
            Dictionary with various performance metrics
        """
        return self.optimization_system.get_performance_report()
    
    def optimize_resource_usage(self) -> Dict[str, Any]:
        """
        Optimize resource usage based on performance metrics.
        
        This could adjust caching parameters, clear unused cache entries,
        or make other optimizations based on system performance.
        
        Returns:
            Dictionary with optimization results
        """
        report = self.get_performance_report()
        
        # Example optimizations
        cache_stats = report["cache"]
        cache_hit_rate = cache_stats.get("hit_rate", 0)
        
        optimizations = {
            "performed": [],
            "recommendations": []
        }
        
        # If cache hit rate is low, we might want to adjust caching strategy
        if cache_hit_rate < 0.5 and cache_stats.get("total_requests", 0) > 100:
            optimizations["recommendations"].append(
                "Low cache hit rate. Consider analyzing query patterns for better caching."
            )
        
        # Identify slow operations
        slow_ops = report["slow_operations"]
        if slow_ops:
            optimizations["recommendations"].append(
                f"Slow operations detected: {', '.join(slow_ops)}. Consider optimizing these intent handlers."
            )
        
        return optimizations

    def handle_agent_failure(self, intent: Intent, failed_agents: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Handle agent failures by applying fallback mechanisms.
        Delegates to the intent classifier's handler.
        
        Args:
            intent: The intent being processed
            failed_agents: List of agent types that have failed
            
        Returns:
            tuple: (updated_agent_list, fallback_mapping)
        """
        return self.intent_classifier.handle_agent_failure(intent, failed_agents)
