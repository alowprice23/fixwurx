#!/usr/bin/env python3
"""
Intent Optimization System for FixWurx

This module implements the intent caching and optimization functionality for the FixWurx 
ecosystem, providing performance improvements and intelligent intent prediction.
"""

import time
import logging
import collections
from typing import Dict, List, Any, Optional, Set, Tuple, Deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("IntentOptimizationSystem")

class IntentOptimizationSystem:
    """Advanced intent caching and optimization system for FixWurx."""
    
    def __init__(
        self, 
        cache_capacity: int = 100, 
        history_size: int = 50,
        window_size: int = 10
    ):
        """
        Initialize the intent optimization system.
        
        Args:
            cache_capacity: Maximum number of cached intent classifications
            history_size: Maximum number of intent sequences to store
            window_size: Window size for next intent prediction
        """
        # Cache of previously classified intents
        self.cache = {}
        self.cache_capacity = cache_capacity
        self.cache_hits = 0
        self.cache_misses = 0
        
        # LRU tracking for cache eviction
        self.lru_queue = collections.deque(maxlen=cache_capacity)
        
        # Intent sequence history for next intent prediction
        self.intent_history = collections.deque(maxlen=history_size)
        self.window_size = window_size
        
        # Intent transition matrix for prediction
        self.transition_matrix = {}
        
        logger.info(f"Intent Optimization System initialized with cache capacity {cache_capacity}")
    
    def get_cached_intent(self, query: str, context_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached intent for the given query and context.
        
        Args:
            query: The user's query
            context_hash: Optional hash of the context for context-aware caching
            
        Returns:
            The cached intent if found, None otherwise
        """
        # Create a cache key
        cache_key = query if context_hash is None else f"{query}|{context_hash}"
        
        # Check if the query is in the cache
        if cache_key in self.cache:
            # Update LRU queue
            if cache_key in self.lru_queue:
                self.lru_queue.remove(cache_key)
            self.lru_queue.append(cache_key)
            
            # Record cache hit and return the cached result
            self.cache_hits += 1
            logger.debug(f"Cache hit for query: {query}")
            
            return self.cache[cache_key]
        
        # Record cache miss
        self.cache_misses += 1
        logger.debug(f"Cache miss for query: {query}")
        
        return None
    
    def cache_intent(self, query: str, intent: Dict[str, Any], context_hash: Optional[str] = None) -> None:
        """
        Cache an intent classification for future use.
        
        Args:
            query: The user's query
            intent: The classified intent
            context_hash: Optional hash of the context for context-aware caching
        """
        # Create a cache key
        cache_key = query if context_hash is None else f"{query}|{context_hash}"
        
        # Add timestamp to the intent
        intent_with_timestamp = intent.copy()
        intent_with_timestamp["cached_at"] = time.time()
        
        # If the cache is full, evict the least recently used item
        if len(self.cache) >= self.cache_capacity and cache_key not in self.cache:
            if self.lru_queue:
                oldest_key = self.lru_queue.popleft()
                if oldest_key in self.cache:
                    logger.debug(f"Evicting LRU cache entry: {oldest_key}")
                    del self.cache[oldest_key]
        
        # Add the new item to the cache
        self.cache[cache_key] = intent_with_timestamp
        
        # Update LRU queue
        if cache_key in self.lru_queue:
            self.lru_queue.remove(cache_key)
        self.lru_queue.append(cache_key)
        
        logger.debug(f"Cached intent for query: {query}")
    
    def record_intent_sequence(self, intent_type: str) -> None:
        """
        Record an intent type in the sequence history for transition analysis.
        
        Args:
            intent_type: The type of the current intent
        """
        # Add intent to history
        self.intent_history.append(intent_type)
        
        # Update transition matrix
        if len(self.intent_history) >= 2:
            prev_intent = self.intent_history[-2]
            curr_intent = self.intent_history[-1]
            
            # Initialize transition counts if needed
            if prev_intent not in self.transition_matrix:
                self.transition_matrix[prev_intent] = {}
            
            if curr_intent not in self.transition_matrix[prev_intent]:
                self.transition_matrix[prev_intent][curr_intent] = 0
            
            # Increment transition count
            self.transition_matrix[prev_intent][curr_intent] += 1
    
    def predict_next_intents(self, current_intent_type: str, top_n: int = 3) -> List[str]:
        """
        Predict the most likely next intent types based on historical patterns.
        
        Args:
            current_intent_type: The type of the current intent
            top_n: Number of predictions to return
            
        Returns:
            List of the most likely next intent types
        """
        # Check if we have transition data for this intent type
        if current_intent_type not in self.transition_matrix:
            return []
        
        # Get all transitions from this intent type
        transitions = self.transition_matrix[current_intent_type]
        
        # Sort transitions by frequency
        sorted_transitions = sorted(
            transitions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return the top N transitions
        return [t[0] for t in sorted_transitions[:top_n]]
    
    def clear_cache(self) -> None:
        """Clear the intent cache."""
        self.cache = {}
        self.lru_queue.clear()
        logger.info("Intent cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache and prediction system.
        
        Returns:
            Dictionary with statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "cache_capacity": self.cache_capacity,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "history_size": len(self.intent_history),
            "transition_patterns": len(self.transition_matrix)
        }
    
    def optimize_intent(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply optimizations to an intent based on historical patterns.
        
        Args:
            intent: The intent to optimize
            
        Returns:
            The optimized intent
        """
        # Record this intent in our sequence history
        self.record_intent_sequence(intent["type"])
        
        # For now, just return the original intent
        # In a more sophisticated system, this could adjust confidence scores,
        # add predicted parameters based on historical patterns, etc.
        return intent

def create_context_hash(context: Dict[str, Any]) -> str:
    """
    Create a hash string from a context dictionary for context-aware caching.
    
    This is a simplified implementation. In a production system, you would want
    to use a more robust hashing mechanism that can handle complex nested structures.
    
    Args:
        context: The context dictionary
    
    Returns:
        A hash string representing the context
    """
    # Extract relevant parts of the context that affect intent classification
    relevant_parts = []
    
    # Consider conversation history if available
    if "history" in context and isinstance(context["history"], list):
        # Use only the last few interactions
        history = context["history"][-3:]
        for interaction in history:
            if isinstance(interaction, dict):
                # Extract user queries from history
                if interaction.get("role") == "user" and "content" in interaction:
                    relevant_parts.append(f"u:{interaction['content']}")
    
    # Consider current state if available
    if "state" in context and isinstance(context["state"], dict):
        # Extract key state information
        for key in ["current_folder", "current_file", "mode"]:
            if key in context["state"]:
                relevant_parts.append(f"{key}:{context['state'][key]}")
    
    # Join all relevant parts and return as a hash string
    return "|".join(relevant_parts)
