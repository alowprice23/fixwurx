#!/usr/bin/env python3
"""
Intent Caching System for FixWurx

This module implements a caching layer for the intent classification system,
improving performance by avoiding redundant processing of similar queries.
"""

import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple
import hashlib

from components.intent_classification_system import Intent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("intent_cache.log"), logging.StreamHandler()]
)
logger = logging.getLogger("IntentCachingSystem")

class IntentCachingSystem:
    """Caching system for intent classification results."""

    def __init__(self, registry: Any, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize the intent caching system.
        
        Args:
            registry: The component registry
            cache_size: Maximum number of entries to store in the cache
            cache_ttl: Time-to-live in seconds for cache entries
        """
        self.registry = registry
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache = {}  # Dictionary to store cached intents
        self.access_times = {}  # Dictionary to track last access time for each entry
        self.access_count = {}  # Dictionary to track access count for each entry
        self.insertion_order = []  # List to track insertion order for cache eviction
        
        logger.info(f"Intent Caching System initialized with cache_size={cache_size}, cache_ttl={cache_ttl}")

    def get_cached_intent(self, query: str, context_hash: Optional[str] = None) -> Optional[Intent]:
        """
        Get a cached intent for a query and context hash if available.
        
        Args:
            query: The user query
            context_hash: Optional hash of relevant context
            
        Returns:
            Cached Intent object if available, None otherwise
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, context_hash)
        
        # Check if key exists and entry is not expired
        if cache_key in self.cache:
            # Check if entry is expired
            if self._is_expired(cache_key):
                # Remove expired entry
                self._remove_entry(cache_key)
                return None
            
            # Update access metrics
            self._update_access(cache_key)
            
            # Return cached intent
            return self.cache[cache_key]
        
        return None

    def cache_intent(self, query: str, intent: Intent, context_hash: Optional[str] = None) -> None:
        """
        Cache an intent for future use.
        
        Args:
            query: The user query
            intent: The Intent object to cache
            context_hash: Optional hash of relevant context
        """
        # Generate cache key
        cache_key = self._generate_cache_key(query, context_hash)
        
        # Check if cache is full
        if len(self.cache) >= self.cache_size and cache_key not in self.cache:
            # Evict least recently used entry
            self._evict_entry()
        
        # Add/update cache entry
        self.cache[cache_key] = intent
        self.access_times[cache_key] = time.time()
        self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
        
        # Update insertion order
        if cache_key in self.insertion_order:
            self.insertion_order.remove(cache_key)
        self.insertion_order.append(cache_key)
        
        logger.debug(f"Cached intent for query: {query[:30]}...")

    def invalidate_cache(self, query_prefix: Optional[str] = None) -> None:
        """
        Invalidate cache entries matching a query prefix.
        
        Args:
            query_prefix: Optional prefix to match for selective invalidation
        """
        if query_prefix is None:
            # Invalidate entire cache
            self.cache.clear()
            self.access_times.clear()
            self.access_count.clear()
            self.insertion_order.clear()
            logger.info("Cache completely invalidated")
        else:
            # Invalidate entries matching prefix
            keys_to_remove = []
            for key in self.cache.keys():
                # Extract query from key (format: "query:context_hash")
                parts = key.split(":", 1)
                if len(parts) > 0 and parts[0].startswith(query_prefix):
                    keys_to_remove.append(key)
            
            # Remove matching entries
            for key in keys_to_remove:
                self._remove_entry(key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching prefix: {query_prefix}")

    def _generate_cache_key(self, query: str, context_hash: Optional[str] = None) -> str:
        """Generate a cache key from query and context hash."""
        # Normalize query (lowercase, remove extra whitespace)
        normalized_query = " ".join(query.lower().split())
        
        if context_hash:
            return f"{normalized_query}:{context_hash}"
        return normalized_query

    def _is_expired(self, cache_key: str) -> bool:
        """Check if a cache entry is expired."""
        # Get last access time
        last_access = self.access_times.get(cache_key, 0)
        # Check if entry is expired
        return (time.time() - last_access) > self.cache_ttl

    def _update_access(self, cache_key: str) -> None:
        """Update access metrics for a cache entry."""
        self.access_times[cache_key] = time.time()
        self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.access_times:
            del self.access_times[cache_key]
        if cache_key in self.access_count:
            del self.access_count[cache_key]
        if cache_key in self.insertion_order:
            self.insertion_order.remove(cache_key)

    def _evict_entry(self) -> None:
        """Evict the least valuable cache entry."""
        # Simple strategy: evict oldest entry
        if self.insertion_order:
            oldest_key = self.insertion_order[0]
            self._remove_entry(oldest_key)
            logger.debug(f"Evicted oldest cache entry")

    def generate_context_hash(self, context: Dict[str, Any]) -> str:
        """
        Generate a hash for context data to use in cache keys.
        
        Args:
            context: Context dictionary
            
        Returns:
            String hash of relevant context
        """
        # Extract only the relevant parts of the context for caching
        relevant_context = {}
        
        # Include conversation history if available
        if "history" in context:
            # Only use the last few messages to avoid too much specificity
            history = context.get("history", [])
            relevant_context["history"] = history[-3:] if len(history) > 3 else history
        
        # Include system state if available
        if "system_state" in context:
            relevant_context["system_state"] = context["system_state"]
        
        # Convert to string and hash
        context_str = json.dumps(relevant_context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()


def initialize_system(registry: Any) -> IntentCachingSystem:
    """
    Initialize the intent caching system.
    
    Args:
        registry: The component registry
        
    Returns:
        An initialized IntentCachingSystem instance
    """
    # Create caching system
    caching_system = IntentCachingSystem(registry)
    
    # Register with registry
    if hasattr(registry, "register_component"):
        registry.register_component("intent_caching_system", caching_system)
    
    return caching_system


if __name__ == "__main__":
    print("Intent Caching System - This module should be imported, not run directly.")
