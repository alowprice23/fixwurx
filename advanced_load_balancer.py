#!/usr/bin/env python3
"""
advanced_load_balancer.py
━━━━━━━━━━━━━━━━━━━━━━━━━
Compatibility layer for the new async load balancer implementation.

This file provides backward compatibility for existing code that imports
AdvancedLoadBalancer and BalancingStrategy from the old implementation.
"""

import logging
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional, Set
from enum import Enum, auto

# Import from the new implementation
from async_load_balancer import (
    AsyncLoadBalancer,
    BalancingStrategy,
    Worker,
    BugRouting,
    State,
    AffinityType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AdvancedLoadBalancer")

# Export BalancingStrategy directly from async_load_balancer
# This is already an Enum which should be compatible

# Create a wrapper class for the new AsyncLoadBalancer that exposes the same
# interface as the old AdvancedLoadBalancer
class AdvancedLoadBalancer:
    """
    Compatibility wrapper for AsyncLoadBalancer that provides the same interface
    as the old AdvancedLoadBalancer implementation.
    """
    
    def __init__(self, strategy: BalancingStrategy, 
                health_check_interval_sec: int = 5, 
                scaling_coordinator=None, 
                resource_optimizer=None,
                config: Optional[Dict[str, Any]] = None,
                testing_mode: bool = False):
        """Initialize the advanced load balancer."""
        self.strategy = strategy
        self.health_check_interval_sec = health_check_interval_sec
        self.scaling_coordinator = scaling_coordinator
        self.resource_optimizer = resource_optimizer
        self.testing_mode = testing_mode
        
        # Set default configuration
        self.config = config or {}
        self.hash_replicas = self.config.get("hash_replicas", 100)
        self.enable_sticky_routing = self.config.get("sticky_bugs", True)
        self.sticky_expiration_sec = self.config.get("sticky_expiration_sec", 300)
        self.enable_affinity_routing = self.config.get("enable_affinity_routing", True)
        self.affinity_weight = self.config.get("affinity_weight", 0.3)
        self.enable_predictive_routing = self.config.get("enable_predictive_routing", True)
        self.prediction_weight = self.config.get("prediction_weight", 0.2)
        self.strategy_update_interval_sec = self.config.get("strategy_update_interval_sec", 60)
        self.auto_strategy_selection = self.config.get("auto_strategy_selection", False)
        
        # Create the async load balancer
        self._async_lb = AsyncLoadBalancer(
            config=self.config,
            test_mode=self.testing_mode
        )
        
        # We need to maintain a synchronous interface, so we'll use an event loop
        # in a separate thread for async operations
        self._loop = asyncio.new_event_loop()
        self._thread = None
        self._is_running = False
        self._is_initialized = False
        
        # Initialize worker bug assignments
        self.worker_bug_assignments = {}
        self.bug_routing = {}
        self.workers = {}
        self.worker_metrics = {}
        self.worker_affinities = {}
        self.worker_load_predictions = {}
        self.strategy_scores = {
            strategy: 0.5 for strategy in BalancingStrategy
        }
        
    def _run_loop(self):
        """Run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        
    def _run_coroutine(self, coro):
        """Run a coroutine in the event loop and return the result."""
        if not self._is_initialized:
            self._initialize_async()
            
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()
        
    def _initialize_async(self):
        """Initialize the async infrastructure."""
        if self._is_initialized:
            return
            
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._is_initialized = True
        
    def start(self):
        """Start the load balancer."""
        if self._is_running:
            return
            
        self._run_coroutine(self._async_lb.start())
        self._is_running = True
        
        # Initial synchronization
        if self.scaling_coordinator:
            self._sync_with_scaling_coordinator()
            
        logger.info("Advanced Load Balancer started.")

    def stop(self):
        """Stop the load balancer."""
        if not self._is_running:
            return
            
        self._run_coroutine(self._async_lb.stop())
        self._is_running = False
        
        logger.info("Advanced Load Balancer stopped.")
        
    def _sync_with_scaling_coordinator(self):
        """Synchronize with scaling coordinator."""
        if not self.scaling_coordinator:
            return
            
        # Get worker status from scaling coordinator
        workers = self.scaling_coordinator.get_worker_status()
        
        # Update worker metrics
        for worker in workers:
            worker_id = worker["worker_id"]
            
            # Add worker to async load balancer
            self._run_coroutine(self._async_lb.add_worker(
                worker_id=worker_id,
                capacity=worker.get("total_agents", 0),
                metadata={
                    "state": worker.get("state", "unknown"),
                    "affinities": worker.get("affinities", {}),
                    "languages": worker.get("languages", {})
                }
            ))
            
            # Update local caches for backward compatibility
            self.workers[worker_id] = {
                "id": worker_id,
                "metadata": worker.get("metadata", {}),
                "active_connections": worker.get("total_agents", 0) - worker.get("free_agents", 0),
                "healthy": worker.get("state") == "ready",
            }
            
            self.worker_bug_assignments[worker_id] = set()

    def add_worker(self, worker_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a worker to the load balancer."""
        if not self._is_running:
            self.start()
            
        self._run_coroutine(self._async_lb.add_worker(
            worker_id=worker_id,
            capacity=10,  # Default capacity
            metadata=metadata or {}
        ))
        
        # Update local caches for backward compatibility
        self.workers[worker_id] = {
            "id": worker_id,
            "metadata": metadata or {},
            "active_connections": 0,
            "healthy": True,
        }
        
        self.worker_bug_assignments[worker_id] = set()
        
        logger.info(f"Added worker {worker_id} to load balancer.")

    def remove_worker(self, worker_id: str):
        """Remove a worker from the load balancer."""
        if not self._is_running:
            return
            
        self._run_coroutine(self._async_lb.remove_worker(worker_id))
        
        # Update local caches for backward compatibility
        if worker_id in self.workers:
            del self.workers[worker_id]
            
        if worker_id in self.worker_bug_assignments:
            del self.worker_bug_assignments[worker_id]
            
        logger.info(f"Removed worker {worker_id} from load balancer.")

    def select_worker(self, bug_id: str, requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Select a worker for a bug."""
        if not self._is_running:
            self.start()
            
        return self._run_coroutine(self._async_lb.select_worker(bug_id, requirements))

    def register_task_completion(self, bug_id: str, worker_id: str, success: bool, processing_time_ms: int):
        """Register a task completion."""
        if not self._is_running:
            return
            
        self._run_coroutine(self._async_lb.register_task_completion(
            bug_id, worker_id, success, processing_time_ms
        ))

    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Get advanced metrics about the load balancer."""
        if not self._is_running:
            return {
                "worker_count": 0,
                "active_bugs_count": 0,
                "sticky_routing_enabled": self.enable_sticky_routing,
                "affinity_routing_enabled": self.enable_affinity_routing,
                "predictive_routing_enabled": self.enable_predictive_routing,
                "current_strategy": self.strategy.name,
                "strategy_scores": {s.name: score for s, score in self.strategy_scores.items()},
                "auto_strategy_selection": self.auto_strategy_selection,
                "hash_ring_size": 0,
                "worker_affinities_count": 0,
                "testing_mode": self.testing_mode
            }
            
        metrics = self._run_coroutine(self._async_lb.get_metrics())
        
        # Convert to format expected by old code
        return {
            "worker_count": metrics.get("worker_count", 0),
            "active_bugs_count": metrics.get("active_tasks_count", 0),
            "sticky_routing_enabled": self.enable_sticky_routing,
            "affinity_routing_enabled": self.enable_affinity_routing,
            "predictive_routing_enabled": self.enable_predictive_routing,
            "current_strategy": self.strategy.name,
            "strategy_scores": {s.name: score for s, score in self.strategy_scores.items()},
            "auto_strategy_selection": self.auto_strategy_selection,
            "hash_ring_size": len(self.workers),
            "worker_affinities_count": len(self.workers),
            "testing_mode": self.testing_mode
        }

    def set_sticky_routing(self, enabled: bool):
        """Enable or disable sticky routing."""
        self.enable_sticky_routing = enabled
        self.config["sticky_bugs"] = enabled

    def set_affinity_routing(self, enabled: bool):
        """Enable or disable affinity-based routing."""
        self.enable_affinity_routing = enabled
        self.config["enable_affinity_routing"] = enabled

    def set_predictive_routing(self, enabled: bool):
        """Enable or disable predictive routing."""
        self.enable_predictive_routing = enabled
        self.config["enable_predictive_routing"] = enabled

    def check_worker_health(self):
        """Check the health of worker nodes."""
        # This is automatically handled by the async load balancer
        pass
        
    def update_worker_status(self, workers):
        """Update worker status based on a list of worker data."""
        if not self._is_running:
            self.start()
            
        for worker in workers:
            worker_id = worker.get("worker_id")
            if not worker_id:
                continue
                
            # Check if worker already exists
            if worker_id in self.workers:
                # Update existing worker
                self.workers[worker_id].update({
                    "active_connections": worker.get("total_agents", 0) - worker.get("free_agents", 0),
                    "healthy": True,
                    "metadata": worker.get("metadata", {})
                })
            else:
                # Add new worker
                self.add_worker(worker_id, worker.get("metadata", {}))
                
            # Update the worker in the async load balancer
            self._run_coroutine(self._async_lb.add_worker(
                worker_id=worker_id,
                capacity=worker.get("total_agents", 10),
                metadata={
                    "free_agents": worker.get("free_agents", 0),
                    "total_agents": worker.get("total_agents", 0),
                    "state": worker.get("state", "ready"),
                    "affinities": worker.get("affinities", {}),
                    "languages": worker.get("languages", {})
                }
            ))
