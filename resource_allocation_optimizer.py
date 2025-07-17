#!/usr/bin/env python3
"""
resource_allocation_optimizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compatibility layer for the resource allocation optimizer.

This file provides the resource allocation optimizer component that analyzes
usage patterns and optimizes resource allocation in the FixWurx system.
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ResourceAllocationOptimizer")

class ResourceAllocationOptimizer:
    """
    Analyzes resource usage patterns and optimizes resource allocation.
    """
    
    def __init__(self, optimization_interval_sec: int = 30, config: Optional[Dict[str, Any]] = None):
        """Initialize the resource allocation optimizer."""
        self.optimization_interval_sec = optimization_interval_sec
        self.config = config or {}
        self.usage_history = []
        self.current_usage_ratio = 0.5  # Default to 50% usage
        self.predicted_utilization = 0.5  # Default to 50% predicted utilization
        self.is_in_burst_mode = False
        self.running = False
        self.thread = None
        
        # Configure burst mode settings
        self.burst_mode_threshold = self.config.get("burst_mode_threshold", 0.85)
        self.burst_mode_cooldown_sec = self.config.get("burst_mode_cooldown_sec", 300)
        self.last_burst_mode_time = 0
        
        logger.info("ResourceAllocationOptimizer initialized")
        
    def start(self):
        """Start the resource optimizer."""
        if not self.running:
            self.running = True
            logger.info("ResourceAllocationOptimizer started")
            
    def stop(self):
        """Stop the resource optimizer."""
        if self.running:
            self.running = False
            logger.info("ResourceAllocationOptimizer stopped")
            
    def collect_usage_snapshot(self) -> Dict[str, Any]:
        """Collect a snapshot of current resource usage."""
        snapshot = {
            "usage_ratio": self.current_usage_ratio,
            "predicted_utilization": self.predicted_utilization,
            "timestamp": time.time(),
            "in_burst_mode": self.is_in_burst_mode
        }
        
        self.usage_history.append(snapshot)
        # Keep history limited to last 100 entries
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]
            
        return snapshot
    
    def update_predictions(self):
        """Update utilization predictions based on recent usage patterns."""
        # This would normally use a more sophisticated algorithm, possibly
        # incorporating machine learning for prediction
        if not self.usage_history:
            return
            
        # Simple moving average of recent usage
        recent_usage = [entry["usage_ratio"] for entry in self.usage_history[-10:]]
        if recent_usage:
            avg_usage = sum(recent_usage) / len(recent_usage)
            # Add a small buffer to prediction
            self.predicted_utilization = min(1.0, avg_usage * 1.1)
            
        # Check for burst mode threshold
        if self.current_usage_ratio > self.burst_mode_threshold:
            # If not already in burst mode, enter burst mode
            if not self.is_in_burst_mode:
                self.set_burst_mode(True)
        elif self.is_in_burst_mode:
            # Check if cooldown period has passed
            current_time = time.time()
            if current_time - self.last_burst_mode_time > self.burst_mode_cooldown_sec:
                self.set_burst_mode(False)
                
    def set_usage_ratio(self, ratio: float):
        """Set the current usage ratio."""
        self.current_usage_ratio = max(0.0, min(1.0, ratio))
        
    # Alias for compatibility with tests that use set_current_usage_ratio
    def set_current_usage_ratio(self, ratio: float):
        """Alias for set_usage_ratio for backward compatibility."""
        self.set_usage_ratio(ratio)
        
    def set_predicted_utilization(self, prediction: float):
        """Set the predicted utilization."""
        self.predicted_utilization = max(0.0, min(1.0, prediction))
        
    def set_burst_mode(self, enabled: bool):
        """Enable or disable burst mode."""
        if enabled != self.is_in_burst_mode:
            self.is_in_burst_mode = enabled
            if enabled:
                self.last_burst_mode_time = time.time()
                logger.info("Burst mode activated")
            else:
                logger.info("Burst mode deactivated")
                
    # Alias for compatibility with tests that use set_in_burst_mode
    def set_in_burst_mode(self, enabled: bool):
        """Alias for set_burst_mode for backward compatibility."""
        self.set_burst_mode(enabled)
        
    def calculate_target_agents(self, current_agents: int, max_agents: int) -> int:
        """Calculate target number of agents based on predictions."""
        if self.is_in_burst_mode:
            # In burst mode, use maximum available
            return max_agents
            
        # Calculate target based on predicted utilization
        predicted_needed = int(current_agents * self.predicted_utilization / 0.7)  # Target 70% utilization
        return min(max_agents, max(1, predicted_needed))
        
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics."""
        return {
            "current_usage_ratio": self.current_usage_ratio,
            "predicted_utilization": self.predicted_utilization,
            "in_burst_mode": self.is_in_burst_mode,
            "burst_mode_threshold": self.burst_mode_threshold,
            "history_size": len(self.usage_history),
            "last_burst_mode_time": self.last_burst_mode_time
        }
        
    def apply_optimization_plan(self, plan: Dict[str, Any]) -> bool:
        """Apply an optimization plan."""
        # This is just a stub in the compatibility layer
        if "burst_mode" in plan:
            self.set_burst_mode(plan["burst_mode"])
            
        if "usage_ratio" in plan:
            self.set_usage_ratio(plan["usage_ratio"])
            
        if "predicted_utilization" in plan:
            self.set_predicted_utilization(plan["predicted_utilization"])
            
        return True
