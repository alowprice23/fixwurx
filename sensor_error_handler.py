"""
FixWurx Auditor - Sensor Error Handler

This module provides mechanisms for handling and recovering from errors 
that occur within sensors themselves.
"""

import logging
import time
import traceback
import json
import os
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger('sensor_error_handler')

class SensorErrorHandler:
    """Handles errors within sensors themselves."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SensorErrorHandler.
        
        Args:
            config: Optional configuration for error handling
        """
        self.config = config or {}
        self.error_dir = self.config.get("error_dir", "auditor_data/sensor_errors")
        os.makedirs(self.error_dir, exist_ok=True)
        
        # Recovery strategies
        self.recovery_strategies = {
            "restart": self._restart_sensor,
            "quarantine": self._quarantine_sensor,
            "fallback": self._use_fallback_data,
            "reset": self._reset_sensor_state,
        }
        
        # Error counts per sensor
        self.error_counts: Dict[str, Dict[str, Any]] = {}
        
        # Error thresholds
        self.thresholds = {
            "max_errors_per_hour": self.config.get("max_errors_per_hour", 5),
            "max_consecutive_errors": self.config.get("max_consecutive_errors", 3),
            "error_window_seconds": self.config.get("error_window_seconds", 3600),
        }
        
        # Recovery callbacks
        self.recovery_callbacks: Dict[str, Callable] = {}
        
        logger.info(f"Initialized SensorErrorHandler with storage at {self.error_dir}")
    
    def register_recovery_callback(self, strategy: str, callback: Callable):
        """
        Register a callback for a recovery strategy.
        
        Args:
            strategy: Name of the recovery strategy
            callback: Function to call for recovery
        """
        self.recovery_callbacks[strategy] = callback
        logger.info(f"Registered recovery callback for strategy: {strategy}")
    
    def handle_sensor_error(self, sensor_id: str, error: Exception, 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """
        Handle an error that occurred within a sensor.
        
        Args:
            sensor_id: ID of the sensor that experienced the error
            error: The exception that was raised
            context: Additional context about the error
            
        Returns:
            The recovery strategy used
        """
        # Record error
        self._record_error(sensor_id, error, context)
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(sensor_id, error)
        
        # Execute recovery
        self._execute_recovery(sensor_id, strategy, context)
        
        logger.info(f"Applied recovery strategy '{strategy}' for sensor {sensor_id}")
        return strategy
    
    def _record_error(self, sensor_id: str, error: Exception, 
                     context: Optional[Dict[str, Any]] = None):
        """
        Record an error for analysis.
        
        Args:
            sensor_id: ID of the sensor
            error: The exception
            context: Additional context
        """
        # Initialize error counts if needed
        if sensor_id not in self.error_counts:
            self.error_counts[sensor_id] = {
                "total": 0,
                "consecutive": 0,
                "timestamps": []
            }
        
        # Update error counts
        self.error_counts[sensor_id]["total"] += 1
        self.error_counts[sensor_id]["consecutive"] += 1
        self.error_counts[sensor_id]["timestamps"].append(time.time())
        
        # Prepare error data
        error_data = {
            "timestamp": time.time(),
            "sensor_id": sensor_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        
        # Save to file
        filename = f"{sensor_id}_{int(time.time())}.json"
        filepath = os.path.join(self.error_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(error_data, f, indent=2)
        
        logger.error(f"Sensor error in {sensor_id}: {str(error)}")
    
    def _determine_recovery_strategy(self, sensor_id: str, error: Exception) -> str:
        """
        Determine the best recovery strategy based on error pattern.
        
        Args:
            sensor_id: ID of the sensor
            error: The exception
            
        Returns:
            Name of the recovery strategy to use
        """
        error_info = self.error_counts[sensor_id]
        
        # Check consecutive errors
        if error_info["consecutive"] >= self.thresholds["max_consecutive_errors"]:
            return "quarantine"
        
        # Check error frequency
        recent_errors = 0
        cutoff_time = time.time() - self.thresholds["error_window_seconds"]
        
        for timestamp in error_info["timestamps"]:
            if timestamp >= cutoff_time:
                recent_errors += 1
        
        if recent_errors >= self.thresholds["max_errors_per_hour"]:
            return "quarantine"
        
        # Default strategy based on error type
        if isinstance(error, (MemoryError, PermissionError)):
            return "restart"
        elif isinstance(error, (ValueError, TypeError)):
            return "reset"
        else:
            return "fallback"
    
    def _execute_recovery(self, sensor_id: str, strategy: str, 
                         context: Optional[Dict[str, Any]] = None):
        """
        Execute the selected recovery strategy.
        
        Args:
            sensor_id: ID of the sensor
            strategy: Name of the recovery strategy
            context: Additional context
        """
        # Log recovery attempt
        logger.info(f"Attempting recovery for {sensor_id} using strategy: {strategy}")
        
        # Execute built-in strategy
        if strategy in self.recovery_strategies:
            self.recovery_strategies[strategy](sensor_id, context)
        
        # Execute callback if registered
        if strategy in self.recovery_callbacks:
            try:
                self.recovery_callbacks[strategy](sensor_id, context)
            except Exception as e:
                logger.error(f"Error in recovery callback for {sensor_id}: {str(e)}")
    
    def _restart_sensor(self, sensor_id: str, context: Optional[Dict[str, Any]] = None):
        """Restart the sensor (placeholder implementation)."""
        logger.info(f"Restarting sensor: {sensor_id}")
        # In a real implementation, this would connect to the SensorThreadingManager
        # to stop and restart the sensor thread
    
    def _quarantine_sensor(self, sensor_id: str, context: Optional[Dict[str, Any]] = None):
        """Quarantine the sensor to prevent further errors (placeholder implementation)."""
        logger.info(f"Quarantining sensor: {sensor_id}")
        # In a real implementation, this would disable the sensor and notify administrators
    
    def _use_fallback_data(self, sensor_id: str, context: Optional[Dict[str, Any]] = None):
        """Use fallback data when the sensor fails (placeholder implementation)."""
        logger.info(f"Using fallback data for sensor: {sensor_id}")
        # In a real implementation, this would provide cached or default data
    
    def _reset_sensor_state(self, sensor_id: str, context: Optional[Dict[str, Any]] = None):
        """Reset the sensor's internal state (placeholder implementation)."""
        logger.info(f"Resetting state for sensor: {sensor_id}")
        # In a real implementation, this would clear the sensor's internal state
        
        # Reset consecutive error count
        if sensor_id in self.error_counts:
            self.error_counts[sensor_id]["consecutive"] = 0
    
    def get_sensor_health(self, sensor_id: str) -> Dict[str, Any]:
        """
        Get health information for a sensor based on its error history.
        
        Args:
            sensor_id: ID of the sensor
            
        Returns:
            Health information dictionary
        """
        if sensor_id not in self.error_counts:
            return {
                "status": "healthy",
                "error_count": 0,
                "consecutive_errors": 0,
                "recent_errors": 0
            }
        
        error_info = self.error_counts[sensor_id]
        
        # Count recent errors
        recent_errors = 0
        cutoff_time = time.time() - self.thresholds["error_window_seconds"]
        
        for timestamp in error_info["timestamps"]:
            if timestamp >= cutoff_time:
                recent_errors += 1
        
        # Determine status
        status = "healthy"
        if error_info["consecutive"] >= self.thresholds["max_consecutive_errors"]:
            status = "failing"
        elif recent_errors >= self.thresholds["max_errors_per_hour"]:
            status = "degraded"
        
        return {
            "status": status,
            "error_count": error_info["total"],
            "consecutive_errors": error_info["consecutive"],
            "recent_errors": recent_errors
        }

# Factory function
def create_sensor_error_handler(config: Optional[Dict[str, Any]] = None) -> SensorErrorHandler:
    """
    Create and initialize a SensorErrorHandler.
    
    Args:
        config: Optional configuration for error handling
        
    Returns:
        Initialized SensorErrorHandler
    """
    return SensorErrorHandler(config)
