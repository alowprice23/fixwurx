"""
FixWurx Auditor Sensor Manager

This module implements the SensorManager which coordinates sensor activities,
error collection, and handles the scheduling of sensor monitoring.
"""

import os
import logging
import datetime
import threading
import time
from typing import Dict, List, Set, Any, Optional, Union, Callable

# Import sensor components
from error_report import ErrorReport
from sensor_base import ErrorSensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SensorManager] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sensor_manager')


class SensorManager:
    """
    Manages sensors and coordinates error collection.
    """
    
    def __init__(self, registry, config: Dict[str, Any] = None):
        """
        Initialize the sensor manager.
        
        Args:
            registry: Sensor registry to use
            config: Configuration options
        """
        self.registry = registry
        self.config = config or {}
        self.enabled = self.config.get("sensors_enabled", True)
        self.collection_interval = self.config.get("collection_interval_seconds", 60)
        self.last_collection_time = None
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitor_lock = threading.Lock()
        self.monitor_callbacks = []  # Callbacks when errors are detected
        self.prioritized_components = self.config.get("prioritized_components", [])
        
        logger.info(f"Initialized SensorManager (enabled: {self.enabled}, interval: {self.collection_interval}s)")
    
    def register_component_sensors(self, component_name: str, 
                                 sensors: List[ErrorSensor]) -> None:
        """
        Register multiple sensors for a component.
        
        Args:
            component_name: Name of the component
            sensors: List of sensors to register
        """
        for sensor in sensors:
            self.registry.register_sensor(sensor)
        
        logger.info(f"Registered {len(sensors)} sensors for component {component_name}")
    
    def collect_errors(self, force: bool = False) -> List[ErrorReport]:
        """
        Collect errors from all sensors.
        
        Args:
            force: Whether to force collection even if the interval hasn't elapsed
            
        Returns:
            List of new error reports
        """
        if not self.enabled:
            logger.info("Sensor Manager is disabled, not collecting errors")
            return []
        
        now = datetime.datetime.now()
        
        # Check if enough time has elapsed since last collection
        if not force and self.last_collection_time:
            elapsed = (now - self.last_collection_time).total_seconds()
            if elapsed < self.collection_interval:
                logger.debug(f"Not collecting errors, only {elapsed}s elapsed since last collection")
                return []
        
        # Collect errors
        self.last_collection_time = now
        reports = self.registry.collect_errors()
        
        # Notify callbacks if errors are found
        if reports and self.monitor_callbacks:
            for callback in self.monitor_callbacks:
                try:
                    callback(reports)
                except Exception as e:
                    logger.error(f"Error in monitor callback: {e}")
        
        return reports
    
    def monitor_component(self, component_name: str, data: Any) -> List[ErrorReport]:
        """
        Monitor a component for errors.
        
        Args:
            component_name: Name of the component to monitor
            data: Component data to monitor
            
        Returns:
            List of new error reports
        """
        if not self.enabled:
            return []
        
        reports = []
        
        # Get sensors for this component
        sensors = self.registry.get_sensors_for_component(component_name)
        
        # Monitor with each sensor
        for sensor in sensors:
            if sensor.enabled and sensor.should_monitor():
                try:
                    sensor_reports = sensor.monitor(data)
                    if sensor_reports:
                        reports.extend(sensor_reports)
                except Exception as e:
                    logger.error(f"Error monitoring with sensor {sensor.sensor_id}: {e}")
        
        return reports
    
    def start_monitoring(self) -> None:
        """Start the background monitoring thread."""
        if self.is_monitoring:
            logger.warning("Monitoring thread already running")
            return
        
        with self.monitor_lock:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Monitoring thread started")
    
    def stop_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        if not self.is_monitoring:
            logger.warning("Monitoring thread not running")
            return
        
        with self.monitor_lock:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
                self.monitoring_thread = None
                logger.info("Monitoring thread stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                # Start with prioritized components
                for component_name in self.prioritized_components:
                    self._monitor_single_component(component_name)
                
                # Then monitor all other components
                components = self.registry.get_all_component_names()
                for component_name in components:
                    if component_name not in self.prioritized_components:
                        self._monitor_single_component(component_name)
                
                # Collect errors
                self.collect_errors()
                
                # Sleep before next cycle
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Sleep longer after error
    
    def _monitor_single_component(self, component_name: str) -> None:
        """
        Monitor a single component.
        
        Args:
            component_name: Name of the component to monitor
        """
        try:
            # Get component data
            data = self._get_component_data(component_name)
            if data:
                # Monitor the component
                reports = self.monitor_component(component_name, data)
                if reports:
                    logger.info(f"Detected {len(reports)} errors in component {component_name}")
        except Exception as e:
            logger.error(f"Error monitoring component {component_name}: {e}")
    
    def _get_component_data(self, component_name: str) -> Any:
        """
        Get data for a component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component data, or None if not available
        """
        # This would be more sophisticated in a real implementation,
        # querying the component for its current state
        # For now, just return a placeholder
        return {"component_name": component_name}
    
    def register_error_callback(self, callback: Callable[[List[ErrorReport]], None]) -> None:
        """
        Register a callback to be called when errors are collected.
        
        Args:
            callback: Function to call with the list of new error reports
        """
        if callback not in self.monitor_callbacks:
            self.monitor_callbacks.append(callback)
            logger.info("Registered error callback")
    
    def unregister_error_callback(self, callback: Callable[[List[ErrorReport]], None]) -> None:
        """
        Unregister an error callback.
        
        Args:
            callback: The callback to unregister
        """
        if callback in self.monitor_callbacks:
            self.monitor_callbacks.remove(callback)
            logger.info("Unregistered error callback")
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the sensor manager.
        
        Args:
            enabled: Whether the manager should be enabled
        """
        self.enabled = enabled
        logger.info(f"Sensor Manager is now {'enabled' if enabled else 'disabled'}")
    
    def set_collection_interval(self, interval: int) -> None:
        """
        Set the collection interval.
        
        Args:
            interval: Collection interval in seconds
        """
        self.collection_interval = max(1, interval)  # Minimum 1 second
        logger.info(f"Collection interval set to {self.collection_interval}s")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the sensor manager.
        
        Returns:
            Status dictionary
        """
        return {
            "enabled": self.enabled,
            "collection_interval": self.collection_interval,
            "last_collection_time": self.last_check_time.isoformat() if self.last_collection_time else None,
            "is_monitoring": self.is_monitoring,
            "prioritized_components": self.prioritized_components,
            "registry_status": self.registry.get_sensor_status()
        }
    
    def add_prioritized_component(self, component_name: str) -> None:
        """
        Add a component to the prioritized list for monitoring.
        
        Args:
            component_name: Name of the component to prioritize
        """
        if component_name not in self.prioritized_components:
            self.prioritized_components.append(component_name)
            logger.info(f"Added {component_name} to prioritized components")
    
    def remove_prioritized_component(self, component_name: str) -> None:
        """
        Remove a component from the prioritized list.
        
        Args:
            component_name: Name of the component to remove from priority
        """
        if component_name in self.prioritized_components:
            self.prioritized_components.remove(component_name)
            logger.info(f"Removed {component_name} from prioritized components")
