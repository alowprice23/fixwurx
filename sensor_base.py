"""
FixWurx Auditor Sensor Base

This module implements the base sensor interface for all error sensors
in the auditor framework. It defines the standard methods and properties
that all sensors must implement.
"""

import datetime
import logging
from typing import Dict, List, Any, Optional

# Import the ErrorReport class
from error_report import ErrorReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SensorBase] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sensor_base')


class ErrorSensor:
    """
    Base interface for all error sensors in the system.
    Defines the standard methods that all sensors must implement.
    """
    
    def __init__(self, sensor_id: str, component_name: str, config: Dict[str, Any] = None):
        """
        Initialize a new error sensor.
        
        Args:
            sensor_id: Unique identifier for this sensor
            component_name: Name of the component being monitored
            config: Configuration options for this sensor
        """
        self.sensor_id = sensor_id
        self.component_name = component_name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.sensitivity = self.config.get("sensitivity", 1.0)
        self.error_reports = []  # Type: List[ErrorReport]
        self.last_check_time = None
        self.threshold_overrides = self.config.get("threshold_overrides", {})
        self.monitor_frequency = self.config.get("monitor_frequency_seconds", 60)
        self.auto_acknowledge = self.config.get("auto_acknowledge", False)
        logger.info(f"Initialized {self.__class__.__name__} for {component_name}")
    
    def monitor(self, data: Any) -> List[ErrorReport]:
        """
        Monitor component data for errors.
        
        Args:
            data: Data to monitor for errors
            
        Returns:
            List of new error reports, if any
        """
        # This method should be overridden by subclasses
        self.last_check_time = datetime.datetime.now()
        return []
    
    def report_error(self, error_type: str, severity: str, details: Dict[str, Any], 
                     context: Optional[Dict[str, Any]] = None) -> ErrorReport:
        """
        Generate and record an error report.
        
        Args:
            error_type: Type of error detected
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            details: Specific details about the error
            context: Additional context information
            
        Returns:
            The generated error report
        """
        # Check if sensor is enabled
        if not self.enabled:
            logger.debug(f"Sensor {self.sensor_id} is disabled, not reporting error")
            return None
        
        # Apply sensitivity filtering - higher severity errors are less affected by low sensitivity
        severity_weights = {
            "CRITICAL": 0.25,
            "HIGH": 0.5,
            "MEDIUM": 0.75,
            "LOW": 1.0
        }
        
        severity_weight = severity_weights.get(severity, 1.0)
        adjusted_sensitivity = 1.0 - ((1.0 - self.sensitivity) * severity_weight)
        
        # Random chance to ignore error based on sensitivity
        import random
        if random.random() > adjusted_sensitivity:
            logger.debug(f"Sensor {self.sensor_id} filtered error due to sensitivity setting")
            return None
        
        # Create error report
        report = ErrorReport(
            sensor_id=self.sensor_id,
            component_name=self.component_name,
            error_type=error_type,
            severity=severity,
            details=details,
            context=context
        )
        
        # Add to local cache
        self.error_reports.append(report)
        
        # Auto-acknowledge if configured
        if self.auto_acknowledge:
            report.acknowledge()
        
        # Log the error
        logger.warning(f"Error detected by {self.sensor_id}: {error_type} - {severity}")
        
        return report
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of this sensor.
        
        Returns:
            Status dictionary
        """
        return {
            "sensor_id": self.sensor_id,
            "component_name": self.component_name,
            "enabled": self.enabled,
            "sensitivity": self.sensitivity,
            "error_count": len(self.error_reports),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "monitor_frequency": self.monitor_frequency,
            "threshold_overrides": self.threshold_overrides
        }
    
    def clear_reports(self) -> None:
        """Clear stored error reports."""
        logger.info(f"Clearing {len(self.error_reports)} reports from {self.sensor_id}")
        self.error_reports = []
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable this sensor.
        
        Args:
            enabled: Whether the sensor should be enabled
        """
        self.enabled = enabled
        logger.info(f"Sensor {self.sensor_id} is now {'enabled' if enabled else 'disabled'}")
    
    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Set the sensitivity level of this sensor.
        
        Args:
            sensitivity: Sensitivity level (0.0 to 1.0)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))  # Clamp to [0.0, 1.0]
        logger.info(f"Sensor {self.sensor_id} sensitivity set to {self.sensitivity}")
    
    def get_pending_reports(self) -> List[ErrorReport]:
        """
        Get all pending error reports.
        
        Returns:
            List of error reports
        """
        return self.error_reports
    
    def set_threshold_override(self, threshold_name: str, value: Any) -> None:
        """
        Override a specific threshold value for this sensor.
        
        Args:
            threshold_name: Name of the threshold to override
            value: New threshold value
        """
        self.threshold_overrides[threshold_name] = value
        logger.info(f"Sensor {self.sensor_id} threshold {threshold_name} set to {value}")
    
    def reset_threshold_overrides(self) -> None:
        """Reset all threshold overrides to their default values."""
        self.threshold_overrides = {}
        logger.info(f"Sensor {self.sensor_id} threshold overrides reset to defaults")
    
    def set_monitor_frequency(self, frequency_seconds: int) -> None:
        """
        Set the monitoring frequency for this sensor.
        
        Args:
            frequency_seconds: Monitoring frequency in seconds
        """
        self.monitor_frequency = max(1, frequency_seconds)  # Minimum 1 second
        logger.info(f"Sensor {self.sensor_id} monitor frequency set to {self.monitor_frequency}s")
    
    def should_monitor(self) -> bool:
        """
        Check if the sensor should perform monitoring based on its frequency.
        
        Returns:
            True if monitoring should be performed, False otherwise
        """
        if not self.last_check_time:
            return True
        
        elapsed = (datetime.datetime.now() - self.last_check_time).total_seconds()
        return elapsed >= self.monitor_frequency
    
    def _get_threshold(self, name: str, default_value: Any) -> Any:
        """
        Get a threshold value, considering overrides.
        
        Args:
            name: Threshold name
            default_value: Default threshold value
            
        Returns:
            The threshold value, considering overrides
        """
        return self.threshold_overrides.get(name, self.config.get(name, default_value))
