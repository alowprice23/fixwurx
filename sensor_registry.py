"""
FixWurx Auditor Sensor Registry

This module implements the SensorRegistry which serves as the central management
system for all error sensors in the auditor framework. It handles sensor registration,
error collection, aggregation, and reporting.
"""

import os
import logging
import yaml
import json
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Type

# Import sensor components
from error_report import ErrorReport
from sensor_base import ErrorSensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [SensorRegistry] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sensor_registry')


class SensorRegistry:
    """
    Central registry for all error sensors in the system.
    Manages sensor registration, data collection, and aggregation.
    """
    
    def __init__(self, storage_path: str = "auditor_data/sensors"):
        """
        Initialize the sensor registry.
        
        Args:
            storage_path: Path to store sensor data
        """
        self.sensors = {}  # type: Dict[str, ErrorSensor]
        self.error_reports = []  # type: List[ErrorReport]
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"Initialized SensorRegistry with storage at {storage_path}")
    
    def register_sensor(self, sensor: ErrorSensor) -> None:
        """
        Register a new sensor.
        
        Args:
            sensor: Sensor instance to register
        """
        if sensor.sensor_id in self.sensors:
            logger.warning(f"Sensor {sensor.sensor_id} already registered, replacing")
        
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Registered sensor: {sensor.sensor_id} for {sensor.component_name}")
    
    def unregister_sensor(self, sensor_id: str) -> None:
        """
        Unregister a sensor.
        
        Args:
            sensor_id: ID of the sensor to unregister
        """
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            logger.info(f"Unregistered sensor: {sensor_id}")
        else:
            logger.warning(f"Sensor {sensor_id} not found, cannot unregister")
    
    def collect_errors(self) -> List[ErrorReport]:
        """
        Collect errors from all sensors.
        
        Returns:
            List of new error reports
        """
        new_reports = []
        
        for sensor_id, sensor in self.sensors.items():
            if not sensor.enabled:
                continue
            
            # Get pending reports from this sensor
            reports = sensor.get_pending_reports()
            if reports:
                new_reports.extend(reports)
                # Clear reports from sensor
                sensor.clear_reports()
        
        # Add to global list
        self.error_reports.extend(new_reports)
        
        # Store reports to disk
        self._store_reports(new_reports)
        
        logger.info(f"Collected {len(new_reports)} new error reports from {len(self.sensors)} sensors")
        return new_reports
    
    def get_sensor(self, sensor_id: str) -> Optional[ErrorSensor]:
        """
        Get a specific sensor by ID.
        
        Args:
            sensor_id: ID of the sensor to get
            
        Returns:
            Sensor instance, or None if not found
        """
        return self.sensors.get(sensor_id)
    
    def get_sensors_for_component(self, component_name: str) -> List[ErrorSensor]:
        """
        Get all sensors for a specific component.
        
        Args:
            component_name: Name of the component
            
        Returns:
            List of sensors for the component
        """
        return [sensor for sensor in self.sensors.values() 
                if sensor.component_name == component_name]
    
    def get_all_component_names(self) -> List[str]:
        """
        Get names of all components that have sensors.
        
        Returns:
            List of component names
        """
        return list(set(sensor.component_name for sensor in self.sensors.values()))
    
    def get_sensor_status(self, sensor_id: str = None) -> Dict[str, Any]:
        """
        Get status of a specific sensor or all sensors.
        
        Args:
            sensor_id: ID of the sensor to get status for, or None for all sensors
            
        Returns:
            Status dictionary
        """
        if sensor_id:
            sensor = self.get_sensor(sensor_id)
            if sensor:
                return sensor.get_status()
            else:
                return {"error": f"Sensor {sensor_id} not found"}
        
        # Get status for all sensors
        return {
            "total_sensors": len(self.sensors),
            "enabled_sensors": sum(1 for s in self.sensors.values() if s.enabled),
            "sensors": [s.get_status() for s in self.sensors.values()]
        }
    
    def get_error_trends(self) -> Dict[str, Any]:
        """
        Analyze error trends across sensors.
        
        Returns:
            Dictionary with trend analysis
        """
        # Group errors by component, type, and severity
        by_component = {}
        by_type = {}
        by_severity = {}
        
        for report in self.error_reports:
            # By component
            if report.component_name not in by_component:
                by_component[report.component_name] = 0
            by_component[report.component_name] += 1
            
            # By type
            if report.error_type not in by_type:
                by_type[report.error_type] = 0
            by_type[report.error_type] += 1
            
            # By severity
            if report.severity not in by_severity:
                by_severity[report.severity] = 0
            by_severity[report.severity] += 1
        
        # Create time series data (errors per day)
        time_series = {}
        for report in self.error_reports:
            date = report.timestamp.split('T')[0]  # Extract date part
            if date not in time_series:
                time_series[date] = 0
            time_series[date] += 1
        
        return {
            "total_errors": len(self.error_reports),
            "by_component": by_component,
            "by_type": by_type,
            "by_severity": by_severity,
            "time_series": time_series
        }
    
    def get_error_report(self, error_id: str) -> Optional[ErrorReport]:
        """
        Get a specific error report by ID.
        
        Args:
            error_id: ID of the error report to get
            
        Returns:
            Error report, or None if not found
        """
        for report in self.error_reports:
            if report.error_id == error_id:
                return report
        
        # Try to load from storage
        return self._load_report(error_id)
    
    def resolve_error(self, error_id: str, resolution: str) -> bool:
        """
        Resolve an error report.
        
        Args:
            error_id: ID of the error report to resolve
            resolution: Description of how the error was resolved
            
        Returns:
            True if the error was resolved, False otherwise
        """
        report = self.get_error_report(error_id)
        if report:
            report.resolve(resolution)
            self._store_report(report)  # Update on disk
            logger.info(f"Resolved error {error_id}: {resolution}")
            return True
        return False
    
    def acknowledge_error(self, error_id: str) -> bool:
        """
        Acknowledge an error report.
        
        Args:
            error_id: ID of the error report to acknowledge
            
        Returns:
            True if the error was acknowledged, False otherwise
        """
        report = self.get_error_report(error_id)
        if report:
            report.acknowledge()
            self._store_report(report)  # Update on disk
            logger.info(f"Acknowledged error {error_id}")
            return True
        return False
    
    def query_errors(self, component_name: str = None, error_type: str = None, 
                     severity: str = None, status: str = None) -> List[ErrorReport]:
        """
        Query error reports with filters.
        
        Args:
            component_name: Filter by component name
            error_type: Filter by error type
            severity: Filter by severity
            status: Filter by status
            
        Returns:
            List of matching error reports
        """
        results = []
        
        for report in self.error_reports:
            if component_name and report.component_name != component_name:
                continue
            if error_type and report.error_type != error_type:
                continue
            if severity and report.severity != severity:
                continue
            if status and report.status != status:
                continue
            
            results.append(report)
        
        logger.info(f"Query returned {len(results)} error reports")
        return results
    
    def add_root_cause_to_error(self, error_id: str, root_cause: str) -> bool:
        """
        Add root cause analysis to an error report.
        
        Args:
            error_id: ID of the error report
            root_cause: Description of the root cause
            
        Returns:
            True if successful, False otherwise
        """
        report = self.get_error_report(error_id)
        if report:
            report.add_root_cause(root_cause)
            self._store_report(report)  # Update on disk
            return True
        return False
    
    def add_impact_to_error(self, error_id: str, impact: Dict[str, Any]) -> bool:
        """
        Add impact assessment to an error report.
        
        Args:
            error_id: ID of the error report
            impact: Description of the impact
            
        Returns:
            True if successful, False otherwise
        """
        report = self.get_error_report(error_id)
        if report:
            report.add_impact(impact)
            self._store_report(report)  # Update on disk
            return True
        return False
    
    def add_related_error(self, error_id: str, related_id: str) -> bool:
        """
        Add a related error to an error report.
        
        Args:
            error_id: ID of the error report
            related_id: ID of the related error
            
        Returns:
            True if successful, False otherwise
        """
        report = self.get_error_report(error_id)
        if report:
            report.add_related_error(related_id)
            self._store_report(report)  # Update on disk
            return True
        return False
    
    def add_recommendation_to_error(self, error_id: str, recommendation: Dict[str, Any]) -> bool:
        """
        Add a recommendation to an error report.
        
        Args:
            error_id: ID of the error report
            recommendation: Recommendation details
            
        Returns:
            True if successful, False otherwise
        """
        report = self.get_error_report(error_id)
        if report:
            report.add_recommendation(recommendation)
            self._store_report(report)  # Update on disk
            return True
        return False
    
    def _store_reports(self, reports: List[ErrorReport]) -> None:
        """
        Store error reports to disk.
        
        Args:
            reports: List of error reports to store
        """
        for report in reports:
            self._store_report(report)
    
    def _store_report(self, report: ErrorReport) -> None:
        """
        Store an error report to disk.
        
        Args:
            report: Error report to store
        """
        try:
            filename = os.path.join(self.storage_path, f"{report.error_id}.yaml")
            with open(filename, 'w') as f:
                yaml.dump(report.to_dict(), f)
        except Exception as e:
            logger.error(f"Failed to store error report {report.error_id}: {e}")
    
    def _load_report(self, error_id: str) -> Optional[ErrorReport]:
        """
        Load an error report from disk.
        
        Args:
            error_id: ID of the error report to load
            
        Returns:
            Error report, or None if not found
        """
        try:
            filename = os.path.join(self.storage_path, f"{error_id}.yaml")
            if not os.path.exists(filename):
                return None
            
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
                return ErrorReport.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load error report {error_id}: {e}")
            return None
    
    def load_all_reports(self) -> None:
        """Load all error reports from disk."""
        try:
            # Get all YAML files in storage directory
            files = [f for f in os.listdir(self.storage_path) if f.endswith('.yaml')]
            
            loaded_reports = []
            for file in files:
                error_id = file.replace('.yaml', '')
                report = self._load_report(error_id)
                if report:
                    loaded_reports.append(report)
            
            # Add to in-memory list, avoiding duplicates
            existing_ids = {report.error_id for report in self.error_reports}
            for report in loaded_reports:
                if report.error_id not in existing_ids:
                    self.error_reports.append(report)
                    existing_ids.add(report.error_id)
            
            logger.info(f"Loaded {len(loaded_reports)} error reports from disk")
        except Exception as e:
            logger.error(f"Failed to load error reports: {e}")


class SensorManager:
    """
    Manager for handling multiple sensor registries across components.
    
    This class provides higher-level sensor management capabilities,
    including threading support, error aggregation, and system-wide metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the sensor manager.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.registries = {}  # type: Dict[str, SensorRegistry]
        self.default_registry = None
        self.thread_pool = None
        self.running = False
        self.collection_interval = self.config.get("collection_interval", 60)  # seconds
        
        # Initialize default registry
        self._init_default_registry()
        
        logger.info("Initialized SensorManager")
    
    def _init_default_registry(self) -> None:
        """Initialize the default sensor registry."""
        default_config = self.config.get("default_registry", {})
        storage_path = default_config.get("storage_path", "auditor_data/sensors")
        
        self.default_registry = SensorRegistry(storage_path=storage_path)
        self.registries["default"] = self.default_registry
    
    def create_registry(self, name: str, config: Dict[str, Any] = None) -> SensorRegistry:
        """
        Create a new sensor registry.
        
        Args:
            name: Name for the registry
            config: Configuration options
            
        Returns:
            The created registry
        """
        if name in self.registries:
            logger.warning(f"Registry {name} already exists, returning existing instance")
            return self.registries[name]
        
        config = config or {}
        storage_path = config.get("storage_path", f"auditor_data/sensors/{name}")
        
        registry = SensorRegistry(storage_path=storage_path)
        self.registries[name] = registry
        
        logger.info(f"Created sensor registry: {name}")
        return registry
    
    def get_registry(self, name: str = "default") -> Optional[SensorRegistry]:
        """
        Get a sensor registry by name.
        
        Args:
            name: Name of the registry
            
        Returns:
            SensorRegistry instance, or None if not found
        """
        return self.registries.get(name)
    
    def register_sensor(self, sensor: ErrorSensor, registry_name: str = "default") -> bool:
        """
        Register a sensor with a specific registry.
        
        Args:
            sensor: Sensor to register
            registry_name: Name of the registry to register with
            
        Returns:
            True if successful, False otherwise
        """
        registry = self.get_registry(registry_name)
        if registry:
            registry.register_sensor(sensor)
            return True
        
        logger.error(f"Registry {registry_name} not found, cannot register sensor")
        return False
    
    def collect_all_errors(self) -> Dict[str, List[ErrorReport]]:
        """
        Collect errors from all registries.
        
        Returns:
            Dictionary mapping registry names to lists of error reports
        """
        results = {}
        
        for name, registry in self.registries.items():
            reports = registry.collect_errors()
            results[name] = reports
        
        return results
    
    def get_all_sensor_status(self) -> Dict[str, Any]:
        """
        Get status of all sensors across all registries.
        
        Returns:
            Status dictionary
        """
        registry_statuses = {}
        total_sensors = 0
        enabled_sensors = 0
        
        for name, registry in self.registries.items():
            status = registry.get_sensor_status()
            registry_statuses[name] = status
            total_sensors += status.get("total_sensors", 0)
            enabled_sensors += status.get("enabled_sensors", 0)
        
        return {
            "total_registries": len(self.registries),
            "total_sensors": total_sensors,
            "enabled_sensors": enabled_sensors,
            "registries": registry_statuses
        }
    
    def get_error_report(self, error_id: str, registry_name: str = None) -> Optional[ErrorReport]:
        """
        Get an error report by ID from a specific registry or all registries.
        
        Args:
            error_id: ID of the error report
            registry_name: Name of the registry, or None to search all
            
        Returns:
            ErrorReport instance, or None if not found
        """
        if registry_name:
            registry = self.get_registry(registry_name)
            if registry:
                return registry.get_error_report(error_id)
            return None
        
        # Search all registries
        for registry in self.registries.values():
            report = registry.get_error_report(error_id)
            if report:
                return report
        
        return None
    
    def query_all_errors(self, component_name: str = None, error_type: str = None, 
                       severity: str = None, status: str = None) -> List[ErrorReport]:
        """
        Query error reports across all registries.
        
        Args:
            component_name: Filter by component name
            error_type: Filter by error type
            severity: Filter by severity
            status: Filter by status
            
        Returns:
            List of matching error reports
        """
        all_results = []
        
        for registry in self.registries.values():
            results = registry.query_errors(
                component_name=component_name,
                error_type=error_type,
                severity=severity,
                status=status
            )
            all_results.extend(results)
        
        return all_results
    
    def start_collection(self) -> bool:
        """
        Start automatic error collection in a background thread.
        
        Returns:
            True if started, False otherwise
        """
        if self.running:
            logger.warning("Error collection already running")
            return False
        
        # Import threading here to avoid issues in environments without it
        import threading
        
        self.thread_pool = threading.Thread(target=self._collection_loop)
        self.thread_pool.daemon = True
        self.running = True
        self.thread_pool.start()
        
        logger.info(f"Started automatic error collection (interval: {self.collection_interval}s)")
        return True
    
    def stop_collection(self) -> bool:
        """
        Stop automatic error collection.
        
        Returns:
            True if stopped, False otherwise
        """
        if not self.running:
            logger.warning("Error collection not running")
            return False
        
        self.running = False
        
        logger.info("Stopped automatic error collection")
        return True
    
    def _collection_loop(self) -> None:
        """
        Background loop for automatic error collection.
        """
        import time
        
        while self.running:
            try:
                # Collect errors from all registries
                self.collect_all_errors()
                
                # Sleep for the collection interval
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(10)  # Sleep briefly before retrying

# Factory function to create sensor registry
def create_sensor_registry(config: Dict[str, Any] = None) -> SensorRegistry:
    """
    Create and configure a sensor registry.
    
    Args:
        config: Configuration options
        
    Returns:
        SensorRegistry instance
    """
    config = config or {}
    sensors_config = config.get("sensors", {})
    
    # Create registry
    storage_path = sensors_config.get("storage_path", "auditor_data/sensors")
    registry = SensorRegistry(storage_path=storage_path)
    
    return registry
