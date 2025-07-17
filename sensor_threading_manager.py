"""
FixWurx Auditor - Sensor Threading Manager

This module implements a threading system for running sensors in isolated threads,
providing fault tolerance and resource control.
"""

import threading
import time
import logging
from typing import Dict, List, Any, Optional
from queue import Queue, Empty

from sensor_base import ErrorSensor
from error_report import ErrorReport
from sensor_registry import SensorRegistry

logger = logging.getLogger('sensor_threading_manager')

class SensorThreadingManager:
    """Manages threaded execution of sensors."""
    
    def __init__(self, registry: SensorRegistry, interval: int = 60):
        """
        Initialize the SensorThreadingManager.
        
        Args:
            registry: The sensor registry containing all sensors
            interval: Default monitoring interval in seconds
        """
        self.registry = registry
        self.default_interval = interval
        self.threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.error_queues: Dict[str, Queue] = {}
        self.report_queue = Queue()  # Central queue for all error reports
        
        # Start report processor thread
        self.report_processor_stop = threading.Event()
        self.report_processor = threading.Thread(
            target=self._process_reports,
            daemon=True,
            name="ReportProcessor"
        )
        self.report_processor.start()
        
        logger.info(f"Initialized SensorThreadingManager with {len(registry.get_all_sensor_ids())} sensors")
    
    def start_sensor_thread(self, sensor_id: str, interval: Optional[int] = None):
        """
        Start a dedicated thread for a sensor.
        
        Args:
            sensor_id: ID of the sensor to run in its own thread
            interval: Optional custom interval for this sensor
        """
        if sensor_id in self.threads and self.threads[sensor_id].is_alive():
            logger.info(f"Thread for sensor {sensor_id} is already running")
            return
        
        sensor = self.registry.get_sensor(sensor_id)
        if not sensor:
            logger.error(f"Sensor {sensor_id} not found in registry")
            return
        
        # Create stop event and error queue
        stop_event = threading.Event()
        error_queue = Queue()
        
        # Create and start thread
        thread = threading.Thread(
            target=self._run_sensor,
            args=(sensor, stop_event, error_queue, interval or self.default_interval),
            daemon=True,
            name=f"Sensor-{sensor_id}"
        )
        
        self.threads[sensor_id] = thread
        self.stop_events[sensor_id] = stop_event
        self.error_queues[sensor_id] = error_queue
        
        thread.start()
        logger.info(f"Started thread for sensor {sensor_id}")
    
    def stop_sensor_thread(self, sensor_id: str):
        """
        Stop a sensor thread.
        
        Args:
            sensor_id: ID of the sensor thread to stop
        """
        if sensor_id in self.stop_events:
            self.stop_events[sensor_id].set()
            if self.threads[sensor_id].is_alive():
                self.threads[sensor_id].join(timeout=5)
                logger.info(f"Stopped thread for sensor {sensor_id}")
            
            # Clean up
            del self.threads[sensor_id]
            del self.stop_events[sensor_id]
            del self.error_queues[sensor_id]
    
    def start_all_sensors(self):
        """Start threads for all sensors in the registry."""
        for sensor_id in self.registry.get_all_sensor_ids():
            self.start_sensor_thread(sensor_id)
    
    def stop_all_sensors(self):
        """Stop all sensor threads."""
        for sensor_id in list(self.threads.keys()):
            self.stop_sensor_thread(sensor_id)
        
        # Stop report processor
        self.report_processor_stop.set()
        if self.report_processor.is_alive():
            self.report_processor.join(timeout=5)
        
        logger.info("All sensor threads stopped")
    
    def _run_sensor(self, sensor: ErrorSensor, stop_event: threading.Event, 
                   error_queue: Queue, interval: int):
        """
        Run the sensor monitoring loop in a separate thread.
        
        Args:
            sensor: The sensor to monitor with
            stop_event: Event to signal thread termination
            error_queue: Queue to report thread errors
            interval: Monitoring interval in seconds
        """
        try:
            logger.info(f"Starting monitoring loop for {sensor.sensor_id}")
            
            while not stop_event.is_set():
                try:
                    # Run the sensor monitor
                    reports = sensor.monitor()
                    
                    # Put reports in the central queue
                    if reports:
                        for report in reports:
                            self.report_queue.put(report)
                        
                        logger.debug(f"Sensor {sensor.sensor_id} generated {len(reports)} reports")
                
                except Exception as e:
                    # Handle sensor failure
                    error_msg = f"Error in sensor {sensor.sensor_id}: {str(e)}"
                    logger.error(error_msg)
                    error_queue.put((sensor.sensor_id, str(e)))
                    
                    # Allow some recovery time
                    time.sleep(5)
                
                # Wait for next interval or until stopped
                stop_event.wait(interval)
                
        except Exception as e:
            # Handle thread failure
            error_msg = f"Thread for sensor {sensor.sensor_id} failed: {str(e)}"
            logger.error(error_msg)
            error_queue.put((sensor.sensor_id, str(e)))
    
    def _process_reports(self):
        """Process reports from the central queue."""
        try:
            while not self.report_processor_stop.is_set():
                try:
                    # Get report from queue with timeout
                    report = self.report_queue.get(timeout=1)
                    
                    # Process the report (store, analyze, alert, etc.)
                    # This would call into the error management system
                    logger.info(f"Processing report: {report.error_type} ({report.severity})")
                    
                    # Mark as done
                    self.report_queue.task_done()
                    
                except Empty:
                    # No reports available, just continue
                    pass
                    
        except Exception as e:
            logger.error(f"Report processor thread failed: {str(e)}")
    
    def get_thread_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all sensor threads."""
        status = {}
        
        for sensor_id, thread in self.threads.items():
            error_count = self.error_queues[sensor_id].qsize()
            
            status[sensor_id] = {
                "running": thread.is_alive(),
                "errors": error_count,
                "name": thread.name,
                "daemon": thread.daemon
            }
        
        return status

# Factory function
def create_sensor_threading_manager(registry: SensorRegistry, interval: int = 60) -> SensorThreadingManager:
    """
    Create and initialize a SensorThreadingManager.
    
    Args:
        registry: The sensor registry to use
        interval: Default monitoring interval in seconds
        
    Returns:
        Initialized SensorThreadingManager
    """
    return SensorThreadingManager(registry, interval)
