#!/usr/bin/env python3
"""
System Monitor Component for Triangulum Integration

This module provides the SystemMonitor class for monitoring system status.
"""

import os
import json
import logging
import time
import threading
from typing import Dict, Any, Callable, Optional

# Import the client for Triangulum connection status
try:
    from triangulum_client import TriangulumClient
except ImportError:
    # Mock if not available
    class TriangulumClient:
        @staticmethod
        def is_connected():
            return False
        
        last_heartbeat = None
        api_calls = 0
        api_errors = 0

# Configure logging if not already configured
logger = logging.getLogger("TriangulumIntegration")

# Mock mode for testing
MOCK_MODE = os.environ.get("TRIANGULUM_TEST_MODE", "0") == "1"

# Import psutil if available
try:
    import psutil
except ImportError:
    # Mock psutil for testing
    if MOCK_MODE:
        class MockPsutil:
            @staticmethod
            def cpu_percent():
                return 25.0
            
            @staticmethod
            def virtual_memory():
                class MemInfo:
                    def __init__(self):
                        self.percent = 40.0
                return MemInfo()
            
            @staticmethod
            def disk_usage(path):
                class DiskInfo:
                    def __init__(self):
                        self.percent = 60.0
                return DiskInfo()
            
            @staticmethod
            def boot_time():
                return time.time() - 3600  # 1 hour ago
        
        psutil = MockPsutil()
    else:
        psutil = None

class SystemMonitor:
    """
    Monitors the system status and reports metrics to Triangulum.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize system monitor.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.metrics = {}
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = self.config.get("monitor_interval", 60)  # seconds
        self.callback = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Initialize metrics
        self._init_metrics()
        
        logger.info("System monitor initialized")
    
    def _init_metrics(self) -> None:
        """
        Initialize metrics dictionary.
        """
        self.metrics = {
            "system": {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "disk_usage": 0.0,
                "uptime": 0
            },
            "application": {
                "tasks_pending": 0,
                "tasks_running": 0,
                "tasks_completed": 0,
                "tasks_failed": 0,
                "error_rate": 0.0,
                "average_task_duration": 0.0
            },
            "resources": {
                "active_workers": 0,
                "total_workers": 0,
                "queue_depth": 0,
                "burst_active": False
            },
            "triangulum": {
                "connection_status": "disconnected",
                "last_heartbeat": None,
                "api_calls": 0,
                "api_errors": 0
            }
        }
    
    def start_monitoring(self, callback: Callable = None) -> bool:
        """
        Start monitoring system status.
        
        Args:
            callback: Callback function to call with metrics
            
        Returns:
            Whether monitoring was started
        """
        with self.lock:
            if self.is_monitoring:
                logger.warning("System monitoring already started")
                return False
            
            self.callback = callback
            self.stop_event.clear()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.is_monitoring = True
            logger.info("System monitoring started")
            return True
    
    def stop_monitoring(self) -> bool:
        """
        Stop monitoring system status.
        
        Returns:
            Whether monitoring was stopped
        """
        with self.lock:
            if not self.is_monitoring:
                logger.warning("System monitoring not running")
                return False
            
            self.stop_event.set()
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
                if self.monitor_thread.is_alive():
                    logger.warning("Monitor thread did not terminate gracefully")
            
            self.is_monitoring = False
            self.monitor_thread = None
            logger.info("System monitoring stopped")
            return True
    
    def _monitor_loop(self) -> None:
        """
        Main monitoring loop.
        """
        while not self.stop_event.is_set():
            try:
                # Update metrics
                self._update_metrics()
                
                # Call callback if provided
                if self.callback:
                    metrics_copy = self.get_metrics()  # Get a thread-safe copy
                    self.callback(metrics_copy)
                
                # Sleep until next interval
                self.stop_event.wait(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                self.stop_event.wait(5)  # Wait a bit before retrying
    
    def _update_metrics(self) -> None:
        """
        Update system metrics.
        """
        try:
            with self.lock:
                if MOCK_MODE or not psutil:
                    # Use mock values in test mode
                    self.metrics["system"]["cpu_usage"] = 25.0
                    self.metrics["system"]["memory_usage"] = 40.0
                    self.metrics["system"]["disk_usage"] = 60.0
                    self.metrics["system"]["uptime"] = 3600
                else:
                    # System metrics from psutil
                    self.metrics["system"]["cpu_usage"] = psutil.cpu_percent()
                    self.metrics["system"]["memory_usage"] = psutil.virtual_memory().percent
                    self.metrics["system"]["disk_usage"] = psutil.disk_usage('/').percent
                    self.metrics["system"]["uptime"] = int(time.time() - psutil.boot_time())
                
                # Update Triangulum connection status
                try:
                    self.metrics["triangulum"]["connection_status"] = "connected" if TriangulumClient.is_connected() else "disconnected"
                    self.metrics["triangulum"]["last_heartbeat"] = TriangulumClient.last_heartbeat
                    self.metrics["triangulum"]["api_calls"] = TriangulumClient.api_calls
                    self.metrics["triangulum"]["api_errors"] = TriangulumClient.api_errors
                except:
                    # If TriangulumClient is not available, use default values
                    self.metrics["triangulum"]["connection_status"] = "disconnected"
                    self.metrics["triangulum"]["last_heartbeat"] = None
                    self.metrics["triangulum"]["api_calls"] = 0
                    self.metrics["triangulum"]["api_errors"] = 0
                
                logger.debug("Updated system metrics")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Current metrics
        """
        with self.lock:
            # Return a deep copy to avoid thread safety issues
            return json.loads(json.dumps(self.metrics))
    
    def update_application_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update application metrics.
        
        Args:
            metrics: Application metrics to update
        """
        with self.lock:
            self.metrics["application"].update(metrics)
    
    def update_resource_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update resource metrics.
        
        Args:
            metrics: Resource metrics to update
        """
        with self.lock:
            self.metrics["resources"].update(metrics)
