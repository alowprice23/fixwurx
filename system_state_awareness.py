#!/usr/bin/env python3
"""
System State Awareness Module

This module provides advanced state awareness capabilities for the auditor agent,
enabling real-time monitoring of system state, component health, and operational status.
"""

import os
import sys
import json
import logging
import time
import threading
import queue
import uuid
import socket
import psutil
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system_state_awareness.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SystemStateAwareness")

class SystemState:
    """
    Represents the complete state of the system at a point in time.
    """
    
    def __init__(self):
        """Initialize system state."""
        self.timestamp = time.time()
        self.components = {}
        self.resources = {}
        self.processes = {}
        self.events = []
        self.alerts = []
        self.metrics = {}
        self.environment = {}
        self.status = "initializing"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert system state to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "timestamp": self.timestamp,
            "components": self.components,
            "resources": self.resources,
            "processes": self.processes,
            "events": self.events,
            "alerts": self.alerts,
            "metrics": self.metrics,
            "environment": self.environment,
            "status": self.status
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """
        Create system state from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            System state
        """
        state = cls()
        state.timestamp = data.get("timestamp", time.time())
        state.components = data.get("components", {})
        state.resources = data.get("resources", {})
        state.processes = data.get("processes", {})
        state.events = data.get("events", [])
        state.alerts = data.get("alerts", [])
        state.metrics = data.get("metrics", {})
        state.environment = data.get("environment", {})
        state.status = data.get("status", "unknown")
        return state

class StateCollector:
    """
    Collects state information from various sources.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize state collector.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.collectors = {}
        self.register_default_collectors()
        
        logger.info("State collector initialized")
    
    def register_default_collectors(self) -> None:
        """Register default state collectors."""
        # Register system collectors
        self.register_collector("system", self._collect_system_info)
        self.register_collector("hardware", self._collect_hardware_info)
        self.register_collector("processes", self._collect_process_info)
        self.register_collector("network", self._collect_network_info)
        self.register_collector("memory", self._collect_memory_info)
        self.register_collector("disk", self._collect_disk_info)
        self.register_collector("environment", self._collect_environment_info)
    
    def register_collector(self, name: str, collector_func: Callable) -> None:
        """
        Register a state collector function.
        
        Args:
            name: Collector name
            collector_func: Collector function
        """
        self.collectors[name] = collector_func
        logger.debug(f"Registered collector: {name}")
    
    def collect_state(self) -> SystemState:
        """
        Collect current system state.
        
        Returns:
            Current system state
        """
        state = SystemState()
        
        # Set status to collecting
        state.status = "collecting"
        
        # Run all collectors
        for name, collector in self.collectors.items():
            try:
                result = collector()
                if name == "system":
                    state.components.update(result)
                elif name == "hardware":
                    state.resources.update(result)
                elif name == "processes":
                    state.processes = result
                elif name == "network":
                    state.resources["network"] = result
                elif name == "memory":
                    state.resources["memory"] = result
                elif name == "disk":
                    state.resources["disk"] = result
                elif name == "environment":
                    state.environment = result
                else:
                    # Add as a component
                    state.components[name] = result
            except Exception as e:
                logger.error(f"Error collecting {name} state: {e}")
                state.alerts.append({
                    "level": "error",
                    "source": f"collector.{name}",
                    "message": f"Error collecting state: {str(e)}",
                    "timestamp": time.time()
                })
        
        # Set status to operational
        state.status = "operational"
        
        return state
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information.
        
        Returns:
            System information
        """
        return {
            "system": {
                "name": platform.node(),
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "boot_time": psutil.boot_time(),
                "uptime": time.time() - psutil.boot_time()
            }
        }
    
    def _collect_hardware_info(self) -> Dict[str, Any]:
        """
        Collect hardware information.
        
        Returns:
            Hardware information
        """
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=0.1, percpu=False),
            "per_cpu_percent": psutil.cpu_percent(interval=0.1, percpu=True)
        }
        
        return {
            "cpu": cpu_info
        }
    
    def _collect_process_info(self) -> Dict[str, Any]:
        """
        Collect process information.
        
        Returns:
            Process information
        """
        processes = {}
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent', 'create_time', 'status']):
            try:
                proc_info = proc.info
                pid = proc_info['pid']
                processes[str(pid)] = {
                    "name": proc_info['name'],
                    "username": proc_info['username'],
                    "memory_percent": proc_info['memory_percent'],
                    "cpu_percent": proc_info['cpu_percent'],
                    "create_time": proc_info['create_time'],
                    "status": proc_info['status'],
                    "is_running": proc.is_running()
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return processes
    
    def _collect_network_info(self) -> Dict[str, Any]:
        """
        Collect network information.
        
        Returns:
            Network information
        """
        network_io = psutil.net_io_counters()
        net_connections = []
        
        try:
            for conn in psutil.net_connections():
                net_connections.append({
                    "fd": conn.fd,
                    "family": conn.family,
                    "type": conn.type,
                    "local_addr": str(conn.laddr) if conn.laddr else None,
                    "remote_addr": str(conn.raddr) if conn.raddr else None,
                    "status": conn.status,
                    "pid": conn.pid
                })
        except (psutil.AccessDenied, PermissionError):
            # Fallback to basic network info if permissions are insufficient
            logger.warning("Insufficient permissions to collect detailed network connections")
        
        network_info = {
            "io_counters": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
                "errin": network_io.errin,
                "errout": network_io.errout,
                "dropin": network_io.dropin,
                "dropout": network_io.dropout
            },
            "connections": net_connections[:100]  # Limit to 100 connections to avoid overwhelming
        }
        
        # Add network interfaces
        network_info["interfaces"] = {}
        for interface_name, interface_addresses in psutil.net_if_addrs().items():
            addresses = []
            for addr in interface_addresses:
                addresses.append({
                    "address": addr.address,
                    "netmask": addr.netmask,
                    "broadcast": addr.broadcast,
                    "ptp": addr.ptp
                })
            
            network_info["interfaces"][interface_name] = {
                "addresses": addresses
            }
        
        return network_info
    
    def _collect_memory_info(self) -> Dict[str, Any]:
        """
        Collect memory information.
        
        Returns:
            Memory information
        """
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        return {
            "virtual": {
                "total": virtual_memory.total,
                "available": virtual_memory.available,
                "used": virtual_memory.used,
                "free": virtual_memory.free,
                "percent": virtual_memory.percent,
                "active": getattr(virtual_memory, 'active', None),
                "inactive": getattr(virtual_memory, 'inactive', None),
                "buffers": getattr(virtual_memory, 'buffers', None),
                "cached": getattr(virtual_memory, 'cached', None),
                "shared": getattr(virtual_memory, 'shared', None)
            },
            "swap": {
                "total": swap_memory.total,
                "used": swap_memory.used,
                "free": swap_memory.free,
                "percent": swap_memory.percent,
                "sin": swap_memory.sin,
                "sout": swap_memory.sout
            }
        }
    
    def _collect_disk_info(self) -> Dict[str, Any]:
        """
        Collect disk information.
        
        Returns:
            Disk information
        """
        disks = {}
        
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks[partition.device] = {
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "opts": partition.opts,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                }
            except (PermissionError, OSError):
                # Skip partitions that can't be accessed
                pass
        
        # Add disk I/O counters
        io_counters = psutil.disk_io_counters(perdisk=True)
        disk_io = {}
        
        for disk_name, counters in io_counters.items():
            disk_io[disk_name] = {
                "read_count": counters.read_count,
                "write_count": counters.write_count,
                "read_bytes": counters.read_bytes,
                "write_bytes": counters.write_bytes,
                "read_time": counters.read_time,
                "write_time": counters.write_time
            }
        
        return {
            "partitions": disks,
            "io": disk_io
        }
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """
        Collect environment information.
        
        Returns:
            Environment information
        """
        # Collect environment variables
        env_vars = {
            key: value for key, value in os.environ.items()
            if not key.lower().startswith(("pass", "key", "secret", "token", "cred"))  # Exclude sensitive variables
        }
        
        # Current directory and Python info
        environment = {
            "current_directory": os.getcwd(),
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform,
                "path": sys.path
            },
            "environment_variables": env_vars
        }
        
        return environment

class StateTracker:
    """
    Tracks system state over time.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize state tracker.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.state_collector = StateCollector(config)
        self.state_history = []
        self.max_history = self.config.get("max_history", 100)
        self.current_state = None
        self.state_dir = os.path.abspath(self.config.get("state_dir", "states"))
        self.tracking = False
        self.tracking_thread = None
        self.tracking_interval = self.config.get("tracking_interval", 60)  # seconds
        self.stop_event = threading.Event()
        self.state_changed_callbacks = []
        
        # Create state directory if it doesn't exist
        os.makedirs(self.state_dir, exist_ok=True)
        
        logger.info("State tracker initialized")
    
    def register_state_changed_callback(self, callback: Callable[[SystemState, SystemState], None]) -> None:
        """
        Register a callback to be called when the state changes.
        
        Args:
            callback: Callback function that takes (old_state, new_state)
        """
        self.state_changed_callbacks.append(callback)
        logger.debug(f"Registered state changed callback: {callback.__name__}")
    
    def update_state(self) -> SystemState:
        """
        Update current state.
        
        Returns:
            Updated state
        """
        old_state = self.current_state
        self.current_state = self.state_collector.collect_state()
        
        # Add to history
        self.state_history.append(self.current_state)
        
        # Trim history if needed
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]
        
        # Call state changed callbacks
        if old_state is not None:
            for callback in self.state_changed_callbacks:
                try:
                    callback(old_state, self.current_state)
                except Exception as e:
                    logger.error(f"Error in state changed callback: {e}")
        
        return self.current_state
    
    def start_tracking(self) -> bool:
        """
        Start tracking system state.
        
        Returns:
            Whether tracking was started
        """
        if self.tracking:
            logger.warning("State tracking already started")
            return False
        
        self.stop_event.clear()
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        self.tracking = True
        logger.info("State tracking started")
        return True
    
    def stop_tracking(self) -> bool:
        """
        Stop tracking system state.
        
        Returns:
            Whether tracking was stopped
        """
        if not self.tracking:
            logger.warning("State tracking not running")
            return False
        
        self.stop_event.set()
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=5)
            if self.tracking_thread.is_alive():
                logger.warning("Tracking thread did not terminate gracefully")
        
        self.tracking = False
        logger.info("State tracking stopped")
        return True
    
    def _tracking_loop(self) -> None:
        """Tracking loop for system state."""
        while not self.stop_event.is_set():
            try:
                # Update state
                self.update_state()
                
                # Save state if needed
                if self.config.get("save_states", False):
                    self.save_current_state()
                
                # Sleep until next update
                self.stop_event.wait(self.tracking_interval)
            except Exception as e:
                logger.error(f"Error in tracking loop: {e}")
                self.stop_event.wait(5)  # Wait a bit before retrying
    
    def save_current_state(self) -> Optional[str]:
        """
        Save current state to disk.
        
        Returns:
            Path to saved state file, or None if saving failed
        """
        if self.current_state is None:
            logger.warning("No current state to save")
            return None
        
        timestamp = self.current_state.timestamp
        state_file = os.path.join(self.state_dir, f"state_{int(timestamp)}.json")
        
        try:
            with open(state_file, "w") as f:
                json.dump(self.current_state.to_dict(), f, indent=2)
            
            logger.debug(f"Saved state to {state_file}")
            return state_file
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return None
    
    def load_state(self, state_file: str) -> Optional[SystemState]:
        """
        Load state from disk.
        
        Args:
            state_file: Path to state file
            
        Returns:
            Loaded state, or None if loading failed
        """
        try:
            with open(state_file, "r") as f:
                state_dict = json.load(f)
            
            state = SystemState.from_dict(state_dict)
            logger.debug(f"Loaded state from {state_file}")
            return state
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return None
    
    def get_current_state(self) -> Optional[SystemState]:
        """
        Get current system state.
        
        Returns:
            Current state, or None if not available
        """
        if self.current_state is None:
            self.update_state()
        
        return self.current_state
    
    def get_state_history(self) -> List[SystemState]:
        """
        Get state history.
        
        Returns:
            List of historical states
        """
        return self.state_history.copy()
    
    def clear_history(self) -> None:
        """Clear state history."""
        self.state_history = []
        logger.info("Cleared state history")

class StateAnalyzer:
    """
    Analyzes system state to detect issues and anomalies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize state analyzer.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.analyzers = {}
        self.register_default_analyzers()
        
        logger.info("State analyzer initialized")
    
    def register_default_analyzers(self) -> None:
        """Register default state analyzers."""
        self.register_analyzer("cpu_usage", self._analyze_cpu_usage)
        self.register_analyzer("memory_usage", self._analyze_memory_usage)
        self.register_analyzer("disk_usage", self._analyze_disk_usage)
        self.register_analyzer("network_io", self._analyze_network_io)
        self.register_analyzer("process_status", self._analyze_process_status)
        self.register_analyzer("system_uptime", self._analyze_system_uptime)
    
    def register_analyzer(self, name: str, analyzer_func: Callable) -> None:
        """
        Register a state analyzer function.
        
        Args:
            name: Analyzer name
            analyzer_func: Analyzer function
        """
        self.analyzers[name] = analyzer_func
        logger.debug(f"Registered analyzer: {name}")
    
    def analyze_state(self, state: SystemState) -> List[Dict[str, Any]]:
        """
        Analyze system state.
        
        Args:
            state: System state to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        
        # Run all analyzers
        for name, analyzer in self.analyzers.items():
            try:
                result = analyzer(state)
                if result:
                    results.append({
                        "analyzer": name,
                        "result": result,
                        "timestamp": time.time()
                    })
            except Exception as e:
                logger.error(f"Error analyzing state with {name}: {e}")
                results.append({
                    "analyzer": name,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        return results
    
    def _analyze_cpu_usage(self, state: SystemState) -> Optional[Dict[str, Any]]:
        """
        Analyze CPU usage.
        
        Args:
            state: System state
            
        Returns:
            Analysis result, or None if no issues
        """
        try:
            cpu_info = state.resources.get("cpu", {})
            cpu_percent = cpu_info.get("usage_percent", 0)
            
            # Check CPU usage
            if cpu_percent > 90:
                return {
                    "level": "critical",
                    "message": f"High CPU usage: {cpu_percent}%",
                    "value": cpu_percent,
                    "threshold": 90
                }
            elif cpu_percent > 75:
                return {
                    "level": "warning",
                    "message": f"Elevated CPU usage: {cpu_percent}%",
                    "value": cpu_percent,
                    "threshold": 75
                }
        except Exception as e:
            logger.error(f"Error analyzing CPU usage: {e}")
        
        return None
    
    def _analyze_memory_usage(self, state: SystemState) -> Optional[Dict[str, Any]]:
        """
        Analyze memory usage.
        
        Args:
            state: System state
            
        Returns:
            Analysis result, or None if no issues
        """
        try:
            memory_info = state.resources.get("memory", {}).get("virtual", {})
            memory_percent = memory_info.get("percent", 0)
            
            # Check memory usage
            if memory_percent > 95:
                return {
                    "level": "critical",
                    "message": f"High memory usage: {memory_percent}%",
                    "value": memory_percent,
                    "threshold": 95
                }
            elif memory_percent > 85:
                return {
                    "level": "warning",
                    "message": f"Elevated memory usage: {memory_percent}%",
                    "value": memory_percent,
                    "threshold": 85
                }
        except Exception as e:
            logger.error(f"Error analyzing memory usage: {e}")
        
        return None
    
    def _analyze_disk_usage(self, state: SystemState) -> Optional[Dict[str, Any]]:
        """
        Analyze disk usage.
        
        Args:
            state: System state
            
        Returns:
            Analysis result, or None if no issues
        """
        try:
            disk_info = state.resources.get("disk", {}).get("partitions", {})
            
            critical_disks = []
            warning_disks = []
            
            for device, info in disk_info.items():
                disk_percent = info.get("percent", 0)
                
                # Check disk usage
                if disk_percent > 95:
                    critical_disks.append({
                        "device": device,
                        "mountpoint": info.get("mountpoint"),
                        "percent": disk_percent
                    })
                elif disk_percent > 85:
                    warning_disks.append({
                        "device": device,
                        "mountpoint": info.get("mountpoint"),
                        "percent": disk_percent
                    })
            
            if critical_disks:
                return {
                    "level": "critical",
                    "message": f"High disk usage on {len(critical_disks)} partitions",
                    "disks": critical_disks,
                    "threshold": 95
                }
            elif warning_disks:
                return {
                    "level": "warning",
                    "message": f"Elevated disk usage on {len(warning_disks)} partitions",
                    "disks": warning_disks,
                    "threshold": 85
                }
        except Exception as e:
            logger.error(f"Error analyzing disk usage: {e}")
        
        return None
    
    def _analyze_network_io(self, state: SystemState) -> Optional[Dict[str, Any]]:
        """
        Analyze network I/O.
        
        Args:
            state: System state
            
        Returns:
            Analysis result, or None if no issues
        """
        # This would normally compare current and previous network stats
        # to detect unusual traffic, but we'll just return None for now
        return None
    
    def _analyze_process_status(self, state: SystemState) -> Optional[Dict[str, Any]]:
        """
        Analyze process status.
        
        Args:
            state: System state
            
        Returns:
            Analysis result, or None if no issues
        """
        try:
            processes = state.processes
            
            # Get non-running processes
            non_running = [
                {
                    "pid": pid,
                    "name": info.get("name"),
                    "status": info.get("status")
                }
                for pid, info in processes.items()
                if info.get("is_running") is False
            ]
            
            if non_running:
                return {
                    "level": "warning",
                    "message": f"Found {len(non_running)} non-running processes",
                    "processes": non_running
                }
        except Exception as e:
            logger.error(f"Error analyzing process status: {e}")
        
        return None
    
    def _analyze_system_uptime(self, state: SystemState) -> Optional[Dict[str, Any]]:
        """
        Analyze system uptime.
        
        Args:
            state: System state
            
        Returns:
            Analysis result, or None if no issues
        """
        try:
            system_info = state.components.get("system", {})
            uptime = system_info.get("uptime", 0)
            
            # Check if system was recently rebooted
            if uptime < 300:  # Less than 5 minutes
                return {
                    "level": "info",
                    "message": f"System recently rebooted, uptime: {int(uptime)} seconds",
                    "uptime": uptime
                }
        except Exception as e:
            logger.error(f"Error analyzing system uptime: {e}")
        
        return None

class ComponentMonitor:
    """
    Monitors the health and status of system components.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize component monitor.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.components = {}
        self.monitoring_thread = None
        self.monitoring_interval = self.config.get("monitoring_interval", 30)  # seconds
        self.stop_event = threading.Event()
        self.component_changed_callbacks = []
        
        logger.info("Component monitor initialized")
    
    def register_component(self, name: str, component_info: Dict[str, Any]) -> None:
        """
        Register a component for monitoring.
        
        Args:
            name: Component name
            component_info: Component information
        """
        if "health_check" not in component_info:
            raise ValueError(f"Component {name} must have a health_check function")
        
        self.components[name] = {
            "info": component_info,
            "status": "unknown",
            "last_check": None,
            "last_success": None,
            "failures": 0,
            "health": 0.0  # 0.0 to 1.0
        }
        
        logger.info(f"Registered component: {name}")
    
    def register_component_changed_callback(self, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], None]) -> None:
        """
        Register a callback to be called when a component's status changes.
        
        Args:
            callback: Callback function that takes (component_name, old_status, new_status)
        """
        self.component_changed_callbacks.append(callback)
        logger.debug(f"Registered component changed callback: {callback.__name__}")
    
    def check_component(self, name: str) -> Dict[str, Any]:
        """
        Check a component's health.
        
        Args:
            name: Component name
            
        Returns:
            Component status
        """
        if name not in self.components:
            logger.warning(f"Component {name} not registered")
            return {
                "status": "unknown",
                "message": f"Component {name} not registered",
                "timestamp": time.time()
            }
        
        component = self.components[name]
        old_status = component.copy()
        
        try:
            # Get health check function
            health_check = component["info"]["health_check"]
            
            # Run health check
            result = health_check()
            
            # Update component status
            component["last_check"] = time.time()
            
            if result.get("status") == "healthy":
                component["status"] = "healthy"
                component["last_success"] = time.time()
                component["failures"] = 0
                component["health"] = 1.0
            else:
                component["status"] = result.get("status", "unhealthy")
                component["failures"] += 1
                
                # Calculate health based on failures
                max_failures = self.config.get("max_failures", 5)
                component["health"] = max(0.0, 1.0 - (component["failures"] / max_failures))
            
            #
