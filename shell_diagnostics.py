#!/usr/bin/env python3
"""
shell_diagnostics.py
──────────────────
Basic diagnostic tools for the FixWurx shell environment.

This module provides diagnostic capabilities to check the health and status 
of various components of the shell environment, identify issues, and help
with troubleshooting.
"""

import os
import sys
import time
import logging
import platform
import threading
import multiprocessing
import json
import datetime
import psutil
import inspect
import importlib
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union
from enum import Enum, auto

# Internal imports
from shell_environment import register_command, emit_event, EventType, get_environment_variable
import shell_scripting
import fixwurx_commands
from error_reporting import report_error
from alert_system import get_alert_system, AlertLevel, AlertCategory, Alert
from resource_manager import get_resource_manager, ResourceType

# Configure logging
logger = logging.getLogger("ShellDiagnostics")

# Constants
DEFAULT_HEALTH_CHECK_INTERVAL = 300  # seconds (5 minutes)
DEFAULT_METRICS_INTERVAL = 60  # seconds (1 minute)
DEFAULT_HISTORY_SIZE = 100  # number of historical metrics to keep
DEFAULT_CPU_WARNING_THRESHOLD = 80  # percentage
DEFAULT_MEMORY_WARNING_THRESHOLD = 80  # percentage
DEFAULT_DISK_WARNING_THRESHOLD = 90  # percentage
DEFAULT_COMMAND_TIMEOUT = 10  # seconds

class ComponentStatus(Enum):
    """Status of a component."""
    HEALTHY = auto()    # Component is healthy
    DEGRADED = auto()   # Component is functioning but with issues
    CRITICAL = auto()   # Component has critical issues
    UNAVAILABLE = auto() # Component is unavailable
    UNKNOWN = auto()    # Status is unknown

class DiagnosticResult:
    """Result of a diagnostic check."""
    
    def __init__(self, component: str, status: ComponentStatus, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a diagnostic result.
        
        Args:
            component: Component name.
            status: Component status.
            message: Status message.
            details: Additional details.
        """
        self.component = component
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "status": self.status.name,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "timestamp_formatted": datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def __str__(self) -> str:
        """Convert to string."""
        status_indicators = {
            ComponentStatus.HEALTHY: "✅",
            ComponentStatus.DEGRADED: "⚠️",
            ComponentStatus.CRITICAL: "❌",
            ComponentStatus.UNAVAILABLE: "❓",
            ComponentStatus.UNKNOWN: "❔"
        }
        
        indicator = status_indicators.get(self.status, "❔")
        
        return f"{indicator} {self.component}: {self.message}"

class SystemMetrics:
    """System metrics."""
    
    def __init__(self):
        """Initialize system metrics."""
        self.timestamp = time.time()
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.memory_available = 0
        self.memory_total = 0
        self.disk_usage = 0.0
        self.disk_available = 0
        self.disk_total = 0
        self.process_count = 0
        self.thread_count = 0
        self.uptime = 0
        self.command_count = 0
        self.event_count = 0
        self.error_count = 0
        self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "timestamp_formatted": datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_available": self.memory_available,
            "memory_total": self.memory_total,
            "disk_usage": self.disk_usage,
            "disk_available": self.disk_available,
            "disk_total": self.disk_total,
            "process_count": self.process_count,
            "thread_count": self.thread_count,
            "uptime": self.uptime,
            "command_count": self.command_count,
            "event_count": self.event_count,
            "error_count": self.error_count,
            "details": self.details
        }

class ShellDiagnostics:
    """
    Shell diagnostics for the FixWurx platform.
    
    This class provides diagnostic capabilities to check the health and status
    of various components of the shell environment, identify issues, and help
    with troubleshooting.
    """
    
    def __init__(self):
        """Initialize shell diagnostics."""
        self._stop_threads = threading.Event()
        self._health_check_thread = None
        self._metrics_thread = None
        self._last_health_check: Dict[str, DiagnosticResult] = {}
        self._metrics_history: List[SystemMetrics] = []
        self._component_checkers: Dict[str, Callable[[], DiagnosticResult]] = {
            "system": self._check_system,
            "shell_environment": self._check_shell_environment,
            "shell_scripting": self._check_shell_scripting,
            "fixwurx_commands": self._check_fixwurx_commands,
            "resource_manager": self._check_resource_manager,
            "error_reporting": self._check_error_reporting,
            "alert_system": self._check_alert_system
        }
        
        # Register commands
        try:
            register_command("diagnose", self.diagnose_command, "Run diagnostic checks on the shell environment")
            register_command("health", self.health_command, "Show health status of shell components")
            register_command("metrics", self.metrics_command, "Show system metrics")
            register_command("doctor", self.doctor_command, "Run diagnostics and attempt to fix issues")
        except Exception as e:
            logger.error(f"Failed to register diagnostic commands: {e}")
        
        # Start health check thread
        self._start_health_check_thread()
        
        # Start metrics thread
        self._start_metrics_thread()
        
        logger.info("Shell diagnostics initialized")
    
    def _start_health_check_thread(self) -> None:
        """Start the health check thread."""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._stop_threads.clear()
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="HealthCheckThread"
            )
            self._health_check_thread.start()
            logger.info("Started health check thread")
    
    def _start_metrics_thread(self) -> None:
        """Start the metrics thread."""
        if self._metrics_thread is None or not self._metrics_thread.is_alive():
            self._stop_threads.clear()
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop,
                daemon=True,
                name="MetricsThread"
            )
            self._metrics_thread.start()
            logger.info("Started metrics thread")
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while not self._stop_threads.is_set():
            try:
                # Run health checks for all components
                for component, checker in self._component_checkers.items():
                    try:
                        result = checker()
                        self._last_health_check[component] = result
                        
                        # Log critical issues
                        if result.status == ComponentStatus.CRITICAL:
                            logger.error(f"Critical issue detected in {component}: {result.message}")
                            
                            # Emit event for critical issues
                            emit_event(EventType.ERROR, {
                                "source": "ShellDiagnostics",
                                "component": component,
                                "message": result.message,
                                "details": result.details
                            })
                    except Exception as e:
                        logger.error(f"Error checking component {component}: {e}")
                        self._last_health_check[component] = DiagnosticResult(
                            component=component,
                            status=ComponentStatus.UNKNOWN,
                            message=f"Error checking component: {e}",
                            details={"error": str(e)}
                        )
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            # Sleep until next check
            time.sleep(DEFAULT_HEALTH_CHECK_INTERVAL)
    
    def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while not self._stop_threads.is_set():
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Add to history
                self._metrics_history.append(metrics)
                
                # Limit history size
                if len(self._metrics_history) > DEFAULT_HISTORY_SIZE:
                    self._metrics_history = self._metrics_history[-DEFAULT_HISTORY_SIZE:]
                
                # Check for threshold violations
                self._check_thresholds(metrics)
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
            
            # Sleep until next collection
            time.sleep(DEFAULT_METRICS_INTERVAL)
    
    def _collect_metrics(self) -> SystemMetrics:
        """
        Collect system metrics.
        
        Returns:
            System metrics.
        """
        metrics = SystemMetrics()
        
        try:
            # CPU usage
            metrics.cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.percent
            metrics.memory_available = memory.available
            metrics.memory_total = memory.total
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.disk_usage = disk.percent
            metrics.disk_available = disk.free
            metrics.disk_total = disk.total
            
            # Process and thread count
            current_process = psutil.Process(os.getpid())
            metrics.process_count = len(psutil.pids())
            metrics.thread_count = threading.active_count()
            
            # Uptime
            metrics.uptime = time.time() - psutil.boot_time()
            
            # Command count (if available)
            try:
                command_count = get_environment_variable("COMMAND_COUNT")
                if command_count is not None:
                    metrics.command_count = int(command_count)
            except Exception:
                pass
            
            # Event count (if available)
            try:
                event_count = get_environment_variable("EVENT_COUNT")
                if event_count is not None:
                    metrics.event_count = int(event_count)
            except Exception:
                pass
            
            # Error count (if available)
            try:
                error_count = get_environment_variable("ERROR_COUNT")
                if error_count is not None:
                    metrics.error_count = int(error_count)
            except Exception:
                pass
            
            # Additional details
            metrics.details = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "processors": psutil.cpu_count(logical=False),
                "logical_processors": psutil.cpu_count(logical=True),
                "process_memory": dict(psutil.Process(os.getpid()).memory_info()._asdict())
            }
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _check_thresholds(self, metrics: SystemMetrics) -> None:
        """
        Check metrics against thresholds.
        
        Args:
            metrics: System metrics.
        """
        alerts = []
        
        # Check CPU usage
        if metrics.cpu_usage > DEFAULT_CPU_WARNING_THRESHOLD:
            alerts.append({
                "component": "System",
                "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                "level": AlertLevel.HIGH if metrics.cpu_usage > 90 else AlertLevel.MEDIUM,
                "details": {
                    "cpu_usage": metrics.cpu_usage,
                    "threshold": DEFAULT_CPU_WARNING_THRESHOLD
                }
            })
        
        # Check memory usage
        if metrics.memory_usage > DEFAULT_MEMORY_WARNING_THRESHOLD:
            alerts.append({
                "component": "System",
                "message": f"High memory usage: {metrics.memory_usage:.1f}%",
                "level": AlertLevel.HIGH if metrics.memory_usage > 90 else AlertLevel.MEDIUM,
                "details": {
                    "memory_usage": metrics.memory_usage,
                    "memory_available": metrics.memory_available,
                    "memory_total": metrics.memory_total,
                    "threshold": DEFAULT_MEMORY_WARNING_THRESHOLD
                }
            })
        
        # Check disk usage
        if metrics.disk_usage > DEFAULT_DISK_WARNING_THRESHOLD:
            alerts.append({
                "component": "System",
                "message": f"High disk usage: {metrics.disk_usage:.1f}%",
                "level": AlertLevel.HIGH if metrics.disk_usage > 95 else AlertLevel.MEDIUM,
                "details": {
                    "disk_usage": metrics.disk_usage,
                    "disk_available": metrics.disk_available,
                    "disk_total": metrics.disk_total,
                    "threshold": DEFAULT_DISK_WARNING_THRESHOLD
                }
            })
        
        # Emit alerts
        for alert in alerts:
            # Log alert
            level = alert["level"]
            if level == AlertLevel.HIGH:
                logger.warning(f"[ALERT] {alert['message']}")
            else:
                logger.info(f"[ALERT] {alert['message']}")
            
            # Emit event
            emit_event(EventType.SYSTEM, {
                "source": "ShellDiagnostics",
                "component": alert["component"],
                "message": alert["message"],
                "alert_level": level.name,
                "details": alert["details"]
            })
    
    def _check_system(self) -> DiagnosticResult:
        """
        Check system health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Get system metrics
            metrics = self._collect_metrics()
            
            # Determine status
            status = ComponentStatus.HEALTHY
            message = "System is healthy"
            
            if metrics.cpu_usage > 90 or metrics.memory_usage > 90 or metrics.disk_usage > 95:
                status = ComponentStatus.CRITICAL
                message = "System resources are critically low"
            elif metrics.cpu_usage > 80 or metrics.memory_usage > 80 or metrics.disk_usage > 90:
                status = ComponentStatus.DEGRADED
                message = "System resources are running low"
            
            # Return diagnostic result
            return DiagnosticResult(
                component="system",
                status=status,
                message=message,
                details={
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "disk_usage": metrics.disk_usage,
                    "memory_available": metrics.memory_available,
                    "memory_total": metrics.memory_total,
                    "disk_available": metrics.disk_available,
                    "disk_total": metrics.disk_total,
                    "process_count": metrics.process_count,
                    "thread_count": metrics.thread_count,
                    "uptime": metrics.uptime,
                    "platform": platform.platform(),
                    "python_version": sys.version
                }
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="system",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking system: {e}",
                details={"error": str(e)}
            )
    
    def _check_shell_environment(self) -> DiagnosticResult:
        """
        Check shell environment health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Import shell environment
            from shell_environment import (
                register_command, register_event_handler, emit_event,
                get_environment_variable, set_environment_variable
            )
            
            # Check if basic functions are available
            if (not callable(register_command) or
                not callable(register_event_handler) or
                not callable(emit_event) or
                not callable(get_environment_variable) or
                not callable(set_environment_variable)):
                return DiagnosticResult(
                    component="shell_environment",
                    status=ComponentStatus.CRITICAL,
                    message="Shell environment functions are not available",
                    details={"missing_functions": "Core shell environment functions are not callable"}
                )
            
            # Test environment variables
            test_var_name = f"TEST_VAR_{int(time.time())}"
            test_var_value = f"test_value_{int(time.time())}"
            
            set_environment_variable(test_var_name, test_var_value)
            retrieved_value = get_environment_variable(test_var_name)
            
            if retrieved_value != test_var_value:
                return DiagnosticResult(
                    component="shell_environment",
                    status=ComponentStatus.DEGRADED,
                    message="Environment variables are not working correctly",
                    details={
                        "expected": test_var_value,
                        "actual": retrieved_value
                    }
                )
            
            # Return healthy result
            return DiagnosticResult(
                component="shell_environment",
                status=ComponentStatus.HEALTHY,
                message="Shell environment is healthy",
                details={
                    "environment_variables_working": True
                }
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="shell_environment",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking shell environment: {e}",
                details={"error": str(e)}
            )
    
    def _check_shell_scripting(self) -> DiagnosticResult:
        """
        Check shell scripting health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Check if shell scripting module is available
            if not hasattr(shell_scripting, "execute_script"):
                return DiagnosticResult(
                    component="shell_scripting",
                    status=ComponentStatus.CRITICAL,
                    message="Shell scripting module is missing execute_script function",
                    details={"missing_functions": "execute_script"}
                )
            
            # Try to execute a simple script
            script = "result = 2 + 2"
            context = {}
            
            try:
                shell_scripting.execute_script(script, context)
                
                if context.get("result") != 4:
                    return DiagnosticResult(
                        component="shell_scripting",
                        status=ComponentStatus.DEGRADED,
                        message="Shell scripting execution is not working correctly",
                        details={
                            "expected": 4,
                            "actual": context.get("result")
                        }
                    )
            except Exception as script_error:
                return DiagnosticResult(
                    component="shell_scripting",
                    status=ComponentStatus.CRITICAL,
                    message=f"Error executing test script: {script_error}",
                    details={"error": str(script_error)}
                )
            
            # Return healthy result
            return DiagnosticResult(
                component="shell_scripting",
                status=ComponentStatus.HEALTHY,
                message="Shell scripting is healthy",
                details={
                    "script_execution_working": True
                }
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="shell_scripting",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking shell scripting: {e}",
                details={"error": str(e)}
            )
    
    def _check_fixwurx_commands(self) -> DiagnosticResult:
        """
        Check FixWurx commands health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Check if fixwurx_commands module is loaded
            if not hasattr(fixwurx_commands, "get_commands"):
                return DiagnosticResult(
                    component="fixwurx_commands",
                    status=ComponentStatus.CRITICAL,
                    message="FixWurx commands module is missing get_commands function",
                    details={"missing_functions": "get_commands"}
                )
            
            # Get available commands
            commands = fixwurx_commands.get_commands()
            
            if not commands:
                return DiagnosticResult(
                    component="fixwurx_commands",
                    status=ComponentStatus.DEGRADED,
                    message="No FixWurx commands are available",
                    details={"commands": commands}
                )
            
            # Return healthy result
            return DiagnosticResult(
                component="fixwurx_commands",
                status=ComponentStatus.HEALTHY,
                message="FixWurx commands are healthy",
                details={
                    "command_count": len(commands),
                    "commands": list(commands.keys())
                }
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="fixwurx_commands",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking FixWurx commands: {e}",
                details={"error": str(e)}
            )
    
    def _check_resource_manager(self) -> DiagnosticResult:
        """
        Check resource manager health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Get resource manager
            resource_manager = get_resource_manager()
            
            if resource_manager is None:
                return DiagnosticResult(
                    component="resource_manager",
                    status=ComponentStatus.CRITICAL,
                    message="Resource manager is not available",
                    details={"error": "get_resource_manager() returned None"}
                )
            
            # Check resource allocation
            try:
                cpu_allocation = resource_manager.get_allocation(ResourceType.CPU)
                memory_allocation = resource_manager.get_allocation(ResourceType.MEMORY)
                
                if cpu_allocation is None or memory_allocation is None:
                    return DiagnosticResult(
                        component="resource_manager",
                        status=ComponentStatus.DEGRADED,
                        message="Resource allocations are not available",
                        details={
                            "cpu_allocation": cpu_allocation,
                            "memory_allocation": memory_allocation
                        }
                    )
            except Exception as allocation_error:
                return DiagnosticResult(
                    component="resource_manager",
                    status=ComponentStatus.DEGRADED,
                    message=f"Error checking resource allocations: {allocation_error}",
                    details={"error": str(allocation_error)}
                )
            
            # Return healthy result
            return DiagnosticResult(
                component="resource_manager",
                status=ComponentStatus.HEALTHY,
                message="Resource manager is healthy",
                details={
                    "cpu_allocation": cpu_allocation,
                    "memory_allocation": memory_allocation
                }
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="resource_manager",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking resource manager: {e}",
                details={"error": str(e)}
            )
    
    def _check_error_reporting(self) -> DiagnosticResult:
        """
        Check error reporting health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Check if error reporting module is available
            if not callable(report_error):
                return DiagnosticResult(
                    component="error_reporting",
                    status=ComponentStatus.CRITICAL,
                    message="Error reporting function is not available",
                    details={"missing_functions": "report_error"}
                )
            
            # Return healthy result
            return DiagnosticResult(
                component="error_reporting",
                status=ComponentStatus.HEALTHY,
                message="Error reporting is healthy",
                details={}
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="error_reporting",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking error reporting: {e}",
                details={"error": str(e)}
            )
    
    def _check_alert_system(self) -> DiagnosticResult:
        """
        Check alert system health.
        
        Returns:
            Diagnostic result.
        """
        try:
            # Get alert system
            alert_system = get_alert_system()
            
            if alert_system is None:
                return DiagnosticResult(
                    component="alert_system",
                    status=ComponentStatus.CRITICAL,
                    message="Alert system is not available",
                    details={"error": "get_alert_system() returned None"}
                )
            
            # Check rules
            try:
                rules = alert_system.get_rules()
                
                if not rules:
                    return DiagnosticResult(
                        component="alert_system",
                        status=ComponentStatus.DEGRADED,
                        message="No alert rules are defined",
                        details={"rules": rules}
                    )
            except Exception as rules_error:
                return DiagnosticResult(
                    component="alert_system",
                    status=ComponentStatus.DEGRADED,
                    message=f"Error checking alert rules: {rules_error}",
                    details={"error": str(rules_error)}
                )
            
            # Return healthy result
            return DiagnosticResult(
                component="alert_system",
                status=ComponentStatus.HEALTHY,
                message="Alert system is healthy",
                details={
                    "rule_count": len(rules)
                }
            )
        except Exception as e:
            # Return error result
            return DiagnosticResult(
                component="alert_system",
                status=ComponentStatus.UNKNOWN,
                message=f"Error checking alert system: {e}",
                details={"error": str(e)}
            )
    
    def diagnose(self, component: Optional[str] = None) -> Dict[str, DiagnosticResult]:
        """
        Run diagnostic checks.
        
        Args:
            component: Component to check, or None for all components.
            
        Returns:
            Dictionary of diagnostic results.
        """
        results = {}
        
        if component is not None:
            # Check specific component
            checker = self._component_checkers.get(component)
            if checker:
                try:
                    results[component] = checker()
                except Exception as e:
                    results[component] = DiagnosticResult(
                        component=component,
                        status=ComponentStatus.UNKNOWN,
                        message=f"Error checking component: {e}",
                        details={"error": str(e)}
                    )
            else:
                # Component not found
                results[component] = DiagnosticResult(
                    component=component,
                    status=ComponentStatus.UNKNOWN,
                    message=f"Unknown component: {component}",
                    details={}
                )
        else:
            # Check all components
            for component, checker in self._component_checkers.items():
                try:
                    results[component] = checker()
                except Exception as e:
                    results[component] = DiagnosticResult(
                        component=component,
                        status=ComponentStatus.UNKNOWN,
                        message=f"Error checking component: {e}",
                        details={"error": str(e)}
                    )
        
        # Update last health check
        for component, result in results.items():
            self._last_health_check[component] = result
        
        return results
    
    def get_health(self) -> Dict[str, DiagnosticResult]:
        """
        Get health status of components.
        
        Returns:
            Dictionary of diagnostic results.
        """
        # If no health checks have been run, run them now
        if not self._last_health_check:
            return self.diagnose()
        
        return self._last_health_check
    
    def get_metrics(self, count: int = 1) -> List[SystemMetrics]:
        """
        Get system metrics.
        
        Args:
            count: Number of historical metrics to return.
            
        Returns:
            List of system metrics.
        """
        if not self._metrics_history:
            # Collect metrics now
            metrics = self._collect_metrics()
            self._metrics_history.append(metrics)
            return [metrics]
        
        # Return latest metrics
        return self._metrics_history[-count:]
    
    def fix_issues(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt to fix issues.
        
        Args:
            component: Component to fix, or None for all components.
            
        Returns:
            Dictionary of fix results.
        """
        results = {}
        
        # First diagnose to get current status
        diagnostics = self.diagnose(component)
        
        for comp, result in diagnostics.items():
            # Skip healthy components
            if result.status == ComponentStatus.HEALTHY:
                results[comp] = {
                    "status": "healthy",
                    "message": "No issues to fix"
                }
                continue
            
            # Try to fix issues
            if comp == "system":
                results[comp] = self._fix_system_issues(result)
            elif comp == "shell_environment":
                results[comp] = self._fix_shell_environment_issues(result)
            elif comp == "shell_scripting":
                results[comp] = self._fix_shell_scripting_issues(result)
            elif comp == "fixwurx_commands":
                results[comp] = self._fix_fixwurx_commands_issues(result)
            elif comp == "resource_manager":
                results[comp] = self._fix_resource_manager_issues(result)
            elif comp == "error_reporting":
                results[comp] = self._fix_error_reporting_issues(result)
            elif comp == "alert_system":
                results[comp] = self._fix_alert_system_issues(result)
            else:
                # Unknown component
                results[comp] = {
                    "status": "unknown",
                    "message": f"Unknown component: {comp}",
                    "fixed": False
                }
        
        return results
    
    def _fix_system_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix system issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        # For system issues, we can't do much automatically
        # but we can provide some recommendations
        recommendations = []
        
        if result.status == ComponentStatus.CRITICAL or result.status == ComponentStatus.DEGRADED:
            if result.details.get("cpu_usage", 0) > 80:
                recommendations.append("Reduce CPU-intensive tasks or allocate more CPU resources")
            
            if result.details.get("memory_usage", 0) > 80:
                recommendations.append("Free up memory by closing unnecessary applications or allocate more memory")
            
            if result.details.get("disk_usage", 0) > 90:
                recommendations.append("Free up disk space by removing temporary files or unused applications")
        
        return {
            "status": "recommendations",
            "message": "System issues require manual intervention",
            "fixed": False,
            "recommendations": recommendations
        }
    
    def _fix_shell_environment_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix shell environment issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        fixed = False
        message = "Unable to fix shell environment issues automatically"
        
        # Try to reload shell environment module
        try:
            if result.status != ComponentStatus.HEALTHY:
                # Attempt to reload the module
                import importlib
                importlib.reload(sys.modules["shell_environment"])
                
                # Check if that fixed the issue
                new_result = self._check_shell_environment()
                if new_result.status == ComponentStatus.HEALTHY:
                    fixed = True
                    message = "Fixed shell environment issues by reloading the module"
        except Exception as e:
            message = f"Error attempting to fix shell environment: {e}"
        
        return {
            "status": "attempted",
            "message": message,
            "fixed": fixed,
            "details": result.details
        }
    
    def _fix_shell_scripting_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix shell scripting issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        fixed = False
        message = "Unable to fix shell scripting issues automatically"
        
        try:
            if result.status != ComponentStatus.HEALTHY:
                # Attempt to reload the module
                import importlib
                importlib.reload(sys.modules["shell_scripting"])
                
                # Check if that fixed the issue
                new_result = self._check_shell_scripting()
                if new_result.status == ComponentStatus.HEALTHY:
                    fixed = True
                    message = "Fixed shell scripting issues by reloading the module"
        except Exception as e:
            message = f"Error attempting to fix shell scripting: {e}"
        
        return {
            "status": "attempted",
            "message": message,
            "fixed": fixed,
            "details": result.details
        }
    
    def _fix_fixwurx_commands_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix FixWurx commands issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        fixed = False
        message = "Unable to fix FixWurx commands issues automatically"
        
        try:
            if result.status != ComponentStatus.HEALTHY:
                # Attempt to reload the module
                import importlib
                importlib.reload(sys.modules["fixwurx_commands"])
                
                # Check if that fixed the issue
                new_result = self._check_fixwurx_commands()
                if new_result.status == ComponentStatus.HEALTHY:
                    fixed = True
                    message = "Fixed FixWurx commands issues by reloading the module"
        except Exception as e:
            message = f"Error attempting to fix FixWurx commands: {e}"
        
        return {
            "status": "attempted",
            "message": message,
            "fixed": fixed,
            "details": result.details
        }
    
    def _fix_resource_manager_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix resource manager issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        fixed = False
        message = "Unable to fix resource manager issues automatically"
        
        try:
            if result.status != ComponentStatus.HEALTHY:
                # Attempt to reset the resource manager
                from resource_manager import reset_resource_manager
                if callable(reset_resource_manager):
                    reset_resource_manager()
                    
                    # Check if that fixed the issue
                    new_result = self._check_resource_manager()
                    if new_result.status == ComponentStatus.HEALTHY:
                        fixed = True
                        message = "Fixed resource manager issues by resetting it"
        except Exception as e:
            message = f"Error attempting to fix resource manager: {e}"
        
        return {
            "status": "attempted",
            "message": message,
            "fixed": fixed,
            "details": result.details
        }
    
    def _fix_error_reporting_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix error reporting issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        fixed = False
        message = "Unable to fix error reporting issues automatically"
        
        try:
            if result.status != ComponentStatus.HEALTHY:
                # Attempt to reload the module
                import importlib
                importlib.reload(sys.modules["error_reporting"])
                
                # Check if that fixed the issue
                new_result = self._check_error_reporting()
                if new_result.status == ComponentStatus.HEALTHY:
                    fixed = True
                    message = "Fixed error reporting issues by reloading the module"
        except Exception as e:
            message = f"Error attempting to fix error reporting: {e}"
        
        return {
            "status": "attempted",
            "message": message,
            "fixed": fixed,
            "details": result.details
        }
    
    def _fix_alert_system_issues(self, result: DiagnosticResult) -> Dict[str, Any]:
        """
        Fix alert system issues.
        
        Args:
            result: Diagnostic result.
            
        Returns:
            Fix result.
        """
        fixed = False
        message = "Unable to fix alert system issues automatically"
        
        try:
            if result.status != ComponentStatus.HEALTHY:
                # Check if it's missing rules
                if result.message.startswith("No alert rules"):
                    # Create default rules
                    from alert_system import get_alert_system
                    alert_system = get_alert_system()
                    
                    # Check if creating default rules is available
                    if hasattr(alert_system, "_create_default_rules"):
                        alert_system._create_default_rules()
                        
                        # Check if that fixed the issue
                        new_result = self._check_alert_system()
                        if new_result.status == ComponentStatus.HEALTHY:
                            fixed = True
                            message = "Fixed alert system issues by creating default rules"
        except Exception as e:
            message = f"Error attempting to fix alert system: {e}"
        
        return {
            "status": "attempted",
            "message": message,
            "fixed": fixed,
            "details": result.details
        }
    
    def diagnose_command(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle diagnose command.
        
        Args:
            args: Command arguments.
            
        Returns:
            Command result.
        """
        component = args[0] if args else None
        results = self.diagnose(component)
        
        # Format results
        formatted_results = []
        for component, result in results.items():
            formatted_results.append(str(result))
        
        return {
            "success": True,
            "results": formatted_results,
            "details": {component: result.to_dict() for component, result in results.items()}
        }
    
    def health_command(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle health command.
        
        Args:
            args: Command arguments.
            
        Returns:
            Command result.
        """
        health = self.get_health()
        
        # Format results
        formatted_results = []
        for component, result in health.items():
            formatted_results.append(str(result))
        
        return {
            "success": True,
            "results": formatted_results,
            "details": {component: result.to_dict() for component, result in health.items()}
        }
    
    def metrics_command(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle metrics command.
        
        Args:
            args: Command arguments.
            
        Returns:
            Command result.
        """
        # Parse count argument
        count = 1
        if args and args[0].isdigit():
            count = int(args[0])
        
        metrics = self.get_metrics(count)
        
        # Format results
        formatted_results = []
        for i, metric in enumerate(metrics):
            if i > 0:
                formatted_results.append("---")
            
            formatted_results.append(f"Time: {datetime.datetime.fromtimestamp(metric.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            formatted_results.append(f"CPU Usage: {metric.cpu_usage:.1f}%")
            formatted_results.append(f"Memory Usage: {metric.memory_usage:.1f}% ({metric.memory_available / (1024 * 1024):.1f} MB available)")
            formatted_results.append(f"Disk Usage: {metric.disk_usage:.1f}% ({metric.disk_available / (1024 * 1024 * 1024):.1f} GB available)")
            formatted_results.append(f"Processes: {metric.process_count}")
            formatted_results.append(f"Threads: {metric.thread_count}")
        
        return {
            "success": True,
            "results": formatted_results,
            "details": [metric.to_dict() for metric in metrics]
        }
    
    def doctor_command(self, args: List[str]) -> Dict[str, Any]:
        """
        Handle doctor command.
        
        Args:
            args: Command arguments.
            
        Returns:
            Command result.
        """
        component = args[0] if args else None
        
        # First diagnose
        results = self.diagnose(component)
        
        # Check if there are issues
        has_issues = False
        for result in results.values():
            if result.status != ComponentStatus.HEALTHY:
                has_issues = True
                break
        
        # If there are issues, try to fix them
        fix_results = {}
        if has_issues:
            fix_results = self.fix_issues(component)
        
        # Format results
        formatted_results = []
        for component, result in results.items():
            formatted_results.append(str(result))
            
            if result.status != ComponentStatus.HEALTHY and component in fix_results:
                fix_result = fix_results[component]
                if fix_result.get("fixed", False):
                    formatted_results.append(f"  ✅ {fix_result.get('message', 'Fixed')}")
                else:
                    formatted_results.append(f"  ❌ {fix_result.get('message', 'Not fixed')}")
                    
                    # Add recommendations if any
                    if "recommendations" in fix_result:
                        formatted_results.append("  Recommendations:")
                        for recommendation in fix_result["recommendations"]:
                            formatted_results.append(f"   - {recommendation}")
        
        return {
            "success": True,
            "results": formatted_results,
            "details": {
                "diagnostics": {component: result.to_dict() for component, result in results.items()},
                "fixes": fix_results
            }
        }


# Create singleton diagnostics
_shell_diagnostics = None

def get_shell_diagnostics() -> ShellDiagnostics:
    """
    Get the singleton shell diagnostics instance.
    
    Returns:
        Shell diagnostics instance.
    """
    global _shell_diagnostics
    
    if _shell_diagnostics is None:
        _shell_diagnostics = ShellDiagnostics()
    
    return _shell_diagnostics

def diagnose(component: Optional[str] = None) -> Dict[str, DiagnosticResult]:
    """
    Run diagnostic checks.
    
    Args:
        component: Component to check, or None for all components.
        
    Returns:
        Dictionary of diagnostic results.
    """
    return get_shell_diagnostics().diagnose(component)

def get_health() -> Dict[str, DiagnosticResult]:
    """
    Get health status of components.
    
    Returns:
        Dictionary of diagnostic results.
    """
    return get_shell_diagnostics().get_health()

def get_metrics(count: int = 1) -> List[SystemMetrics]:
    """
    Get system metrics.
    
    Args:
        count: Number of historical metrics to return.
        
    Returns:
        List of system metrics.
    """
    return get_shell_diagnostics().get_metrics(count)

def fix_issues(component: Optional[str] = None) -> Dict[str, Any]:
    """
    Attempt to fix issues.
    
    Args:
        component: Component to fix, or None for all components.
        
    Returns:
        Dictionary of fix results.
    """
    return get_shell_diagnostics().fix_issues(component)

# Initialize shell diagnostics if not in a test environment
if not any(arg.endswith('test.py') for arg in sys.argv):
    # Initialize shell diagnostics
    get_shell_diagnostics()
