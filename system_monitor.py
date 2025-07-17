"""
monitoring/system_monitor.py
────────────────────────────
Light-weight *telemetry bridge* that publishes core runtime metrics and
structured error logs to various backends (Prometheus, Datadog, ELK, stdout).

Enhanced Features:
- Core runtime metrics: tick counter, scope entropy (H₀), agent utilization
- Structured error logging with severity levels
- Rotating buffer of the last 100 errors
- Severity-based filtering and query capabilities
- Dashboard integration for visualization
- Daily log file persistence

Abstractions
────────────
`MetricBus` is a *very thin* dependency-inversion layer: an object exposing a
single

    send(metric_name:str, value:float, tags:dict[str,str]|None) -> None

method. In production we inject a Prometheus/StatsD/OTLP implementation; in
unit-tests we pass a tiny stub that records calls.

Usage
─────
```python
bus = PromMetricBus()          # your impl
mon = SystemMonitor(engine, metric_bus=bus)

while True:
    await engine.execute_tick()
    mon.emit_tick()            # push metrics for this tick
    
    # Log an error if one occurred
    if error_condition:
        mon.log_error("Error description", component="component_name")
    
    # Query recent errors
    recent_errors = mon.get_recent_errors(min_severity="WARNING")
```
"""
from __future__ import annotations

import os
import time
from typing import Protocol, Dict, Optional, List, Any, Union

from monitoring.error_log import ErrorLog, ErrorSeverity

class Engine(Protocol):
    """Protocol for the Triangulum engine, defining the interface for monitoring."""
    @property
    def tick_count(self) -> int:
        ...

    @property
    def scope_entropy(self) -> float:
        ...

    @property
    def agent_utilization(self) -> Dict[str, float]:
        ...
        
    @property
    def error_state(self) -> Dict[str, Any]:
        """Current error state of the engine."""
        ...
    
    @property
    def planner_metrics(self) -> Dict[str, Any]:
        """Metrics from the planner agent."""
        ...
        
    def get_path_metrics(self, path_id: str) -> Dict[str, Any]:
        """Get metrics for a specific solution path."""
        ...
    
    def get_all_path_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all solution paths."""
        ...
    
    def get_family_tree(self) -> Dict[str, Any]:
        """Get the agent family tree."""
        ...
    
    def get_agent_assignments(self) -> Dict[str, Dict[str, Any]]:
        """Get agent assignments to solution paths."""
        ...

class MetricBus(Protocol):
    """Protocol for a metric bus, defining the interface for sending metrics."""
    def send(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        ...

class SystemMonitor:
    """
    Publishes core runtime metrics and error logs from the Triangulum engine.
    
    Features:
    - Metric collection and emission
    - Structured error logging
    - Query and filtering capabilities
    - Dashboard integration
    """
    def __init__(
        self, 
        engine: Engine, 
        metric_bus: MetricBus, 
        env: str = "dev",
        log_dir: Optional[str] = None,
        max_error_entries: int = 100
    ):
        """
        Initialize the SystemMonitor.

        Args:
            engine: The Triangulum engine to monitor.
            metric_bus: The metric bus to send metrics to.
            env: Environment identifier (dev, test, prod, etc.)
            log_dir: Directory for error log persistence
            max_error_entries: Maximum number of error entries to keep in memory
        """
        self.engine = engine
        self.metric_bus = metric_bus
        self.env = env
        
        # Initialize error log
        self.error_log = ErrorLog(
            max_entries=max_error_entries,
            log_dir=log_dir
        )
        
        # Track last error check time
        self.last_error_check = time.time()
        
        # Initialize metrics tracking
        self.metrics_history = {
            "tick_count": [],
            "scope_entropy": [],
            "agent_utilization": {},
            "planner": {
                "path_completion_rate": [],
                "path_success_rate": [],
                "average_path_complexity": [],
                "fallback_activation_rate": [],
                "paths_in_progress": [],
                "paths_completed": [],
                "paths_failed": []
            },
            "solution_paths": {},
            "family_tree": {
                "agent_count": [],
                "tree_depth": [],
                "branch_factor": []
            }
        }
        self.metrics_max_history = 1000  # Keep the last 1000 data points
        
        # Keep track of active solution paths
        self.active_solution_paths = set()

    def emit_tick(self) -> None:
        """
        Emit metrics for the current engine tick and check for new errors.
        """
        # Emit core metrics
        tick_count = self.engine.tick_count
        scope_entropy = self.engine.scope_entropy
        
        self.metric_bus.send("triangulum.tick_count", tick_count)
        self.metric_bus.send("triangulum.scope_entropy", scope_entropy)
        
        # Store metrics history
        self._update_metrics_history("tick_count", tick_count)
        self._update_metrics_history("scope_entropy", scope_entropy)

        # Track agent utilization
        for agent_id, utilization in self.engine.agent_utilization.items():
            self.metric_bus.send(
                "triangulum.agent_utilization",
                utilization,
                tags={"agent_id": agent_id}
            )
            
            # Store agent utilization history
            if agent_id not in self.metrics_history["agent_utilization"]:
                self.metrics_history["agent_utilization"][agent_id] = []
                
            self._update_metrics_history(
                f"agent_utilization.{agent_id}", 
                utilization
            )
        
        # Emit planner metrics if available
        self._emit_planner_metrics()
        
        # Check for new errors in the engine state
        self._check_engine_errors()
    
    def _update_metrics_history(self, metric_name: str, value: float) -> None:
        """Update the metrics history with a new data point."""
        if "." in metric_name:
            # Handle nested metrics like agent_utilization.agent_id
            category, key = metric_name.split(".", 1)
            if category in self.metrics_history and isinstance(self.metrics_history[category], dict):
                if key not in self.metrics_history[category]:
                    self.metrics_history[category][key] = []
                
                history = self.metrics_history[category][key]
                history.append((time.time(), value))
                
                # Trim if needed
                if len(history) > self.metrics_max_history:
                    self.metrics_history[category][key] = history[-self.metrics_max_history:]
        else:
            # Handle top-level metrics
            if metric_name in self.metrics_history:
                history = self.metrics_history[metric_name]
                history.append((time.time(), value))
                
                # Trim if needed
                if len(history) > self.metrics_max_history:
                    self.metrics_history[metric_name] = history[-self.metrics_max_history:]
    
    def _check_engine_errors(self) -> None:
        """Check for new errors in the engine state and log them."""
        # Skip if engine doesn't have error_state
        if not hasattr(self.engine, 'error_state'):
            return
            
        try:
            error_state = self.engine.error_state
            
            # Skip if error_state is empty or not a dict
            if not error_state or not isinstance(error_state, dict):
                return
                
            # Process new errors
            for component, errors in error_state.items():
                if not errors:
                    continue
                    
                for error in errors:
                    # Extract error details
                    message = error.get("message", "Unknown error")
                    severity_str = error.get("severity", "ERROR").upper()
                    timestamp = error.get("timestamp", time.time())
                    context = error.get("context", {})
                    
                    # Skip if we've already processed this error
                    if timestamp <= self.last_error_check:
                        continue
                    
                    # Convert severity string to ErrorSeverity
                    try:
                        severity = ErrorSeverity.from_string(severity_str)
                    except ValueError:
                        severity = ErrorSeverity.ERROR
                    
                    # Log the error
                    self.error_log.add_entry(
                        message=message,
                        severity=severity,
                        component=component,
                        **context
                    )
            
            # Update last error check time
            self.last_error_check = time.time()
        except Exception as e:
            # Log internally if error checking fails
            self.error_log.error(
                f"Failed to check engine errors: {e}",
                component="system_monitor",
                exception=str(e)
            )
    
    def log_debug(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log a DEBUG level message."""
        return self.error_log.debug(message, component, **context)
    
    def log_info(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log an INFO level message."""
        return self.error_log.info(message, component, **context)
    
    def log_warning(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log a WARNING level message."""
        return self.error_log.warning(message, component, **context)
    
    def log_error(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log an ERROR level message."""
        return self.error_log.error(message, component, **context)
    
    def log_critical(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log a CRITICAL level message."""
        return self.error_log.critical(message, component, **context)
    
    def get_recent_errors(
        self,
        count: int = 10,
        min_severity: str = "WARNING"
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent errors.
        
        Args:
            count: Maximum number of errors to return
            min_severity: Minimum severity level
            
        Returns:
            List of recent errors
        """
        return self.error_log.query(
            min_severity=min_severity,
            limit=count
        )
    
    def query_errors(
        self,
        min_severity: Optional[Union[str, ErrorSeverity]] = None,
        max_severity: Optional[Union[str, ErrorSeverity]] = None,
        component: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        contains_text: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Query errors with filtering (delegates to error_log.query)."""
        return self.error_log.query(
            min_severity=min_severity,
            max_severity=max_severity,
            component=component,
            start_time=start_time,
            end_time=end_time,
            contains_text=contains_text,
            limit=limit
        )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about logged errors."""
        return self.error_log.get_stats()
    
    def get_errors_for_dashboard(
        self,
        hours: int = 24,
        min_severity: str = "WARNING"
    ) -> List[Dict[str, Any]]:
        """Get errors formatted for dashboard display."""
        return self.error_log.get_entries_for_dashboard(
            hours=hours,
            min_severity=min_severity
        )
    
    def _emit_planner_metrics(self) -> None:
        """
        Emit planner-specific metrics.
        """
        # Skip if engine doesn't have planner metrics
        if not hasattr(self.engine, 'planner_metrics'):
            return
        
        try:
            # Get planner metrics
            planner_metrics = self.engine.planner_metrics
            
            # Skip if no planner metrics available
            if not planner_metrics:
                return
            
            # Emit core planner metrics
            for key, value in planner_metrics.items():
                if isinstance(value, (int, float)):
                    self.metric_bus.send(
                        f"triangulum.planner.{key}",
                        value
                    )
                    self._update_metrics_history(f"planner.{key}", value)
            
            # Track solution path metrics
            if hasattr(self.engine, 'get_all_path_metrics'):
                path_metrics = self.engine.get_all_path_metrics()
                if path_metrics:
                    # Calculate aggregated metrics
                    active_paths = 0
                    completed_paths = 0
                    failed_paths = 0
                    total_complexity = 0.0
                    successful_paths = 0
                    
                    for path_id, metrics in path_metrics.items():
                        status = metrics.get("status", "unknown")
                        
                        # Track individual path metrics
                        if path_id not in self.metrics_history["solution_paths"]:
                            self.metrics_history["solution_paths"][path_id] = {}
                            
                        # Store path metrics history
                        for metric_key, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                metric_path = f"solution_paths.{path_id}.{metric_key}"
                                self._update_metrics_history(metric_path, metric_value)
                        
                        # Count paths by status
                        if status == "active":
                            active_paths += 1
                            self.active_solution_paths.add(path_id)
                        elif status == "completed":
                            completed_paths += 1
                            if path_id in self.active_solution_paths:
                                self.active_solution_paths.remove(path_id)
                            if metrics.get("success", False):
                                successful_paths += 1
                        elif status == "failed":
                            failed_paths += 1
                            if path_id in self.active_solution_paths:
                                self.active_solution_paths.remove(path_id)
                        
                        # Sum complexity for average calculation
                        complexity = metrics.get("complexity", 0.0)
                        if isinstance(complexity, (int, float)):
                            total_complexity += complexity
                    
                    # Calculate derived metrics
                    total_paths = active_paths + completed_paths + failed_paths
                    avg_complexity = total_complexity / max(1, total_paths)
                    completion_rate = completed_paths / max(1, completed_paths + active_paths)
                    success_rate = successful_paths / max(1, completed_paths)
                    fallback_rate = sum(1 for m in path_metrics.values() if m.get("is_fallback", False)) / max(1, total_paths)
                    
                    # Emit aggregated metrics
                    self.metric_bus.send("triangulum.planner.paths_in_progress", active_paths)
                    self.metric_bus.send("triangulum.planner.paths_completed", completed_paths)
                    self.metric_bus.send("triangulum.planner.paths_failed", failed_paths)
                    self.metric_bus.send("triangulum.planner.path_completion_rate", completion_rate)
                    self.metric_bus.send("triangulum.planner.path_success_rate", success_rate)
                    self.metric_bus.send("triangulum.planner.average_path_complexity", avg_complexity)
                    self.metric_bus.send("triangulum.planner.fallback_activation_rate", fallback_rate)
                    
                    # Store aggregated metrics history
                    self._update_metrics_history("planner.paths_in_progress", active_paths)
                    self._update_metrics_history("planner.paths_completed", completed_paths)
                    self._update_metrics_history("planner.paths_failed", failed_paths)
                    self._update_metrics_history("planner.path_completion_rate", completion_rate)
                    self._update_metrics_history("planner.path_success_rate", success_rate)
                    self._update_metrics_history("planner.average_path_complexity", avg_complexity)
                    self._update_metrics_history("planner.fallback_activation_rate", fallback_rate)
            
            # Track family tree metrics if available
            if hasattr(self.engine, 'get_family_tree'):
                family_tree = self.engine.get_family_tree()
                if family_tree:
                    # Calculate tree metrics
                    agent_count = len(family_tree.get("agents", []))
                    tree_depth = family_tree.get("max_depth", 0)
                    branch_factor = family_tree.get("avg_branch_factor", 0.0)
                    
                    # Emit family tree metrics
                    self.metric_bus.send("triangulum.family_tree.agent_count", agent_count)
                    self.metric_bus.send("triangulum.family_tree.tree_depth", tree_depth)
                    self.metric_bus.send("triangulum.family_tree.branch_factor", branch_factor)
                    
                    # Store family tree metrics history
                    self._update_metrics_history("family_tree.agent_count", agent_count)
                    self._update_metrics_history("family_tree.tree_depth", tree_depth)
                    self._update_metrics_history("family_tree.branch_factor", branch_factor)
            
            # Track agent assignments if available
            if hasattr(self.engine, 'get_agent_assignments'):
                assignments = self.engine.get_agent_assignments()
                if assignments:
                    # Emit assignment metrics
                    self.metric_bus.send("triangulum.planner.assigned_agents", len(assignments))
                    
                    # Calculate assignment statistics
                    bugs_with_assignments = set()
                    paths_with_assignments = set()
                    
                    for agent_id, assignment in assignments.items():
                        bug_id = assignment.get("bug_id")
                        path_id = assignment.get("path_id")
                        
                        if bug_id:
                            bugs_with_assignments.add(bug_id)
                        if path_id:
                            paths_with_assignments.add(path_id)
                    
                    # Emit assignment statistics
                    self.metric_bus.send("triangulum.planner.bugs_with_assignments", len(bugs_with_assignments))
                    self.metric_bus.send("triangulum.planner.paths_with_assignments", len(paths_with_assignments))
                    
        except Exception as e:
            # Log error if planner metrics emission fails
            self.log_error(
                f"Failed to emit planner metrics: {e}",
                component="system_monitor",
                exception=str(e)
            )
    
    def get_metrics_for_dashboard(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get metrics formatted for dashboard display.
        
        Args:
            hours: Hours of history to include
            
        Returns:
            Dictionary of metric data for the dashboard
        """
        # Calculate start time
        start_time = time.time() - (hours * 3600)
        
        # Extract metrics in the time range
        result = {
            "tick_count": [],
            "scope_entropy": [],
            "agent_utilization": {},
            "planner": {},
            "family_tree": {},
            "solution_paths": {}
        }
        
        # Process top-level metrics
        for metric in ["tick_count", "scope_entropy"]:
            if metric in self.metrics_history:
                result[metric] = [
                    (ts, val) for ts, val in self.metrics_history[metric]
                    if ts >= start_time
                ]
        
        # Process agent utilization
        for agent_id, history in self.metrics_history["agent_utilization"].items():
            result["agent_utilization"][agent_id] = [
                (ts, val) for ts, val in history
                if ts >= start_time
            ]
        
        # Process planner metrics
        for metric, history in self.metrics_history["planner"].items():
            result["planner"][metric] = [
                (ts, val) for ts, val in history
                if ts >= start_time
            ]
        
        # Process family tree metrics
        for metric, history in self.metrics_history["family_tree"].items():
            result["family_tree"][metric] = [
                (ts, val) for ts, val in history
                if ts >= start_time
            ]
        
        # Process solution path metrics (limit to the most active paths)
        active_paths = list(self.active_solution_paths)
        # Include up to 10 most active paths
        for path_id in active_paths[:10]:
            if path_id in self.metrics_history["solution_paths"]:
                result["solution_paths"][path_id] = {}
                for metric, history in self.metrics_history["solution_paths"][path_id].items():
                    result["solution_paths"][path_id][metric] = [
                        (ts, val) for ts, val in history
                        if ts >= start_time
                    ]
        
        # Add some aggregate statistics
        if result["scope_entropy"]:
            result["avg_entropy"] = sum(v for _, v in result["scope_entropy"]) / len(result["scope_entropy"])
        else:
            result["avg_entropy"] = 0
            
        # Calculate planner aggregates if available
        if "path_success_rate" in result["planner"] and result["planner"]["path_success_rate"]:
            result["avg_path_success_rate"] = sum(v for _, v in result["planner"]["path_success_rate"]) / len(result["planner"]["path_success_rate"])
        else:
            result["avg_path_success_rate"] = 0
            
        # Add error counts by severity
        error_stats = self.error_log.get_stats()
        result["errors"] = error_stats.get("by_severity", {})
        
        return result
    
    def export_error_logs(self, filepath: str) -> bool:
        """Export error logs to a file."""
        return self.error_log.export_to_file(filepath)
