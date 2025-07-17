"""
monitoring/error_log.py
──────────────────────
Structured error logging system with a rotating buffer, severity-based filtering,
and dashboard integration capabilities.

Features:
- Circular buffer maintaining the last 100 errors
- Error severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Timestamp and context information for each log entry
- Query and filtering capabilities by severity, component, and time range
- JSON-serializable format for dashboard integration

Usage:
    ```python
    # Create a logger
    error_log = ErrorLog()
    
    # Log errors with different severity levels
    error_log.debug("Connection established", component="network")
    error_log.info("Processing started", component="engine", context={"job_id": "123"})
    error_log.warning("Resource usage high", component="memory", usage=85)
    error_log.error("Failed to connect", component="database", exception=str(ex))
    error_log.critical("System shutdown", component="core", reason="power failure")
    
    # Query errors by severity
    critical_errors = error_log.query(min_severity="CRITICAL")
    
    # Query errors by component
    db_errors = error_log.query(component="database")
    
    # Get recent errors for dashboard
    recent = error_log.get_recent(10)
    
    # Export all logs for persistence
    error_log.export_to_file("logs/error_history.json")
    ```
"""

import time
import json
import os
import threading
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque


class ErrorSeverity(Enum):
    """Error severity levels in ascending order of importance."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    
    @classmethod
    def from_string(cls, severity_str: str) -> 'ErrorSeverity':
        """Convert a string to an ErrorSeverity enum."""
        try:
            return cls[severity_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid severity level: {severity_str}")


class ErrorLog:
    """
    Structured error logging system with a rotating buffer and filtering capabilities.
    
    Maintains a thread-safe circular buffer of the most recent errors, with
    severity levels, timestamps, and context information.
    """
    
    def __init__(self, max_entries: int = 100, log_dir: Optional[str] = None):
        """
        Initialize the error log.
        
        Args:
            max_entries: Maximum number of log entries to keep in memory
            log_dir: Directory for persistent log files (if None, logs are not persisted)
        """
        self.max_entries = max_entries
        self.log_dir = log_dir
        self.entries = deque(maxlen=max_entries)
        self.lock = threading.RLock()
        
        # Create log directory if specified
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def add_entry(
        self,
        message: str,
        severity: ErrorSeverity,
        component: str,
        **context
    ) -> Dict[str, Any]:
        """
        Add an error log entry.
        
        Args:
            message: Error message
            severity: Severity level
            component: System component that generated the error
            **context: Additional context as keyword arguments
            
        Returns:
            The created log entry
        """
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "severity": severity.name,
            "severity_level": severity.value,
            "component": component,
            "message": message,
            "context": {**context}
        }
        
        with self.lock:
            self.entries.append(entry)
            
            # If persistence is enabled, append to daily log file
            if self.log_dir:
                self._persist_entry(entry)
        
        return entry
    
    def debug(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log a DEBUG level message."""
        return self.add_entry(message, ErrorSeverity.DEBUG, component, **context)
    
    def info(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log an INFO level message."""
        return self.add_entry(message, ErrorSeverity.INFO, component, **context)
    
    def warning(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log a WARNING level message."""
        return self.add_entry(message, ErrorSeverity.WARNING, component, **context)
    
    def error(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log an ERROR level message."""
        return self.add_entry(message, ErrorSeverity.ERROR, component, **context)
    
    def critical(self, message: str, component: str, **context) -> Dict[str, Any]:
        """Log a CRITICAL level message."""
        return self.add_entry(message, ErrorSeverity.CRITICAL, component, **context)
    
    def query(
        self,
        min_severity: Optional[Union[str, ErrorSeverity]] = None,
        max_severity: Optional[Union[str, ErrorSeverity]] = None,
        component: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        contains_text: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query log entries with filtering.
        
        Args:
            min_severity: Minimum severity level (inclusive)
            max_severity: Maximum severity level (inclusive)
            component: Filter by component
            start_time: Filter by start timestamp (Unix time)
            end_time: Filter by end timestamp (Unix time)
            contains_text: Filter entries containing this text in message
            limit: Maximum number of entries to return (newest first)
            
        Returns:
            List of matching log entries
        """
        # Convert string severity to enum if needed
        if isinstance(min_severity, str):
            min_severity = ErrorSeverity.from_string(min_severity)
        if isinstance(max_severity, str):
            max_severity = ErrorSeverity.from_string(max_severity)
        
        # Get min/max severity values for comparison
        min_level = min_severity.value if min_severity else float('-inf')
        max_level = max_severity.value if max_severity else float('inf')
        
        with self.lock:
            # Apply filters
            filtered = []
            for entry in reversed(self.entries):  # newest first
                # Check severity range
                severity_level = entry["severity_level"]
                if not (min_level <= severity_level <= max_level):
                    continue
                
                # Check component
                if component and entry["component"] != component:
                    continue
                
                # Check time range
                timestamp = entry["timestamp"]
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                
                # Check text content
                if contains_text and contains_text.lower() not in entry["message"].lower():
                    continue
                
                filtered.append(entry)
                
                # Apply limit
                if limit and len(filtered) >= limit:
                    break
            
            return filtered
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent log entries.
        
        Args:
            count: Maximum number of entries to return
            
        Returns:
            List of recent log entries, newest first
        """
        with self.lock:
            # Create a copy of all entries
            all_entries = list(self.entries)
            
            # Sort entries by timestamp (newest first)
            # Use a more robust sorting to ensure proper ordering
            all_entries = sorted(all_entries, key=lambda e: float(e.get("timestamp", 0)), reverse=True)
            
            # Take the first 'count' entries or all if fewer are available
            return all_entries[:min(count, len(all_entries))]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the error log.
        
        Returns:
            Dictionary with statistics
        """
        with self.lock:
            severity_counts = {level.name: 0 for level in ErrorSeverity}
            component_counts = {}
            
            for entry in self.entries:
                severity = entry["severity"]
                component = entry["component"]
                
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                component_counts[component] = component_counts.get(component, 0) + 1
            
            # Get time range
            if self.entries:
                oldest = min(entry["timestamp"] for entry in self.entries)
                newest = max(entry["timestamp"] for entry in self.entries)
                time_span = newest - oldest
            else:
                oldest = 0
                newest = 0
                time_span = 0
            
            return {
                "total_entries": len(self.entries),
                "by_severity": severity_counts,
                "by_component": component_counts,
                "oldest_entry": oldest,
                "newest_entry": newest,
                "time_span_seconds": time_span,
                "max_capacity": self.max_entries
            }
    
    def clear(self) -> None:
        """Clear all log entries."""
        with self.lock:
            self.entries.clear()
    
    def export_to_file(self, filepath: str) -> bool:
        """
        Export all log entries to a JSON file.
        
        Args:
            filepath: Path to the output file
            
        Returns:
            True if export was successful
        """
        try:
            with self.lock:
                with open(filepath, 'w') as f:
                    json.dump(list(self.entries), f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting logs: {e}")
            return False
    
    def _persist_entry(self, entry: Dict[str, Any]) -> None:
        """
        Persist a log entry to the daily log file.
        
        Args:
            entry: Log entry to persist
        """
        try:
            # Create filename based on date
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"error_log_{date_str}.jsonl"
            filepath = os.path.join(self.log_dir, filename)
            
            # Append entry as JSON line
            with open(filepath, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            # Don't use self.error here to avoid potential recursion
            print(f"Error persisting log entry: {e}")
    
    def get_entries_for_dashboard(
        self,
        hours: int = 24,
        min_severity: str = "WARNING",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get entries formatted for dashboard display.
        
        Args:
            hours: Hours of history to include
            min_severity: Minimum severity to include
            limit: Maximum number of entries
            
        Returns:
            List of log entries formatted for dashboard
        """
        # Calculate start time
        start_time = time.time() - (hours * 3600)
        
        # Query with dashboard-specific formatting
        entries = self.query(
            min_severity=min_severity,
            start_time=start_time,
            limit=limit
        )
        
        # Format for dashboard
        for entry in entries:
            # Convert timestamp to readable format
            entry["time"] = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
            entry["date"] = datetime.fromtimestamp(entry["timestamp"]).strftime("%Y-%m-%d")
            
            # Add severity class for CSS styling
            entry["severity_class"] = entry["severity"].lower()
            
            # Format context as string if present
            if entry["context"]:
                context_items = [f"{k}={v}" for k, v in entry["context"].items()]
                entry["context_str"] = ", ".join(context_items)
            else:
                entry["context_str"] = ""
        
        return entries


# Global singleton instance for convenience
global_error_log = ErrorLog()


# Example usage
if __name__ == "__main__":
    # Create a logger
    logger = ErrorLog(max_entries=10)
    
    # Log some test entries
    logger.debug("Application started", component="app")
    logger.info("User logged in", component="auth", user_id="user123")
    logger.warning("High memory usage", component="system", usage_percent=85)
    
    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        logger.error(
            "Division error occurred",
            component="calculator",
            exception=str(e),
            traceback=str(e.__traceback__)
        )
    
    logger.critical(
        "Database connection failed",
        component="database",
        connection_string="postgres://localhost:5432"
    )
    
    # Query for warnings and above
    high_severity = logger.query(min_severity="WARNING")
    print(f"Found {len(high_severity)} high severity logs:")
    for entry in high_severity:
        print(f"[{entry['severity']}] {entry['message']}")
    
    # Export logs
    logger.export_to_file("error_log_export.json")
    print("Logs exported to error_log_export.json")
