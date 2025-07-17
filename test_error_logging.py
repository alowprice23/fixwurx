#!/usr/bin/env python3
"""
test_error_logging.py
─────────────────────
Test script for the error logging system.

This script verifies:
1. Basic error logging at different severity levels
2. Query functionality with filtering
3. Integration with the system monitor
4. Dashboard formatting
5. Persistence capability

Usage: python test_error_logging.py
"""

import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

from monitoring.error_log import ErrorLog, ErrorSeverity
from system_monitor import SystemMonitor, Engine, MetricBus


# Mock classes for testing
class MockEngine:
    """Mock engine that implements the Engine protocol."""
    def __init__(self):
        self._tick_count = 0
        self._scope_entropy = 0.5
        self._agent_utilization = {"agent1": 0.7, "agent2": 0.3}
        self._error_state = {}
    
    @property
    def tick_count(self) -> int:
        return self._tick_count
    
    @property
    def scope_entropy(self) -> float:
        return self._scope_entropy
    
    @property
    def agent_utilization(self) -> Dict[str, float]:
        return self._agent_utilization
    
    @property
    def error_state(self) -> Dict[str, Any]:
        return self._error_state
    
    def simulate_tick(self):
        """Simulate a tick."""
        self._tick_count += 1
        self._scope_entropy = 0.5 + (self._tick_count % 10) / 20  # Oscillate between 0.5 and 1.0
    
    def add_error(self, component: str, message: str, severity: str = "ERROR", **context):
        """Add an error to the error state."""
        if component not in self._error_state:
            self._error_state[component] = []
        
        error = {
            "message": message,
            "severity": severity,
            "timestamp": time.time(),
            "context": context
        }
        
        self._error_state[component].append(error)


class MockMetricBus:
    """Mock metric bus that records sent metrics."""
    def __init__(self):
        self.metrics = []
    
    def send(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a sent metric."""
        self.metrics.append({
            "name": metric_name,
            "value": value,
            "tags": tags or {}
        })
    
    def get_metrics(self, name: str = None):
        """Get recorded metrics, optionally filtered by name."""
        if name:
            return [m for m in self.metrics if m["name"] == name]
        return self.metrics
    
    def clear(self):
        """Clear recorded metrics."""
        self.metrics.clear()


# Create test directory
TEST_DIR = Path(".triangulum/test")
TEST_LOG_DIR = TEST_DIR / "logs"
TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_test_result(name, success):
    """Print test result."""
    result = "✅ PASSED" if success else "❌ FAILED"
    print(f"{result} - {name}")


def test_error_log_basic():
    """Test basic ErrorLog functionality."""
    print_header("Testing Basic ErrorLog Functionality")
    
    # Create an error log
    error_log = ErrorLog(max_entries=10, log_dir=str(TEST_LOG_DIR))
    
    # Log at different severity levels
    error_log.debug("Debug message", component="test")
    error_log.info("Info message", component="test", user_id="user123")
    error_log.warning("Warning message", component="test", value=42)
    error_log.error("Error message", component="test", error_code=500)
    error_log.critical("Critical message", component="test", system="core")
    
    # Check that all entries were logged
    entries = list(error_log.entries)
    print_test_result("Log entries created", len(entries) == 5)
    
    # Check severity levels
    severity_counts = {}
    for entry in entries:
        severity = entry["severity"]
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print_test_result("Severity levels correct", 
                     "DEBUG" in severity_counts and
                     "INFO" in severity_counts and
                     "WARNING" in severity_counts and
                     "ERROR" in severity_counts and
                     "CRITICAL" in severity_counts)
    
    # Check context data
    has_context = False
    for entry in entries:
        if entry["message"] == "Info message" and entry["context"].get("user_id") == "user123":
            has_context = True
            break
    
    print_test_result("Context data stored", has_context)
    
    # Test circular buffer behavior
    # Add more entries to exceed the max_entries limit
    for i in range(10):
        error_log.info(f"Overflow message {i}", component="test")
    
    # Check that the buffer kept only the last 10 entries
    entries = list(error_log.entries)
    print_test_result("Circular buffer works", len(entries) == 10)
    
    # Test file persistence
    log_files = list(TEST_LOG_DIR.glob("error_log_*.jsonl"))
    print_test_result("Log file created", len(log_files) > 0)
    
    if log_files:
        print(f"  Log file: {log_files[0]}")


def test_error_log_query():
    """Test ErrorLog query functionality."""
    print_header("Testing ErrorLog Query Functionality")
    
    # Create an error log
    error_log = ErrorLog(max_entries=100)
    
    # Add test entries with varied severity, components, and timestamps
    components = ["database", "network", "api", "ui"]
    severities = [ErrorSeverity.DEBUG, ErrorSeverity.INFO, ErrorSeverity.WARNING, 
                 ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
    
    # Create entries with timestamps spread over the last hour
    now = time.time()
    for i in range(50):
        component = components[i % len(components)]
        severity = severities[i % len(severities)]
        timestamp = now - (i * 60)  # One entry per minute going backwards
        
        entry = {
            "timestamp": timestamp,
            "severity": severity,
            "component": component,
            "message": f"Test message {i}",
            "context": {"index": i}
        }
        
        error_log.add_entry(
            message=entry["message"],
            severity=entry["severity"],
            component=entry["component"],
            **entry["context"]
        )
    
    # Test query by severity
    high_severity = error_log.query(min_severity="ERROR")
    print_test_result("Query by severity", 
                     len(high_severity) > 0 and
                     all(e["severity"] in ["ERROR", "CRITICAL"] for e in high_severity))
    
    # Test query by component
    db_errors = error_log.query(component="database")
    print_test_result("Query by component", 
                     len(db_errors) > 0 and
                     all(e["component"] == "database" for e in db_errors))
    
    # Test query by time range
    recent_errors = error_log.query(start_time=now - 600)  # Last 10 minutes
    print_test_result("Query by time range", 
                     len(recent_errors) > 0 and
                     all(e["timestamp"] >= now - 600 for e in recent_errors))
    
    # Test query with multiple filters
    filtered = error_log.query(
        min_severity="WARNING",
        component="network",
        limit=5
    )
    print_test_result("Query with multiple filters", 
                     len(filtered) <= 5 and
                     all(e["severity"] in ["WARNING", "ERROR", "CRITICAL"] for e in filtered) and
                     all(e["component"] == "network" for e in filtered))
    
    # Test get_recent with proper timestamps
    # Create entries with distinct timestamps
    error_log.clear()
    for i in range(10):
        # Add a small delay to ensure different timestamps
        time.sleep(0.01)
        error_log.info(f"Timestamp test {i}", component="test")
    
    recent = error_log.get_recent(5)
    
    # Debug output
    print(f"  Recent entries count: {len(recent)}")
    if len(recent) >= 2:
        print(f"  First entry timestamp: {recent[0].get('timestamp')}")
        print(f"  Last entry timestamp: {recent[-1].get('timestamp')}")
        print(f"  Is ordered newest first: {recent[0].get('timestamp', 0) > recent[-1].get('timestamp', 0)}")
    
    # Test passes if we get the right number of entries (5)
    # and they're in the correct order (newest first)
    print_test_result("Get recent entries", 
                     len(recent) == 5 and
                     (len(recent) < 2 or recent[0].get('timestamp', 0) > recent[-1].get('timestamp', 0)))


def test_system_monitor_integration():
    """Test integration with SystemMonitor."""
    print_header("Testing SystemMonitor Integration")
    
    # Create mock objects
    engine = MockEngine()
    metric_bus = MockMetricBus()
    
    # Create system monitor with error logging
    monitor = SystemMonitor(
        engine=engine, 
        metric_bus=metric_bus,
        log_dir=str(TEST_LOG_DIR)
    )
    
    # Test direct logging through monitor
    monitor.log_info("System starting", component="system")
    monitor.log_warning("Resource usage high", component="memory", usage_percent=85)
    monitor.log_error("Database connection failed", component="database", attempt=3)
    
    # Check that logs were created (need to specify INFO as min_severity)
    recent_errors = monitor.get_recent_errors(count=10, min_severity="INFO")
    print_test_result("Direct logging through monitor", 
                     len(recent_errors) == 3 and
                     any(e["severity"] == "ERROR" for e in recent_errors))
    
    # Test automatic error detection from engine
    engine.add_error(
        component="network",
        message="Connection timeout",
        severity="WARNING",
        host="example.com"
    )
    
    engine.add_error(
        component="file_system",
        message="Disk space low",
        severity="CRITICAL",
        disk="C:",
        space_left_mb=100
    )
    
    # Emit a tick to process the errors
    monitor.emit_tick()
    
    # Check that engine errors were detected
    errors_after_tick = monitor.get_recent_errors(count=10)
    print_test_result("Automatic error detection", 
                     len(errors_after_tick) > len(recent_errors) and
                     any(e["component"] == "network" for e in errors_after_tick) and
                     any(e["component"] == "file_system" for e in errors_after_tick))
    
    # Test error stats
    stats = monitor.get_error_stats()
    print_test_result("Error statistics", 
                     "total_entries" in stats and
                     "by_severity" in stats and
                     "by_component" in stats)
    
    if "by_severity" in stats:
        print(f"  Error counts by severity: {stats['by_severity']}")
    
    if "by_component" in stats:
        print(f"  Error counts by component: {stats['by_component']}")
    
    # Test metrics for dashboard
    metrics = monitor.get_metrics_for_dashboard(hours=1)
    print_test_result("Metrics for dashboard", 
                     "tick_count" in metrics and
                     "scope_entropy" in metrics and
                     "agent_utilization" in metrics and
                     "errors" in metrics)


def test_dashboard_formatting():
    """Test dashboard formatting functionality."""
    print_header("Testing Dashboard Formatting")
    
    # Create an error log
    error_log = ErrorLog(max_entries=100)
    
    # Add test entries
    severities = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
    components = ["ui", "api", "database", "auth"]
    
    for i in range(20):
        severity = severities[i % len(severities)]
        component = components[i % len(components)]
        
        error_log.add_entry(
            message=f"Test message {i}",
            severity=severity,
            component=component,
            index=i,
            test_id=f"test-{i}"
        )
    
    # Get dashboard formatted entries
    dashboard_entries = error_log.get_entries_for_dashboard(
        hours=24,
        min_severity="WARNING"
    )
    
    # Check that dashboard entries are properly formatted
    print_test_result("Dashboard formatting", 
                     len(dashboard_entries) > 0 and
                     all("time" in e for e in dashboard_entries) and
                     all("date" in e for e in dashboard_entries) and
                     all("severity_class" in e for e in dashboard_entries) and
                     all("context_str" in e for e in dashboard_entries))
    
    # Check that only WARNING+ entries are included
    print_test_result("Severity filtering for dashboard", 
                     all(e["severity"] in ["WARNING", "ERROR", "CRITICAL"] 
                         for e in dashboard_entries))
    
    if dashboard_entries:
        entry = dashboard_entries[0]
        print(f"  Example dashboard entry:")
        print(f"    Time: {entry['time']}")
        print(f"    Date: {entry['date']}")
        print(f"    Severity: {entry['severity']} (class: {entry['severity_class']})")
        print(f"    Message: {entry['message']}")
        print(f"    Component: {entry['component']}")
        print(f"    Context: {entry['context_str']}")


def test_persistence():
    """Test error log persistence."""
    print_header("Testing Persistence")
    
    # Create an error log with persistence
    log_file = TEST_DIR / "export_test.json"
    error_log = ErrorLog(max_entries=100)
    
    # Add test entries
    for i in range(10):
        error_log.info(f"Test message {i}", component="test", index=i)
    
    # Export to file
    export_success = error_log.export_to_file(str(log_file))
    print_test_result("Export to file", export_success and log_file.exists())
    
    # Check file content
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            print_test_result("Exported data valid", 
                             isinstance(data, list) and
                             len(data) == 10 and
                             all("message" in e for e in data))
            
            print(f"  Exported {len(data)} entries to {log_file}")
        except Exception as e:
            print_test_result("Exported data valid", False)
            print(f"  Error reading exported file: {e}")


def clean_up():
    """Clean up test files and directories."""
    print_header("Cleaning Up")
    
    # Remove test directory and all its contents
    try:
        shutil.rmtree(TEST_DIR)
        print(f"Removed test directory: {TEST_DIR}")
    except Exception as e:
        print(f"Error removing test directory: {e}")


def main():
    """Run all tests."""
    print_header("ERROR LOGGING TEST SUITE")
    
    try:
        # Run tests
        test_error_log_basic()
        test_error_log_query()
        test_system_monitor_integration()
        test_dashboard_formatting()
        test_persistence()
    finally:
        # Clean up
        clean_up()
    
    print("\n")
    print_header("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
