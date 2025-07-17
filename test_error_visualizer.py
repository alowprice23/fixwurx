#!/usr/bin/env python3
"""
test_error_visualizer.py
────────────────────────
Test suite for the error visualizer and enhanced error log visualization.

This test suite validates:
1. Error trend analysis
2. Severity distribution visualization
3. Component-based error grouping
4. Pattern detection in error logs
5. Export capabilities (CSV, HTML)
6. Integration with the system monitor

Usage: python test_error_visualizer.py
"""

import os
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from monitoring.error_log import ErrorLog, ErrorSeverity
from monitoring.error_visualizer import ErrorVisualizer
from system_monitor import SystemMonitor


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


# Mock classes for testing
class MockMetricBus:
    """Mock metric bus that records sent metrics."""
    def __init__(self):
        self.metrics = []
    
    def send(self, metric_name, value, tags=None):
        """Record a sent metric."""
        self.metrics.append({
            "name": metric_name,
            "value": value,
            "tags": tags or {}
        })


class MockEngine:
    """Mock engine that implements the Engine protocol."""
    def __init__(self):
        self._tick_count = 0
        self._scope_entropy = 0.5
        self._agent_utilization = {"agent1": 0.7, "agent2": 0.3}
        self._error_state = {}
    
    @property
    def tick_count(self):
        return self._tick_count
    
    @property
    def scope_entropy(self):
        return self._scope_entropy
    
    @property
    def agent_utilization(self):
        return self._agent_utilization
    
    @property
    def error_state(self):
        return self._error_state


def create_test_error_log(days_back=7):
    """Create a test error log with varied entries over time."""
    error_log = ErrorLog(max_entries=1000)
    
    # List of components and severities for test data
    components = ["database", "network", "api", "ui", "auth", "storage", "core"]
    severities = [
        ErrorSeverity.DEBUG, 
        ErrorSeverity.INFO, 
        ErrorSeverity.WARNING,
        ErrorSeverity.ERROR, 
        ErrorSeverity.CRITICAL
    ]
    
    # Generate errors with timestamps spread over the specified days
    now = time.time()
    total_errors = 100
    
    for i in range(total_errors):
        # Generate timestamp (newer errors more frequent)
        if i < total_errors * 0.7:  # 70% of errors in last day
            days_ago = (i / (total_errors * 0.7)) * 1
        else:  # 30% of errors spread over remaining days
            days_ago = 1 + ((i - total_errors * 0.7) / (total_errors * 0.3)) * (days_back - 1)
        
        timestamp = now - (days_ago * 86400)
        
        # Select component and severity with intentional patterns
        component_idx = i % len(components)
        severity_idx = int((i / 10) % len(severities))
        
        component = components[component_idx]
        severity = severities[severity_idx]
        
        # Create message with some patterns
        if i % 20 < 5:
            message = f"Connection failed to {component} service"
        elif i % 20 < 10:
            message = f"Validation error in {component} module"
        elif i % 20 < 15:
            message = f"Timeout occurred while processing {component} request"
        else:
            message = f"Unexpected error in {component}: error code {i % 100}"
        
        # Add context data
        context = {
            "error_id": f"ERR-{i:04d}",
            "user_id": f"user-{i % 10}",
            "duration_ms": i * 10,
            "timestamp": timestamp
        }
        
        # Add entry with timestamp override
        entry = error_log.add_entry(
            message=message,
            severity=severity,
            component=component,
            **context
        )
        
        # Override the timestamp for testing
        entry["timestamp"] = timestamp
    
    return error_log


def test_error_trends():
    """Test error trend analysis."""
    print_header("Testing Error Trend Analysis")
    
    # Create test error log
    error_log = create_test_error_log(days_back=7)
    visualizer = ErrorVisualizer(error_log)
    
    # Test daily trends
    daily_trends = visualizer.get_error_trends(days=7, group_by="day")
    
    print_test_result("Get daily trends", 
                     len(daily_trends) > 0 and
                     all(isinstance(trends, list) for trends in daily_trends.values()))
    
    # Test hourly trends
    hourly_trends = visualizer.get_error_trends(days=1, group_by="hour")
    
    print_test_result("Get hourly trends", 
                     len(hourly_trends) > 0 and
                     all(isinstance(trends, list) for trends in hourly_trends.values()))
    
    # Test weekly trends
    weekly_trends = visualizer.get_error_trends(days=30, group_by="week")
    
    print_test_result("Get weekly trends", 
                     len(weekly_trends) > 0 and
                     all(isinstance(trends, list) for trends in weekly_trends.values()))
    
    # Check if trends contain valid data
    has_valid_data = False
    for component, trends in daily_trends.items():
        if trends and isinstance(trends[0], tuple) and len(trends[0]) == 2:
            date_str, count = trends[0]
            if isinstance(date_str, str) and isinstance(count, int):
                has_valid_data = True
                break
    
    print_test_result("Trends contain valid data", has_valid_data)
    
    # Output sample data for visual inspection
    if daily_trends:
        component = next(iter(daily_trends))
        trends = daily_trends[component]
        print(f"\nSample trend data for component '{component}':")
        for date_str, count in trends:
            print(f"  {date_str}: {count} errors")


def test_distribution_analysis():
    """Test error distribution analysis."""
    print_header("Testing Error Distribution Analysis")
    
    # Create test error log
    error_log = create_test_error_log(days_back=7)
    visualizer = ErrorVisualizer(error_log)
    
    # Test severity distribution
    severity_dist = visualizer.get_severity_distribution()
    
    print_test_result("Get severity distribution", 
                     len(severity_dist) > 0 and
                     all(isinstance(count, int) for count in severity_dist.values()))
    
    # Test component distribution
    component_dist = visualizer.get_component_distribution()
    
    print_test_result("Get component distribution", 
                     len(component_dist) > 0 and
                     all(isinstance(count, int) for count in component_dist.values()))
    
    # Test filtered distributions
    warning_component_dist = visualizer.get_component_distribution(min_severity="WARNING")
    
    print_test_result("Get filtered component distribution", 
                     len(warning_component_dist) > 0 and
                     sum(warning_component_dist.values()) <= sum(component_dist.values()))
    
    # Test time-filtered distributions
    recent_severity_dist = visualizer.get_severity_distribution(days=1)
    
    print_test_result("Get recent severity distribution", 
                     len(recent_severity_dist) > 0)
    
    # Output sample data for visual inspection
    print("\nSeverity distribution:")
    for severity, count in severity_dist.items():
        print(f"  {severity}: {count} errors")
    
    print("\nComponent distribution:")
    for component, count in sorted(component_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {component}: {count} errors")


def test_error_patterns():
    """Test error pattern detection."""
    print_header("Testing Error Pattern Detection")
    
    # Create test error log
    error_log = create_test_error_log(days_back=7)
    visualizer = ErrorVisualizer(error_log)
    
    # Test pattern detection
    patterns = visualizer.get_error_patterns(min_occurrences=2)
    
    print_test_result("Detect error patterns", 
                     len(patterns) > 0 and
                     all(isinstance(pattern, dict) for pattern in patterns))
    
    # Check if patterns contain required fields
    valid_patterns = all(
        "pattern" in p and "count" in p and "examples" in p
        for p in patterns
    )
    
    print_test_result("Patterns contain required fields", valid_patterns)
    
    # Test filtered pattern detection
    error_patterns = visualizer.get_error_patterns(
        min_severity="ERROR",
        min_occurrences=2
    )
    
    print_test_result("Detect filtered error patterns", 
                     len(error_patterns) > 0)
    
    # Output sample data for visual inspection
    if patterns:
        print("\nTop error patterns:")
        for i, pattern in enumerate(patterns[:3], 1):
            print(f"  {i}. Pattern: \"{pattern['pattern']}\"")
            print(f"     Count: {pattern['count']}")
            print(f"     Components: {', '.join(pattern['components'])}")
            print(f"     Severity levels: {', '.join(pattern['severity_levels'])}")


def test_error_summary():
    """Test error summary generation."""
    print_header("Testing Error Summary")
    
    # Create test error log
    error_log = create_test_error_log(days_back=7)
    visualizer = ErrorVisualizer(error_log)
    
    # Test summary generation
    summary = visualizer.get_error_summary(days=7)
    
    print_test_result("Generate error summary", 
                     isinstance(summary, dict) and
                     "total_errors" in summary and
                     "severity_distribution" in summary and
                     "component_distribution" in summary)
    
    # Test filtered summary
    warning_summary = visualizer.get_error_summary(
        days=7,
        min_severity="WARNING"
    )
    
    print_test_result("Generate filtered summary", 
                     isinstance(warning_summary, dict) and
                     warning_summary["total_errors"] <= summary["total_errors"])
    
    # Output sample data for visual inspection
    print("\nError summary:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Error rate: {summary['error_rate_per_day']:.2f} per day")
    if summary.get('peak_day'):
        print(f"  Peak day: {summary['peak_day']} ({summary['peak_day_count']} errors)")


def test_export_capabilities():
    """Test export capabilities."""
    print_header("Testing Export Capabilities")
    
    # Create test error log and temporary files
    error_log = create_test_error_log(days_back=7)
    visualizer = ErrorVisualizer(error_log)
    
    csv_file = TEST_DIR / "errors.csv"
    html_file = TEST_DIR / "errors.html"
    
    # Test CSV export
    csv_result = visualizer.export_to_csv(str(csv_file))
    
    print_test_result("Export to CSV", 
                     csv_result and
                     csv_file.exists() and
                     csv_file.stat().st_size > 0)
    
    # Test HTML export
    html_result = visualizer.export_to_html(str(html_file))
    
    print_test_result("Export to HTML", 
                     html_result and
                     html_file.exists() and
                     html_file.stat().st_size > 0)
    
    # Test HTML export with summary
    html_summary_file = TEST_DIR / "errors_with_summary.html"
    html_summary_result = visualizer.export_to_html(
        str(html_summary_file),
        include_summary=True
    )
    
    print_test_result("Export to HTML with summary", 
                     html_summary_result and
                     html_summary_file.exists() and
                     html_summary_file.stat().st_size > html_file.stat().st_size)
    
    # Check file content
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            header = f.readline().strip()
        
        print_test_result("CSV has correct header", 
                         "timestamp" in header and
                         "severity" in header and
                         "component" in header)
    
    if html_file.exists():
        with open(html_file, 'r') as f:
            content = f.read()
        
        print_test_result("HTML has correct structure", 
                         "<html>" in content and
                         "<table>" in content and
                         "Error Log Report" in content)
    
    # Output file paths
    print(f"\nExported files:")
    print(f"  CSV: {csv_file}")
    print(f"  HTML: {html_file}")
    print(f"  HTML with summary: {html_summary_file}")


def test_system_monitor_integration():
    """Test integration with system monitor."""
    print_header("Testing System Monitor Integration")
    
    # Create mock objects and system monitor
    engine = MockEngine()
    metric_bus = MockMetricBus()
    
    monitor = SystemMonitor(
        engine=engine,
        metric_bus=metric_bus,
        log_dir=str(TEST_LOG_DIR)
    )
    
    # Log some test errors
    monitor.log_info("System starting", component="system")
    monitor.log_warning("Resource usage high", component="memory", usage_percent=85)
    monitor.log_error("Database connection failed", component="database", attempt=3)
    monitor.log_critical("System shutdown imminent", component="core", reason="resource exhaustion")
    
    # Create visualizer with monitor's error log
    visualizer = ErrorVisualizer(monitor.error_log)
    
    # Test summary generation
    summary = visualizer.get_error_summary(min_severity="INFO")
    
    print_test_result("Generate summary from monitor", 
                     isinstance(summary, dict) and
                     summary["total_errors"] >= 4)
    
    # Test component distribution
    component_dist = visualizer.get_component_distribution()
    
    print_test_result("Get component distribution from monitor", 
                     "database" in component_dist and
                     "memory" in component_dist and
                     "system" in component_dist and
                     "core" in component_dist)
    
    # Test HTML export
    html_file = TEST_DIR / "monitor_errors.html"
    html_result = visualizer.export_to_html(str(html_file), include_summary=True)
    
    print_test_result("Export monitor errors to HTML", 
                     html_result and
                     html_file.exists() and
                     html_file.stat().st_size > 0)
    
    # Output summary data
    print("\nSystem monitor error summary:")
    print(f"  Total errors: {summary['total_errors']}")
    print(f"  Components: {', '.join(component_dist.keys())}")


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
    print_header("ERROR VISUALIZER TEST SUITE")
    
    try:
        # Run tests
        test_error_trends()
        test_distribution_analysis()
        test_error_patterns()
        test_error_summary()
        test_export_capabilities()
        test_system_monitor_integration()
    finally:
        # Clean up
        clean_up()
    
    print("\n")
    print_header("TEST SUITE COMPLETE")


if __name__ == "__main__":
    main()
