#!/usr/bin/env python3
"""
Auditor Commands Module

This module registers command handlers for the Auditor Agent within the shell environment,
enabling comprehensive monitoring, auditing, and quality enforcement capabilities.
"""

import os
import sys
import json
import yaml
import time
import logging
import datetime
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

logger = logging.getLogger("AuditorCommands")

def register_auditor_commands(registry):
    """
    Register Auditor command handlers with the component registry.
    
    Args:
        registry: Component registry instance
    """
    # Try to import Auditor Agent
    try:
        from auditor_agent import AuditorAgent
        
        # Create and register auditor agent instance
        config_path = "auditor_config.yaml"
        auditor_instance = AuditorAgent(config_path)
        registry.register_component("auditor", auditor_instance)
        
        # Register command handlers
        registry.register_command_handler("audit", audit_command, "auditor")
        registry.register_command_handler("monitor", monitor_command, "auditor")
        registry.register_command_handler("trace", trace_command, "auditor")
        registry.register_command_handler("sensors", sensors_command, "auditor")
        registry.register_command_handler("metrics", metrics_command, "auditor")
        registry.register_command_handler("report", report_command, "auditor")
        registry.register_command_handler("errors", errors_command, "auditor")
        registry.register_command_handler("patch", patch_command, "auditor")
        
        registry.register_command_handler("auditor:audit", audit_command, "auditor")
        registry.register_command_handler("auditor:monitor", monitor_command, "auditor")
        registry.register_command_handler("auditor:trace", trace_command, "auditor")
        registry.register_command_handler("auditor:sensors", sensors_command, "auditor")
        registry.register_command_handler("auditor:metrics", metrics_command, "auditor")
        registry.register_command_handler("auditor:report", report_command, "auditor")
        registry.register_command_handler("auditor:errors", errors_command, "auditor")
        registry.register_command_handler("auditor:patch", patch_command, "auditor")
        registry.register_command_handler("auditor:log", log_command, "auditor")
        registry.register_command_handler("auditor:alerts", alerts_command, "auditor")
        registry.register_command_handler("auditor:status", status_command, "auditor")
        
        # Register event handlers
        registry.register_event_handler("system_audit", system_audit_handler, "auditor")
        registry.register_event_handler("error_detected", error_detected_handler, "auditor")
        registry.register_event_handler("benchmark_completed", benchmark_completed_handler, "auditor")
        
        logger.info("Auditor commands registered")
    except ImportError:
        logger.warning("Failed to import Auditor Agent module")
    except Exception as e:
        logger.error(f"Error registering Auditor commands: {e}")

def audit_command(args: str) -> int:
    """
    Run an audit using the Auditor Agent.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run an audit")
    parser.add_argument("--full", action="store_true", help="Force a full audit")
    parser.add_argument("--component", help="Focus on specific component")
    parser.add_argument("--save", help="Save audit result to file")
    parser.add_argument("--report", choices=["text", "json", "yaml"], default="text", 
                       help="Report format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    full_audit = cmd_args.full
    component = cmd_args.component
    save_file = cmd_args.save
    report_format = cmd_args.report
    
    print(f"Running {'full' if full_audit else 'incremental'} audit...")
    if component:
        print(f"Focusing on component: {component}")
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # Perform the audit
        audit_options = {}
        if component:
            audit_options["component"] = component
        
        result = auditor.run_audit(force_full_audit=full_audit, **audit_options)
        
        # Display the results
        if report_format == "json":
            print(json.dumps(result, indent=2, default=str))
        elif report_format == "yaml":
            print(yaml.dump(result, default_flow_style=False))
        else:  # text format
            print("\nAudit Results:")
            print(f"  Status: {result.get('status', 'Unknown')}")
            print(f"  Issues Found: {result.get('issues_found', 0)}")
            print(f"  Passed Checks: {result.get('checks_passed', 0)}/{result.get('total_checks', 0)}")
            
            # Display issues if any
            issues = result.get("issues", [])
            if issues:
                print("\nIssues:")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue.get('severity', 'UNKNOWN')}: {issue.get('message', 'No message')}")
                    print(f"     Component: {issue.get('component', 'Unknown')}")
                    if 'details' in issue:
                        print(f"     Details: {issue['details']}")
            
            # Display recommendations if any
            recommendations = result.get("recommendations", [])
            if recommendations:
                print("\nRecommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec.get('title', 'Unknown')}")
                    print(f"     {rec.get('description', 'No description')}")
        
        # Save to file if requested
        if save_file:
            try:
                save_dir = Path(save_file).parent
                if save_dir != Path('.'):
                    save_dir.mkdir(parents=True, exist_ok=True)
                
                with open(save_file, 'w') as f:
                    if save_file.endswith('.json'):
                        json.dump(result, f, indent=2, default=str)
                    elif save_file.endswith('.yaml') or save_file.endswith('.yml'):
                        yaml.dump(result, f, default_flow_style=False)
                    else:
                        f.write(f"Audit Results - {datetime.datetime.now()}\n\n")
                        f.write(f"Status: {result.get('status', 'Unknown')}\n")
                        f.write(f"Issues Found: {result.get('issues_found', 0)}\n")
                        f.write(f"Passed Checks: {result.get('checks_passed', 0)}/{result.get('total_checks', 0)}\n")
                        
                        # Write issues
                        if issues:
                            f.write("\nIssues:\n")
                            for i, issue in enumerate(issues, 1):
                                f.write(f"{i}. {issue.get('severity', 'UNKNOWN')}: {issue.get('message', 'No message')}\n")
                                f.write(f"   Component: {issue.get('component', 'Unknown')}\n")
                                if 'details' in issue:
                                    f.write(f"   Details: {issue['details']}\n")
                        
                        # Write recommendations
                        if recommendations:
                            f.write("\nRecommendations:\n")
                            for i, rec in enumerate(recommendations, 1):
                                f.write(f"{i}. {rec.get('title', 'Unknown')}\n")
                                f.write(f"   {rec.get('description', 'No description')}\n")
                
                print(f"Audit result saved to {save_file}")
            except Exception as e:
                print(f"Error saving audit result: {e}")
        
        return 0
    except Exception as e:
        print(f"Error running audit: {e}")
        return 1

def monitor_command(args: str) -> int:
    """
    Control the Auditor Agent monitoring.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    cmd_args = shlex.split(args)
    
    # No arguments - show status
    if not cmd_args:
        is_monitoring = auditor.is_running
        print(f"Auditor monitoring is currently {'active' if is_monitoring else 'inactive'}")
        print("Use 'monitor start' to start monitoring or 'monitor stop' to stop")
        return 0
    
    command = cmd_args[0].lower()
    
    # Start monitoring
    if command == "start":
        if auditor.is_running:
            print("Monitoring is already running")
        else:
            auditor.start()
            print("Monitoring started")
        return 0
    
    # Stop monitoring
    elif command == "stop":
        if not auditor.is_running:
            print("Monitoring is already stopped")
        else:
            auditor.stop()
            print("Monitoring stopped")
        return 0
    
    # Watch monitoring (continuous status)
    elif command == "watch":
        try:
            print("Live monitoring (press Ctrl+C to stop)...")
            while True:
                status = auditor.get_status()
                
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"Auditor Status - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                
                is_monitoring = status.get("is_monitoring", False)
                print(f"Monitoring: {'Active' if is_monitoring else 'Inactive'}")
                print(f"Last Audit: {status.get('last_audit_time', 'Never')}")
                print(f"Queue Size: {status.get('queue_size', 0)}")
                
                # Display active sensors
                sensors = status.get("active_sensors", [])
                print(f"\nActive Sensors: {len(sensors)}")
                for i, sensor in enumerate(sensors[:5], 1):
                    sensor_id = sensor.get("sensor_id", "Unknown")
                    component = sensor.get("component_name", "Unknown")
                    last_check = sensor.get("last_check_time", "Unknown")
                    print(f"  {i}. {sensor_id} - {component} - Last check: {last_check}")
                
                # Display recent alerts
                alerts = status.get("recent_alerts", [])
                print(f"\nRecent Alerts: {len(alerts)}")
                for i, alert in enumerate(alerts[:5], 1):
                    alert_type = alert.get("type", "Unknown")
                    severity = alert.get("severity", "Unknown")
                    timestamp = alert.get("timestamp", "Unknown")
                    message = alert.get("message", "No message")
                    print(f"  {i}. [{severity}] {alert_type} - {timestamp}")
                    print(f"     {message}")
                
                # Display performance metrics
                metrics = status.get("performance_metrics", {})
                print("\nPerformance:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value}")
                
                # Wait before updating
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nMonitoring watch stopped")
        return 0
    
    # Unknown command
    else:
        print(f"Unknown monitor command: {command}")
        print("Available commands: start, stop, watch")
        return 1

def trace_command(args: str) -> int:
    """
    Trace execution or activity through the system.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Trace execution or activity")
    parser.add_argument("execution_id", nargs="?", help="Execution ID to trace")
    parser.add_argument("--component", help="Component to trace")
    parser.add_argument("--depth", type=int, default=3, help="Trace depth")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    execution_id = cmd_args.execution_id
    component = cmd_args.component
    depth = cmd_args.depth
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # Get trace data
        if execution_id:
            print(f"Tracing execution: {execution_id}")
            trace_data = auditor.trace_execution(execution_id, depth=depth)
        elif component:
            print(f"Tracing component: {component}")
            trace_data = auditor.trace_component(component, depth=depth)
        else:
            print("Error: Either execution_id or component must be specified")
            return 1
        
        # Display the results
        if output_format == "json":
            print(json.dumps(trace_data, indent=2, default=str))
        elif output_format == "yaml":
            print(yaml.dump(trace_data, default_flow_style=False))
        else:  # text format
            print("\nTrace Results:")
            
            # Display execution info
            if "execution" in trace_data:
                execution = trace_data["execution"]
                print(f"Execution ID: {execution.get('id', 'Unknown')}")
                print(f"Type: {execution.get('type', 'Unknown')}")
                print(f"Status: {execution.get('status', 'Unknown')}")
                print(f"Start Time: {execution.get('start_time', 'Unknown')}")
                print(f"End Time: {execution.get('end_time', 'Unknown')}")
                print(f"Duration: {execution.get('duration', 'Unknown')}")
            
            # Display component info
            if "component" in trace_data:
                component = trace_data["component"]
                print(f"\nComponent: {component.get('name', 'Unknown')}")
                print(f"Type: {component.get('type', 'Unknown')}")
                print(f"Health: {component.get('health', 'Unknown')}")
            
            # Display trace steps
            steps = trace_data.get("steps", [])
            if steps:
                print("\nTrace Steps:")
                for i, step in enumerate(steps, 1):
                    print(f"  {i}. {step.get('type', 'Unknown')} - {step.get('timestamp', 'Unknown')}")
                    print(f"     Status: {step.get('status', 'Unknown')}")
                    print(f"     Component: {step.get('component', 'Unknown')}")
                    if "message" in step:
                        print(f"     Message: {step['message']}")
                    if "data" in step:
                        print(f"     Data: {step['data']}")
            
            # Display dependencies
            dependencies = trace_data.get("dependencies", [])
            if dependencies:
                print("\nDependencies:")
                for i, dep in enumerate(dependencies, 1):
                    print(f"  {i}. {dep.get('name', 'Unknown')} - {dep.get('type', 'Unknown')}")
                    print(f"     Status: {dep.get('status', 'Unknown')}")
        
        return 0
    except Exception as e:
        print(f"Error tracing execution: {e}")
        return 1

def sensors_command(args: str) -> int:
    """
    Manage and view sensor information.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    cmd_args = shlex.split(args)
    
    # No arguments - list all sensors
    if not cmd_args:
        # Get sensor information from registry
        sensor_registry = getattr(auditor, "sensor_registry", None)
        if not sensor_registry:
            print("Error: Sensor registry not available")
            return 1
        
        sensors = list(sensor_registry.sensors.values())
        
        print(f"\nRegistered Sensors ({len(sensors)}):")
        
        for i, sensor in enumerate(sensors, 1):
            status = sensor.get_status() if hasattr(sensor, 'get_status') else {"sensor_id": sensor.sensor_id}
            enabled = "Enabled" if hasattr(sensor, 'enabled') and sensor.enabled else "Disabled" if hasattr(sensor, 'enabled') else "Unknown"
            print(f"  {i}. {status.get('sensor_id', 'Unknown')} - "
                  f"{status.get('component_name', 'Unknown')} - {enabled}")
        
        print("\nUse 'sensors info <sensor_id>' for more information")
        return 0
    
    command = cmd_args[0].lower()
    
    # Show sensor info
    if command == "info" and len(cmd_args) > 1:
        sensor_id = cmd_args[1]
        
        # Get sensor registry
        sensor_registry = getattr(auditor, "sensor_registry", None)
        if not sensor_registry:
            print("Error: Sensor registry not available")
            return 1
        
        # Get the sensor
        sensor = sensor_registry.get_sensor(sensor_id)
        if not sensor:
            print(f"Error: Sensor '{sensor_id}' not found")
            return 1
        
        # Get sensor status
        status = sensor.get_status() if hasattr(sensor, 'get_status') else {"sensor_id": sensor.sensor_id}
        
        print(f"\nSensor Information - {sensor_id}")
        print("=" * 50)
        
        # Format and print status dictionary
        for key, value in status.items():
            if isinstance(value, (dict, list)):
                print(f"  {key}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    for item in value:
                        print(f"    • {item}")
            else:
                print(f"  {key}: {value}")
        
        return 0
    
    # Enable a sensor
    elif command == "enable" and len(cmd_args) > 1:
        sensor_id = cmd_args[1]
        
        # Get sensor registry
        sensor_registry = getattr(auditor, "sensor_registry", None)
        if not sensor_registry:
            print("Error: Sensor registry not available")
            return 1
        
        # Get the sensor
        sensor = sensor_registry.get_sensor(sensor_id)
        if not sensor:
            print(f"Error: Sensor '{sensor_id}' not found")
            return 1
        
        # Enable the sensor
        if hasattr(sensor, 'enable'):
            sensor.enable()
            print(f"Sensor '{sensor_id}' enabled")
        else:
            print(f"Error: Sensor '{sensor_id}' doesn't support enabling")
        
        return 0
    
    # Disable a sensor
    elif command == "disable" and len(cmd_args) > 1:
        sensor_id = cmd_args[1]
        
        # Get sensor registry
        sensor_registry = getattr(auditor, "sensor_registry", None)
        if not sensor_registry:
            print("Error: Sensor registry not available")
            return 1
        
        # Get the sensor
        sensor = sensor_registry.get_sensor(sensor_id)
        if not sensor:
            print(f"Error: Sensor '{sensor_id}' not found")
            return 1
        
        # Disable the sensor
        if hasattr(sensor, 'disable'):
            sensor.disable()
            print(f"Sensor '{sensor_id}' disabled")
        else:
            print(f"Error: Sensor '{sensor_id}' doesn't support disabling")
        
        return 0
    
    # Unknown command
    else:
        print(f"Unknown sensors command: {command}")
        print("Available commands: info <sensor_id>, enable <sensor_id>, disable <sensor_id>")
        return 1

def metrics_command(args: str) -> int:
    """
    View and analyze metrics.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="View and analyze metrics")
    parser.add_argument("action", nargs="?", choices=["list", "trend", "top"], default="list", 
                        help="Action to perform")
    parser.add_argument("metric", nargs="?", help="Metric name for trend analysis")
    parser.add_argument("--component", help="Filter by component")
    parser.add_argument("--period", choices=["hour", "day", "week", "month"], default="day",
                       help="Time period")
    parser.add_argument("--count", type=int, default=5, help="Number of metrics to show")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    metric_name = cmd_args.metric
    component = cmd_args.component
    period = cmd_args.period
    count = cmd_args.count
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # List metrics
        if action == "list":
            # Get metrics
            metrics = auditor.get_metrics(component=component, period=period)
            
            # Display metrics
            if output_format == "json":
                print(json.dumps(metrics, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(metrics, default_flow_style=False))
            else:  # text format
                print(f"\nMetrics ({period}):")
                
                if component:
                    print(f"Component: {component}")
                
                # Group metrics by category
                categories = {
                    "Detection and Fix": ["bug_detection_recall", "bug_fix_yield", "mttd", "mttr"],
                    "Energy and Convergence": ["energy_reduction_pct", "proof_coverage_delta", "residual_risk_improvement"],
                    "Test and Quality": ["test_pass_ratio", "regression_introduction_rate", "duplicate_module_ratio"],
                    "Agent and Resource": ["agent_coordination_overhead", "token_per_fix_efficiency", "hallucination_rate"],
                    "Time and Stability": ["lyapunov_descent_consistency", "meta_guard_breach_count"],
                    "Overall": ["aggregate_confidence_score"]
                }
                
                # Check if metrics is a dictionary with metric values or a list of snapshots
                if isinstance(metrics, dict):
                    metric_values = metrics
                elif isinstance(metrics, list) and metrics and isinstance(metrics[-1], dict):
                    metric_values = metrics[-1].get("metrics", {})
                else:
                    metric_values = {}
                
                # Print metrics by category
                for category, metric_names in categories.items():
                    category_metrics = {}
                    for metric in metric_names:
                        if metric in metric_values:
                            category_metrics[metric] = metric_values[metric]
                    
                    if category_metrics:
                        print(f"\n{category}:")
                        for metric, value in category_metrics.items():
                            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                            print(f"  {metric}: {value_str}")
        
        # Show trend for a specific metric
        elif action == "trend" and metric_name:
            # Get metric history
            metric_history = auditor.get_metric_history(metric_name, component=component, period=period)
            
            # Display trend
            if output_format == "json":
                print(json.dumps(metric_history, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(metric_history, default_flow_style=False))
            else:  # text format
                print(f"\nTrend for '{metric_name}' ({period}):")
                
                if component:
                    print(f"Component: {component}")
                
                # Extract values for trend
                values = []
                if isinstance(metric_history, list):
                    for snapshot in metric_history:
                        if isinstance(snapshot, dict):
                            timestamp = snapshot.get("timestamp", "Unknown")
                            if isinstance(timestamp, (int, float)):
                                timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                            
                            value = snapshot.get("value")
                            if value is not None:
                                values.append((timestamp, value))
                
                if not values:
                    print("No data available for trend analysis")
                    return 0
                
                # Calculate stats
                metric_values = [v[1] for v in values]
                min_value = min(metric_values)
                max_value = max(metric_values)
                avg_value = sum(metric_values) / len(metric_values)
                
                print(f"Min: {min_value:.2f}, Max: {max_value:.2f}, Avg: {avg_value:.2f}")
                print(f"Change: {values[-1][1] - values[0][1]:.2f}")
                
                # Show ASCII trend
                print("\nTrend:")
                width = 50  # Width of the ASCII chart
                
                # Normalize values to chart width
                if min_value == max_value:
                    normalized = [width // 2 for _ in metric_values]
                else:
                    normalized = [int((v - min_value) / (max_value - min_value) * width) for v in metric_values]
                
                # Draw the chart
                for i, (timestamp, value) in enumerate(values):
                    bar = "#" * normalized[i]
                    print(f"{timestamp}: {bar} {value:.2f}")
        
        # Show top metrics by improvement
        elif action == "top":
            # Get metrics
            metrics = auditor.get_metrics(component=component, period=period)
            
            # Check if we have historical data
            if not isinstance(metrics, list) or len(metrics) < 2:
                print("Insufficient metrics data for top analysis")
                return 0
            
            # Get first and last metrics
            first_metrics = metrics[0].get("metrics", {})
            last_metrics = metrics[-1].get("metrics", {})
            
            # Calculate improvements
            improvements = []
            for key in set(first_metrics.keys()).intersection(last_metrics.keys()):
                if isinstance(first_metrics[key], (int, float)) and isinstance(last_metrics[key], (int, float)):
                    improvement = last_metrics[key] - first_metrics[key]
                    improvements.append((key, improvement, first_metrics[key], last_metrics[key]))
            
            # Sort by absolute improvement
            improvements.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Display top metrics
            if output_format == "json":
                top_metrics = {item[0]: {"improvement": item[1], "initial": item[2], "final": item[3]} 
                              for item in improvements[:count]}
                print(json.dumps(top_metrics, indent=2, default=str))
            elif output_format == "yaml":
                top_metrics = {item[0]: {"improvement": item[1], "initial": item[2], "final": item[3]} 
                              for item in improvements[:count]}
                print(yaml.dump(top_metrics, default_flow_style=False))
            else:  # text format
                print(f"\nTop {count} Metrics by Improvement:")
                for i, (key, improvement, first, last) in enumerate(improvements[:count], 1):
                    direction = "+" if improvement >= 0 else ""
                    print(f"  {i}. {key}: {first:.2f} → {last:.2f} ({direction}{improvement:.2f})")
        
        return 0
    except Exception as e:
        print(f"Error analyzing metrics: {e}")
        return 1

def report_command(args: str) -> int:
    """
    Generate system reports.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate system reports")
    parser.add_argument("report_type", choices=["audit", "health", "performance", "security"], 
                        help="Type of report to generate")

def errors_command(args: str) -> int:
    """
    Manage error reports and issues.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage error reports and issues")
    parser.add_argument("action", nargs="?", choices=["list", "details", "trends", "fix"], default="list", 
                        help="Action to perform")
    parser.add_argument("error_id", nargs="?", help="Error ID for details/fix actions")
    parser.add_argument("--status", choices=["open", "closed", "in_progress", "all"], default="open", 
                       help="Filter by error status")
    parser.add_argument("--severity", choices=["critical", "high", "medium", "low", "all"], default="all", 
                       help="Filter by error severity")
    parser.add_argument("--component", help="Filter by component")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    error_id = cmd_args.error_id
    status = cmd_args.status
    severity = cmd_args.severity
    component = cmd_args.component
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # List errors
        if action == "list":
            # Get errors
            errors_filter = {}
            if status != "all":
                errors_filter["status"] = status
            if severity != "all":
                errors_filter["severity"] = severity
            if component:
                errors_filter["component"] = component
            
            errors = auditor.get_errors(filter=errors_filter)
            
            # Display errors
            if output_format == "json":
                print(json.dumps(errors, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(errors, default_flow_style=False))
            else:  # text format
                print("\nError Reports:")
                
                if not errors:
                    print("No errors found matching the criteria")
                    return 0
                
                print(f"Found {len(errors)} errors:")
                for i, error in enumerate(errors, 1):
                    error_id = error.get("id", "Unknown")
                    severity = error.get("severity", "Unknown")
                    status = error.get("status", "Unknown")
                    component = error.get("component", "Unknown")
                    message = error.get("message", "No message")
                    timestamp = error.get("timestamp", "Unknown")
                    
                    print(f"  {i}. [{severity}] {error_id} - {status}")
                    print(f"     Component: {component}")
                    print(f"     Time: {timestamp}")
                    print(f"     Message: {message}")
        
        # Show error details
        elif action == "details" and error_id:
            # Get error details
            error = auditor.get_error_details(error_id)
            
            if not error:
                print(f"Error '{error_id}' not found")
                return 1
            
            # Display error
            if output_format == "json":
                print(json.dumps(error, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(error, default_flow_style=False))
            else:  # text format
                print(f"\nError Details - {error_id}")
                print("=" * 60)
                
                # Print error information
                print(f"Status: {error.get('status', 'Unknown')}")
                print(f"Severity: {error.get('severity', 'Unknown')}")
                print(f"Component: {error.get('component', 'Unknown')}")
                print(f"Timestamp: {error.get('timestamp', 'Unknown')}")
                print(f"Message: {error.get('message', 'No message')}")
                
                # Print error details
                if "details" in error:
                    print("\nDetails:")
                    details = error["details"]
                    if isinstance(details, str):
                        print(details)
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            print(f"  {key}: {value}")
                    elif isinstance(details, list):
                        for item in details:
                            print(f"  • {item}")
                
                # Print stack trace
                if "stack_trace" in error:
                    print("\nStack Trace:")
                    print(error["stack_trace"])
                
                # Print suggested fixes
                if "suggested_fixes" in error:
                    print("\nSuggested Fixes:")
                    fixes = error["suggested_fixes"]
                    if isinstance(fixes, list):
                        for i, fix in enumerate(fixes, 1):
                            print(f"  {i}. {fix}")
                    else:
                        print(f"  {fixes}")
        
        # Show error trends
        elif action == "trends":
            # Get error trends
            trends = auditor.get_error_trends(component=component)
            
            # Display trends
            if output_format == "json":
                print(json.dumps(trends, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(trends, default_flow_style=False))
            else:  # text format
                print("\nError Trends:")
                print("=" * 60)
                
                # Print trend summary
                summary = trends.get("summary", {})
                if summary:
                    print("Summary:")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")
                
                # Print trends by severity
                by_severity = trends.get("by_severity", {})
                if by_severity:
                    print("\nBy Severity:")
                    for severity, count in by_severity.items():
                        print(f"  {severity}: {count}")
                
                # Print trends by component
                by_component = trends.get("by_component", {})
                if by_component:
                    print("\nBy Component:")
                    for component, count in by_component.items():
                        print(f"  {component}: {count}")
                
                # Print trend over time
                over_time = trends.get("over_time", [])
                if over_time:
                    print("\nOver Time:")
                    for point in over_time:
                        timestamp = point.get("timestamp", "Unknown")
                        count = point.get("count", 0)
                        print(f"  {timestamp}: {count}")
        
        # Fix an error
        elif action == "fix" and error_id:
            # Fix the error
            result = auditor.fix_error(error_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\nFix result for error {error_id}:")
                
                if success:
                    print("✓ Error fixed successfully")
                else:
                    print("✗ Failed to fix error")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                
                # Print fix details
                if "details" in result:
                    print("\nDetails:")
                    details = result["details"]
                    if isinstance(details, str):
                        print(details)
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            print(f"  {key}: {value}")
                    elif isinstance(details, list):
                        for item in details:
                            print(f"  • {item}")
        
        # Unknown action
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  errors list [--status <status>] [--severity <severity>] [--component <component>]")
            print("  errors details <error_id>")
            print("  errors trends [--component <component>]")
            print("  errors fix <error_id>")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing errors: {e}")
        return 1

def log_command(args: str) -> int:
    """
    View and search audit logs.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="View and search audit logs")
    parser.add_argument("--level", choices=["debug", "info", "warning", "error", "critical", "all"], default="info", 
                       help="Minimum log level to display")
    parser.add_argument("--component", help="Filter by component")
    parser.add_argument("--search", help="Search string")
    parser.add_argument("--count", "-n", type=int, default=20, help="Number of log lines to show")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    parser.add_argument("--output", help="Output file for logs")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    level = cmd_args.level
    component = cmd_args.component
    search = cmd_args.search
    count = cmd_args.count
    follow = cmd_args.follow
    output_file = cmd_args.output
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # Get logs
        logs = auditor.get_logs(level=level, component=component, search=search, count=count)
        
        # Display logs
        print(f"\nAudit Logs (level: {level}):")
        
        if not logs:
            print("No logs found matching the criteria")
            return 0
        
        for log in logs:
            timestamp = log.get("timestamp", "Unknown")
            level = log.get("level", "Unknown")
            component = log.get("component", "Unknown")
            message = log.get("message", "No message")
            
            print(f"[{timestamp}] {level:7} {component}: {message}")
        
        # Follow logs if requested
        if follow:
            try:
                print("\nFollowing logs (press Ctrl+C to stop)...")
                last_timestamp = logs[-1].get("timestamp") if logs else None
                
                while True:
                    new_logs = auditor.get_logs(level=level, component=component, search=search,
                                             count=10, since=last_timestamp)
                    
                    if new_logs:
                        for log in new_logs:
                            timestamp = log.get("timestamp", "Unknown")
                            level = log.get("level", "Unknown")
                            component = log.get("component", "Unknown")
                            message = log.get("message", "No message")
                            
                            print(f"[{timestamp}] {level:7} {component}: {message}")
                        
                        last_timestamp = new_logs[-1].get("timestamp")
                    
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nLog following stopped")
        
        # Save logs to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    for log in logs:
                        timestamp = log.get("timestamp", "Unknown")
                        level = log.get("level", "Unknown")
                        component = log.get("component", "Unknown")
                        message = log.get("message", "No message")
                        
                        f.write(f"[{timestamp}] {level:7} {component}: {message}\n")
                
                print(f"\nLogs saved to {output_file}")
            except Exception as e:
                print(f"Error saving logs: {e}")
        
        return 0
    except Exception as e:
        print(f"Error viewing logs: {e}")
        return 1

def alerts_command(args: str) -> int:
    """
    View and manage alerts.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="View and manage alerts")
    parser.add_argument("action", nargs="?", choices=["list", "ack", "clear"], default="list", 
                        help="Action to perform")
    parser.add_argument("alert_id", nargs="?", help="Alert ID for ack/clear actions")
    parser.add_argument("--severity", choices=["critical", "high", "medium", "low", "all"], default="all", 
                       help="Filter by alert severity")
    parser.add_argument("--component", help="Filter by component")
    parser.add_argument("--status", choices=["active", "acknowledged", "cleared", "all"], default="active", 
                       help="Filter by alert status")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    alert_id = cmd_args.alert_id
    severity = cmd_args.severity
    component = cmd_args.component
    status = cmd_args.status
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # List alerts
        if action == "list":
            # Get alerts
            alerts_filter = {}
            if severity != "all":
                alerts_filter["severity"] = severity
            if component:
                alerts_filter["component"] = component
            if status != "all":
                alerts_filter["status"] = status
            
            alerts = auditor.get_alerts(filter=alerts_filter)
            
            # Display alerts
            if output_format == "json":
                print(json.dumps(alerts, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(alerts, default_flow_style=False))
            else:  # text format
                print("\nAlerts:")
                
                if not alerts:
                    print("No alerts found matching the criteria")
                    return 0
                
                print(f"Found {len(alerts)} alerts:")
                for i, alert in enumerate(alerts, 1):
                    alert_id = alert.get("id", "Unknown")
                    severity = alert.get("severity", "Unknown")
                    status = alert.get("status", "Unknown")
                    component = alert.get("component", "Unknown")
                    message = alert.get("message", "No message")
                    timestamp = alert.get("timestamp", "Unknown")
                    
                    print(f"  {i}. [{severity}] {alert_id} - {status}")
                    print(f"     Component: {component}")
                    print(f"     Time: {timestamp}")
                    print(f"     Message: {message}")
        
        # Acknowledge an alert
        elif action == "ack" and alert_id:
            # Acknowledge the alert
            result = auditor.acknowledge_alert(alert_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\nAcknowledge result for alert {alert_id}:")
                
                if success:
                    print("✓ Alert acknowledged successfully")
                else:
                    print("✗ Failed to acknowledge alert")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
        
        # Clear an alert
        elif action == "clear" and alert_id:
            # Clear the alert
            result = auditor.clear_alert(alert_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\nClear result for alert {alert_id}:")
                
                if success:
                    print("✓ Alert cleared successfully")
                else:
                    print("✗ Failed to clear alert")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
        
        # Unknown action
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  alerts list [--severity <severity>] [--component <component>] [--status <status>]")
            print("  alerts ack <alert_id>")
            print("  alerts clear <alert_id>")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing alerts: {e}")
        return 1

def status_command(args: str) -> int:
    """
    Display auditor status.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Display auditor status")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # Get status
        status = auditor.get_status()
        
        # Display status
        if output_format == "json":
            print(json.dumps(status, indent=2, default=str))
        elif output_format == "yaml":
            print(yaml.dump(status, default_flow_style=False))
        else:  # text format
            print("\nAuditor Status:")
            print("=" * 60)
            
            # Print general status
            is_running = status.get("is_running", False)
            print(f"Running: {'Yes' if is_running else 'No'}")
            print(f"Uptime: {status.get('uptime', 'Unknown')}")
            print(f"Last Audit: {status.get('last_audit_time', 'Never')}")
            print(f"Queue Size: {status.get('queue_size', 0)}")
            
            # Print active sensors
            active_sensors = status.get("active_sensors", [])
            print(f"\nActive Sensors: {len(active_sensors)}")
            for i, sensor in enumerate(active_sensors[:5], 1):
                sensor_id = sensor.get("sensor_id", "Unknown")
                component = sensor.get("component_name", "Unknown")
                last_check = sensor.get("last_check_time", "Unknown")
                print(f"  {i}. {sensor_id} - {component} - Last check: {last_check}")
            
            if len(active_sensors) > 5:
                print(f"  ... and {len(active_sensors) - 5} more")
            
            # Print recent alerts
            recent_alerts = status.get("recent_alerts", [])
            print(f"\nRecent Alerts: {len(recent_alerts)}")
            for i, alert in enumerate(recent_alerts[:5], 1):
                alert_type = alert.get("type", "Unknown")
                severity = alert.get("severity", "Unknown")
                timestamp = alert.get("timestamp", "Unknown")
                message = alert.get("message", "No message")
                print(f"  {i}. [{severity}] {alert_type} - {timestamp}")
                print(f"     {message}")
            
            if len(recent_alerts) > 5:
                print(f"  ... and {len(recent_alerts) - 5} more")
            
            # Print performance metrics
            metrics = status.get("performance_metrics", {})
            if metrics:
                print("\nPerformance Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            # Print health status
            health = status.get("health", {})
            if health:
                print("\nHealth Status:")
                for key, value in health.items():
                    print(f"  {key}: {value}")
        
        return 0
    except Exception as e:
        print(f"Error getting auditor status: {e}")
        return 1

def system_audit_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle system audit events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info("System audit event received")
        
        # Get auditor instance
        registry = sys.modules.get("__main__").registry
        auditor = registry.get_component("auditor")
        
        if not auditor:
            return {"success": False, "error": "Auditor agent not available"}
        
        # Extract event data
        component = event_data.get("component")
        full_audit = event_data.get("full_audit", False)
        
        # Run audit
        audit_options = {}
        if component:
            audit_options["component"] = component
        
        result = auditor.run_audit(force_full_audit=full_audit, **audit_options)
        
        # Log result
        issues_found = result.get("issues_found", 0)
        logger.info(f"System audit completed with {issues_found} issues found")
        
        return {
            "success": True,
            "issues_found": issues_found,
            "audit_result": result
        }
    except Exception as e:
        logger.error(f"Error handling system audit event: {e}")
        return {"success": False, "error": str(e)}

def error_detected_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle error detected events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info("Error detected event received")
        
        # Get auditor instance
        registry = sys.modules.get("__main__").registry
        auditor = registry.get_component("auditor")
        
        if not auditor:
            return {"success": False, "error": "Auditor agent not available"}
        
        # Extract event data
        error_data = event_data.get("error", {})
        
        # Process error
        error_id = error_data.get("id")
        severity = error_data.get("severity", "Unknown")
        component = error_data.get("component", "Unknown")
        message = error_data.get("message", "No message")
        
        logger.info(f"Processing error: {error_id} - {severity} - {component} - {message}")
        
        # Add error to auditor
        result = auditor.record_error(error_data)
        
        # Generate alert if needed
        if severity in ["critical", "high"]:
            alert_data = {
                "type": "error_detected",
                "severity": severity,
                "component": component,
                "message": f"Error detected: {message}",
                "error_id": error_id
            }
            
            auditor.generate_alert(alert_data)
        
        return {
            "success": True,
            "error_id": error_id,
            "recorded": result.get("success", False)
        }
    except Exception as e:
        logger.error(f"Error handling error detected event: {e}")
        return {"success": False, "error": str(e)}

def benchmark_completed_handler(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle benchmark completed events.
    
    Args:
        event_data: Event data
        
    Returns:
        Result data
    """
    try:
        logger.info("Benchmark completed event received")
        
        # Get auditor instance
        registry = sys.modules.get("__main__").registry
        auditor = registry.get_component("auditor")
        
        if not auditor:
            return {"success": False, "error": "Auditor agent not available"}
        
        # Extract event data
        benchmark_id = event_data.get("benchmark_id")
        benchmark_type = event_data.get("type", "Unknown")
        component = event_data.get("component", "Unknown")
        results = event_data.get("results", {})
        
        logger.info(f"Processing benchmark: {benchmark_id} - {benchmark_type} - {component}")
        
        # Record benchmark results
        result = auditor.record_benchmark(event_data)
        
        # Update metrics
        metrics = results.get("metrics", {})
        if metrics:
            auditor.update_metrics(metrics)
        
        # Check for issues
        threshold = event_data.get("threshold", 0.75)
        score = results.get("score", 0)
        
        if score < threshold:
            # Generate alert for failed benchmark
            alert_data = {
                "type": "benchmark_failed",
                "severity": "medium",
                "component": component,
                "message": f"Benchmark failed: {benchmark_type} - Score: {score} (Threshold: {threshold})",
                "benchmark_id": benchmark_id
            }
            
            auditor.generate_alert(alert_data)
        
        return {
            "success": True,
            "benchmark_id": benchmark_id,
            "recorded": result.get("success", False),
            "score": score
        }
    except Exception as e:
        logger.error(f"Error handling benchmark completed event: {e}")
        return {"success": False, "error": str(e)}

def patch_command(args: str) -> int:
    """
    Manage and apply patches.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    import shlex
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Manage and apply patches")
    parser.add_argument("action", nargs="?", choices=["list", "view", "apply", "revert"], default="list", 
                        help="Action to perform")
    parser.add_argument("patch_id", nargs="?", help="Patch ID for view/apply/revert actions")
    parser.add_argument("--status", choices=["pending", "applied", "failed", "reverted", "all"], default="pending", 
                       help="Filter by patch status")
    parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                       help="Output format")
    
    try:
        cmd_args = parser.parse_args(shlex.split(args))
    except SystemExit:
        return 1
    
    action = cmd_args.action
    patch_id = cmd_args.patch_id
    status = cmd_args.status
    output_format = cmd_args.format
    
    # Get auditor instance
    registry = sys.modules.get("__main__").registry
    auditor = registry.get_component("auditor")
    
    if not auditor:
        print("Error: Auditor agent not available")
        return 1
    
    try:
        # List patches
        if action == "list":
            # Get patches
            patches_filter = {}
            if status != "all":
                patches_filter["status"] = status
            
            patches = auditor.get_patches(filter=patches_filter)
            
            # Display patches
            if output_format == "json":
                print(json.dumps(patches, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(patches, default_flow_style=False))
            else:  # text format
                print("\nPatches:")
                
                if not patches:
                    print("No patches found matching the criteria")
                    return 0
                
                print(f"Found {len(patches)} patches:")
                for i, patch in enumerate(patches, 1):
                    patch_id = patch.get("id", "Unknown")
                    status = patch.get("status", "Unknown")
                    target = patch.get("target", "Unknown")
                    created_at = patch.get("created_at", "Unknown")
                    description = patch.get("description", "No description")
                    
                    print(f"  {i}. {patch_id} - {status}")
                    print(f"     Target: {target}")
                    print(f"     Created: {created_at}")
                    print(f"     Description: {description}")
        
        # View patch details
        elif action == "view" and patch_id:
            # Get patch details
            patch = auditor.get_patch_details(patch_id)
            
            if not patch:
                print(f"Patch '{patch_id}' not found")
                return 1
            
            # Display patch
            if output_format == "json":
                print(json.dumps(patch, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(patch, default_flow_style=False))
            else:  # text format
                print(f"\nPatch Details - {patch_id}")
                print("=" * 60)
                
                # Print patch information
                print(f"Status: {patch.get('status', 'Unknown')}")
                print(f"Target: {patch.get('target', 'Unknown')}")
                print(f"Created: {patch.get('created_at', 'Unknown')}")
                print(f"Applied: {patch.get('applied_at', 'Never')}")
                print(f"Description: {patch.get('description', 'No description')}")
                
                # Print patch changes
                if "changes" in patch:
                    print("\nChanges:")
                    changes = patch["changes"]
                    if isinstance(changes, list):
                        for i, change in enumerate(changes, 1):
                            print(f"  {i}. {change.get('file', 'Unknown')}:")
                            print(f"     Type: {change.get('type', 'Unknown')}")
                            if "line" in change:
                                print(f"     Line: {change['line']}")
                            if "content" in change:
                                print(f"     Content: {change['content']}")
                    elif isinstance(changes, dict):
                        for file, file_changes in changes.items():
                            print(f"  {file}:")
                            if isinstance(file_changes, list):
                                for change in file_changes:
                                    print(f"     {change}")
                            else:
                                print(f"     {file_changes}")
                
                # Print patch verification
                if "verification" in patch:
                    print("\nVerification:")
                    verification = patch["verification"]
                    if isinstance(verification, dict):
                        for key, value in verification.items():
                            print(f"  {key}: {value}")
                    else:
                        print(f"  {verification}")
        
        # Apply a patch
        elif action == "apply" and patch_id:
            # Apply the patch
            result = auditor.apply_patch(patch_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\nApply result for patch {patch_id}:")
                
                if success:
                    print("[SUCCESS] Patch applied successfully")
                else:
                    print("[FAILED] Failed to apply patch")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                
                # Print application details
                if "details" in result:
                    print("\nDetails:")
                    details = result["details"]
                    if isinstance(details, str):
                        print(details)
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            print(f"  {key}: {value}")
                    elif isinstance(details, list):
                        for item in details:
                            print(f"  - {item}")
        
        # Revert a patch
        elif action == "revert" and patch_id:
            # Revert the patch
            result = auditor.revert_patch(patch_id)
            
            # Display result
            if output_format == "json":
                print(json.dumps(result, indent=2, default=str))
            elif output_format == "yaml":
                print(yaml.dump(result, default_flow_style=False))
            else:  # text format
                success = result.get("success", False)
                print(f"\nRevert result for patch {patch_id}:")
                
                if success:
                    print("[SUCCESS] Patch reverted successfully")
                else:
                    print("[FAILED] Failed to revert patch")
                    if "error" in result:
                        print(f"  Error: {result['error']}")
                
                # Print revert details
                if "details" in result:
                    print("\nDetails:")
                    details = result["details"]
                    if isinstance(details, str):
                        print(details)
                    elif isinstance(details, dict):
                        for key, value in details.items():
                            print(f"  {key}: {value}")
                    elif isinstance(details, list):
                        for item in details:
                            print(f"  - {item}")
        
        # Unknown action
        else:
            print("Error: Invalid action or missing required arguments")
            print("Usage examples:")
            print("  patch list [--status <status>]")
            print("  patch view <patch_id>")
            print("  patch apply <patch_id>")
            print("  patch revert <patch_id>")
            return 1
        
        return 0
    except Exception as e:
        print(f"Error managing patches: {e}")
        return 1
