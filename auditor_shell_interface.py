#!/usr/bin/env python3
"""
FixWurx Auditor Shell Interface for Sensors

This module implements a shell interface for managing sensors, monitoring errors,
and analyzing benchmark metrics directly from the command line.
"""

import os
import sys
import cmd
import json
import time
import datetime
import argparse
import logging
import textwrap
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import core components
from sensor_registry import SensorRegistry, create_sensor_registry
from sensor_manager import SensorManager
from error_report import ErrorReport
from benchmark_storage import BenchmarkStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AuditorShell] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auditor_shell')


class AuditorShellInterface(cmd.Cmd):
    """
    Interactive shell interface for managing sensors, errors, and benchmarks.
    
    This shell extends the standard Python cmd module to provide commands for:
    - Listing active sensors and their status
    - Viewing and acknowledging error reports
    - Analyzing benchmark metrics and trends
    - Creating visualization reports
    """
    
    intro = "\n".join([
        "="*70,
        "FixWurx Auditor Sensor Shell Interface",
        "Type 'help' or '?' to list commands. Type 'exit' to exit.",
        "="*70
    ])
    prompt = "(auditor) $ "
    
    def __init__(self, 
                sensor_registry: Optional[SensorRegistry] = None,
                sensor_manager: Optional[SensorManager] = None,
                benchmark_storage: Optional[BenchmarkStorage] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the shell interface.
        
        Args:
            sensor_registry: SensorRegistry instance or None to create
            sensor_manager: SensorManager instance or None to create
            benchmark_storage: BenchmarkStorage instance or None to create
            config: Optional configuration dictionary
        """
        super().__init__()
        
        self.config = config or {}
        
        # Get or create components
        self.sensor_registry = sensor_registry or self._create_sensor_registry()
        self.sensor_manager = sensor_manager or self._create_sensor_manager()
        self.benchmark_storage = benchmark_storage or self._create_benchmark_storage()
        
        # Initialize state
        self.current_project = self.config.get("default_project", "DefaultProject")
        self.current_session = None
        self.cached_errors = []
        self.cached_metrics = {}
        self.last_refresh = 0
        self.refresh_interval = self.config.get("refresh_interval", 60)  # seconds
        
        logger.info("Initialized Auditor Shell Interface")
    
    def _create_sensor_registry(self) -> SensorRegistry:
        """Create and initialize the sensor registry."""
        registry_config = self.config.get("sensor_registry", {})
        return create_sensor_registry(registry_config)
    
    def _create_sensor_manager(self) -> SensorManager:
        """Create and initialize the sensor manager."""
        manager_config = self.config.get("sensor_manager", {})
        return SensorManager(self.sensor_registry, manager_config)
    
    def _create_benchmark_storage(self) -> BenchmarkStorage:
        """Create and initialize the benchmark storage."""
        storage_path = self.config.get("benchmark_storage_path", "auditor_data/benchmarks")
        return BenchmarkStorage(storage_path)
    
    def _refresh_data(self, force: bool = False) -> None:
        """
        Refresh cached data if needed.
        
        Args:
            force: Force refresh even if within interval
        """
        current_time = time.time()
        if force or (current_time - self.last_refresh) >= self.refresh_interval:
            # Refresh error reports
            self.cached_errors = self._get_error_reports()
            
            # Refresh metrics if we have a current session
            if self.current_session:
                self.cached_metrics = self._get_session_metrics(self.current_session)
            
            self.last_refresh = current_time
    
    def _get_error_reports(self) -> List[Dict[str, Any]]:
        """Get current error reports from all sensors."""
        reports = []
        
        # Get all sensors by iterating through the registry's sensors dictionary
        sensors = list(self.sensor_registry.sensors.values())
        
        # Collect error reports
        for sensor in sensors:
            sensor_reports = sensor.get_error_reports() if hasattr(sensor, 'get_error_reports') else []
            for report in sensor_reports:
                # Convert to dict if it's an ErrorReport object
                if isinstance(report, ErrorReport):
                    report_dict = report.to_dict()
                else:
                    report_dict = report
                
                reports.append(report_dict)
        
        return reports
    
    def _get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            # Get metrics from benchmark storage
            metric_list = self.benchmark_storage.get_session_metrics(session_id)
            
            if metric_list:
                # Use most recent metrics
                metrics = metric_list[-1].get("metrics", {})
        except Exception as e:
            logger.error(f"Error getting session metrics: {str(e)}")
        
        return metrics
    
    # Command methods
    
    def do_exit(self, arg):
        """Exit the shell."""
        print("Exiting Auditor Shell Interface...")
        return True
    
    def do_quit(self, arg):
        """Alias for exit."""
        return self.do_exit(arg)
    
    def do_refresh(self, arg):
        """Refresh all cached data."""
        print("Refreshing data...")
        self._refresh_data(force=True)
        print(f"Data refreshed. {len(self.cached_errors)} errors, "
              f"{len(self.cached_metrics)} metrics.")
    
    def do_status(self, arg):
        """Show the status of the auditor system."""
        self._refresh_data()
        
        print("\n" + "="*70)
        print(f"AUDITOR SYSTEM STATUS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Get all sensors by iterating through the registry's sensors dictionary
        sensors = list(self.sensor_registry.sensors.values())
        
        # Show sensor status
        print(f"\nActive Sensors: {len(sensors)}")
        for i, sensor in enumerate(sensors):
            status = sensor.get_status() if hasattr(sensor, 'get_status') else {"sensor_id": sensor.sensor_id}
            last_check = status.get("last_check_time", "Unknown")
            if isinstance(last_check, (int, float)):
                last_check = datetime.datetime.fromtimestamp(last_check).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"  {i+1}. {status.get('sensor_id', 'Unknown')} - "
                  f"{status.get('component_name', 'Unknown')} - "
                  f"Last check: {last_check}")
        
        # Show error summary
        print(f"\nActive Errors: {len(self.cached_errors)}")
        severity_counts = {}
        for report in self.cached_errors:
            severity = report.get("severity", "UNKNOWN")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            print(f"  {severity}: {count}")
        
        # Show current session
        print(f"\nCurrent Project: {self.current_project}")
        if self.current_session:
            print(f"Current Session: {self.current_session}")
            
            # Show key metrics if available
            if self.cached_metrics:
                print("\nKey Metrics:")
                for key in ["bug_detection_recall", "bug_fix_yield", "test_pass_ratio", 
                           "energy_reduction_pct", "aggregate_confidence_score"]:
                    if key in self.cached_metrics:
                        print(f"  {key}: {self.cached_metrics[key]:.2f}")
        else:
            print("No active session selected")
        
        print("\n" + "="*70)
    
    # Alias for singular form of 'sensors'
    def do_sensor(self, arg):
        """Alias for 'sensors' command."""
        return self.do_sensors(arg)
        
    def do_sensors(self, arg):
        """
        List and manage sensors.
        
        Usage:
          sensors list              - List all sensors
          sensors info <sensor_id>  - Show detailed info for a sensor
          sensors enable <sensor_id> - Enable a sensor
          sensors disable <sensor_id> - Disable a sensor
        """
        args = arg.split()
        if not args:
            print("Error: Missing subcommand. Use 'sensors list', 'sensors info <sensor_id>', etc.")
            return
        
        subcommand = args[0].lower()
        
        # List all sensors
        if subcommand == "list":
            sensors = list(self.sensor_registry.sensors.values())
            print(f"\nRegistered Sensors ({len(sensors)}):")
            
            for i, sensor in enumerate(sensors):
                status = sensor.get_status() if hasattr(sensor, 'get_status') else {"sensor_id": sensor.sensor_id}
                enabled = "Enabled" if hasattr(sensor, 'enabled') and sensor.enabled else "Disabled" if hasattr(sensor, 'enabled') else "Unknown"
                print(f"  {i+1}. {status.get('sensor_id', 'Unknown')} - "
                      f"{status.get('component_name', 'Unknown')} - {enabled}")
        
        # Show detailed info for a sensor
        elif subcommand == "info" and len(args) > 1:
            sensor_id = args[1]
            sensor = self.sensor_registry.get_sensor(sensor_id)
            
            if not sensor:
                print(f"Error: Sensor '{sensor_id}' not found")
                return
            
            status = sensor.get_status() if hasattr(sensor, 'get_status') else {"sensor_id": sensor.sensor_id}
            
            print(f"\nSensor Information - {sensor_id}")
            print("="*50)
            
            # Format and print status dictionary
            for key, value in status.items():
                if isinstance(value, (dict, list)):
                    print(f"  {key}:")
                    json_str = json.dumps(value, indent=2)
                    print(textwrap.indent(json_str, "    "))
                else:
                    print(f"  {key}: {value}")
        
        # Enable a sensor
        elif subcommand == "enable" and len(args) > 1:
            sensor_id = args[1]
            sensor = self.sensor_registry.get_sensor(sensor_id)
            
            if not sensor:
                print(f"Error: Sensor '{sensor_id}' not found")
                return
            
            if hasattr(sensor, 'enable'):
                sensor.enable()
                print(f"Sensor '{sensor_id}' enabled")
            else:
                print(f"Error: Sensor '{sensor_id}' doesn't support enabling")
        
        # Disable a sensor
        elif subcommand == "disable" and len(args) > 1:
            sensor_id = args[1]
            sensor = self.sensor_registry.get_sensor(sensor_id)
            
            if not sensor:
                print(f"Error: Sensor '{sensor_id}' not found")
                return
            
            if hasattr(sensor, 'disable'):
                sensor.disable()
                print(f"Sensor '{sensor_id}' disabled")
            else:
                print(f"Error: Sensor '{sensor_id}' doesn't support disabling")
        
        else:
            print("Error: Unknown subcommand or missing arguments")
            print("Usage:")
            print("  sensors list              - List all sensors")
            print("  sensors info <sensor_id>  - Show detailed info for a sensor")
            print("  sensors enable <sensor_id> - Enable a sensor")
            print("  sensors disable <sensor_id> - Disable a sensor")
    
    # Alias for singular form of 'errors'
    def do_error(self, arg):
        """Alias for 'errors' command."""
        return self.do_errors(arg)
        
    def do_errors(self, arg):
        """
        View and manage error reports.
        
        Usage:
          errors list              - List all error reports
          errors view <error_id>   - View detailed information for an error
          errors ack <error_id>    - Acknowledge an error
          errors resolve <error_id> <resolution> - Resolve an error
        """
        args = arg.split()
        if not args:
            print("Error: Missing subcommand. Use 'errors list', 'errors view <error_id>', etc.")
            return
        
        subcommand = args[0].lower()
        self._refresh_data()
        
        # List all errors
        if subcommand == "list":
            # Show current context
            print(f"\nCurrent Project: {self.current_project}")
            print(f"Current Session: {self.current_session or 'None'}")
            print("-" * 40)
            
            if not self.cached_errors:
                print("No active error reports found")
                return
            
            print(f"\nActive Error Reports ({len(self.cached_errors)}):")
            for i, report in enumerate(self.cached_errors):
                error_id = report.get("error_id", "Unknown")
                error_type = report.get("error_type", "Unknown")
                severity = report.get("severity", "Unknown")
                timestamp = report.get("timestamp", "Unknown")
                if isinstance(timestamp, (int, float)):
                    timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                # Get message from details if available
                details = report.get("details", {})
                message = details.get("message", "No details") if isinstance(details, dict) else "No details"
                
                print(f"  {i+1}. [{severity}] {error_id} - {error_type} - {timestamp}")
                print(f"     {message}")
        
        # View detailed error info
        elif subcommand == "view" and len(args) > 1:
            error_id = args[1]
            
            # Find the error report
            report = None
            for r in self.cached_errors:
                if r.get("error_id") == error_id:
                    report = r
                    break
            
            if not report:
                print(f"Error: Error report '{error_id}' not found")
                return
            
            print(f"\nError Report - {error_id}")
            print("="*50)
            
            # Print basic information
            timestamp = report.get("timestamp", "Unknown")
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Type: {report.get('error_type', 'Unknown')}")
            print(f"Severity: {report.get('severity', 'Unknown')}")
            print(f"Timestamp: {timestamp}")
            print(f"Component: {report.get('component_name', 'Unknown')}")
            print(f"Sensor: {report.get('sensor_id', 'Unknown')}")
            print(f"Status: {report.get('status', 'Unknown')}")
            
            # Print details
            details = report.get("details", {})
            if details:
                print("\nDetails:")
                if isinstance(details, dict):
                    for key, value in details.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {details}")
            
            # Print context
            context = report.get("context", {})
            if context:
                print("\nContext:")
                if isinstance(context, dict):
                    json_str = json.dumps(context, indent=2)
                    print(textwrap.indent(json_str, "  "))
                else:
                    print(f"  {context}")
        
        # Acknowledge an error
        elif subcommand == "ack" and len(args) > 1:
            error_id = args[1]
            
            # Find the error report
            report = None
            for r in self.cached_errors:
                if r.get("error_id") == error_id:
                    report = r
                    break
            
            if not report:
                print(f"Error: Error report '{error_id}' not found")
                return
            
            # Get the sensor that generated the report
            sensor_id = report.get("sensor_id")
            sensor = self.sensor_registry.get_sensor(sensor_id) if sensor_id else None
            
            if not sensor or not hasattr(sensor, 'acknowledge_error'):
                print(f"Error: Cannot acknowledge error - sensor '{sensor_id}' not found or doesn't support acknowledgement")
                return
            
            # Acknowledge the error
            try:
                sensor.acknowledge_error(error_id)
                print(f"Error '{error_id}' acknowledged")
                self._refresh_data(force=True)
            except Exception as e:
                print(f"Error acknowledging error: {str(e)}")
        
        # Resolve an error
        elif subcommand == "resolve" and len(args) > 2:
            error_id = args[1]
            resolution = " ".join(args[2:])
            
            # Find the error report
            report = None
            for r in self.cached_errors:
                if r.get("error_id") == error_id:
                    report = r
                    break
            
            if not report:
                print(f"Error: Error report '{error_id}' not found")
                return
            
            # Get the sensor that generated the report
            sensor_id = report.get("sensor_id")
            sensor = self.sensor_registry.get_sensor(sensor_id) if sensor_id else None
            
            if not sensor or not hasattr(sensor, 'resolve_error'):
                print(f"Error: Cannot resolve error - sensor '{sensor_id}' not found or doesn't support resolution")
                return
            
            # Resolve the error
            try:
                sensor.resolve_error(error_id, resolution)
                print(f"Error '{error_id}' resolved: {resolution}")
                self._refresh_data(force=True)
            except Exception as e:
                print(f"Error resolving error: {str(e)}")
        
        else:
            print("Error: Unknown subcommand or missing arguments")
            print("Usage:")
            print("  errors list              - List all error reports")
            print("  errors view <error_id>   - View detailed information for an error")
            print("  errors ack <error_id>    - Acknowledge an error")
            print("  errors resolve <error_id> <resolution> - Resolve an error")
    
    # Numeric command handler for selecting projects by number
    def default(self, line):
        """Handle commands not matched by other do_* methods."""
        # Check if it's a numeric command for project selection
        if line.isdigit():
            project_num = int(line)
            # Get list of projects
            projects = self.benchmark_storage.get_all_projects()
            if projects and 1 <= project_num <= len(projects):
                # Use the project at that index (1-based)
                project_name = sorted(projects)[project_num - 1]
                print(f"Selecting project: {project_name}")
                return self.do_project(f"use {project_name}")
            else:
                print(f"Error: No project at position {project_num}")
                return False
        
        # Default behavior for unknown commands
        print(f"*** Unknown syntax: {line}")
        return False
        
    def do_project(self, arg):
        """
        Manage benchmark projects.
        
        Usage:
          project list             - List all projects
          project use <name>       - Set current project
          project create <name>    - Create a new project
        """
        args = arg.split()
        if not args:
            print("Error: Missing subcommand. Use 'project list', 'project use <name>', etc.")
            return
        
        subcommand = args[0].lower()
        
        # List all projects
        if subcommand == "list":
            projects = self.benchmark_storage.get_all_projects()
            
            if not projects:
                print("No benchmark projects found")
                return
            
            print(f"\nBenchmark Projects ({len(projects)}):")
            for i, project in enumerate(projects):
                sessions = self.benchmark_storage.get_project_sessions(project)
                print(f"  {i+1}. {project} - {len(sessions)} sessions")
        
        # Set current project
        elif subcommand == "use" and len(args) > 1:
            project_name = args[1]
            
            # Validate project exists
            projects = self.benchmark_storage.get_all_projects()
            if project_name not in projects:
                print(f"Warning: Project '{project_name}' doesn't exist. Creating it.")
            
            self.current_project = project_name
            self.current_session = None  # Reset current session
            print(f"Current project set to '{project_name}'")
        
        # Create a new project
        elif subcommand == "create" and len(args) > 1:
            project_name = args[1]
            
            # Create a new session to ensure the project exists
            session_id = self.benchmark_storage.create_session(
                project_name=project_name,
                metadata={
                    "description": f"Project created from shell interface on {datetime.datetime.now().isoformat()}"
                }
            )
            
            print(f"Created project '{project_name}' with initial session '{session_id}'")
            self.current_project = project_name
            self.current_session = session_id
        
        else:
            print("Error: Unknown subcommand or missing arguments")
            print("Usage:")
            print("  project list             - List all projects")
            print("  project use <name>       - Set current project")
            print("  project create <name>    - Create a new project")
    
    def do_session(self, arg):
        """
        Manage benchmark sessions.
        
        Usage:
          session list                 - List sessions for current project
          session create [name]        - Create a new session
          session use <session_id>     - Set current session
          session info <session_id>    - Show session information
          session export <session_id>  - Export session data
        """
        args = arg.split()
        if not args:
            print("Error: Missing subcommand. Use 'session list', 'session create', etc.")
            return
        
        subcommand = args[0].lower()
        
        # List sessions for current project
        if subcommand == "list":
            if not self.current_project:
                print("Error: No current project set. Use 'project use <name>' first.")
                return
            
            sessions = self.benchmark_storage.get_project_sessions(self.current_project)
            
            if not sessions:
                print(f"No sessions found for project '{self.current_project}'")
                return
            
            print(f"\nSessions for Project '{self.current_project}' ({len(sessions)}):")
            for i, session_id in enumerate(sessions):
                # Try to get session metadata
                try:
                    summary = self.benchmark_storage.generate_session_summary(session_id)
                    created_at = summary.get("created_at", "Unknown")
                    last_updated = summary.get("last_updated", "Unknown")
                    metrics_count = summary.get("metrics_count", 0)
                    
                    print(f"  {i+1}. {session_id}")
                    print(f"     Created: {created_at}")
                    print(f"     Last updated: {last_updated}")
                    print(f"     Metrics snapshots: {metrics_count}")
                except Exception as e:
                    print(f"  {i+1}. {session_id} - Error: {str(e)}")
        
        # Create a new session
        elif subcommand == "create":
            if not self.current_project:
                print("Error: No current project set. Use 'project use <name>' first.")
                return
            
            # Optional session name
            metadata = {}
            if len(args) > 1:
                metadata["name"] = args[1]
            
            # Create the session
            session_id = self.benchmark_storage.create_session(
                project_name=self.current_project,
                metadata=metadata
            )
            
            print(f"Created new session '{session_id}' for project '{self.current_project}'")
            self.current_session = session_id
        
        # Set current session
        elif subcommand == "use" and len(args) > 1:
            session_id = args[1]
            
            # Validate session exists
            try:
                summary = self.benchmark_storage.generate_session_summary(session_id)
                if not summary:
                    print(f"Error: Session '{session_id}' not found")
                    return
                
                self.current_session = session_id
                print(f"Current session set to '{session_id}'")
                
                # Refresh metrics for the new session
                self._refresh_data(force=True)
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Show session information
        elif subcommand == "info" and len(args) > 1:
            session_id = args[1]
            
            try:
                summary = self.benchmark_storage.generate_session_summary(session_id)
                if not summary:
                    print(f"Error: Session '{session_id}' not found")
                    return
                
                print(f"\nSession Information - {session_id}")
                print("="*50)
                
                # Print basic information
                print(f"Project: {summary.get('project_name', 'Unknown')}")
                print(f"Created: {summary.get('created_at', 'Unknown')}")
                print(f"Last updated: {summary.get('last_updated', 'Unknown')}")
                print(f"Metrics snapshots: {summary.get('metrics_count', 0)}")
                print(f"Error reports: {summary.get('reports_count', 0)}")
                
                # Print key metrics
                if "key_metrics" in summary and summary["key_metrics"]:
                    print("\nKey Metrics Changes:")
                    for metric, values in summary["key_metrics"].items():
                        change = values["delta"]
                        change_str = f"+{change:.2f}" if change >= 0 else f"{change:.2f}"
                        print(f"  {metric}: {values['initial']:.2f} → {values['final']:.2f} ({change_str})")
            except Exception as e:
                print(f"Error retrieving session information: {str(e)}")
        
        # Export session data
        elif subcommand == "export" and len(args) > 1:
            session_id = args[1]
            
            # Get export directory
            export_dir = f"auditor_data/exports/{session_id}_{int(time.time())}"
            
            try:
                success = self.benchmark_storage.export_session_data(
                    session_id=session_id,
                    export_path=export_dir
                )
                
                if success:
                    print(f"Session data exported to: {export_dir}")
                else:
                    print(f"Error exporting session data")
            except Exception as e:
                print(f"Error exporting session data: {str(e)}")
        
        else:
            print("Error: Unknown subcommand or missing arguments")
            print("Usage:")
            print("  session list                 - List sessions for current project")
            print("  session create [name]        - Create a new session")
            print("  session use <session_id>     - Set current session")
            print("  session info <session_id>    - Show session information")
            print("  session export <session_id>  - Export session data")
    
    def do_metrics(self, arg):
        """
        View and analyze benchmark metrics.
        
        Usage:
          metrics list                  - List metrics for current session
          metrics trend <metric_name>   - Show trend for a specific metric
          metrics top [count]           - Show top metrics by improvement
        """
        args = arg.split()
        if not args:
            print("Error: Missing subcommand. Use 'metrics list', 'metrics trend <name>', etc.")
            return
        
        subcommand = args[0].lower()
        
        # List metrics for current session
        if subcommand == "list":
            if not self.current_session:
                print("Error: No current session set. Use 'session use <session_id>' first.")
                return
            
            self._refresh_data()
            
            if not self.cached_metrics:
                print(f"No metrics found for session '{self.current_session}'")
                return
            
            print(f"\nMetrics for Session '{self.current_session}':")
            
            # Group metrics by category
            categories = {
                "Detection and Fix": ["bug_detection_recall", "bug_fix_yield", "mttd", "mttr"],
                "Energy and Convergence": ["energy_reduction_pct", "proof_coverage_delta", "residual_risk_improvement"],
                "Test and Quality": ["test_pass_ratio", "regression_introduction_rate", "duplicate_module_ratio"],
                "Agent and Resource": ["agent_coordination_overhead", "token_per_fix_efficiency", "hallucination_rate"],
                "Time and Stability": ["lyapunov_descent_consistency", "meta_guard_breach_count"],
                "Overall": ["aggregate_confidence_score"]
            }
            
            # Print metrics by category
            for category, metric_names in categories.items():
                print(f"\n{category}:")
                for metric in metric_names:
                    if metric in self.cached_metrics:
                        value = self.cached_metrics[metric]
                        value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                        print(f"  {metric}: {value_str}")
        
        # Show trend for a specific metric
        elif subcommand == "trend" and len(args) > 1:
            metric_name = args[1]
            
            if not self.current_session:
                print("Error: No current session set. Use 'session use <session_id>' first.")
                return
            
            try:
                metrics_list = self.benchmark_storage.get_session_metrics(self.current_session)
                
                if not metrics_list:
                    print(f"No metrics found for session '{self.current_session}'")
                    return
                
                # Extract the metric values
                values = []
                for snapshot in metrics_list:
                    metrics = snapshot.get("metrics", {})
                    if metric_name in metrics:
                        timestamp = snapshot.get("timestamp", "Unknown")
                        if isinstance(timestamp, str):
                            try:
                                timestamp = datetime.datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
                            except:
                                pass
                        values.append((timestamp, metrics[metric_name]))
                
                if not values:
                    print(f"Metric '{metric_name}' not found in session data")
                    return
                
                print(f"\nTrend for '{metric_name}' in Session '{self.current_session}':")
                
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
                
            except Exception as e:
                print(f"Error generating trend: {str(e)}")
        
        # Show top metrics by improvement
        elif subcommand == "top":
            if not self.current_session:
                print("Error: No current session set. Use 'session use <session_id>' first.")
                return
            
            try:
                # Default to top 5
                count = 5
                if len(args) > 1:
                    try:
                        count = int(args[1])
                    except ValueError:
                        print(f"Warning: Invalid count '{args[1]}', using default of 5")
                
                # Get metrics from benchmark storage
                metrics_list = self.benchmark_storage.get_session_metrics(self.current_session)
                
                if not metrics_list or len(metrics_list) < 2:
                    print(f"Insufficient metrics data for session '{self.current_session}'")
                    return
                
                # Get first and last metrics
                first_metrics = metrics_list[0].get("metrics", {})
                last_metrics = metrics_list[-1].get("metrics", {})
                
                # Calculate improvements
                improvements = []
                for key in set(first_metrics.keys()).intersection(last_metrics.keys()):
                    if isinstance(first_metrics[key], (int, float)) and isinstance(last_metrics[key], (int, float)):
                        improvement = last_metrics[key] - first_metrics[key]
                        improvements.append((key, improvement, first_metrics[key], last_metrics[key]))
                
                # Sort by absolute improvement
                improvements.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print(f"\nTop {count} Metrics by Improvement:")
                for i, (key, improvement, first, last) in enumerate(improvements[:count]):
                    direction = "+" if improvement >= 0 else ""
                    print(f"  {i+1}. {key}: {first:.2f} → {last:.2f} ({direction}{improvement:.2f})")
                
            except Exception as e:
                print(f"Error generating top metrics: {str(e)}")
        
        else:
            print("Error: Unknown subcommand or missing arguments")
            print("Usage:")
            print("  metrics list                  - List metrics for current session")
            print("  metrics trend <metric_name>   - Show trend for a specific metric")
            print("  metrics top [count]           - Show top metrics by improvement")
    
    def do_visualize(self, arg):
        """
        Generate visualizations of benchmark data.
        
        Usage:
          visualize summary           - Show system health summary
          visualize compare <metric>  - Compare a metric across projects
        """
        args = arg.split()
        if not args:
            print("Error: Missing subcommand. Use 'visualize summary', 'visualize compare <metric>', etc.")
            return
        
        subcommand = args[0].lower()
        
        # Show system health summary
        if subcommand == "summary":
            self._refresh_data()
            
            print("\n" + "="*70)
            print("SYSTEM HEALTH SUMMARY")
            print("="*70)
            
            # Get all sensors
            sensors = list(self.sensor_registry.sensors.values())
            
            # Collect health scores
            health_scores = []
            for sensor in sensors:
                if hasattr(sensor, 'get_status'):
                    status = sensor.get_status()
                    if "health_score" in status:
                        health_scores.append((status["component_name"], status["health_score"]))
            
            if not health_scores:
                print("No health scores available")
                return
            
            # Calculate overall health
            overall_health = sum(score for _, score in health_scores) / len(health_scores)
            
            # Determine status
            status = "CRITICAL"
            if overall_health >= 90:
                status = "EXCELLENT"
            elif overall_health >= 75:
                status = "GOOD"
            elif overall_health >= 60:
                status = "FAIR"
            elif overall_health >= 40:
                status = "POOR"
            
            print(f"\nOverall System Health: {overall_health:.1f}/100 - {status}")
            
            # Show component health
            print("\nComponent Health:")
            for component, score in sorted(health_scores, key=lambda x: x[1]):
                # Create a simple bar chart
                bar_width = 40
                filled = int(score / 100 * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                
                # Determine status
                comp_status = "CRITICAL"
                if score >= 90:
                    comp_status = "EXCELLENT"
                elif score >= 75:
                    comp_status = "GOOD"
                elif score >= 60:
                    comp_status = "FAIR"
                elif score >= 40:
                    comp_status = "POOR"
                
                print(f"  {component:<20} [{bar}] {score:.1f}/100 - {comp_status}")
            
            # Show error summary
            print(f"\nActive Errors: {len(self.cached_errors)}")
            severity_counts = {}
            for report in self.cached_errors:
                severity = report.get("severity", "UNKNOWN")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            for severity, count in severity_counts.items():
                print(f"  {severity}: {count}")
        
        # Compare a metric across projects
        elif subcommand == "compare" and len(args) > 1:
            metric_name = args[1]
            
            # Get all projects
            projects = self.benchmark_storage.get_all_projects()
            
            if not projects:
                print("No projects found")
                return
            
            print(f"\nComparing '{metric_name}' Across Projects:")
            
            # Collect data for each project
            project_data = {}
            for project in projects:
                sessions = self.benchmark_storage.get_project_sessions(project)
                if not sessions:
                    continue
                
                # Use the most recent session
                session_id = sessions[-1]
                
                try:
                    metrics_list = self.benchmark_storage.get_session_metrics(session_id)
                    if not metrics_list:
                        continue
                    
                    # Get most recent metrics
                    metrics = metrics_list[-1].get("metrics", {})
                    
                    if metric_name in metrics:
                        project_data[project] = metrics[metric_name]
                except Exception as e:
                    print(f"Error retrieving data for project '{project}': {str(e)}")
            
            if not project_data:
                print(f"No data found for metric '{metric_name}'")
                return
            
            # Find min and max for scaling
            min_value = min(project_data.values())
            max_value = max(project_data.values())
            
            # Print comparison
            width = 40  # Width of the ASCII chart
            
            for project, value in sorted(project_data.items(), key=lambda x: x[1], reverse=True):
                # Normalize the value
                if min_value == max_value:
                    bar_width = width // 2
                else:
                    bar_width = int((value - min_value) / (max_value - min_value) * width)
                
                bar = "#" * bar_width
                print(f"  {project:<20} {bar} {value:.2f}")
        
        else:
            print("Error: Unknown subcommand or missing arguments")
            print("Usage:")
            print("  visualize summary           - Show system health summary")
            print("  visualize compare <metric>  - Compare a metric across projects")


# Main function to run the shell
def main():
    """Run the auditor shell interface."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Auditor Shell Interface")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {str(e)}")
    
    # Create and run the shell
    shell = AuditorShellInterface(config=config)
    shell.cmdloop()


if __name__ == "__main__":
    main()
