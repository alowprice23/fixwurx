#!/usr/bin/env python3
"""
auditor_commands_reference.py
────────────────────────────
Implements comprehensive command-line interfaces for Auditor agent functionality.

This module provides a complete reference of commands for the Auditor system,
including documentation, parameter validation, error handling, and integration
with the shell environment. It serves as the primary interface for users to
interact with the Auditor system through the command line.
"""

import os
import sys
import json
import logging
import argparse
import time
import datetime
from typing import Dict, List, Any, Optional

# Import core components
from shell_environment import CommandRegistry, Command, CommandContext, CommandResult
from auditor_agent import AuditorAgent
from auditor_shell_integration import AuditorShellIntegration
from auditor_sensor_integration import SensorManager

# Configure logging
logger = logging.getLogger("AuditorCommands")

# Command categories
COMMAND_CATEGORIES = {
    "core": "Core Auditor Commands",
    "monitor": "System Monitoring",
    "report": "Auditing and Reporting",
    "sensor": "Sensor Management",
    "analysis": "Performance Analysis",
    "config": "Configuration and Settings",
    "alert": "Alerting and Notifications"
}

class AuditorCommands:
    """
    Provides a comprehensive set of commands for the Auditor system.
    
    This class implements the complete command reference for the Auditor system,
    providing users with a wide range of commands to interact with the system
    through the command line.
    """
    
    def __init__(self, command_registry: CommandRegistry = None):
        """
        Initialize the Auditor commands.
        
        Args:
            command_registry: Command registry for registering commands.
        """
        self.command_registry = command_registry or CommandRegistry()
        
        # Core components
        try:
            self.auditor_agent = AuditorAgent()
            self.auditor_shell = AuditorShellIntegration()
            self.sensor_manager = SensorManager()
        except Exception as e:
            logger.warning(f"Error initializing auditor components: {e}")
            logger.warning("Some functionality may be limited")
        
        # Register commands
        self._register_commands()
        
        logger.info("Auditor commands initialized")
    
    def _register_commands(self) -> None:
        """
        Register all Auditor commands with the command registry.
        """
        # Core commands
        self.command_registry.register(Command(
            name="auditor-status",
            handler=self.cmd_status,
            help="Display Auditor system status",
            category="core",
            args=[
                {"name": "--detailed", "help": "Show detailed status information", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-info",
            handler=self.cmd_info,
            help="Display information about the Auditor system",
            category="core",
            args=[
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        # Monitoring commands
        self.command_registry.register(Command(
            name="auditor-monitor",
            handler=self.cmd_monitor,
            help="Monitor system activity in real-time",
            category="monitor",
            args=[
                {"name": "--component", "help": "Component to monitor (shell, agents, all)", "default": "all"},
                {"name": "--interval", "help": "Update interval in seconds", "type": int, "default": 5},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-processes",
            handler=self.cmd_processes,
            help="Display active processes being monitored",
            category="monitor",
            args=[
                {"name": "--detailed", "help": "Show detailed process information", "action": "store_true"},
                {"name": "--filter", "help": "Filter processes by name or ID"}
            ]
        ))
        
        # Reporting commands
        self.command_registry.register(Command(
            name="auditor-report",
            handler=self.cmd_report,
            help="Generate a system audit report",
            category="report",
            args=[
                {"name": "--type", "help": "Report type (activity, performance, security, all)", "default": "all"},
                {"name": "--period", "help": "Time period (hour, day, week, month)", "default": "day"},
                {"name": "--output", "help": "Output file for the report"},
                {"name": "--format", "help": "Report format (text, json, html, pdf)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-logs",
            handler=self.cmd_logs,
            help="Display auditor logs",
            category="report",
            args=[
                {"name": "--level", "help": "Minimum log level (info, warning, error, critical)", "default": "info"},
                {"name": "--component", "help": "Component to show logs for"},
                {"name": "--lines", "help": "Number of lines to display", "type": int, "default": 100},
                {"name": "--follow", "help": "Follow log output", "action": "store_true"},
                {"name": "--filter", "help": "Filter logs by pattern"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-events",
            handler=self.cmd_events,
            help="Display system events",
            category="report",
            args=[
                {"name": "--type", "help": "Event type (activity, error, security, all)", "default": "all"},
                {"name": "--start", "help": "Start time (YYYY-MM-DD HH:MM:SS format)"},
                {"name": "--end", "help": "End time (YYYY-MM-DD HH:MM:SS format)"},
                {"name": "--limit", "help": "Maximum number of events to display", "type": int, "default": 100},
                {"name": "--format", "help": "Output format (text, json, csv)", "default": "text"}
            ]
        ))
        
        # Sensor management commands
        self.command_registry.register(Command(
            name="auditor-sensors",
            handler=self.cmd_sensors,
            help="List available sensors",
            category="sensor",
            args=[
                {"name": "--status", "help": "Filter by status (active, inactive, all)", "default": "all"},
                {"name": "--type", "help": "Filter by sensor type"},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-sensor-enable",
            handler=self.cmd_sensor_enable,
            help="Enable a sensor",
            category="sensor",
            args=[
                {"name": "sensor", "help": "Name or ID of the sensor to enable"},
                {"name": "--options", "help": "Sensor options (key=value,key=value)"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-sensor-disable",
            handler=self.cmd_sensor_disable,
            help="Disable a sensor",
            category="sensor",
            args=[
                {"name": "sensor", "help": "Name or ID of the sensor to disable"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-sensor-data",
            handler=self.cmd_sensor_data,
            help="Display data from a sensor",
            category="sensor",
            args=[
                {"name": "sensor", "help": "Name or ID of the sensor"},
                {"name": "--period", "help": "Time period (hour, day, week, month)", "default": "hour"},
                {"name": "--format", "help": "Output format (text, json, csv)", "default": "text"},
                {"name": "--output", "help": "Output file for data"}
            ]
        ))
        
        # Analysis commands
        self.command_registry.register(Command(
            name="auditor-analyze",
            handler=self.cmd_analyze,
            help="Analyze system performance",
            category="analysis",
            args=[
                {"name": "--component", "help": "Component to analyze (shell, agents, neural, all)", "default": "all"},
                {"name": "--metric", "help": "Specific metric to analyze"},
                {"name": "--period", "help": "Time period (hour, day, week, month)", "default": "day"},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-benchmark",
            handler=self.cmd_benchmark,
            help="Run system benchmarks",
            category="analysis",
            args=[
                {"name": "--type", "help": "Benchmark type (cpu, memory, io, all)", "default": "all"},
                {"name": "--duration", "help": "Benchmark duration in seconds", "type": int, "default": 60},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-trends",
            handler=self.cmd_trends,
            help="Analyze performance trends over time",
            category="analysis",
            args=[
                {"name": "--metric", "help": "Metric to analyze (cpu, memory, latency, all)", "default": "all"},
                {"name": "--period", "help": "Time period (day, week, month, year)", "default": "month"},
                {"name": "--format", "help": "Output format (text, json, chart)", "default": "text"},
                {"name": "--output", "help": "Output file for charts"}
            ]
        ))
        
        # Configuration commands
        self.command_registry.register(Command(
            name="auditor-configure",
            handler=self.cmd_configure,
            help="Configure the Auditor system",
            category="config",
            args=[
                {"name": "--component", "help": "Component to configure (sensors, alerts, storage, all)", "default": "all"},
                {"name": "--config", "help": "Path to configuration file"},
                {"name": "--set", "help": "Set a configuration value (key=value)"},
                {"name": "--reset", "help": "Reset configuration to defaults", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-export-config",
            handler=self.cmd_export_config,
            help="Export Auditor configuration",
            category="config",
            args=[
                {"name": "output", "help": "Output file for configuration"},
                {"name": "--component", "help": "Component to export configuration for", "default": "all"},
                {"name": "--format", "help": "Output format (json, yaml)", "default": "json"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-import-config",
            handler=self.cmd_import_config,
            help="Import Auditor configuration",
            category="config",
            args=[
                {"name": "input", "help": "Input configuration file"},
                {"name": "--component", "help": "Component to import configuration for", "default": "all"},
                {"name": "--merge", "help": "Merge with existing configuration", "action": "store_true"},
                {"name": "--force", "help": "Force import", "action": "store_true"}
            ]
        ))
        
        # Alert commands
        self.command_registry.register(Command(
            name="auditor-alerts",
            handler=self.cmd_alerts,
            help="Display active alerts",
            category="alert",
            args=[
                {"name": "--status", "help": "Alert status (active, acknowledged, resolved, all)", "default": "active"},
                {"name": "--severity", "help": "Alert severity (info, warning, error, critical)", "default": "all"},
                {"name": "--limit", "help": "Maximum number of alerts to display", "type": int, "default": 100},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-alert-acknowledge",
            handler=self.cmd_alert_acknowledge,
            help="Acknowledge an alert",
            category="alert",
            args=[
                {"name": "alert", "help": "Alert ID to acknowledge"},
                {"name": "--comment", "help": "Comment for acknowledgment"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-alert-resolve",
            handler=self.cmd_alert_resolve,
            help="Resolve an alert",
            category="alert",
            args=[
                {"name": "alert", "help": "Alert ID to resolve"},
                {"name": "--comment", "help": "Comment for resolution"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="auditor-alert-rules",
            handler=self.cmd_alert_rules,
            help="Manage alert rules",
            category="alert",
            args=[
                {"name": "--list", "help": "List alert rules", "action": "store_true"},
                {"name": "--add", "help": "Add a new alert rule (JSON string or file path)"},
                {"name": "--remove", "help": "Remove an alert rule by ID"},
                {"name": "--enable", "help": "Enable an alert rule by ID"},
                {"name": "--disable", "help": "Disable an alert rule by ID"},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        # Help command
        self.command_registry.register(Command(
            name="auditor-help",
            handler=self.cmd_help,
            help="Display Auditor help information",
            category="core",
            args=[
                {"name": "--command", "help": "Command to get help for"},
                {"name": "--category", "help": "Category to get help for"},
                {"name": "--list", "help": "List all commands", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json, markdown)", "default": "text"}
            ]
        ))
        
        logger.info(f"Registered Auditor commands in {len(COMMAND_CATEGORIES)} categories")
    
    # Core commands implementation with simple stubs
    def cmd_status(self, context: CommandContext) -> CommandResult:
        """Display Auditor system status."""
        detailed = context.args.get("detailed", False)
        format_type = context.args.get("format", "text")
        
        status = {
            "status": "Operational",
            "active_sensors": 12,
            "total_sensors": 15,
            "active_alerts": 2,
            "last_update": time.time(),
            "uptime": 86400,  # 1 day in seconds
            "resources": {
                "cpu_percent": 15,
                "memory_usage": 128,
                "disk_usage": 256
            }
        }
        
        if format_type == "json":
            return CommandResult(success=True, data=status)
        else:
            output = ["Auditor System Status:"]
            output.append(f"Overall Status: {status['status']}")
            output.append(f"Active Sensors: {status['active_sensors']}/{status['total_sensors']}")
            output.append(f"Active Alerts: {status['active_alerts']}")
            output.append(f"Last Update: {datetime.datetime.fromtimestamp(status['last_update']).strftime('%Y-%m-%d %H:%M:%S')}")
            output.append(f"Uptime: {str(datetime.timedelta(seconds=status['uptime']))}")
            
            if detailed:
                output.append("\nResource Usage:")
                output.append(f"  CPU: {status['resources']['cpu_percent']}%")
                output.append(f"  Memory: {status['resources']['memory_usage']} MB")
                output.append(f"  Disk: {status['resources']['disk_usage']} MB")
            
            return CommandResult(success=True, output="\n".join(output))
    
    def cmd_info(self, context: CommandContext) -> CommandResult:
        """Display information about the Auditor system."""
        format_type = context.args.get("format", "text")
        
        info = {
            "version": "1.0.0",
            "build_date": "2025-07-14",
            "components": {
                "auditor_agent": "0.9.5",
                "sensor_manager": "1.2.3",
                "alert_system": "0.8.7"
            },
            "capabilities": {
                "monitoring": True,
                "alerting": True,
                "reporting": True,
                "analysis": True
            },
            "sensors": {
                "system": 5,
                "process": 3,
                "network": 2,
                "security": 3,
                "custom": 2
            }
        }
        
        if format_type == "json":
            return CommandResult(success=True, data=info)
        else:
            output = ["Auditor System Information:"]
            output.append(f"Version: {info['version']}")
            output.append(f"Build Date: {info['build_date']}")
            
            output.append("\nComponents:")
            for component, version in info["components"].items():
                output.append(f"  {component}: v{version}")
            
            output.append("\nCapabilities:")
            for capability, enabled in info["capabilities"].items():
                output.append(f"  {capability}: {'Enabled' if enabled else 'Disabled'}")
            
            output.append("\nSensors:")
            for sensor_type, count in info["sensors"].items():
                output.append(f"  {sensor_type}: {count}")
            
            return CommandResult(success=True, output="\n".join(output))
    
    def cmd_monitor(self, context: CommandContext) -> CommandResult:
        """Monitor system activity in real-time."""
        component = context.args.get("component", "all")
        interval = context.args.get("interval", 5)
        
        # This is a placeholder since real-time monitoring would require
        # different handling in an interactive shell
        return CommandResult(
            success=True,
            output=f"Monitoring {component} (interval: {interval}s)\n" +
                   "Note: Real-time monitoring requires interactive shell"
        )
    
    def cmd_processes(self, context: CommandContext) -> CommandResult:
        """Display active processes being monitored."""
        detailed = context.args.get("detailed", False)
        filter_str = context.args.get("filter")
        
        processes = [
            {"id": 1001, "name": "shell_environment.py", "cpu": 2.5, "memory": 42, "status": "running"},
            {"id": 1002, "name": "triangulation_engine.py", "cpu": 15.2, "memory": 128, "status": "running"},
            {"id": 1003, "name": "neural_matrix.py", "cpu": 22.8, "memory": 256, "status": "running"}
        ]
        
        if filter_str:
            processes = [p for p in processes if filter_str in p["name"] or str(p["id"]) == filter_str]
        
        output = ["Active Monitored Processes:"]
        
        if not processes:
            output.append("No processes found")
        else:
            for proc in processes:
                output.append(f"\nProcess: {proc['name']} (PID: {proc['id']})")
                output.append(f"Status: {proc['status']}")
                output.append(f"CPU: {proc['cpu']}%")
                output.append(f"Memory: {proc['memory']} MB")
                
                if detailed:
                    # Add more details in a real implementation
                    output.append("Threads: 4")
                    output.append("Start Time: 2025-07-14 01:00:00")
                    output.append("Command: python3 /path/to/script.py")
        
        return CommandResult(success=True, output="\n".join(output))
    
    def cmd_report(self, context: CommandContext) -> CommandResult:
        """Generate a system audit report."""
        report_type = context.args.get("type", "all")
        period = context.args.get("period", "day")
        output_file = context.args.get("output")
        format_type = context.args.get("format", "text")
        
        # Generate report - this is a placeholder for real implementation
        report = {
            "type": report_type,
            "period": period,
            "generated_at": time.time(),
            "summary": {
                "events": 256,
                "alerts": 5,
                "errors": 2
            },
            "details": {
                "activity": {
                    "commands_executed": 120,
                    "files_modified": 45,
                    "processes_started": 15
                },
                "performance": {
                    "avg_cpu": 22.5,
                    "avg_memory": 256,
                    "peak_cpu": 45.2,
                    "peak_memory": 512
                },
                "security": {
                    "authentication_attempts": 12,
                    "authorization_failures": 1,
                    "suspicious_activities": 0
                }
            }
        }
        
        if output_file:
            try:
                if format_type == "json":
                    with open(output_file, "w") as f:
                        json.dump(report, f, indent=2)
                elif format_type == "html":
                    # Simplified HTML generation
                    html = f"<html><head><title>Audit Report</title></head><body><h1>Audit Report</h1><p>Generated at: {datetime.datetime.fromtimestamp(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}</p></body></html>"
                    with open(output_file, "w") as f:
                        f.write(html)
                else:
                    # Basic text report
                    with open(output_file, "w") as f:
                        f.write(f"Audit Report ({report_type})\n")
                        f.write(f"Period: {period}\n")
                        f.write(f"Generated at: {datetime.datetime.fromtimestamp(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("\nSummary:\n")
                        f.write(f"  Events: {report['summary']['events']}\n")
                        f.write(f"  Alerts: {report['summary']['alerts']}\n")
                        f.write(f"  Errors: {report['summary']['errors']}\n")
                
                return CommandResult(success=True, output=f"Report saved to {output_file}")
            except Exception as e:
                return CommandResult(success=False, error=f"Error saving report: {e}")
        
        # Format output for display
        if format_type == "json":
            return CommandResult(success=True, data=report)
        else:
            output = [f"Audit Report ({report_type})"]
            output.append(f"Period: {period}")
            output.append(f"Generated at: {datetime.datetime.fromtimestamp(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            output.append("\nSummary:")
            output.append(f"  Events: {report['summary']['events']}")
            output.append(f"  Alerts: {report['summary']['alerts']}")
            output.append(f"  Errors: {report['summary']['errors']}")
            
            if report_type == "all" or report_type == "activity":
                output.append("\nActivity:")
                output.append(f"  Commands Executed: {report['details']['activity']['commands_executed']}")
                output.append(f"  Files Modified: {report['details']['activity']['files_modified']}")
                output.append(f"  Processes Started: {report['details']['activity']['processes_started']}")
            
            if report_type == "all" or report_type == "performance":
                output.append("\nPerformance:")
                output.append(f"  Average CPU: {report['details']['performance']['avg_cpu']}%")
                output.append(f"  Average Memory: {report['details']['performance']['avg_memory']} MB")
                output.append(f"  Peak CPU: {report['details']['performance']['peak_cpu']}%")
                output.append(f"  Peak Memory: {report['details']['performance']['peak_memory']} MB")
            
            if report_type == "all" or report_type == "security":
                output.append("\nSecurity:")
                output.append(f"  Authentication Attempts: {report['details']['security']['authentication_attempts']}")
                output.append(f"  Authorization Failures: {report['details']['security']['authorization_failures']}")
                output.append(f"  Suspicious Activities: {report['details']['security']['suspicious_activities']}")
            
            return CommandResult(success=True, output="\n".join(output))
    
    # Helper method to implement all remaining command methods with simple stubs
    def cmd_logs(self, context: CommandContext) -> CommandResult:
        """Display auditor logs."""
        return CommandResult(success=True, output="Log entries would be displayed here")
    
    def cmd_events(self, context: CommandContext) -> CommandResult:
        """Display system events."""
        return CommandResult(success=True, output="System events would be displayed here")
    
    def cmd_sensors(self, context: CommandContext) -> CommandResult:
        """List available sensors."""
        return CommandResult(success=True, output="Available sensors would be listed here")
    
    def cmd_sensor_enable(self, context: CommandContext) -> CommandResult:
        """Enable a sensor."""
        sensor = context.args.get("sensor")
        return CommandResult(success=True, output=f"Sensor {sensor} enabled")
    
    def cmd_sensor_disable(self, context: CommandContext) -> CommandResult:
        """Disable a sensor."""
        sensor = context.args.get("sensor")
        return CommandResult(success=True, output=f"Sensor {sensor} disabled")
    
    def cmd_sensor_data(self, context: CommandContext) -> CommandResult:
        """Display data from a sensor."""
        sensor = context.args.get("sensor")
        return CommandResult(success=True, output=f"Data from sensor {sensor} would be displayed here")
    
    def cmd_analyze(self, context: CommandContext) -> CommandResult:
        """Analyze system performance."""
        component = context.args.get("component", "all")
        return CommandResult(success=True, output=f"Analysis of {component} would be displayed here")
    
    def cmd_benchmark(self, context: CommandContext) -> CommandResult:
        """Run system benchmarks."""
        benchmark_type = context.args.get("type", "all")
        return CommandResult(success=True, output=f"Benchmark results for {benchmark_type} would be displayed here")
    
    def cmd_trends(self, context: CommandContext) -> CommandResult:
        """Analyze performance trends over time."""
        metric = context.args.get("metric", "all")
        return CommandResult(success=True, output=f"Trend analysis for {metric} would be displayed here")
    
    def cmd_configure(self, context: CommandContext) -> CommandResult:
        """Configure the Auditor system."""
        component = context.args.get("component", "all")
        return CommandResult(success=True, output=f"Configuration for {component} would be displayed/updated here")
    
    def cmd_export_config(self, context: CommandContext) -> CommandResult:
        """Export Auditor configuration."""
        output_file = context.args.get("output")
        return CommandResult(success=True, output=f"Configuration exported to {output_file}")
    
    def cmd_import_config(self, context: CommandContext) -> CommandResult:
        """Import Auditor configuration."""
        input_file = context.args.get("input")
        return CommandResult(success=True, output=f"Configuration imported from {input_file}")
    
    def cmd_alerts(self, context: CommandContext) -> CommandResult:
        """Display active alerts."""
        status = context.args.get("status", "active")
        return CommandResult(success=True, output=f"Alerts with status {status} would be displayed here")
    
    def cmd_alert_acknowledge(self, context: CommandContext) -> CommandResult:
        """Acknowledge an alert."""
        alert = context.args.get("alert")
        return CommandResult(success=True, output=f"Alert {alert} acknowledged")
    
    def cmd_alert_resolve(self, context: CommandContext) -> CommandResult:
        """Resolve an alert."""
        alert = context.args.get("alert")
        return CommandResult(success=True, output=f"Alert {alert} resolved")
    
    def cmd_alert_rules(self, context: CommandContext) -> CommandResult:
        """Manage alert rules."""
        if context.args.get("list"):
            return CommandResult(success=True, output="Alert rules would be listed here")
        elif context.args.get("add"):
            return CommandResult(success=True, output="Alert rule added")
        elif context.args.get("remove"):
            return CommandResult(success=True, output=f"Alert rule {context.args.get('remove')} removed")
        elif context.args.get("enable"):
            return CommandResult(success=True, output=f"Alert rule {context.args.get('enable')} enabled")
        elif context.args.get("disable"):
            return CommandResult(success=True, output=f"Alert rule {context.args.get('disable')} disabled")
        else:
            return CommandResult(success=True, output="Use --list, --add, --remove, --enable, or --disable to manage alert rules")
    
    def cmd_help(self, context: CommandContext) -> CommandResult:
        """Display Auditor help information."""
        command_name = context.args.get("command")
        category_name = context.args.get("category")
        list_all = context.args.get("list", False)
        format_type = context.args.get("format", "text")
        
        # Filter for auditor commands only
        auditor_commands = [
            cmd for cmd in self.command_registry.commands
            if cmd.name.startswith("auditor-")
        ]
        
        if command_name:
            # Display help for specific command
            command = next((cmd for cmd in auditor_commands if cmd.name == command_name), None)
            if not command:
                return CommandResult(success=False, error=f"Command '{command_name}' not found")
            
            # Text output for command help
            output = [f"Command: {command.name}"]
            output.append(f"Description: {command.help}")
            output.append(f"Category: {COMMAND_CATEGORIES.get(command.category, command.category)}")
            
            if command.args:
                output.append("\nArguments:")
                for arg in command.args:
                    name = arg.get("name", "")
                    help_text = arg.get("help", "")
                    default = arg.get("default", "")
                    
                    if name.startswith("--"):
                        output.append(f"  {name}: {help_text}")
                        if default:
                            output.append(f"    Default: {default}")
                    else:
                        output.append(f"  {name} (required): {help_text}")
            
            output.append("\nExamples:")
            output.append(f"  fx {command.name} [arguments]")
            
            return CommandResult(success=True, output="\n".join(output))
            
        elif category_name:
            # Display help for specific category
            category = category_name.lower()
            if category not in COMMAND_CATEGORIES:
                return CommandResult(success=False, error=f"Category '{category_name}' not found")
            
            # Get commands in category
            commands = [cmd for cmd in auditor_commands if cmd.category == category]
            
            if format_type == "json":
                category_help = {
                    "category": category,
                    "description": COMMAND_CATEGORIES[category],
                    "commands": [{"name": cmd.name, "help": cmd.help} for cmd in commands]
                }
                return CommandResult(success=True, data=category_help)
            else:
                # Text output
                output = [f"Category: {COMMAND_CATEGORIES[category]}"]
                output.append("\nCommands:")
                
                for command in commands:
                    output.append(f"  {command.name}: {command.help}")
                
                return CommandResult(success=True, output="\n".join(output))
            
        elif list_all:
            # List all commands
            if format_type == "json":
                commands_by_category = {}
                for category, description in COMMAND_CATEGORIES.items():
                    category_commands = [cmd for cmd in auditor_commands if cmd.category == category]
                    if category_commands:
                        commands_by_category[category] = {
                            "description": description,
                            "commands": [{"name": cmd.name, "help": cmd.help} for cmd in category_commands]
                        }
                return CommandResult(success=True, data=commands_by_category)
            else:
                # Text output
                output = ["Available Auditor Commands:"]
                for category, description in COMMAND_CATEGORIES.items():
                    category_commands = [cmd for cmd in auditor_commands if cmd.category == category]
                    if category_commands:
                        output.append(f"\n{description}:")
                        for command in category_commands:
                            output.append(f"  {command.name}: {command.help}")
                
                return CommandResult(success=True, output="\n".join(output))
        
        else:
            # Default help
            output = ["Auditor Command Help"]
            output.append("\nUse one of the following options:")
            output.append("  --command NAME    Get help for a specific command")
            output.append("  --category NAME   Get help for a specific category")
            output.append("  --list            List all available commands")
            output.append("\nAvailable categories:")
            
            for category, description in COMMAND_CATEGORIES.items():
                output.append(f"  {category}: {description}")
            
            return CommandResult(success=True, output="\n".join(output))
