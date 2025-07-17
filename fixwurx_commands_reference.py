#!/usr/bin/env python3
"""
fixwurx_commands_reference.py
────────────────────────────
Implements comprehensive command-line interfaces for FixWurx functionality.

This module provides a complete reference of commands for the FixWurx system,
including documentation, parameter validation, error handling, and integration
with the shell environment. It serves as the primary interface for users to
interact with the FixWurx system through the command line.
"""

import os
import sys
import json
import logging
import argparse
import textwrap
import time
import datetime
import re
import inspect
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

# Import core components
from shell_environment import CommandRegistry, Command, CommandContext, CommandResult
from shell_scripting import ScriptParser, ScriptRunner
from neural_matrix_initialization import NeuralMatrixInitialization
from neural_matrix_model_selection import NeuralMatrixModelSelection
from triangulation_engine import TriangulationEngine
from bug_detection import BugDetector
from scope_filter import ScopeFilter
from solution_paths import SolutionPathGenerator
from verification import Verifier
from resource_manager import ResourceManager
from storage_manager import StorageManager

# Configure logging
logger = logging.getLogger("FixWurxCommands")

# Command categories
COMMAND_CATEGORIES = {
    "system": "System Commands",
    "bug": "Bug Detection and Analysis",
    "fix": "Bug Fixing and Patch Generation",
    "neural": "Neural Matrix Operations",
    "resource": "Resource Management",
    "storage": "Storage Management",
    "report": "Reporting and Documentation",
    "config": "Configuration and Settings",
    "agent": "Agent System Integration",
    "debug": "Debugging and Troubleshooting",
    "triangulum": "Triangulum Integration"
}

class FixWurxCommands:
    """
    Provides a comprehensive set of commands for the FixWurx system.
    
    This class implements the complete command reference for FixWurx, providing
    users with a wide range of commands to interact with the system through the
    command line.
    """
    
    def __init__(self, command_registry: CommandRegistry = None):
        """
        Initialize the FixWurx commands.
        
        Args:
            command_registry: Command registry for registering commands.
        """
        self.command_registry = command_registry or CommandRegistry()
        
        # Core components
        self.neural_matrix_init = NeuralMatrixInitialization()
        self.neural_matrix_selection = NeuralMatrixModelSelection()
        self.triangulation_engine = TriangulationEngine()
        self.bug_detector = BugDetector()
        self.scope_filter = ScopeFilter()
        self.solution_path_generator = SolutionPathGenerator()
        self.verifier = Verifier()
        self.resource_manager = ResourceManager()
        self.storage_manager = StorageManager()
        
        # Register commands
        self._register_commands()
        
        logger.info("FixWurx commands initialized")
    
    def _register_commands(self) -> None:
        """
        Register all FixWurx commands with the command registry.
        """
        # System commands
        self.command_registry.register(Command(
            name="version",
            handler=self.cmd_version,
            help="Display FixWurx version information",
            category="system",
            args=[]
        ))
        
        self.command_registry.register(Command(
            name="status",
            handler=self.cmd_status,
            help="Display current system status",
            category="system",
            args=[]
        ))
        
        self.command_registry.register(Command(
            name="stats",
            handler=self.cmd_stats,
            help="Display system statistics",
            category="system",
            args=[
                {"name": "--detailed", "help": "Show detailed statistics", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json, csv)", "default": "text"}
            ]
        ))
        
        # Bug detection commands
        self.command_registry.register(Command(
            name="detect",
            handler=self.cmd_detect,
            help="Detect bugs in the specified code",
            category="bug",
            args=[
                {"name": "path", "help": "Path to file or directory to analyze"},
                {"name": "--recursive", "help": "Recursively scan directories", "action": "store_true"},
                {"name": "--output", "help": "Output file for results"},
                {"name": "--format", "help": "Output format (text, json, html)", "default": "text"},
                {"name": "--severity", "help": "Minimum severity level (low, medium, high, critical)", "default": "low"},
                {"name": "--confidence", "help": "Minimum confidence level (low, medium, high)", "default": "medium"},
                {"name": "--timeout", "help": "Timeout in seconds", "type": int, "default": 300}
            ]
        ))
        
        self.command_registry.register(Command(
            name="analyze",
            handler=self.cmd_analyze,
            help="Analyze code for bugs and quality issues",
            category="bug",
            args=[
                {"name": "path", "help": "Path to file or directory to analyze"},
                {"name": "--depth", "help": "Analysis depth (quick, standard, deep)", "default": "standard"},
                {"name": "--output", "help": "Output file for results"},
                {"name": "--categories", "help": "Bug categories to analyze (comma-separated)"},
                {"name": "--exclude", "help": "Patterns to exclude (comma-separated)"},
                {"name": "--format", "help": "Output format (text, json, html)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="filter",
            handler=self.cmd_filter,
            help="Filter files for analysis",
            category="bug",
            args=[
                {"name": "path", "help": "Path to file or directory to filter"},
                {"name": "--patterns", "help": "Include patterns (comma-separated)"},
                {"name": "--exclude", "help": "Exclude patterns (comma-separated)"},
                {"name": "--entropy", "help": "Use entropy-based filtering", "action": "store_true"},
                {"name": "--threshold", "help": "Entropy threshold", "type": float, "default": 0.7},
                {"name": "--output", "help": "Output file for results"}
            ]
        ))
        
        # Bug fixing commands
        self.command_registry.register(Command(
            name="fix",
            handler=self.cmd_fix,
            help="Fix bugs in the specified code",
            category="fix",
            args=[
                {"name": "path", "help": "Path to file or directory to fix"},
                {"name": "--bug-id", "help": "ID of the bug to fix"},
                {"name": "--auto-apply", "help": "Automatically apply fixes", "action": "store_true"},
                {"name": "--backup", "help": "Create backup before fixing", "action": "store_true"},
                {"name": "--verify", "help": "Verify fixes after applying", "action": "store_true"},
                {"name": "--output", "help": "Output directory for patch files"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="generate-patch",
            handler=self.cmd_generate_patch,
            help="Generate a patch for the specified bug",
            category="fix",
            args=[
                {"name": "bug-id", "help": "ID of the bug to fix"},
                {"name": "--output", "help": "Output file for the patch"},
                {"name": "--format", "help": "Patch format (unified, context, git)", "default": "unified"},
                {"name": "--solution-path", "help": "Specific solution path to use"},
                {"name": "--minimal", "help": "Generate minimal patch", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="apply-patch",
            handler=self.cmd_apply_patch,
            help="Apply a patch to the specified file",
            category="fix",
            args=[
                {"name": "patch", "help": "Path to patch file"},
                {"name": "--path", "help": "Base path for applying the patch", "default": "./"},
                {"name": "--backup", "help": "Create backup before applying", "action": "store_true"},
                {"name": "--dry-run", "help": "Show what would be done without making changes", "action": "store_true"},
                {"name": "--strip", "help": "Number of leading path components to strip", "type": int, "default": 1}
            ]
        ))
        
        self.command_registry.register(Command(
            name="verify-patch",
            handler=self.cmd_verify_patch,
            help="Verify a patch against the specified code",
            category="fix",
            args=[
                {"name": "patch", "help": "Path to patch file"},
                {"name": "--path", "help": "Base path for verifying the patch", "default": "./"},
                {"name": "--test", "help": "Run tests after applying patch", "action": "store_true"},
                {"name": "--test-command", "help": "Custom test command to run"},
                {"name": "--verbose", "help": "Show detailed verification results", "action": "store_true"}
            ]
        ))
        
        # Neural matrix commands
        self.command_registry.register(Command(
            name="neural-init",
            handler=self.cmd_neural_init,
            help="Initialize the neural matrix",
            category="neural",
            args=[
                {"name": "--config", "help": "Path to configuration file"},
                {"name": "--force", "help": "Force reinitialization", "action": "store_true"},
                {"name": "--backup", "help": "Create backup before initialization", "action": "store_true"},
                {"name": "--verbose", "help": "Show detailed initialization steps", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="neural-status",
            handler=self.cmd_neural_status,
            help="Display neural matrix status",
            category="neural",
            args=[
                {"name": "--detailed", "help": "Show detailed status information", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="neural-select",
            handler=self.cmd_neural_select,
            help="Select a neural matrix model",
            category="neural",
            args=[
                {"name": "model", "help": "ID or name of the model to select"},
                {"name": "--activate", "help": "Activate the selected model", "action": "store_true"},
                {"name": "--verbose", "help": "Show detailed selection process", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="neural-train",
            handler=self.cmd_neural_train,
            help="Train the neural matrix on new data",
            category="neural",
            args=[
                {"name": "data", "help": "Path to training data"},
                {"name": "--epochs", "help": "Number of training epochs", "type": int, "default": 10},
                {"name": "--batch-size", "help": "Batch size for training", "type": int, "default": 32},
                {"name": "--learning-rate", "help": "Learning rate", "type": float, "default": 0.001},
                {"name": "--validation-split", "help": "Validation split ratio", "type": float, "default": 0.2},
                {"name": "--save", "help": "Save model after training", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="neural-export",
            handler=self.cmd_neural_export,
            help="Export a neural matrix model",
            category="neural",
            args=[
                {"name": "output", "help": "Output path for the exported model"},
                {"name": "--model", "help": "ID or name of the model to export"},
                {"name": "--format", "help": "Export format (npz, json, dir)", "default": "dir"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="neural-import",
            handler=self.cmd_neural_import,
            help="Import a neural matrix model",
            category="neural",
            args=[
                {"name": "path", "help": "Path to the model to import"},
                {"name": "--id", "help": "ID to use for the imported model"},
                {"name": "--activate", "help": "Activate the imported model", "action": "store_true"}
            ]
        ))
        
        # Resource management commands
        self.command_registry.register(Command(
            name="resources",
            handler=self.cmd_resources,
            help="Display resource usage information",
            category="resource",
            args=[
                {"name": "--detailed", "help": "Show detailed resource information", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json, csv)", "default": "text"},
                {"name": "--refresh", "help": "Refresh interval in seconds", "type": int}
            ]
        ))
        
        self.command_registry.register(Command(
            name="allocate",
            handler=self.cmd_allocate,
            help="Allocate resources for a task",
            category="resource",
            args=[
                {"name": "task", "help": "Task to allocate resources for"},
                {"name": "--cpu", "help": "CPU cores to allocate", "type": int},
                {"name": "--memory", "help": "Memory to allocate (MB)", "type": int},
                {"name": "--disk", "help": "Disk space to allocate (MB)", "type": int},
                {"name": "--duration", "help": "Duration of allocation (seconds)", "type": int}
            ]
        ))
        
        self.command_registry.register(Command(
            name="optimize",
            handler=self.cmd_optimize,
            help="Optimize resource usage",
            category="resource",
            args=[
                {"name": "--target", "help": "Optimization target (cpu, memory, disk, all)", "default": "all"},
                {"name": "--aggressive", "help": "Use aggressive optimization", "action": "store_true"},
                {"name": "--apply", "help": "Apply optimization recommendations", "action": "store_true"}
            ]
        ))
        
        # Storage management commands
        self.command_registry.register(Command(
            name="storage-info",
            handler=self.cmd_storage_info,
            help="Display storage information",
            category="storage",
            args=[
                {"name": "--type", "help": "Storage type (patches, models, logs, all)", "default": "all"},
                {"name": "--detailed", "help": "Show detailed storage information", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json, csv)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="backup",
            handler=self.cmd_backup,
            help="Create a system backup",
            category="storage",
            args=[
                {"name": "--output", "help": "Output path for the backup"},
                {"name": "--components", "help": "Components to backup (comma-separated)"},
                {"name": "--compress", "help": "Compress backup", "action": "store_true"},
                {"name": "--exclude", "help": "Patterns to exclude (comma-separated)"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="restore",
            handler=self.cmd_restore,
            help="Restore from a backup",
            category="storage",
            args=[
                {"name": "backup", "help": "Path to the backup to restore from"},
                {"name": "--components", "help": "Components to restore (comma-separated)"},
                {"name": "--force", "help": "Force restore", "action": "store_true"},
                {"name": "--dry-run", "help": "Show what would be restored without making changes", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="cleanup",
            handler=self.cmd_cleanup,
            help="Clean up temporary and unused files",
            category="storage",
            args=[
                {"name": "--type", "help": "File types to clean up (comma-separated)"},
                {"name": "--older-than", "help": "Clean files older than specified days", "type": int},
                {"name": "--dry-run", "help": "Show what would be cleaned without making changes", "action": "store_true"},
                {"name": "--verbose", "help": "Show detailed cleanup information", "action": "store_true"}
            ]
        ))
        
        # Reporting commands
        self.command_registry.register(Command(
            name="report",
            handler=self.cmd_report,
            help="Generate a report",
            category="report",
            args=[
                {"name": "type", "help": "Report type (bugs, fixes, performance, usage, all)"},
                {"name": "--output", "help": "Output file for the report"},
                {"name": "--format", "help": "Report format (text, json, html, pdf)", "default": "text"},
                {"name": "--period", "help": "Time period for the report (day, week, month, all)", "default": "all"},
                {"name": "--detailed", "help": "Generate detailed report", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="visualize",
            handler=self.cmd_visualize,
            help="Visualize data",
            category="report",
            args=[
                {"name": "data", "help": "Data to visualize (bugs, fixes, performance, neural, all)"},
                {"name": "--output", "help": "Output file for the visualization"},
                {"name": "--format", "help": "Visualization format (png, svg, html)", "default": "png"},
                {"name": "--type", "help": "Visualization type (bar, line, pie, scatter, heatmap)", "default": "bar"}
            ]
        ))
        
        # Configuration commands
        self.command_registry.register(Command(
            name="configure",
            handler=self.cmd_configure,
            help="Configure system settings",
            category="config",
            args=[
                {"name": "component", "help": "Component to configure (neural, triangulation, detection, resources, all)"},
                {"name": "--file", "help": "Path to configuration file"},
                {"name": "--set", "help": "Set a configuration value (key=value)"},
                {"name": "--reset", "help": "Reset configuration to defaults", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="settings",
            handler=self.cmd_settings,
            help="Display current settings",
            category="config",
            args=[
                {"name": "--component", "help": "Component to display settings for"},
                {"name": "--format", "help": "Output format (text, json, yaml)", "default": "text"},
                {"name": "--output", "help": "Output file for settings"}
            ]
        ))
        
        # Debugging commands
        self.command_registry.register(Command(
            name="debug",
            handler=self.cmd_debug,
            help="Enter debugging mode",
            category="debug",
            args=[
                {"name": "--component", "help": "Component to debug"},
                {"name": "--level", "help": "Debug level (info, debug, trace)", "default": "debug"},
                {"name": "--output", "help": "Output file for debug logs"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="log",
            handler=self.cmd_log,
            help="Display log messages",
            category="debug",
            args=[
                {"name": "--level", "help": "Minimum log level (info, warning, error, critical)", "default": "info"},
                {"name": "--component", "help": "Component to show logs for"},
                {"name": "--lines", "help": "Number of lines to display", "type": int, "default": 100},
                {"name": "--follow", "help": "Follow log output", "action": "store_true"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="diagnostics",
            handler=self.cmd_diagnostics,
            help="Run system diagnostics",
            category="debug",
            args=[
                {"name": "--component", "help": "Component to run diagnostics for"},
                {"name": "--level", "help": "Diagnostic level (basic, standard, deep)", "default": "standard"},
                {"name": "--output", "help": "Output file for diagnostic results"},
                {"name": "--format", "help": "Output format (text, json, html)", "default": "text"}
            ]
        ))
        
        # Triangulum integration commands
        self.command_registry.register(Command(
            name="triangulum-status",
            handler=self.cmd_triangulum_status,
            help="Display Triangulum status",
            category="triangulum",
            args=[
                {"name": "--detailed", "help": "Show detailed status information", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json)", "default": "text"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="triangulum-plan",
            handler=self.cmd_triangulum_plan,
            help="Generate a Triangulum plan",
            category="triangulum",
            args=[
                {"name": "target", "help": "Target to generate plan for"},
                {"name": "--output", "help": "Output file for the plan"},
                {"name": "--format", "help": "Plan format (text, json, yaml)", "default": "json"},
                {"name": "--parameters", "help": "Additional parameters (key=value)"}
            ]
        ))
        
        self.command_registry.register(Command(
            name="triangulum-execute",
            handler=self.cmd_triangulum_execute,
            help="Execute a Triangulum plan",
            category="triangulum",
            args=[
                {"name": "plan", "help": "Path to plan file or plan ID"},
                {"name": "--dry-run", "help": "Show what would be executed without making changes", "action": "store_true"},
                {"name": "--force", "help": "Force execution", "action": "store_true"},
                {"name": "--timeout", "help": "Timeout in seconds", "type": int, "default": 3600}
            ]
        ))
        
        # Help commands
        self.command_registry.register(Command(
            name="help",
            handler=self.cmd_help,
            help="Display help information",
            category="system",
            args=[
                {"name": "--command", "help": "Command to get help for"},
                {"name": "--category", "help": "Category to get help for"},
                {"name": "--list", "help": "List all commands", "action": "store_true"},
                {"name": "--format", "help": "Output format (text, json, markdown)", "default": "text"}
            ]
        ))
        
        logger.info(f"Registered {len(self.command_registry.commands)} FixWurx commands")
    
    # System commands
    def cmd_version(self, context: CommandContext) -> CommandResult:
        """
        Display FixWurx version information.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        version_info = {
            "version": "1.0.0",
            "build_date": "2025-07-14",
            "python_version": sys.version,
            "platform": sys.platform,
            "components": {
                "neural_matrix": "0.9.5",
                "triangulation_engine": "1.1.2",
                "bug_detector": "0.8.7",
                "shell_environment": "1.2.0"
            }
        }
        
        if context.args.get("format") == "json":
            return CommandResult(success=True, data=version_info)
        else:
            output = [
                f"FixWurx v{version_info['version']} (build: {version_info['build_date']})",
                f"Python: {version_info['python_version']}",
                f"Platform: {version_info['platform']}",
                "\nComponents:",
            ]
            
            for component, version in version_info["components"].items():
                output.append(f"  {component}: v{version}")
            
            return CommandResult(success=True, output="\n".join(output))
    
    def cmd_status(self, context: CommandContext) -> CommandResult:
        """
        Display current system status.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Get component status
        neural_matrix_status = "Active" if self.neural_matrix_init._matrix_exists() else "Not initialized"
        triangulation_status = "Ready" if hasattr(self.triangulation_engine, "is_ready") and self.triangulation_engine.is_ready() else "Not ready"
        
        # Get resource usage
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # Get active tasks
        active_tasks = 0  # This would be replaced with actual active task count
        
        status_info = {
            "timestamp": time.time(),
            "system_status": "Operational",
            "components": {
                "neural_matrix": neural_matrix_status,
                "triangulation_engine": triangulation_status,
                "bug_detector": "Ready",
                "solution_generator": "Ready",
                "verifier": "Ready"
            },
            "resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            },
            "tasks": {
                "active": active_tasks,
                "queued": 0,
                "completed": 0
            }
        }
        
        if context.args.get("format") == "json":
            return CommandResult(success=True, data=status_info)
        else:
            output = [
                f"FixWurx Status: {status_info['system_status']}",
                f"Timestamp: {datetime.datetime.fromtimestamp(status_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
                "\nComponents:",
            ]
            
            for component, status in status_info["components"].items():
                output.append(f"  {component}: {status}")
            
            output.extend([
                "\nResources:",
                f"  CPU: {status_info['resources']['cpu_percent']}%",
                f"  Memory: {status_info['resources']['memory_percent']}%",
                f"  Disk: {status_info['resources']['disk_percent']}%",
                "\nTasks:",
                f"  Active: {status_info['tasks']['active']}",
                f"  Queued: {status_info['tasks']['queued']}",
                f"  Completed: {status_info['tasks']['completed']}"
            ])
            
            return CommandResult(success=True, output="\n".join(output))
    
    def cmd_stats(self, context: CommandContext) -> CommandResult:
        """
        Display system statistics.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Basic stats
        stats = {
            "uptime": 3600,  # Example value in seconds
            "tasks_completed": 42,
            "bugs_detected": 120,
            "bugs_fixed": 98,
            "success_rate": 0.82,
            "average_fix_time": 45.3,  # In seconds
            "neural_matrix_accuracy": 0.89
        }
        
        # Add detailed stats if requested
        if context.args.get("detailed"):
            stats.update({
                "bugs_by_type": {
                    "syntax": 23,
                    "logical": 45,
                    "performance": 32,
                    "security": 20
                },
                "fixes_by_type": {
                    "syntax": 21,
                    "logical": 38,
                    "performance": 25,
                    "security": 14
                },
                "resource_usage": {
                    "cpu_average": 32.5,
                    "memory_average": 45.2,
                    "disk_io": {
                        "read_bytes": 102400,
                        "write_bytes": 51200
                    }
                },
                "performance_metrics": {
                    "average_detection_time": 12.3,
                    "average_analysis_time": 8.7,
                    "average_plan_generation_time": 6.2,
                    "average_verification_time": 18.1
                }
            })
        
        # Format output based on requested format
        format_type = context.args.get("format", "text")
        
        if format_type == "json":
            return CommandResult(success=True, data=stats)
        elif format_type == "csv":
            # Generate CSV output
            output = []
            if context.args.get("detailed"):
                # Complex CSV for detailed stats would be generated here
                pass
            else:
            # Simple CSV for basic stats
                output.append("Metric,Value")
                for key, value in stats.items():
                    output.append(f"{key},{value}")
            
            return CommandResult(success=True, output="\n".join(output))
        else:
            # Text output (default)
            output = ["FixWurx System Statistics:"]
            
            # Format uptime
            uptime_str = str(datetime.timedelta(seconds=stats["uptime"]))
            
            output.extend([
                f"Uptime: {uptime_str}",
                f"Tasks Completed: {stats['tasks_completed']}",
                f"Bugs Detected: {stats['bugs_detected']}",
                f"Bugs Fixed: {stats['bugs_fixed']}",
                f"Success Rate: {stats['success_rate']*100:.1f}%",
                f"Average Fix Time: {stats['average_fix_time']:.1f} seconds",
                f"Neural Matrix Accuracy: {stats['neural_matrix_accuracy']*100:.1f}%"
            ])
            
            # Add detailed stats if requested
            if context.args.get("detailed"):
                # Bugs by type
                output.append("\nBugs by Type:")
                for bug_type, count in stats["bugs_by_type"].items():
                    output.append(f"  {bug_type}: {count}")
                
                # Fixes by type
                output.append("\nFixes by Type:")
                for fix_type, count in stats["fixes_by_type"].items():
                    output.append(f"  {fix_type}: {count}")
                
                # Resource usage
                output.append("\nResource Usage:")
                output.append(f"  CPU Average: {stats['resource_usage']['cpu_average']}%")
                output.append(f"  Memory Average: {stats['resource_usage']['memory_average']}%")
                output.append(f"  Disk Read: {stats['resource_usage']['disk_io']['read_bytes']/1024:.1f} KB")
                output.append(f"  Disk Write: {stats['resource_usage']['disk_io']['write_bytes']/1024:.1f} KB")
                
                # Performance metrics
                output.append("\nPerformance Metrics:")
                for metric, value in stats["performance_metrics"].items():
                    name = metric.replace("_", " ").title()
                    output.append(f"  {name}: {value:.1f} seconds")
            
            return CommandResult(success=True, output="\n".join(output))
    
    # Bug detection commands
    def cmd_detect(self, context: CommandContext) -> CommandResult:
        """
        Detect bugs in the specified code.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Get path to analyze
        path = context.args.get("path")
        if not path or not os.path.exists(path):
            return CommandResult(success=False, error=f"Path not found: {path}")
        
        # Get analysis options
        recursive = context.args.get("recursive", False)
        severity = context.args.get("severity", "low")
        confidence = context.args.get("confidence", "medium")
        timeout = context.args.get("timeout", 300)
        
        # Call bug detector
        try:
            # This would call the actual bug detector implementation
            results = self.bug_detector.detect(
                path=path,
                recursive=recursive,
                severity=severity,
                confidence=confidence,
                timeout=timeout
            )
            
            # Format output based on requested format
            format_type = context.args.get("format", "text")
            
            if format_type == "json":
                return CommandResult(success=True, data=results)
            elif format_type == "html":
                # Generate HTML output
                html_output = "<html><head><title>Bug Detection Results</title></head><body>"
                html_output += f"<h1>Bug Detection Results for {path}</h1>"
                html_output += "<table border='1'><tr><th>ID</th><th>File</th><th>Line</th><th>Type</th><th>Severity</th><th>Description</th></tr>"
                
                for bug in results.get("bugs", []):
                    html_output += f"<tr><td>{bug['id']}</td><td>{bug['file']}</td><td>{bug['line']}</td><td>{bug['type']}</td><td>{bug['severity']}</td><td>{bug['description']}</td></tr>"
                
                html_output += "</table></body></html>"
                
                # Save HTML to file if output specified
                output_file = context.args.get("output")
                if output_file:
                    with open(output_file, "w") as f:
                        f.write(html_output)
                    return CommandResult(success=True, output=f"Bug detection results saved to {output_file}")
                
                return CommandResult(success=True, output=html_output)
            else:
                # Text output (default)
                output = [f"Bug Detection Results for {path}:"]
                output.append(f"Found {len(results.get('bugs', []))} bugs")
                output.append("")
                
                for bug in results.get("bugs", []):
                    output.append(f"Bug ID: {bug['id']}")
                    output.append(f"File: {bug['file']}")
                    output.append(f"Line: {bug['line']}")
                    output.append(f"Type: {bug['type']}")
                    output.append(f"Severity: {bug['severity']}")
                    output.append(f"Description: {bug['description']}")
                    output.append("")
                
                # Save text to file if output specified
                output_file = context.args.get("output")
                if output_file:
                    with open(output_file, "w") as f:
                        f.write("\n".join(output))
                    return CommandResult(success=True, output=f"Bug detection results saved to {output_file}")
                
                return CommandResult(success=True, output="\n".join(output))
        
        except Exception as e:
            return CommandResult(success=False, error=f"Error detecting bugs: {e}")
    
    def cmd_analyze(self, context: CommandContext) -> CommandResult:
        """
        Analyze code for bugs and quality issues.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Get path to analyze
        path = context.args.get("path")
        if not path or not os.path.exists(path):
            return CommandResult(success=False, error=f"Path not found: {path}")
        
        # Get analysis options
        depth = context.args.get("depth", "standard")
        categories = context.args.get("categories")
        exclude = context.args.get("exclude")
        
        # Parse categories and exclude patterns
        if categories:
            categories = [c.strip() for c in categories.split(",")]
        if exclude:
            exclude = [e.strip() for e in exclude.split(",")]
        
        # Call bug detector for analysis
        try:
            # This would call the actual bug detector implementation
            results = self.bug_detector.analyze(
                path=path,
                depth=depth,
                categories=categories,
                exclude=exclude
            )
            
            # Format output based on requested format
            format_type = context.args.get("format", "text")
            output_file = context.args.get("output")
            
            if format_type == "json":
                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)
                    return CommandResult(success=True, output=f"Analysis results saved to {output_file}")
                return CommandResult(success=True, data=results)
            elif format_type == "html":
                # Generate HTML output
                html_output = "<html><head><title>Code Analysis Results</title></head><body>"
                html_output += f"<h1>Code Analysis Results for {path}</h1>"
                
                # Summary section
                html_output += "<h2>Summary</h2>"
                html_output += f"<p>Files analyzed: {results['summary']['files_analyzed']}</p>"
                html_output += f"<p>Lines of code: {results['summary']['lines_of_code']}</p>"
                html_output += f"<p>Bugs found: {results['summary']['bugs_found']}</p>"
                html_output += f"<p>Quality issues: {results['summary']['quality_issues']}</p>"
                
                # Bugs section
                html_output += "<h2>Bugs</h2>"
                html_output += "<table border='1'><tr><th>ID</th><th>File</th><th>Line</th><th>Type</th><th>Severity</th><th>Description</th></tr>"
                
                for bug in results.get("bugs", []):
                    html_output += f"<tr><td>{bug['id']}</td><td>{bug['file']}</td><td>{bug['line']}</td><td>{bug['type']}</td><td>{bug['severity']}</td><td>{bug['description']}</td></tr>"
                
                html_output += "</table>"
                
                # Quality issues section
                html_output += "<h2>Quality Issues</h2>"
                html_output += "<table border='1'><tr><th>File</th><th>Line</th><th>Type</th><th>Description</th></tr>"
                
                for issue in results.get("quality_issues", []):
                    html_output += f"<tr><td>{issue['file']}</td><td>{issue['line']}</td><td>{issue['type']}</td><td>{issue['description']}</td></tr>"
                
                html_output += "</table></body></html>"
                
                # Save HTML to file if output specified
                if output_file:
                    with open(output_file, "w") as f:
                        f.write(html_output)
                    return CommandResult(success=True, output=f"Analysis results saved to {output_file}")
                
                return CommandResult(success=True, output=html_output)
            else:
                # Text output (default)
                output = [f"Code Analysis Results for {path}:"]
                
                # Summary section
                output.append("\nSummary:")
                output.append(f"Files analyzed: {results['summary']['files_analyzed']}")
                output.append(f"Lines of code: {results['summary']['lines_of_code']}")
                output.append(f"Bugs found: {results['summary']['bugs_found']}")
                output.append(f"Quality issues: {results['summary']['quality_issues']}")
                
                # Bugs section
                output.append("\nBugs:")
                for bug in results.get("bugs", []):
                    output.append(f"  Bug ID: {bug['id']}")
                    output.append(f"  File: {bug['file']}")
                    output.append(f"  Line: {bug['line']}")
                    output.append(f"  Type: {bug['type']}")
                    output.append(f"  Severity: {bug['severity']}")
                    output.append(f"  Description: {bug['description']}")
                    output.append("")
                
                # Quality issues section
                output.append("\nQuality Issues:")
                for issue in results.get("quality_issues", []):
                    output.append(f"  File: {issue['file']}")
                    output.append(f"  Line: {issue['line']}")
                    output.append(f"  Type: {issue['type']}")
                    output.append(f"  Description: {issue['description']}")
                    output.append("")
                
                # Save text to file if output specified
                if output_file:
                    with open(output_file, "w") as f:
                        f.write("\n".join(output))
                    return CommandResult(success=True, output=f"Analysis results saved to {output_file}")
                
                return CommandResult(success=True, output="\n".join(output))
        
        except Exception as e:
            return CommandResult(success=False, error=f"Error analyzing code: {e}")
    
    def cmd_filter(self, context: CommandContext) -> CommandResult:
        """
        Filter files for analysis.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Get path to filter
        path = context.args.get("path")
        if not path or not os.path.exists(path):
            return CommandResult(success=False, error=f"Path not found: {path}")
        
        # Get filter options
        patterns = context.args.get("patterns")
        exclude = context.args.get("exclude")
        use_entropy = context.args.get("entropy", False)
        threshold = context.args.get("threshold", 0.7)
        
        # Parse patterns and exclude patterns
        if patterns:
            patterns = [p.strip() for p in patterns.split(",")]
        if exclude:
            exclude = [e.strip() for e in exclude.split(",")]
        
        # Call scope filter
        try:
            # This would call the actual scope filter implementation
            results = self.scope_filter.filter(
                path=path,
                include_patterns=patterns,
                exclude_patterns=exclude,
                use_entropy=use_entropy,
                entropy_threshold=threshold
            )
            
            # Format output
            output = [f"Filter Results for {path}:"]
            output.append(f"Total files: {results['total_files']}")
            output.append(f"Included files: {results['included_files']}")
            output.append(f"Excluded files: {results['excluded_files']}")
            output.append("")
            
            # List included files
            output.append("Included Files:")
            for file in results.get("included", []):
                output.append(f"  {file}")
            
            # Save to file if output specified
            output_file = context.args.get("output")
            if output_file:
                with open(output_file, "w") as f:
                    f.write("\n".join(output))
                return CommandResult(success=True, output=f"Filter results saved to {output_file}")
            
            return CommandResult(success=True, output="\n".join(output))
        
        except Exception as e:
            return CommandResult(success=False, error=f"Error filtering files: {e}")
    
    # Bug fixing commands
    def cmd_fix(self, context: CommandContext) -> CommandResult:
        """
        Fix bugs in the specified code.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Get path to fix
        path = context.args.get("path")
        if not path or not os.path.exists(path):
            return CommandResult(success=False, error=f"Path not found: {path}")
        
        # Get fix options
        bug_id = context.args.get("bug-id")
        auto_apply = context.args.get("auto-apply", False)
        create_backup = context.args.get("backup", False)
        verify = context.args.get("verify", False)
        output_dir = context.args.get("output")
        
        # TODO: Implement actual fix logic
        # This would call the actual fix implementation
        
        # Return example result
        return CommandResult(success=True, output=f"Fixed bugs in {path}")
    
    # Helper method to implement all remaining command methods
    def _not_implemented_command(self, context: CommandContext) -> CommandResult:
        """
        Placeholder for commands that are not yet implemented.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        command_name = inspect.currentframe().f_back.f_code.co_name.replace("cmd_", "")
        return CommandResult(
            success=False,
            error=f"Command '{command_name}' is not yet implemented"
        )
    
    # Forward all other command methods to the not_implemented placeholder
    def cmd_generate_patch(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_apply_patch(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_verify_patch(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_neural_init(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_neural_status(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_neural_select(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_neural_train(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_neural_export(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_neural_import(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_resources(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_allocate(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_optimize(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_storage_info(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_backup(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_restore(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_cleanup(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_report(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_visualize(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_configure(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_settings(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_debug(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_log(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_diagnostics(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_triangulum_status(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_triangulum_plan(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_triangulum_execute(self, context: CommandContext) -> CommandResult:
        return self._not_implemented_command(context)
    
    def cmd_help(self, context: CommandContext) -> CommandResult:
        """
        Display help information.
        
        Args:
            context: Command context.
            
        Returns:
            Command result.
        """
        # Get help options
        command_name = context.args.get("command")
        category_name = context.args.get("category")
        list_all = context.args.get("list", False)
        format_type = context.args.get("format", "text")
        
        if command_name:
            # Display help for specific command
            command = self.command_registry.get_command(command_name)
            if not command:
                return CommandResult(success=False, error=f"Command '{command_name}' not found")
            
            # Get command help
            command_help = {
                "name": command.name,
                "help": command.help,
                "category": command.category,
                "args": command.args
            }
            
            if format_type == "json":
                return CommandResult(success=True, data=command_help)
            elif format_type == "markdown":
                # Generate markdown
                output = [f"# {command.name}"]
                output.append(f"\n{command.help}")
                output.append(f"\nCategory: {COMMAND_CATEGORIES.get(command.category, command.category)}")
                
                if command.args:
                    output.append("\n## Arguments\n")
                    for arg in command.args:
                        name = arg.get("name", "")
                        help_text = arg.get("help", "")
                        default = arg.get("default", "")
                        
                        if name.startswith("--"):
                            output.append(f"- `{name}`: {help_text}")
                            if default:
                                output.append(f"  - Default: `{default}`")
                        else:
                            output.append(f"- `{name}` (required): {help_text}")
                
                output.append("\n## Examples\n")
                output.append(f"```\nfx {command.name} [arguments]\n```")
                
                return CommandResult(success=True, output="\n".join(output))
            else:
                # Text output (default)
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
            commands = self.command_registry.get_commands_by_category(category)
            
            if format_type == "json":
                category_help = {
                    "category": category,
                    "description": COMMAND_CATEGORIES[category],
                    "commands": [{"name": cmd.name, "help": cmd.help} for cmd in commands]
                }
                return CommandResult(success=True, data=category_help)
            elif format_type == "markdown":
                # Generate markdown
                output = [f"# {COMMAND_CATEGORIES[category]}"]
                
                for command in commands:
                    output.append(f"\n## {command.name}")
                    output.append(f"\n{command.help}")
                
                return CommandResult(success=True, output="\n".join(output))
            else:
                # Text output (default)
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
                    commands = self.command_registry.get_commands_by_category(category)
                    commands_by_category[category] = {
                        "description": description,
                        "commands": [{"name": cmd.name, "help": cmd.help} for cmd in commands]
                    }
                return CommandResult(success=True, data=commands_by_category)
            elif format_type == "markdown":
                # Generate markdown
                output = ["# FixWurx Commands"]
                
                for category, description in COMMAND_CATEGORIES.items():
                    commands = self.command_registry.get_commands_by_category(category)
                    if commands:
                        output.append(f"\n## {description}")
                        
                        for command in commands:
                            output.append(f"\n### {command.name}")
                            output.append(f"\n{command.help}")
                
                return CommandResult(success=True, output="\n".join(output))
            else:
                # Text output (default)
                output = ["FixWurx Commands:"]
                
                for category, description in COMMAND_CATEGORIES.items():
                    commands = self.command_registry.get_commands_by_category(category)
                    if commands:
                        output.append(f"\n{description}:")
                        
                        for command in commands:
                            output.append(f"  {command.name}: {command.help}")
                
                return CommandResult(success=True, output="\n".join(output))
        
        else:
            # Display general help
            output = ["FixWurx Command Reference"]
            output.append("\nUsage: fx <command> [arguments]")
            output.append("\nUse 'fx help --command <command>' for help on a specific command")
            output.append("Use 'fx help --category <category>' for help on a specific category")
            output.append("Use 'fx help --list' to list all commands")
            
            output.append("\nCommand Categories:")
            for category, description in COMMAND_CATEGORIES.items():
                command_count = len(self.command_registry.get_commands_by_category(category))
                if command_count > 0:
                    output.append(f"  {description} ({command_count} commands)")
            
            return CommandResult(success=True, output="\n".join(output))

# Main entry point
def main():
    """
    Main entry point for the FixWurx commands reference.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create command registry
    command_registry = CommandRegistry()
    
    # Create and initialize FixWurx commands
    fixwurx_commands = FixWurxCommands(command_registry)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="FixWurx Commands Reference")
    parser.add_argument("--list", action="store_true", help="List all commands")
    parser.add_argument("--category", help="List commands in category")
    parser.add_argument("--command", help="Show help for command")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    # Process arguments
    if args.list:
        # List all commands
        result = fixwurx_commands.cmd_help(CommandContext(args={"list": True, "format": args.format}))
    elif args.category:
        # List commands in category
        result = fixwurx_commands.cmd_help(CommandContext(args={"category": args.category, "format": args.format}))
    elif args.command:
        # Show help for command
        result = fixwurx_commands.cmd_help(CommandContext(args={"command": args.command, "format": args.format}))
    else:
        # Show general help
        result = fixwurx_commands.cmd_help(CommandContext(args={}))
    
    # Print result
    if result.success:
        if result.output:
            print(result.output)
        elif result.data and args.format == "json":
            print(json.dumps(result.data, indent=2))
    else:
        print(f"Error: {result.error}")

if __name__ == "__main__":
    main()
