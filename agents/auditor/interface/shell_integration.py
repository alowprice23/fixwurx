#!/usr/bin/env python3
"""
FixWurx Auditor Shell Integration

This module integrates the Auditor Agent with the FixWurx launchpad shell environment,
allowing direct interaction between the user and the auditor through command-line
interfaces.
"""

import os
import sys
import cmd
import json
import yaml
import datetime
import argparse
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

# Force UTF-8 encoding for stdout/stderr on Windows
if sys.platform == 'win32':
    import codecs
    # Use utf-8 encoding for console output
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')

# Import auditor components
from auditor_agent import AuditorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [AuditorShell] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('auditor_shell')


class AuditorShell(cmd.Cmd):
    """
    Interactive shell for interacting with the FixWurx Auditor Agent.
    """
    
    intro = """
    ╔═══════════════════════════════════════════════╗
    ║               FixWurx Auditor Shell           ║
    ╚═══════════════════════════════════════════════╝
    
    Type 'help' or '?' to list commands.
    Type 'exit' or 'quit' to exit.
    """
    
    prompt = '\n[FixWurx Auditor] >> '
    
    def __init__(self, agent: AuditorAgent):
        """
        Initialize the Auditor Shell.
        
        Args:
            agent: The Auditor Agent instance
        """
        super().__init__()
        self.agent = agent
        self.status_thread = None
        self.show_status = False
        self.last_command = None
        self.results_cache = {}
        
        # Initialize shell environment
        self.env = {
            "config_path": "auditor_config.yaml",
            "output_format": "text",  # text, json, yaml
            "verbose": False,
            "last_audit_time": None,
            "last_audit_result": None,
            "last_error": None
        }
        
        logger.info("Auditor Shell initialized")
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def do_exit(self, arg):
        """Exit the Auditor Shell"""
        self._stop_status_thread()
        print("Exiting Auditor Shell...")
        return True
    
    def do_quit(self, arg):
        """Exit the Auditor Shell"""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        print()  # Print newline
        return self.do_exit(arg)
    
    def do_status(self, arg):
        """
        Show the current status of the Auditor Agent.
        
        Usage: status [--watch]
            --watch: Continuously update status
        """
        args = arg.split()
        watch_mode = "--watch" in args
        
        if watch_mode:
            if self.status_thread and self.status_thread.is_alive():
                print("Already watching status. Type 'status --stop' to stop.")
                return
            
            self.show_status = True
            self.status_thread = threading.Thread(target=self._watch_status)
            self.status_thread.daemon = True
            self.status_thread.start()
            print("Status watch started. Type 'status --stop' to stop.")
        elif arg == "--stop":
            self._stop_status_thread()
            print("Status watch stopped.")
        else:
            self._display_status()
    
    def do_audit(self, arg):
        """
        Run an audit using the Auditor Agent.
        
        Usage: audit [--full] [--save FILENAME]
            --full: Force a full audit even if incremental would suffice
            --save FILENAME: Save audit result to file
        """
        args = arg.split()
        force_full = "--full" in args
        save_file = None
        
        for i, a in enumerate(args):
            if a == "--save" and i + 1 < len(args):
                save_file = args[i + 1]
        
        print(f"Running {'full' if force_full else 'incremental'} audit...")
        
        try:
            result = self.agent.run_audit(force_full_audit=force_full)
            self.env["last_audit_time"] = datetime.datetime.now()
            self.env["last_audit_result"] = result
            
            # Display result
            self._display_result(result, "Audit Result")
            
            # Save result if requested
            if save_file:
                self._save_result(result, save_file)
                print(f"Audit result saved to {save_file}")
            
            return result
        except Exception as e:
            self.env["last_error"] = str(e)
            print(f"Error running audit: {e}")
    
    def do_monitor(self, arg):
        """
        Start or stop proactive monitoring.
        
        Usage: monitor [start|stop]
        """
        if not arg or arg.lower() == "status":
            status = "running" if self.agent.is_running else "stopped"
            print(f"Proactive monitoring is currently {status}")
        elif arg.lower() == "start":
            if self.agent.is_running:
                print("Monitoring is already running")
            else:
                self.agent.start()
                print("Proactive monitoring started")
        elif arg.lower() == "stop":
            if not self.agent.is_running:
                print("Monitoring is already stopped")
            else:
                self.agent.stop()
                print("Proactive monitoring stopped")
        else:
            print("Usage: monitor [start|stop|status]")
    
    def do_trigger(self, arg):
        """
        Trigger an event to be processed by the agent.
        
        Usage: trigger EVENT_TYPE [--data JSON_STRING]
            EVENT_TYPE: One of: code_change, build_failure, performance_alert, audit_request
            --data: JSON string with event data
        """
        args = arg.split()
        if not args:
            print("Usage: trigger EVENT_TYPE [--data JSON_STRING]")
            print("Event types: code_change, build_failure, performance_alert, audit_request")
            return
        
        event_type = args[0]
        valid_events = ["code_change", "build_failure", "performance_alert", "audit_request"]
        
        if event_type not in valid_events:
            print(f"Invalid event type: {event_type}")
            print(f"Valid event types: {', '.join(valid_events)}")
            return
        
        # Parse event data
        event_data = {}
        data_index = -1
        
        for i, a in enumerate(args):
            if a == "--data" and i + 1 < len(args):
                data_index = i + 1
                break
        
        if data_index >= 0:
            try:
                data_str = ' '.join(args[data_index:])
                event_data = json.loads(data_str)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON data: {e}")
                return
        
        # Trigger event
        try:
            self.agent.trigger_event(event_type, event_data)
            print(f"Event '{event_type}' triggered successfully")
        except Exception as e:
            self.env["last_error"] = str(e)
            print(f"Error triggering event: {e}")
    
    def do_config(self, arg):
        """
        View or modify configuration settings.
        
        Usage: config [SETTING [VALUE]]
            With no arguments, shows all settings
            With SETTING only, shows that setting
            With SETTING and VALUE, sets that setting to the given value
        """
        args = arg.split()
        if not args:
            # Show all settings
            print("Current configuration settings:")
            for key, value in self.env.items():
                print(f"  {key} = {value}")
        elif len(args) == 1:
            # Show specific setting
            setting = args[0]
            if setting in self.env:
                print(f"{setting} = {self.env[setting]}")
            else:
                print(f"Unknown setting: {setting}")
        else:
            # Set specific setting
            setting = args[0]
            value = ' '.join(args[1:])
            
            # Convert value to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            
            if setting in self.env:
                self.env[setting] = value
                print(f"Set {setting} = {value}")
            else:
                print(f"Unknown setting: {setting}")
    
    def do_actions(self, arg):
        """
        List recent actions taken by the Auditor Agent.
        
        Usage: actions [--limit N]
            --limit N: Limit to N most recent actions (default: 10)
        """
        args = arg.split()
        limit = 10
        
        for i, a in enumerate(args):
            if a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    print(f"Invalid limit: {args[i + 1]}")
                    return
        
        status = self.agent.get_status()
        actions = status.get("action_history", [])
        
        if not actions:
            print("No actions recorded")
            return
        
        print(f"Recent actions (limit: {limit}):")
        for i, action in enumerate(reversed(actions[:limit])):
            action_type = action.get("type", "unknown")
            timestamp = action.get("timestamp", "unknown")
            status = action.get("status", "unknown")
            
            details = []
            for key, value in action.items():
                if key not in ["type", "timestamp", "status"]:
                    details.append(f"{key}={value}")
            
            details_str = ", ".join(details)
            print(f"  {i+1}. [{timestamp}] {action_type} - {status} {details_str}")
    
    def do_test(self, arg):
        """
        Run functionality tests on system components.
        
        Usage: test [component1 component2 ...]
            With no arguments, tests all components
            With component names, tests only those components
        """
        args = arg.split()
        components = args if args else ["all"]
        
        print(f"Testing components: {', '.join(components)}")
        
        # In a real implementation, this would actually test the components
        # For now, we just simulate it
        results = {
            "tests_run": 5,
            "passed": 4,
            "failed": 1,
            "details": {
                "component1": {"status": "pass", "time": 0.35},
                "component2": {"status": "pass", "time": 0.42},
                "component3": {"status": "pass", "time": 0.28},
                "component4": {"status": "fail", "time": 0.51, "error": "Expected value not returned"},
                "component5": {"status": "pass", "time": 0.39}
            }
        }
        
        self._display_result(results, "Test Results")
        self.results_cache["last_test"] = results
    
    def do_patch(self, arg):
        """
        Manage system patches.
        
        Usage: patch [list|apply|rollback|info] [--id PATCH_ID]
        """
        args = arg.split()
        if not args:
            print("Usage: patch [list|apply|rollback|info] [--id PATCH_ID]")
            return
        
        command = args[0].lower()
        patch_id = None
        
        for i, a in enumerate(args):
            if a == "--id" and i + 1 < len(args):
                patch_id = args[i + 1]
        
        if command == "list":
            # In a real implementation, this would list actual patches
            # For now, we just simulate it
            patches = [
                {"id": "PATCH-001", "component": "component1", "status": "applied", "date": "2025-07-10"},
                {"id": "PATCH-002", "component": "component4", "status": "pending", "date": "2025-07-12"}
            ]
            
            print("Available patches:")
            for patch in patches:
                print(f"  {patch['id']} - {patch['component']} - {patch['status']} - {patch['date']}")
        
        elif command == "apply" and patch_id:
            print(f"Applying patch {patch_id}...")
            print("Patch applied successfully")
        
        elif command == "rollback" and patch_id:
            print(f"Rolling back patch {patch_id}...")
            print("Patch rolled back successfully")
        
        elif command == "info" and patch_id:
            # In a real implementation, this would show actual patch info
            patch_info = {
                "id": patch_id,
                "component": "component4",
                "status": "pending",
                "date": "2025-07-12",
                "description": "Fix for issue #123",
                "changes": [
                    {"file": "component4.py", "lines": "45-52"},
                    {"file": "component4_utils.py", "lines": "18-20"}
                ]
            }
            
            self._display_result(patch_info, f"Patch {patch_id} Info")
        
        else:
            print("Usage: patch [list|apply|rollback|info] [--id PATCH_ID]")
    
    def do_errors(self, arg):
        """
        List and manage system errors.
        
        Usage: errors [list|details|resolve] [--id ERROR_ID] [--limit N]
        """
        args = arg.split()
        if not args:
            print("Usage: errors [list|details|resolve] [--id ERROR_ID] [--limit N]")
            return
        
        command = args[0].lower()
        error_id = None
        limit = 10
        
        for i, a in enumerate(args):
            if a == "--id" and i + 1 < len(args):
                error_id = args[i + 1]
            elif a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    print(f"Invalid limit: {args[i + 1]}")
                    return
        
        if command == "list":
            # In a real implementation, this would list actual errors
            # For now, we just simulate it
            errors = [
                {"id": "ERR-001", "component": "component2", "severity": "low", "status": "resolved"},
                {"id": "ERR-002", "component": "component4", "severity": "high", "status": "open"},
                {"id": "ERR-003", "component": "component1", "severity": "medium", "status": "in_progress"}
            ]
            
            print(f"Recent errors (limit: {limit}):")
            for error in errors[:limit]:
                print(f"  {error['id']} - {error['component']} - {error['severity']} - {error['status']}")
        
        elif command == "details" and error_id:
            # In a real implementation, this would show actual error details
            error_details = {
                "id": error_id,
                "component": "component4",
                "severity": "high",
                "status": "open",
                "message": "Function returned unexpected value",
                "timestamp": "2025-07-12T22:15:30",
                "stack_trace": "Traceback (most recent call last):\n  File \"component4.py\", line 52...",
                "affected_components": ["component4", "component6"]
            }
            
            self._display_result(error_details, f"Error {error_id} Details")
        
        elif command == "resolve" and error_id:
            print(f"Resolving error {error_id}...")
            print("Error marked as resolved")
        
        else:
            print("Usage: errors [list|details|resolve] [--id ERROR_ID] [--limit N]")
    
    def do_metrics(self, arg):
        """
        Display system metrics.
        
        Usage: metrics [--component COMPONENT] [--period PERIOD]
            --component: Filter by component name
            --period: Time period (hour, day, week, month)
        """
        args = arg.split()
        component = None
        period = "day"
        
        for i, a in enumerate(args):
            if a == "--component" and i + 1 < len(args):
                component = args[i + 1]
            elif a == "--period" and i + 1 < len(args):
                period = args[i + 1]
        
        if period not in ["hour", "day", "week", "month"]:
            print(f"Invalid period: {period}")
            print("Valid periods: hour, day, week, month")
            return
        
        component_str = f"for component {component}" if component else "for all components"
        print(f"Displaying metrics {component_str} over the last {period}")
        
        # In a real implementation, this would show actual metrics
        # For now, we just simulate it
        metrics = {
            "time_period": period,
            "component": component or "all",
            "cpu_usage": {
                "average": 35.2,
                "peak": 78.5,
                "trend": "stable"
            },
            "memory_usage": {
                "average": 420.8,
                "peak": 850.2,
                "trend": "increasing"
            },
            "response_time": {
                "average": 0.35,
                "p95": 0.82,
                "trend": "decreasing"
            },
            "error_rate": {
                "average": 0.02,
                "peak": 0.05,
                "trend": "stable"
            }
        }
        
        self._display_result(metrics, "System Metrics")
    
    def do_report(self, arg):
        """
        Generate system reports.
        
        Usage: report [audit|health|performance|security] [--output FILENAME]
        """
        args = arg.split()
        if not args:
            print("Usage: report [audit|health|performance|security] [--output FILENAME]")
            return
        
        report_type = args[0].lower()
        output_file = None
        
        for i, a in enumerate(args):
            if a == "--output" and i + 1 < len(args):
                output_file = args[i + 1]
        
        valid_types = ["audit", "health", "performance", "security"]
        if report_type not in valid_types:
            print(f"Invalid report type: {report_type}")
            print(f"Valid report types: {', '.join(valid_types)}")
            return
        
        print(f"Generating {report_type} report...")
        
        # In a real implementation, this would generate actual reports
        # For now, we just simulate it
        report = {
            "type": report_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "status": "good",
                "issues_found": 3,
                "critical_issues": 0,
                "recommendations": 2
            },
            "details": {
                # Report-specific details would go here
            }
        }
        
        if output_file:
            self._save_result(report, output_file)
            print(f"Report saved to {output_file}")
        else:
            self._display_result(report, f"{report_type.capitalize()} Report")
    
    def do_help(self, arg):
        """
        List available commands with help text or show help for a specific command.
        """
        if arg:
            # Display help for specific command
            super().do_help(arg)
        else:
            # Display all commands grouped by category
            categories = {
                "System Control": ["status", "monitor"],
                "Auditing": ["audit", "test", "errors", "metrics", "report"],
                "Management": ["trigger", "patch", "actions"],
                "Configuration": ["config"],
                "Shell": ["help", "exit", "quit"]
            }
            
            for category, commands in categories.items():
                print(f"\n{category}:")
                for cmd in commands:
                    doc = getattr(self, f"do_{cmd}").__doc__
                    if doc:
                        # Extract first line of docstring
                        desc = doc.strip().split('\n')[0]
                        print(f"  {cmd:<10} - {desc}")
                    else:
                        print(f"  {cmd}")
    
    def _display_status(self):
        """Display the current status of the Auditor Agent"""
        status = self.agent.get_status()
        
        autonomous_mode = "Enabled" if status.get("autonomous_mode", False) else "Disabled"
        monitoring_status = "Running" if status.get("is_monitoring", False) else "Stopped"
        last_audit = status.get("last_audit_time", "Never")
        
        action_count = len(status.get("action_history", []))
        memory_usage = status.get("memory_usage", {})
        rss = memory_usage.get("rss_mb", 0)
        vms = memory_usage.get("vms_mb", 0)
        
        print("\nAuditor Agent Status:")
        print(f"  Autonomous Mode: {autonomous_mode}")
        print(f"  Monitoring: {monitoring_status}")
        print(f"  Last Audit: {last_audit}")
        print(f"  Actions Taken: {action_count}")
        print(f"  Memory Usage: {rss:.2f} MB (RSS), {vms:.2f} MB (VMS)")
        print(f"  Event Queue Size: {status.get('queue_size', 0)}")
    
    def _watch_status(self):
        """Continuously update status display"""
        try:
            while self.show_status:
                # Clear the terminal
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Display the status
                print("FixWurx Auditor Agent - Live Status")
                print("="*40)
                self._display_status()
                print("\nPress Ctrl+C or type 'status --stop' to stop watching")
                
                # Wait a bit
                threading.Event().wait(2.0)
        except Exception as e:
            print(f"Error watching status: {e}")
            self.show_status = False
    
    def _stop_status_thread(self):
        """Stop the status watching thread"""
        if self.status_thread and self.status_thread.is_alive():
            self.show_status = False
            self.status_thread.join(timeout=1.0)
            self.status_thread = None
    
    def _display_result(self, result: Dict[str, Any], title: str):
        """
        Display a result dict in the appropriate format.
        
        Args:
            result: Result dictionary
            title: Result title
        """
        format_type = self.env.get("output_format", "text").lower()
        
        print(f"\n{title}:")
        if format_type == "json":
            print(json.dumps(result, indent=2, default=str))
        elif format_type == "yaml":
            print(yaml.dump(result, default_flow_style=False))
        else:  # text
            self._print_dict(result)
    
    def _print_dict(self, data: Dict[str, Any], indent: int = 2):
        """
        Print a dictionary recursively in a readable format.
        
        Args:
            data: Dictionary to print
            indent: Current indentation level
        """
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{' ' * indent}{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, list):
                print(f"{' ' * indent}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        self._print_dict(item, indent + 2)
                    else:
                        print(f"{' ' * (indent + 2)}- {item}")
            else:
                print(f"{' ' * indent}{key}: {value}")
    
    def _save_result(self, result: Dict[str, Any], filename: str):
        """
        Save a result dict to a file.
        
        Args:
            result: Result dictionary
            filename: Output filename
        """
        try:
            format_type = "json"
            if "." in filename:
                ext = filename.split(".")[-1].lower()
                if ext in ["json", "yaml", "yml", "txt"]:
                    if ext == "yml":
                        format_type = "yaml"
                    else:
                        format_type = ext
            
            with open(filename, 'w') as f:
                if format_type == "json":
                    json.dump(result, f, indent=2, default=str)
                elif format_type in ["yaml", "yml"]:
                    yaml.dump(result, f, default_flow_style=False)
                else:  # text
                    f.write(f"{filename} - Generated on {datetime.datetime.now()}\n\n")
                    self._write_dict_to_file(f, result)
        except Exception as e:
            print(f"Error saving to {filename}: {e}")
    
    def _write_dict_to_file(self, file, data: Dict[str, Any], indent: int = 0):
        """
        Write a dictionary recursively to a file.
        
        Args:
            file: File object
            data: Dictionary to write
            indent: Current indentation level
        """
        for key, value in data.items():
            if isinstance(value, dict):
                file.write(f"{' ' * indent}{key}:\n")
                self._write_dict_to_file(file, value, indent + 2)
            elif isinstance(value, list):
                file.write(f"{' ' * indent}{key}:\n")
                for item in value:
                    if isinstance(item, dict):
                        self._write_dict_to_file(file, item, indent + 2)
                    else:
                        file.write(f"{' ' * (indent + 2)}- {item}\n")
            else:
                file.write(f"{' ' * indent}{key}: {value}\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='FixWurx Auditor Shell')
    
    parser.add_argument(
        '--config',
        type=str,
        default='auditor_config.yaml',
        help='Path to configuration file (default: auditor_config.yaml)'
    )
    
    parser.add_argument(
        '--no-auto',
        action='store_true',
        help='Disable autonomous mode'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize the Auditor Agent
        agent = AuditorAgent(args.config, autonomous_mode=not args.no_auto)
        
        # Create and run the shell
        shell = AuditorShell(agent)
        
        # Start the agent (will start monitoring if autonomous mode is enabled)
        agent.start()
        
        try:
            # Run the shell
            shell.cmdloop()
        finally:
            # Stop the agent when the shell exits
            agent.stop()
        
        return 0
    except Exception as e:
        logger.error(f"Error in Auditor Shell: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
