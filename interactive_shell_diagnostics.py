#!/usr/bin/env python3
"""
interactive_shell_diagnostics.py
────────────────────────────────
Interactive diagnostic tools for the FixWurx shell environment.

This module provides an interactive diagnostic console for real-time
troubleshooting, component inspection, and system testing within the
shell environment.
"""

import os
import sys
import cmd
import time
import psutil
import logging
import readline
import threading
import json
import importlib
import inspect
import traceback
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, Union

# Internal imports
from shell_environment import register_command, emit_event, EventType, get_environment_variable
import shell_scripting
from shell_diagnostics import (
    ComponentStatus, DiagnosticResult, SystemMetrics, 
    get_shell_diagnostics, diagnose, get_health, get_metrics, fix_issues
)

# Configure logging
logger = logging.getLogger("InteractiveShellDiagnostics")
handler = logging.FileHandler("interactive_diagnostics.log")
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class InteractiveDiagnosticConsole(cmd.Cmd):
    """Interactive diagnostic console for the FixWurx shell environment."""
    
    intro = """
   ______ _      __          __               ____  _                           _   _          
  |  ____(_)     \ \        / /              |  _ \(_)                         | | (_)         
  | |__   ___  __ \ \  /\  / /   _ _ __ __  _| |_) |_  __ _  __ _ _ __   ___  ___ |_ ___ ___ 
  |  __| | \ \/ /  \ \/  \/ / | | | '__/ _` |  _ <| |/ _` |/ _` | '_ \ / _ \/ __| | / __/ __|
  | |    | |>  <    \  /\  /| |_| | | | (_| | |_) | | (_| | (_| | | | | (_) \__ \ | \__ \__ \
  |_|    |_/_/\_\    \/  \/  \__,_|_|  \__,_|____/|_|\__,_|\__, |_| |_|\___/|___/_|_|___/___/
                                                             __/ |                            
                                                            |___/                             
                                                            
  Interactive Diagnostic Console

  Type 'help' or '?' to list commands.
  Type 'exit' or 'quit' to exit the console.
"""
    prompt = "diagnostics> "
    
    def __init__(self, shell_diagnostics=None):
        """Initialize the interactive diagnostic console."""
        super().__init__()
        self.shell_diagnostics = shell_diagnostics or get_shell_diagnostics()
        self.components = {}
        self.active_monitoring = False
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.history_file = os.path.expanduser("~/.fx_diagnostics_history")
        self.load_history()
        
        # Register main shell command
        try:
            register_command("interactive_diagnostics", self.start_console_command, 
                            "Start the interactive diagnostic console")
        except Exception as e:
            logger.error(f"Failed to register interactive diagnostics command: {e}")
    
    def load_history(self):
        """Load command history from file."""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                readline.set_history_length(1000)
        except Exception as e:
            logger.error(f"Error loading history: {e}")
    
    def save_history(self):
        """Save command history to file."""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def do_diagnose(self, arg):
        """
        Run diagnostic checks on specific components or all components.
        
        Usage:
          diagnose [component]
          
        Examples:
          diagnose                  # Diagnose all components
          diagnose shell_environment # Diagnose only the shell environment component
        """
        component = arg.strip() if arg else None
        
        try:
            print(f"\nRunning diagnostics{f' for component: {component}' if component else ''}...")
            print("-" * 60)
            
            results = diagnose(component)
            
            # Group results by status
            by_status = {
                ComponentStatus.HEALTHY: [],
                ComponentStatus.DEGRADED: [],
                ComponentStatus.CRITICAL: [],
                ComponentStatus.UNAVAILABLE: [],
                ComponentStatus.UNKNOWN: []
            }
            
            for comp, result in results.items():
                by_status[result.status].append(result)
            
            # Print results by status
            if by_status[ComponentStatus.CRITICAL]:
                print("\n🔴 Critical Issues:")
                for result in by_status[ComponentStatus.CRITICAL]:
                    print(f"  {result}")
            
            if by_status[ComponentStatus.DEGRADED]:
                print("\n🟠 Degraded Components:")
                for result in by_status[ComponentStatus.DEGRADED]:
                    print(f"  {result}")
            
            if by_status[ComponentStatus.UNAVAILABLE] or by_status[ComponentStatus.UNKNOWN]:
                print("\n⚪ Unavailable/Unknown Components:")
                for result in by_status[ComponentStatus.UNAVAILABLE] + by_status[ComponentStatus.UNKNOWN]:
                    print(f"  {result}")
            
            if by_status[ComponentStatus.HEALTHY]:
                print("\n🟢 Healthy Components:")
                for result in by_status[ComponentStatus.HEALTHY]:
                    print(f"  {result}")
            
            # Save component results for inspection
            self.components = results
            
            # Print summary
            critical = len(by_status[ComponentStatus.CRITICAL])
            degraded = len(by_status[ComponentStatus.DEGRADED])
            unavailable = len(by_status[ComponentStatus.UNAVAILABLE]) + len(by_status[ComponentStatus.UNKNOWN])
            healthy = len(by_status[ComponentStatus.HEALTHY])
            total = critical + degraded + unavailable + healthy
            
            print("\nDiagnostic Summary:")
            print(f"  Total Components: {total}")
            print(f"  🟢 Healthy: {healthy}")
            print(f"  🟠 Degraded: {degraded}")
            print(f"  🔴 Critical: {critical}")
            print(f"  ⚪ Unavailable/Unknown: {unavailable}")
            
            if critical > 0 or degraded > 0:
                print("\nUse 'inspect <component>' for details on issues.")
                print("Use 'fix <component>' to attempt automatic fixes.")
            
            return False
        except Exception as e:
            print(f"Error running diagnostics: {e}")
            logger.error(f"Error running diagnostics: {e}\n{traceback.format_exc()}")
            return False
    
    def do_inspect(self, arg):
        """
        Inspect a component in detail.
        
        Usage:
          inspect <component>
          
        Example:
          inspect shell_environment
        """
        component = arg.strip()
        
        if not component:
            print("Error: Component name required.")
            print("Usage: inspect <component>")
            return False
        
        if not self.components:
            print("No diagnostic data available. Run 'diagnose' first.")
            return False
        
        if component not in self.components:
            print(f"Error: Component '{component}' not found in diagnostic data.")
            print("Available components:")
            for comp in sorted(self.components.keys()):
                print(f"  {comp}")
            return False
        
        result = self.components[component]
        
        print(f"\nDetailed Inspection of Component: {component}")
        print("-" * 60)
        print(f"Status: {result.status.name}")
        print(f"Message: {result.message}")
        
        if result.details:
            print("\nDetails:")
            for key, value in result.details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                elif isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")
        
        if result.status != ComponentStatus.HEALTHY:
            print("\nRecommended Action:")
            if result.status == ComponentStatus.CRITICAL:
                print("  This component has critical issues that need immediate attention.")
                print("  Use 'fix {component}' to attempt automatic fixes.")
            elif result.status == ComponentStatus.DEGRADED:
                print("  This component is working but with issues that may affect performance.")
                print("  Use 'fix {component}' to attempt automatic fixes.")
            else:
                print("  This component is unavailable or in an unknown state.")
                print("  Check the component's implementation and dependencies.")
        
        return False
    
    def do_fix(self, arg):
        """
        Attempt to fix issues with a component.
        
        Usage:
          fix [component]
          
        Examples:
          fix                    # Fix all components with issues
          fix shell_environment  # Fix only the shell environment component
        """
        component = arg.strip() if arg else None
        
        try:
            print(f"\nAttempting to fix{f' component: {component}' if component else ' all components with issues'}...")
            print("-" * 60)
            
            results = fix_issues(component)
            
            fixed = []
            not_fixed = []
            
            for comp, result in results.items():
                if result.get("status") == "healthy":
                    # Already healthy, nothing to fix
                    print(f"  ℹ️ {comp}: {result.get('message', 'Already healthy')}")
                elif result.get("fixed", False):
                    # Successfully fixed
                    fixed.append(comp)
                    print(f"  ✅ {comp}: {result.get('message', 'Fixed successfully')}")
                else:
                    # Could not fix
                    not_fixed.append(comp)
                    print(f"  ❌ {comp}: {result.get('message', 'Could not fix')}")
                    
                    # Print recommendations if any
                    if "recommendations" in result:
                        print("    Recommendations:")
                        for recommendation in result["recommendations"]:
                            print(f"     - {recommendation}")
            
            # Print summary
            print("\nFix Summary:")
            print(f"  Components checked: {len(results)}")
            print(f"  Components fixed: {len(fixed)}")
            print(f"  Components not fixed: {len(not_fixed)}")
            
            if not_fixed:
                print("\nThe following components could not be fixed automatically:")
                for comp in not_fixed:
                    print(f"  - {comp}")
                print("\nManual intervention may be required for these components.")
            
            # Re-run diagnostics to update component status
            if fixed:
                print("\nRe-running diagnostics to verify fixes...")
                self.do_diagnose("")
            
            return False
        except Exception as e:
            print(f"Error fixing issues: {e}")
            logger.error(f"Error fixing issues: {e}\n{traceback.format_exc()}")
            return False
    
    def do_metrics(self, arg):
        """
        Display system metrics.
        
        Usage:
          metrics [count]
          
        Examples:
          metrics     # Show the latest metrics
          metrics 5   # Show the last 5 metrics entries
        """
        try:
            args = arg.strip().split()
            count = int(args[0]) if args and args[0].isdigit() else 1
            
            print(f"\nSystem Metrics (Last {count}):")
            print("-" * 60)
            
            metrics_list = get_metrics(count)
            
            for i, metrics in enumerate(metrics_list):
                if i > 0:
                    print("\n" + "-" * 30)
                
                # Format timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metrics.timestamp))
                
                print(f"Time: {timestamp}")
                print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
                print(f"Memory: {metrics.memory_usage:.1f}% used ({metrics.memory_available / (1024 * 1024):.1f} MB available)")
                print(f"Disk: {metrics.disk_usage:.1f}% used ({metrics.disk_available / (1024 * 1024 * 1024):.1f} GB available)")
                print(f"Processes: {metrics.process_count}")
                print(f"Threads: {metrics.thread_count}")
                print(f"Command Count: {metrics.command_count}")
                print(f"Event Count: {metrics.event_count}")
                print(f"Error Count: {metrics.error_count}")
                
                # Additional details if available
                if metrics.details:
                    if "python_version" in metrics.details:
                        print(f"Python Version: {metrics.details['python_version'].split()[0]}")
                    if "platform" in metrics.details:
                        print(f"Platform: {metrics.details['platform']}")
                    if "processors" in metrics.details and "logical_processors" in metrics.details:
                        print(f"CPU: {metrics.details['processors']} physical cores, {metrics.details['logical_processors']} logical cores")
            
            return False
        except Exception as e:
            print(f"Error getting metrics: {e}")
            logger.error(f"Error getting metrics: {e}\n{traceback.format_exc()}")
            return False
    
    def do_monitor(self, arg):
        """
        Start or stop real-time monitoring of system metrics.
        
        Usage:
          monitor [start|stop] [interval]
          
        Examples:
          monitor start    # Start monitoring with default interval (5 seconds)
          monitor start 2  # Start monitoring with 2-second interval
          monitor stop     # Stop monitoring
        """
        args = arg.strip().split()
        
        if not args:
            if self.active_monitoring:
                print("Monitoring is currently active.")
                print(f"Use 'monitor stop' to stop monitoring.")
            else:
                print("Monitoring is not active.")
                print(f"Use 'monitor start [interval]' to start monitoring.")
            return False
        
        action = args[0].lower()
        
        if action == "start":
            if self.active_monitoring:
                print("Monitoring is already active. Use 'monitor stop' first.")
                return False
            
            interval = int(args[1]) if len(args) > 1 and args[1].isdigit() else 5
            
            print(f"Starting real-time monitoring (interval: {interval} seconds)...")
            print("Press Ctrl+C or type 'monitor stop' to stop monitoring.")
            
            # Start monitoring thread
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True
            )
            self.active_monitoring = True
            self.monitoring_thread.start()
            
        elif action == "stop":
            if not self.active_monitoring:
                print("Monitoring is not active.")
                return False
            
            print("Stopping monitoring...")
            self.stop_monitoring.set()
            self.active_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1.0)
            print("Monitoring stopped.")
            
        else:
            print(f"Unknown action: {action}")
            print("Usage: monitor [start|stop] [interval]")
        
        return False
    
    def _monitoring_loop(self, interval):
        """
        Real-time monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        try:
            while not self.stop_monitoring.is_set():
                # Get latest metrics
                metrics = get_metrics(1)[0]
                
                # Format timestamp
                timestamp = time.strftime("%H:%M:%S", time.localtime(metrics.timestamp))
                
                # Clear line and print metrics
                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear line
                sys.stdout.write(
                    f"[{timestamp}] CPU: {metrics.cpu_usage:5.1f}% | "
                    f"Mem: {metrics.memory_usage:5.1f}% | "
                    f"Disk: {metrics.disk_usage:5.1f}% | "
                    f"Proc: {metrics.process_count:4d} | "
                    f"Thrd: {metrics.thread_count:4d} | "
                    f"Cmd: {metrics.command_count:4d} | "
                    f"Err: {metrics.error_count:4d}"
                )
                sys.stdout.flush()
                
                # Sleep for interval
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}\n{traceback.format_exc()}")
        finally:
            self.active_monitoring = False
    
    def do_test(self, arg):
        """
        Run specific tests on components.
        
        Usage:
          test <component> [test_name]
          
        Examples:
          test shell_environment           # Run all tests for shell_environment
          test shell_environment variables # Run the 'variables' test for shell_environment
        """
        args = arg.strip().split()
        
        if not args:
            print("Error: Component name required.")
            print("Usage: test <component> [test_name]")
            return False
        
        component = args[0]
        test_name = args[1] if len(args) > 1 else None
        
        # Get available tests for component
        tests = self._get_component_tests(component)
        
        if not tests:
            print(f"No tests available for component: {component}")
            return False
        
        if test_name and test_name not in tests:
            print(f"Test '{test_name}' not found for component: {component}")
            print("Available tests:")
            for test in sorted(tests.keys()):
                print(f"  {test}: {tests[test].__doc__ or 'No description'}")
            return False
        
        # Run specified test or all tests
        try:
            if test_name:
                print(f"\nRunning test '{test_name}' for component: {component}")
                print("-" * 60)
                
                test_func = tests[test_name]
                result = test_func()
                
                print(f"\nTest Result: {'PASSED' if result else 'FAILED'}")
            else:
                print(f"\nRunning all tests for component: {component}")
                print("-" * 60)
                
                passed = 0
                failed = 0
                
                for name, test_func in sorted(tests.items()):
                    print(f"\nTest: {name}")
                    print(f"Description: {test_func.__doc__ or 'No description'}")
                    
                    try:
                        result = test_func()
                        if result:
                            print(f"Result: ✅ PASSED")
                            passed += 1
                        else:
                            print(f"Result: ❌ FAILED")
                            failed += 1
                    except Exception as e:
                        print(f"Result: ❌ ERROR: {e}")
                        failed += 1
                
                print(f"\nTest Summary for {component}:")
                print(f"  Total Tests: {passed + failed}")
                print(f"  Passed: {passed}")
                print(f"  Failed: {failed}")
            
            return False
        except Exception as e:
            print(f"Error running tests: {e}")
            logger.error(f"Error running tests: {e}\n{traceback.format_exc()}")
            return False
    
    def _get_component_tests(self, component):
        """
        Get tests available for a component.
        
        Args:
            component: Component name
            
        Returns:
            Dictionary of test name to test function
        """
        tests = {}
        
        # Shell Environment tests
        if component == "shell_environment":
            tests = {
                "variables": self._test_shell_environment_variables,
                "commands": self._test_shell_environment_commands,
                "events": self._test_shell_environment_events
            }
        
        # Shell Scripting tests
        elif component == "shell_scripting":
            tests = {
                "execution": self._test_shell_scripting_execution,
                "variables": self._test_shell_scripting_variables,
                "conditions": self._test_shell_scripting_conditions,
                "loops": self._test_shell_scripting_loops
            }
        
        # Fixwurx Commands tests
        elif component == "fixwurx_commands":
            tests = {
                "availability": self._test_fixwurx_commands_availability,
                "execution": self._test_fixwurx_commands_execution
            }
        
        # System tests
        elif component == "system":
            tests = {
                "resources": self._test_system_resources,
                "processes": self._test_system_processes
            }
        
        return tests
    
    def _test_shell_environment_variables(self):
        """Test shell environment variables functionality."""
        from shell_environment import set_environment_variable, get_environment_variable
        
        print("Testing environment variable setting and retrieval...")
        
        # Generate unique test variable name
        test_var = f"TEST_VAR_{int(time.time())}"
        test_value = f"test_value_{int(time.time())}"
        
        # Set variable
        set_environment_variable(test_var, test_value)
        print(f"Set variable: {test_var} = {test_value}")
        
        # Get variable
        retrieved_value = get_environment_variable(test_var)
        print(f"Retrieved value: {retrieved_value}")
        
        # Check if values match
        if retrieved_value == test_value:
            print("✅ Variable values match")
            return True
        else:
            print("❌ Variable values do not match")
            return False
    
    def _test_shell_environment_commands(self):
        """Test shell environment command registration and execution."""
        from shell_environment import register_command
        
        print("Testing command registration and execution...")
        
        # Generate unique test command name
        test_cmd = f"test_cmd_{int(time.time())}"
        
        # Test value to verify command execution
        test_value = [False]
        
        # Define test command function
        def test_command(args):
            print(f"Test command executed with args: {args}")
            test_value[0] = True
            return 0
        
        # Register command
        try:
            register_command(test_cmd, test_command, "Test command")
            print(f"✅ Command registered: {test_cmd}")
        except Exception as e:
            print(f"❌ Command registration failed: {e}")
            return False
        
        # Import and get registry
        try:
            from shell_environment import registry
            
            # Check if command exists in registry
            if test_cmd in registry.command_handlers:
                print(f"✅ Command found in registry")
            else:
                print(f"❌ Command not found in registry")
                return False
            
            # Execute command
            handler_info = registry.get_command_handler(test_cmd)
            if handler_info:
                handler = handler_info["handler"]
                handler("test_args")
                
                if test_value[0]:
                    print("✅ Command executed successfully")
                    return True
                else:
                    print("❌ Command execution failed")
                    return False
            else:
                print("❌ Command handler not found")
                return False
            
        except Exception as e:
            print(f"❌ Error accessing registry: {e}")
            return False
    
    def _test_shell_environment_events(self):
        """Test shell environment event system."""
        from shell_environment import register_event_handler, emit_event
        
        print("Testing event system...")
        
        # Generate unique test event type
        test_event = f"test_event_{int(time.time())}"
        
        # Test value to verify event handling
        test_value = [False]
        test_data = {"test_key": f"test_value_{int(time.time())}"}
        
        # Define test event handler
        def test_handler(event_data):
            print(f"Test event handler called with data: {event_data}")
            if event_data.get("test_key") == test_data["test_key"]:
                test_value[0] = True
            return {"success": True, "handled": True}
        
        # Register event handler
        try:
            register_event_handler(test_event, test_handler, "test")
            print(f"✅ Event handler registered: {test_event}")
        except Exception as e:
            print(f"❌ Event handler registration failed: {e}")
            return False
        
        # Emit event
        try:
            results = emit_event(test_event, test_data)
            
            if results:
                print(f"✅ Event emitted with {len(results)} handler results")
            else:
                print(f"❌ Event emitted but no handlers responded")
                return False
            
            if test_value[0]:
                print("✅ Event handler executed successfully with correct data")
                return True
            else:
                print("❌ Event handler did not set test value")
                return False
            
        except Exception as e:
            print(f"❌ Error emitting event: {e}")
            return False
    
    def _test_shell_scripting_execution(self):
        """Test shell scripting basic execution."""
        import shell_scripting
        
        print("Testing shell script execution...")
        
        # Simple test script
        script = "result = 2 + 3"
        context = {}
        
        try:
            shell_scripting.execute_script(script, context)
            
            if "result" in context:
                print(f"Script execution result: {context['result']}")
                if context["result"] == 5:
                    print("✅ Script execution successful")
                    return True
                else:
                    print(f"❌ Script execution produced incorrect result: {context['result']}")
                    return False
            else:
                print("❌ Script execution did not set 'result' in context")
                return False
        except Exception as e:
            print(f"❌ Error executing script: {e}")
            return False
    
    def _test_shell_scripting_variables(self):
        """Test shell scripting variable functionality."""
        import shell_scripting
        
        print("Testing shell script variables...")
        
        # Test script with variables
        script = """
        x = 10
        y = 20
        sum = x + y
        product = x * y
        """
        
        context = {}
        
        try:
            shell_scripting.execute_script(script, context)
            
            expected = {
                "x": 10,
                "y": 20,
                "sum": 30,
                "product": 200
            }
            
            for var, value in expected.items():
                if var not in context:
                    print(f"❌ Variable '{var}' not set in context")
                    return False
                
                if context[var] != value:
                    print(f"❌ Variable '{var}' has incorrect value: {context[var]} (expected: {value})")
                    return False
            
            print("✅ All variables have correct values")
            return True
            
        except Exception as e:
            print(f"❌ Error testing variables: {e}")
            return False
    
    def _test_shell_scripting_conditions(self):
        """Test shell scripting conditional statements."""
        import shell_scripting
        
        print("Testing shell script conditional statements...")
        
        # Test script with if/else
        script = """
        x = 10
        
        if x > 5:
            result = "greater"
        else:
            result = "less or equal"
            
        if x < 0:
            negative = True
        elif x == 0:
            zero = True
        else:
            positive = True
        """
        
        context = {}
        
        try:
            shell_scripting.execute_script(script, context)
            
            if context.get("result") != "greater":
                print(f"❌ If statement failed, result = {context.get('result')}")
                return False
            
            if not context.get("positive", False):
                print("❌ If-elif-else statement failed")
                return False
            
            print("✅ Conditional statements working correctly")
            return True
            
        except Exception as e:
            print(f"❌ Error testing conditionals: {e}")
            return False
    
    def _test_shell_scripting_loops(self):
        """Test shell scripting loop functionality."""
        import shell_scripting
        
        print("Testing shell script loops...")
        
        # Test script with loops
        script = """
        # For loop
        sum = 0
        for i in range(1, 6):  # 1 to 5
            sum += i
            
        # While loop
        product = 1
        j = 1
        while j <= 5:
            product *= j
            j += 1
        """
        
        context = {}
        
        try:
            shell_scripting.execute_script(script, context)
            
            if context.get("sum") != 15:  # 1+2+3+4+5 = 15
                print(f"❌ For loop failed, sum = {context.get('sum')}")
                return False
            
            if context.get("product") != 120:  # 1*2*3*4*5 = 120
                print(f"❌ While loop failed, product = {context.get('product')}")
                return False
            
            print("✅ Loops working correctly")
            return True
            
        except Exception as e:
            print(f"❌ Error testing loops: {e}")
            return False
    
    def _test_fixwurx_commands_availability(self):
        """Test availability of FixWurx commands."""
        import fixwurx_commands
        
        print("Testing FixWurx commands availability...")
        
        try:
            # Get available commands
            commands = fixwurx_commands.get_commands()
            
            if not commands:
                print("❌ No FixWurx commands available")
                return False
            
            print(f"✅ Found {len(commands)} FixWurx commands:")
            for i, (cmd, _) in enumerate(list(commands.items())[:10], 1):  # Show first 10 commands
                print(f"  {i}. {cmd}")
            
            if len(commands) > 10:
                print(f"  ... and {len(commands) - 10} more commands")
            
            return True
        except Exception as e:
            print(f"❌ Error getting FixWurx commands: {e}")
            return False
    
    def _test_fixwurx_commands_execution(self):
        """Test execution of a simple FixWurx command."""
        import fixwurx_commands
        
        print("Testing FixWurx command execution...")
        
        try:
            # Get available commands
            commands = fixwurx_commands.get_commands()
            
            if not commands:
                print("❌ No FixWurx commands available")
                return False
            
            # Try to find a simple command to test
            test_cmd = None
            for cmd, handler in commands.items():
                if cmd in ["version", "help", "status", "info"]:
                    test_cmd = cmd
                    break
            
            if not test_cmd:
                # If no simple command found, use the first command
                test_cmd = list(commands.keys())[0]
            
            print(f"Testing execution of command: {test_cmd}")
            
            # Get command handler
            handler = commands[test_cmd]
            
            # Execute command
            result = handler("")
            
            print(f"Command execution result: {result}")
            print("✅ Command executed without errors")
            
            return True
        except Exception as e:
            print(f"❌ Error executing FixWurx command: {e}")
            return False
    
    def _test_system_resources(self):
        """Test system resource availability."""
        print("Testing system resource availability...")
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"CPU Usage: {cpu_percent:.1f}%")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            print(f"Memory Usage: {memory.percent:.1f}% ({memory.available / (1024 * 1024):.1f} MB available)")
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            print(f"Disk Usage: {disk.percent:.1f}% ({disk.free / (1024 * 1024 * 1024):.1f} GB free)")
            
            # All resources are available
            print("✅ All system resources are available")
            return True
        except Exception as e:
            print(f"❌ Error checking system resources: {e}")
            return False
    
    def _test_system_processes(self):
        """Test system processes."""
        print("Testing system processes...")
        
        try:
            # Get current process
            current_process = psutil.Process()
            
            # Get process info
            print(f"Current Process ID: {current_process.pid}")
            print(f"Process Name: {current_process.name()}")
            print(f"Process Status: {current_process.status()}")
            
            # Get child processes
            children = current_process.children(recursive=True)
            print(f"Child Processes: {len(children)}")
            
            # Get all Python processes
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            print(f"Python Processes: {len(python_processes)}")
            
            # Process test passed
            print("✅ Process information retrieved successfully")
            return True
        except Exception as e:
            print(f"❌ Error checking system processes: {e}")
            return False
    
    def do_exit(self, arg):
        """Exit the diagnostic console."""
        print("Exiting diagnostic console...")
        self.save_history()
        return True
    
    def do_quit(self, arg):
        """Exit the diagnostic console."""
        return self.do_exit(arg)
    
    def do_help(self, arg):
        """Display help information."""
        if arg:
            # Help for a specific command
            super().do_help(arg)
        else:
            # Custom help display
            print("\nInteractive Diagnostic Console Commands:")
            print("-" * 60)
            
            # Group commands by category
            categories = {
                "Diagnostic Commands": ["diagnose", "inspect", "fix"],
                "Monitoring Commands": ["metrics", "monitor"],
                "Testing Commands": ["test"],
                "General Commands": ["help", "exit", "quit"]
            }
            
            for category, commands in categories.items():
                print(f"\n{category}:")
                for cmd in commands:
                    doc = getattr(self, f"do_{cmd}").__doc__ or ""
                    doc = doc.strip().split("\n")[0]  # Get first line
                    print(f"  {cmd:<10} - {doc}")
            
            print("\nType 'help <command>' for detailed help on a specific command.")
    
    def start_console_command(self, args):
        """
        Handle the interactive_diagnostics command.
        
        Args:
            args: Command arguments
            
        Returns:
            Exit code
        """
        try:
            # Start the console
            self.cmdloop()
            return 0
        except KeyboardInterrupt:
            print("\nExiting interactive diagnostic console...")
            self.save_history()
            return 0
        except Exception as e:
            print(f"Error running interactive diagnostic console: {e}")
            logger.error(f"Error running interactive diagnostic console: {e}\n{traceback.format_exc()}")
            return 1


def register_interactive_diagnostics():
    """Register the interactive diagnostics command."""
    try:
        # Create console instance
        console = InteractiveDiagnosticConsole()
        logger.info("Interactive diagnostics registered")
        return console
    except Exception as e:
        logger.error(f"Failed to register interactive diagnostics: {e}\n{traceback.format_exc()}")
        return None


# Initialize interactive diagnostics if not in a test environment
if not any(arg.endswith('test.py') for arg in sys.argv):
    interactive_diagnostics_console = register_interactive_diagnostics()
